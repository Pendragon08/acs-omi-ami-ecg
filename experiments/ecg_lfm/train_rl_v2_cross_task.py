"""Cross-task Clinical RL v2A for the corrected ECG-LFM joint task.

This runner is deliberately a small extension of the audited R1-v2 runner:
the policy remains ``Categorical(ACS-4) x Bernoulli(OMI)``, every candidate is
reloaded from W0, and only the Fold0 inner-validation split is ever loaded.
The new terms are the NSTE/NSTEMI--OMI reward and sampled ranking
preservation.  Outer-test access is intentionally absent from this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES,
    acs_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint import _aggregate, _read_csv  # noqa: E402
from experiments.ecg_lfm.rl.adaptive_kl_controller import update_kl_controller  # noqa: E402
from experiments.ecg_lfm.rl.clinical_constraints import SafetyGuard  # noqa: E402
from experiments.ecg_lfm.rl.clinical_policy import (  # noqa: E402
    joint_log_probability,
    probabilities,
    sample as sample_policy,
)
from experiments.ecg_lfm.rl.clinical_reward import (  # noqa: E402
    RewardWeights,
    hard_reward,
    soft_f1_shaping,
)
from experiments.ecg_lfm.rl.cross_task_reward import (  # noqa: E402
    CrossTaskRewardConfig,
    cross_task_reward,
)
from experiments.ecg_lfm.rl.kl_regularization import factorized_kl  # noqa: E402
from experiments.ecg_lfm.rl.ranking_preservation import pairwise_ranking_losses  # noqa: E402
from experiments.ecg_lfm.rl.reference_guard import patient_bootstrap_margins  # noqa: E402
from models.ecg_lfm_adapter import load_finetuned_model, parameter_summary  # noqa: E402


def _targets(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    if raw.shape[1] != 5 or bool((raw[:, :4].sum(dim=1) != 1).any()):
        raise ValueError(f"corrected task requires [B,5] labels, got {tuple(raw.shape)}")
    return raw[:, :4].argmax(dim=1).long(), raw[:, 4].long()


def _move(sample: dict, device: torch.device) -> dict:
    from fairseq_signals.utils import utils

    return utils.move_to_cuda(sample) if device.type == "cuda" else sample


def _iterator(task, dataset, cfg: dict, epoch: int, shuffle: bool):
    return task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_signals=int(cfg["optimization"]["batch_size"]),
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=int(cfg["run"]["seed"]),
        num_shards=1,
        shard_id=0,
        num_workers=0,
        data_buffer_size=0,
        epoch=epoch,
    ).next_epoch_itr(shuffle=shuffle)


def _cache_logits(model, task, dataset, cfg: dict, device: torch.device, epoch: int) -> np.ndarray:
    model.eval()
    result = np.full((len(dataset), 5), np.nan, dtype=np.float32)
    with torch.no_grad():
        for batch in _iterator(task, dataset, cfg, epoch, False):
            if not batch:
                continue
            ids = batch["id"].cpu().numpy()
            batch = _move(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch["net_input"])
            result[ids] = model.get_logits(output, normalize=False).float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("reference inference left empty logits")
    return result


def _rows(logits: np.ndarray, segment_csv: Path, labels: np.ndarray) -> tuple[list[dict], np.ndarray]:
    records, record_logits = _aggregate(logits, _read_csv(segment_csv))
    p_acs = torch.softmax(torch.from_numpy(record_logits[:, :4]), dim=1).numpy()
    p_omi = torch.sigmoid(torch.from_numpy(record_logits[:, 4])).numpy()
    rows = []
    for record, acs_prob, omi_prob in zip(records, p_acs, p_omi):
        target = labels[int(record["master_idx"])]
        rows.append(
            {
                "record_id": record["record_id"],
                "patient_id": record["patient_id"],
                "master_idx": int(record["master_idx"]),
                "y_acs": int(np.argmax(target[:4])),
                "y_omi": int(target[4]),
                **{f"p_{name}": float(acs_prob[k]) for k, name in enumerate(ACS_CLASSES)},
                "p_omi": float(omi_prob),
            }
        )
    return rows, record_logits


def _binary_subset(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    if len(y) == 0:
        return {"n": 0, "status": "empty"}
    return binary_metrics(y, p, threshold)


def _reward_hacking(rows: list[dict], threshold: float) -> dict:
    """Report OMI positive-rate shifts and subset diagnostic metrics."""
    y_acs = np.asarray([int(row["y_acs"]) for row in rows], dtype=np.int64)
    y_omi = np.asarray([int(row["y_omi"]) for row in rows], dtype=np.int64)
    p_omi = np.asarray([float(row["p_omi"]) for row in rows], dtype=np.float64)
    pred = (p_omi >= threshold).astype(np.int64)
    result = {"global_pred_omi_rate": float(pred.mean()), "by_true_acs": {}}
    for index, name in enumerate(ACS_CLASSES):
        mask = y_acs == index
        result["by_true_acs"][name] = {
            "n": int(mask.sum()),
            "true_omi_rate": float(y_omi[mask].mean()) if mask.any() else None,
            "pred_omi_rate": float(pred[mask].mean()) if mask.any() else None,
            "omi_metrics": _binary_subset(y_omi[mask], p_omi[mask], threshold),
        }
    nste = np.isin(y_acs, (2, 3))
    result["nste"] = {
        "n": int(nste.sum()),
        "pred_omi_rate": float(pred[nste].mean()) if nste.any() else None,
        "true_omi_rate": float(y_omi[nste].mean()) if nste.any() else None,
        "omi_metrics": _binary_subset(y_omi[nste], p_omi[nste], threshold),
    }
    return result


def _reward_hacking_guard(candidate: dict, reference: dict, cfg: dict) -> dict:
    if not bool(cfg.get("enabled", True)):
        return {"pass": True, "enabled": False}
    c = candidate["nste"]
    r = reference["nste"]
    c_rate = float(c["pred_omi_rate"] or 0.0)
    r_rate = float(r["pred_omi_rate"] or 0.0)
    c_ppv = float(c["omi_metrics"].get("ppv", 0.0))
    r_ppv = float(r["omi_metrics"].get("ppv", 0.0))
    ratio = c_rate / max(r_rate, 1e-8)
    rate_explosion = ratio > float(cfg.get("max_prediction_rate_ratio", 1.5))
    ppv_drop = r_ppv - c_ppv
    ppv_collapse = ppv_drop > float(cfg.get("min_ppv_drop", 0.05))
    return {
        "pass": bool(not (rate_explosion and ppv_collapse)),
        "enabled": True,
        "candidate_nste_pred_omi_rate": c_rate,
        "reference_nste_pred_omi_rate": r_rate,
        "candidate_to_reference_rate_ratio": float(ratio),
        "rate_explosion": bool(rate_explosion),
        "candidate_nste_ppv": c_ppv,
        "reference_nste_ppv": r_ppv,
        "ppv_drop": float(ppv_drop),
        "ppv_collapse": bool(ppv_collapse),
        "max_prediction_rate_ratio": float(cfg.get("max_prediction_rate_ratio", 1.5)),
        "min_ppv_drop": float(cfg.get("min_ppv_drop", 0.05)),
    }


def _bundle(
    rows: list[dict],
    record_logits: np.ndarray,
    reference_record_logits: np.ndarray,
    omi_threshold: float | None = None,
) -> dict:
    y_acs = np.asarray([row["y_acs"] for row in rows], dtype=np.int64)
    p_acs = np.asarray([[row[f"p_{name}"] for name in ACS_CLASSES] for row in rows])
    y_omi = np.asarray([row["y_omi"] for row in rows], dtype=np.int64)
    p_omi = np.asarray([row["p_omi"] for row in rows])
    if omi_threshold is None:
        omi_threshold, details = select_omi_threshold(y_omi, p_omi, "youden_j")
    else:
        _, details = select_omi_threshold(y_omi, p_omi, "fixed_0.5")
        details = {"method": "frozen_global_inner_validation_threshold", "threshold": float(omi_threshold)}
    if omi_threshold is None:
        raise RuntimeError("inner-validation Youden threshold infeasible")
    kl = factorized_kl(torch.from_numpy(record_logits), torch.from_numpy(reference_record_logits))
    nste = np.isin(y_acs, (2, 3))
    nstemi = y_acs == 2
    ua = y_acs == 3
    return {
        "n": int(len(rows)),
        "acs": acs_metrics(y_acs, p_acs),
        "omi": binary_metrics(y_omi, p_omi, float(omi_threshold)),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs": _binary_subset(y_omi[nste], p_omi[nste], float(omi_threshold)),
        "omi_among_nstemi": _binary_subset(y_omi[nstemi], p_omi[nstemi], float(omi_threshold)),
        "omi_among_ua": _binary_subset(y_omi[ua], p_omi[ua], float(omi_threshold)),
        "calibration": {
            "acs": multiclass_calibration_metrics(y_acs, p_acs),
            "omi": binary_calibration_metrics(y_omi, p_omi),
        },
        "reward_hacking": _reward_hacking(rows, float(omi_threshold)),
        "omi_threshold": float(omi_threshold),
        "omi_threshold_detail": details,
        "kl": {name: float(value.mean()) for name, value in kl.items()},
    }


def _guard_view(bundle: dict) -> dict[str, float]:
    return {
        "acs_macro_auprc": float(bundle["acs"]["macro_auprc_ovr"]),
        "omi_auprc": float(bundle["omi"]["auprc"]),
        "omi_ppv": float(bundle["omi"]["ppv"]),
        "omi_specificity": float(bundle["omi"]["specificity"]),
        "stemi_f1": float(bundle["acs"]["per_class"]["STEMI"]["f1"]),
        "acs_brier": float(bundle["calibration"]["acs"]["multiclass_brier"]),
        "omi_brier": float(bundle["calibration"]["omi"]["brier"]),
    }


def _utility(bundle: dict, cfg: dict) -> tuple[float, dict]:
    values = {
        "nstemi_f1": float(bundle["acs"]["per_class"]["NSTEMI"]["f1"]),
        "omi_f1": float(bundle["omi"]["f1"]),
        "nste_omi_auprc": float(bundle["omi_among_nste_acs"]["auprc"]),
    }
    terms = cfg["selection"]["terms"]
    return float(sum(float(terms[name]) * values[name] for name in terms)), values


def _safe_payload(value):
    if isinstance(value, dict):
        return {str(k): _safe_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_payload(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_safe_payload(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe_payload(payload), ensure_ascii=False, allow_nan=False) + "\n")


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_safe_payload(rows))


def _head_only_audit(model) -> dict:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    invalid = [name for name in names if not name.startswith("proj.")]
    if invalid or set(names) != {"proj.weight", "proj.bias"}:
        raise RuntimeError(f"Head-only invariant failed; trainable={names[:20]}")
    return {"trainable_names": names, **parameter_summary(model)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--target_kl", type=float, required=True)
    parser.add_argument("--ranking_coefficient", type=float, required=True)
    parser.add_argument("--run_suffix", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_updates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("cross-task RL v2 requires exactly one CUDA_VISIBLE_DEVICES value")
    if not torch.cuda.is_available():
        raise RuntimeError("cross-task RL v2 requires CUDA")
    if args.learning_rate <= 0 or args.ranking_coefficient < 0:
        raise ValueError("learning rate must be positive and ranking coefficient non-negative")
    if float(args.target_kl) not in (0.005, 0.01):
        raise ValueError("target_KL must be 0.005 or 0.01")
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if int(cfg["run"]["fold"]) != 0:
        raise ValueError("cross-task v2A is registered for nested Fold0 only")
    if cfg["policy"]["trainable"] != "head_only" or int(cfg["policy"]["update_epochs"]) != 1:
        raise ValueError("v2A requires head_only and one update epoch")
    suffix = str(args.run_suffix)
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in suffix):
        raise ValueError("run_suffix contains unsafe characters")
    cfg["optimization"]["learning_rate"] = float(args.learning_rate)
    cfg["kl"]["target_kl"] = float(args.target_kl)
    cfg["ranking"]["coefficient"] = float(args.ranking_coefficient)
    if args.max_updates is not None:
        if args.max_updates < 1:
            raise ValueError("max_updates must be positive")
        cfg["optimization"]["max_updates"] = int(args.max_updates)
    run_name = str(cfg["run"]["name"]) + suffix
    if args.smoke:
        run_name = "smoke_" + run_name
    seed = int(cfg["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    manifest_dir = ROOT / cfg["run"]["manifest_root"] / "fold0"
    checkpoint = ROOT / cfg["run"]["checkpoint"]
    output = ROOT / cfg["run"]["output_root"] / run_name / "fold0"
    output.mkdir(parents=True, exist_ok=True)
    required = [
        checkpoint,
        manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv",
        manifest_dir / "train_segments.csv",
        manifest_dir / "valid_segments.csv",
        manifest_dir.parent / "y_joint_5.npy",
        manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    started = time.time()

    model, saved_cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.proj.parameters():
        parameter.requires_grad = True
    trainable = _head_only_audit(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial_trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    task.load_dataset("train", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=True)
    task.load_dataset("valid", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=False)
    train_dataset, valid_dataset = task.dataset("train"), task.dataset("valid")
    labels = np.load(manifest_dir.parent / "y_joint_5.npy", mmap_mode="r")
    reference_train = _cache_logits(model, task, train_dataset, cfg, device, 1)
    reference_valid = _cache_logits(model, task, valid_dataset, cfg, device, 1)
    reference_rows, reference_record_logits = _rows(
        reference_valid, manifest_dir / "valid_segments.csv", labels
    )
    bootstrap_margins, bootstrap = patient_bootstrap_margins(
        reference_rows,
        int(cfg["safety"]["bootstrap_replicates"]),
        seed,
        float(cfg["safety"]["minimum_absolute_margin"]),
    )
    reference_threshold = float(bootstrap["reference"]["omi_threshold"])
    guard = SafetyGuard.from_reference(
        bootstrap["reference"], bootstrap_margins, max_kl=float(cfg["kl"]["max_inner_validation_mean"])
    )
    baseline_bundle = _bundle(
        reference_rows, reference_record_logits, reference_record_logits, reference_threshold
    )
    baseline_utility, baseline_terms = _utility(baseline_bundle, cfg)
    baseline_hacking = baseline_bundle["reward_hacking"]
    provenance = {
        "status": "running",
        "started_at_unix": started,
        "run_name": run_name,
        "task": "factorized Categorical(ACS 4-way) x Bernoulli(OMI)",
        "algorithm": "cross-task update-budget clipped contextual policy optimization",
        "trainable_policy": "classification head only; LoRA and base encoder frozen",
        "outer_test_loaded": False,
        "config": cfg,
        "config_sha256": sha256(cfg_path),
        "w0_checkpoint": str(checkpoint),
        "w0_checkpoint_sha256": sha256(checkpoint),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "cuda_visible_devices": visible,
        "trainable": trainable,
        "reference_policy": "exact W0 logits cached before update 1",
        "reference_and_policy_equal_at_init": True,
        "cross_task_reward": cfg["reward"]["cross_task"],
        "ranking_preservation": cfg["ranking"],
        "reward_hacking_guard": cfg["reward_hacking"],
        "hard_reward_scale": float(cfg["reward"]["hard_scale"]),
        "update_budget": int(cfg["optimization"]["max_updates"]),
        "validation_interval_updates": int(cfg["optimization"]["eval_interval_updates"]),
        "target_kl": float(args.target_kl),
        "hard_kl_threshold": float(args.target_kl) * float(cfg["kl"]["hard_stop_multiplier"]),
        "ranking_coefficient": float(args.ranking_coefficient),
        "reward_interpretation": cfg["metadata"]["interpretation"],
        "smoke": bool(args.smoke),
    }
    _write(output / "launch_provenance.json", provenance)
    _write(
        output / "reference_safety_guard.json",
        {
            **bootstrap,
            "margins": bootstrap_margins,
            "lower_bounds": guard.lower_bounds,
            "upper_bounds": guard.upper_bounds,
            "max_kl": guard.max_kl,
            "baseline_bundle": baseline_bundle,
            "baseline_utility": baseline_utility,
            "baseline_utility_terms": baseline_terms,
            "baseline_reward_hacking": baseline_hacking,
        },
    )

    optimizer = torch.optim.Adam(
        [model.proj.weight, model.proj.bias],
        lr=float(args.learning_rate),
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["optimization"]["fp16"]))
    reward_weights = RewardWeights.from_mapping(cfg["reward"])
    cross_config = CrossTaskRewardConfig.from_mapping(cfg["reward"]["cross_task"])
    clip_epsilon = float(cfg["policy"]["clip_epsilon"])
    kl_coefficient = float(cfg["kl"]["initial_coefficient"])
    target_kl = float(args.target_kl)
    hard_threshold = target_kl * float(cfg["kl"]["hard_stop_multiplier"])
    hard_patience = int(cfg["kl"]["hard_stop_patience"])
    max_updates = int(cfg["optimization"]["max_updates"])
    eval_interval = int(cfg["optimization"]["eval_interval_updates"])
    max_wall_seconds = float(cfg["optimization"].get("max_wall_seconds", 7920))
    log_path = output / "train_log.jsonl"
    ranking_history: list[dict] = []
    cross_history: list[dict] = []
    kl_history: list[dict] = []
    validation_updates: list[int] = []
    best = None
    updates = 0
    hard_kl_streak = 0
    status = "completed_update_budget"
    stop_reason = None
    epoch = 1
    iterator = _iterator(task, train_dataset, cfg, epoch, True)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    while updates < max_updates and time.time() - started < max_wall_seconds:
        raw_batch = next(iterator, None)
        if raw_batch is None:
            epoch += 1
            iterator = _iterator(task, train_dataset, cfg, epoch, True)
            continue
        if not raw_batch:
            continue
        ids = raw_batch["id"].cpu().numpy()
        batch = _move(raw_batch, device)
        reference_logits = torch.from_numpy(reference_train[ids]).to(device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])
        ):
            old_output = model(**batch["net_input"])
            old_logits = model.get_logits(old_output, normalize=False).float()
            action = sample_policy(old_logits)
            acs_target, omi_target = _targets(model.get_targets(batch, old_output))
            base_reward, base_parts = hard_reward(
                action.acs_action, action.omi_action, acs_target, omi_target, reward_weights
            )
            cross_reward, cross_parts = cross_task_reward(
                action.acs_action, action.omi_action, acs_target, omi_target, cross_config
            )
            reward = float(cfg["reward"]["hard_scale"]) * base_reward + cross_reward
            advantage = reward - reward.mean()
            old_log_probability = action.joint_log_probability.detach()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])
        ):
            current_output = model(**batch["net_input"])
            current_logits = model.get_logits(current_output, normalize=False).float()
            log_probability = joint_log_probability(
                current_logits, action.acs_action, action.omi_action
            )
            ratio = torch.exp(log_probability - old_log_probability)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_objective = torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
            current_acs, current_omi = probabilities(current_logits)
            shaping, shaping_parts = soft_f1_shaping(
                current_acs, current_omi, acs_target, omi_target, reward_weights
            )
            kl_values = factorized_kl(current_logits, reference_logits)
            batch_kl = kl_values["total"].mean()
            rank_generator = torch.Generator()
            rank_generator.manual_seed(seed + updates * 1009 + 17)
            rank = pairwise_ranking_losses(
                current_logits,
                acs_target,
                omi_target,
                cfg["ranking"],
                generator=rank_generator,
            )
            ranking_loss = rank["rank_loss"]
            loss = (
                -(policy_objective + shaping - kl_coefficient * batch_kl)
                + float(args.ranking_coefficient) * ranking_loss
            )
        if not bool(torch.isfinite(loss)):
            status = "nonfinite_rollback"
            stop_reason = "nonfinite_rl_loss"
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [model.proj.weight, model.proj.bias], float(cfg["optimization"]["max_grad_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            status = "nonfinite_rollback"
            stop_reason = "nonfinite_gradient"
            optimizer.zero_grad(set_to_none=True)
            break
        scaler.step(optimizer)
        scaler.update()
        updates += 1
        batch_kl_value = float(batch_kl.detach())
        if batch_kl_value > hard_threshold:
            hard_kl_streak += 1
        else:
            hard_kl_streak = 0
        previous_kl_coefficient = kl_coefficient
        kl_coefficient = update_kl_controller(kl_coefficient, batch_kl_value, target_kl, cfg["kl"])
        update_detail = {
            "event": "update",
            "update": updates,
            "epoch": epoch,
            "loss": float(loss.detach()),
            "policy_objective": float(policy_objective.detach()),
            "soft_f1": float(shaping.detach()),
            "reward_mean": float(reward.mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "ratio_mean": float(ratio.mean()),
            "clip_fraction": float(
                ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).float().mean()
            ),
            "kl_acs": float(kl_values["acs"].mean()),
            "kl_omi": float(kl_values["omi"].mean()),
            "kl_total": batch_kl_value,
            "target_kl": target_kl,
            "kl_coefficient_before": previous_kl_coefficient,
            "kl_coefficient": kl_coefficient,
            "hard_kl_threshold": hard_threshold,
            "hard_kl_streak": hard_kl_streak,
            "grad_norm": float(grad_norm),
            "acs_entropy": float(action.acs_entropy.mean()),
            "omi_entropy": float(action.omi_entropy.mean()),
            "actions": {str(k): int((action.joint_action == k).sum()) for k in range(8)},
            "reward_components": {
                name: float(value.mean()) for name, value in base_parts.items()
            },
            "cross_reward_components": {
                name: float(value.mean()) for name, value in cross_parts.items()
            },
            "soft_f1_components": {name: float(value) for name, value in shaping_parts.items()},
            "ranking_coefficient": float(args.ranking_coefficient),
            "ranking": {
                key: (float(value.detach()) if torch.is_tensor(value) else value)
                for key, value in rank.items()
            },
        }
        _append_jsonl(log_path, update_detail)
        ranking_history.append(
            {
                "update": updates,
                "rank_loss": float(ranking_loss.detach()),
                "acs_rank_loss": float(rank["acs_rank_loss"].detach()),
                "omi_rank_loss": float(rank["omi_rank_loss"].detach()),
                "pairwise_accuracy": float(rank["pairwise_accuracy"]),
                "pair_count": int(rank["pair_count"]),
                "ranking_coefficient": float(args.ranking_coefficient),
            }
        )
        cross_history.append(
            {
                "update": updates,
                "cross_reward_mean": float(cross_reward.mean()),
                "cross_reward_std": float(cross_reward.std(unbiased=False)),
                "cross_nstemi_omi_mean": float(cross_parts["cross_nstemi_omi"].mean()),
                "cross_ua_omi_mean": float(cross_parts["cross_ua_omi"].mean()),
            }
        )
        kl_history.append(
            {
                "update": updates,
                "batch_kl": batch_kl_value,
                "target_kl": target_kl,
                "coefficient_before": previous_kl_coefficient,
                "coefficient_after": kl_coefficient,
                "hard_threshold": hard_threshold,
                "hard_streak": hard_kl_streak,
            }
        )
        if hard_kl_streak >= hard_patience:
            status = "hard_kl_stop_rollback"
            stop_reason = "hard_kl_threshold_exceeded_consecutively"
            break
        should_validate = updates % eval_interval == 0 or (args.smoke and updates == max_updates)
        if should_validate:
            model.eval()
            valid_logits = _cache_logits(model, task, valid_dataset, cfg, device, epoch)
            valid_rows, valid_record_logits = _rows(
                valid_logits, manifest_dir / "valid_segments.csv", labels
            )
            bundle = _bundle(
                valid_rows,
                valid_record_logits,
                reference_record_logits,
                reference_threshold,
            )
            safety = guard.evaluate(_guard_view(bundle), bundle["kl"]["total"])
            hacking = _reward_hacking_guard(
                bundle["reward_hacking"], baseline_hacking, cfg["reward_hacking"]
            )
            safety["reward_hacking_guard"] = hacking
            safety["pass"] = bool(safety["pass"] and hacking["pass"])
            utility, utility_terms = _utility(bundle, cfg)
            eval_result = {
                "event": "inner_validation",
                "update": updates,
                "clinical_utility": utility,
                "clinical_utility_terms": utility_terms,
                "safety": safety,
                "metrics": bundle,
                "kl_controller": {
                    "target_kl": target_kl,
                    "coefficient": kl_coefficient,
                    "hard_streak": hard_kl_streak,
                },
                "ranking": ranking_history[-1],
                "elapsed_seconds": float(time.time() - started),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            _append_jsonl(log_path, eval_result)
            _write(output / f"inner_validation_update{updates}.json", eval_result)
            validation_updates.append(updates)
            if safety["pass"] and (
                best is None or utility > best["utility"]
            ):
                best = {
                    "update": updates,
                    "utility": utility,
                    "utility_terms": utility_terms,
                    "metrics": bundle,
                    "safety": safety,
                    "state": {
                        name: model.state_dict()[name].detach().cpu().clone()
                        for name in trainable_names
                    },
                }
            model.train()
        if updates >= max_updates:
            break

    if updates >= max_updates and stop_reason is None:
        status = "completed_update_budget"
    if time.time() - started >= max_wall_seconds and stop_reason is None:
        status = "wall_budget_stop"
        stop_reason = "max_wall_seconds"
    if best is None and status == "completed_update_budget":
        status = "no_safety_passing_candidate"
    rollback_performed = status in {"hard_kl_stop_rollback", "nonfinite_rollback"}
    rollback_state = best["state"] if best is not None else initial_trainable_state
    if rollback_performed:
        for name, value in rollback_state.items():
            dict(model.named_parameters())[name].data.copy_(value.to(device))

    _write_history(output / "ranking_history.csv", ranking_history)
    _write_history(output / "cross_reward_history.csv", cross_history)
    _write_history(output / "kl_controller_history.csv", kl_history)
    summary = {
        "status": status,
        "stop_reason": stop_reason,
        "updates": updates,
        "outer_test_loaded": False,
        "completed_validation_updates": validation_updates,
        "elapsed_seconds": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "baseline_inner_validation": baseline_bundle,
        "baseline_utility": baseline_utility,
        "baseline_utility_terms": baseline_terms,
        "best": None if best is None else {key: value for key, value in best.items() if key != "state"},
        "rollback": {
            "performed": rollback_performed,
            "source": "last_safety_passing_state_or_W0_head",
            "state_saved": False,
        },
        "ranking_coefficient": float(args.ranking_coefficient),
        "cross_task_reward_config": cfg["reward"]["cross_task"],
        "reward_hacking_reference": baseline_hacking,
        "smoke": bool(args.smoke),
    }
    _write(output / "training_summary.json", summary)
    if best is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name, value in best["state"].items():
            payload["model"][name] = value
        payload["rl_post_training"] = {
            "study": "Cross-task Clinical RL v2A",
            "trainable_policy": "head_only",
            "source_checkpoint_sha256": sha256(checkpoint),
            "selected_update": best["update"],
            "inner_validation_safety": best["safety"],
            "outer_test_loaded": False,
            "config": cfg,
        }
        temporary = output / "checkpoint_best.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, output / "checkpoint_best.pt")
        summary["checkpoint_saved"] = True
        summary["rollback"]["state_saved"] = True
        _write(output / "frozen_candidate_config.json", {
            "status": "candidate_frozen_before_outer_test",
            "outer_test_evaluations": 0,
            "checkpoint_sha256": sha256(output / "checkpoint_best.pt"),
            "selected_update": best["update"],
            "config": cfg,
            "safety": best["safety"],
            "inner_validation_metrics": best["metrics"],
        })
        _write(output / "training_summary.json", summary)
    provenance.update({"status": status, "finished_at_unix": time.time(), "outer_test_loaded": False})
    _write(output / "launch_provenance.json", provenance)
    print(json.dumps({
        "status": status,
        "run_name": run_name,
        "updates": updates,
        "best": None if best is None else best["update"],
        "outer_test_loaded": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
