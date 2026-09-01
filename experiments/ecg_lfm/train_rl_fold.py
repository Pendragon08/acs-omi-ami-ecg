"""Fold0 direct RL post-training from the exact LoRA Joint W0 checkpoint.

This implements one-step clipped contextual policy optimization, not sequence
PPO/GRPO/DAPO. It deliberately never loads the outer-test split.
"""
from __future__ import annotations

import argparse
import json
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
    ACS_CLASSES, acs_metrics, binary_calibration_metrics, binary_metrics,
    multiclass_calibration_metrics, nstemi_vs_ua_metrics,
    select_omi_threshold, sha256,
)
from experiments.ecg_lfm.evaluate_joint import _aggregate, _read_csv  # noqa: E402
from experiments.ecg_lfm.rl.clinical_constraints import SafetyGuard  # noqa: E402
from experiments.ecg_lfm.rl.clinical_policy import (  # noqa: E402
    joint_log_probability, probabilities, sample as sample_policy,
)
from experiments.ecg_lfm.rl.clinical_reward import (  # noqa: E402
    RewardWeights, hard_reward, soft_f1_shaping,
)
from experiments.ecg_lfm.rl.kl_regularization import factorized_kl  # noqa: E402
from experiments.ecg_lfm.rl.reference_guard import patient_bootstrap_margins  # noqa: E402
from models.ecg_lfm_adapter import load_finetuned_model, parameter_summary  # noqa: E402


def _targets(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    if raw.shape[1] != 5 or bool((raw[:, :4].sum(dim=1) != 1).any()):
        raise ValueError(f"RL requires corrected [B,5] labels, got {tuple(raw.shape)}")
    return raw[:, :4].argmax(dim=1).long(), raw[:, 4].long()


def _move(sample: dict, device: torch.device) -> dict:
    from fairseq_signals.utils import utils
    return utils.move_to_cuda(sample) if device.type == "cuda" else sample


def _iterator(task, dataset, cfg: dict, epoch: int, shuffle: bool):
    return task.get_batch_iterator(
        dataset=dataset, max_tokens=None,
        max_signals=int(cfg["optimization"]["batch_size"]),
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=int(cfg["run"]["seed"]), num_shards=1, shard_id=0,
        num_workers=0, data_buffer_size=0, epoch=epoch,
    ).next_epoch_itr(shuffle=shuffle)


def _cache_logits(model, task, dataset, cfg: dict, device: torch.device, epoch: int = 1) -> np.ndarray:
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
        raise RuntimeError("inference did not fill every manifest segment")
    return result


def _validation_rows(logits: np.ndarray, segment_csv: Path, labels: np.ndarray) -> tuple[list[dict], np.ndarray]:
    records, record_logits = _aggregate(logits, _read_csv(segment_csv))
    p_acs = torch.softmax(torch.from_numpy(record_logits[:, :4]), dim=1).numpy()
    p_omi = torch.sigmoid(torch.from_numpy(record_logits[:, 4])).numpy()
    rows = []
    for record, acs_prob, omi_prob in zip(records, p_acs, p_omi):
        target = labels[int(record["master_idx"])]
        rows.append({
            "record_id": record["record_id"], "patient_id": record["patient_id"],
            "y_acs": int(np.argmax(target[:4])), "y_omi": int(target[4]),
            **{f"p_{name}": float(acs_prob[k]) for k, name in enumerate(ACS_CLASSES)},
            "p_omi": float(omi_prob),
        })
    return rows, record_logits


def _metric_bundle(rows: list[dict], record_logits: np.ndarray, reference_logits: np.ndarray) -> dict:
    y_acs = np.asarray([row["y_acs"] for row in rows], dtype=np.int64)
    p_acs = np.asarray([[row[f"p_{name}"] for name in ACS_CLASSES] for row in rows])
    y_omi = np.asarray([row["y_omi"] for row in rows], dtype=np.int64)
    p_omi = np.asarray([row["p_omi"] for row in rows])
    threshold, threshold_detail = select_omi_threshold(y_omi, p_omi, "youden_j")
    if threshold is None:
        raise RuntimeError("inner-validation Youden threshold is infeasible")
    acs = acs_metrics(y_acs, p_acs)
    omi = binary_metrics(y_omi, p_omi, threshold)
    nste = np.isin(y_acs, (2, 3))
    kl = factorized_kl(torch.from_numpy(record_logits), torch.from_numpy(reference_logits))
    return {
        "acs": acs, "omi": omi,
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs": binary_metrics(y_omi[nste], p_omi[nste], threshold),
        "calibration": {
            "acs": multiclass_calibration_metrics(y_acs, p_acs),
            "omi": binary_calibration_metrics(y_omi, p_omi),
        },
        "omi_threshold": float(threshold), "omi_threshold_detail": threshold_detail,
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


def _trainable_audit(model) -> dict:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    invalid = [name for name in names if "lora_A" not in name and "lora_B" not in name and not name.startswith("proj.")]
    if invalid:
        raise RuntimeError(f"base encoder parameters unexpectedly trainable: {invalid[:10]}")
    if not any("lora_A" in name for name in names) or not any(name.startswith("proj.") for name in names):
        raise RuntimeError("RL requires LoRA matrices plus classification head")
    return {"trainable_names": names, **parameter_summary(model)}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_updates", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--kl_coefficient", type=float)
    parser.add_argument("--hard_reward_scale", type=float, default=1.0)
    parser.add_argument("--run_suffix", default="")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("RL runs require exactly one CUDA_VISIBLE_DEVICES value")
    if not torch.cuda.is_available():
        raise RuntimeError("RL post-training requires CUDA")
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if int(cfg["run"]["fold"]) != 0:
        raise ValueError("direct-RL gate is pre-registered for nested Fold0 only")
    if int(cfg["optimization"].get("update_frequency", 1)) != 1:
        raise ValueError("current clipped contextual runner requires update_frequency=1")
    if args.learning_rate is not None: cfg["optimization"]["learning_rate"] = args.learning_rate
    if args.kl_coefficient is not None: cfg["kl"]["coefficient"] = args.kl_coefficient
    if args.hard_reward_scale < 0: raise ValueError("hard reward scale must be nonnegative")

    seed = int(cfg["run"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    manifest_dir = ROOT / cfg["run"]["manifest_root"] / "fold0"
    checkpoint = ROOT / cfg["run"]["checkpoint"]
    safe_suffix = str(args.run_suffix).strip()
    if safe_suffix and any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in safe_suffix):
        raise ValueError("run_suffix may contain only letters, digits, underscore and hyphen")
    run_name = str(cfg["run"]["name"]) + ("_smoke" if args.smoke else "") + safe_suffix
    output = ROOT / cfg["run"]["output_root"] / run_name / "fold0"
    output.mkdir(parents=True, exist_ok=True)
    required = [checkpoint, manifest_dir / "train.tsv", manifest_dir / "valid.tsv",
                manifest_dir / "train_segments.csv", manifest_dir / "valid_segments.csv",
                manifest_dir.parent / "y_joint_5.npy", manifest_dir / "split_provenance.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("; ".join(missing))

    started = time.time()
    model, saved_cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.to(device).eval()
    trainable = _trainable_audit(model)
    task.load_dataset("train", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=True)
    task.load_dataset("valid", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=False)
    train_dataset, valid_dataset = task.dataset("train"), task.dataset("valid")
    labels = np.load(manifest_dir.parent / "y_joint_5.npy", mmap_mode="r")

    reference_train = _cache_logits(model, task, train_dataset, cfg, device)
    reference_valid = _cache_logits(model, task, valid_dataset, cfg, device)
    reference_rows, reference_record_logits = _validation_rows(reference_valid, manifest_dir / "valid_segments.csv", labels)
    margins, bootstrap = patient_bootstrap_margins(
        reference_rows, int(cfg["safety"]["bootstrap_replicates"]), seed,
        float(cfg["safety"]["minimum_absolute_margin"]),
    )
    guard = SafetyGuard.from_reference(
        bootstrap["reference"], margins, float(cfg["kl"]["max_inner_validation_mean"])
    )
    baseline_bundle = _metric_bundle(reference_rows, reference_record_logits, reference_record_logits)
    baseline_utility, baseline_terms = _utility(baseline_bundle, cfg)
    initial_kl = factorized_kl(torch.from_numpy(reference_valid), torch.from_numpy(reference_valid))["total"]
    if float(initial_kl.abs().max()) > 1e-7:
        raise RuntimeError("policy/reference KL must be zero at initialization")

    provenance = {
        "status": "running", "smoke": bool(args.smoke), "started_at_unix": started,
        "task": "factorized Categorical(ACS 4-way) x Bernoulli(OMI)",
        "algorithm": "one-step clipped contextual policy optimization (not PPO/GRPO/DAPO)",
        "outer_test_loaded": False, "config": cfg, "config_path": str(cfg_path),
        "config_sha256": sha256(cfg_path), "w0_checkpoint": str(checkpoint),
        "w0_checkpoint_sha256": sha256(checkpoint),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "cuda_visible_devices": visible, "trainable": trainable,
        "reference_policy": "exact W0 logits cached before first optimizer step",
        "reference_and_policy_equal_at_init": True, "initial_max_abs_kl": float(initial_kl.abs().max()),
        "reward_interpretation": cfg["metadata"]["interpretation"],
        "hard_reward_scale": float(args.hard_reward_scale),
    }
    _write(output / "launch_provenance.json", provenance)
    _write(output / "reference_safety_guard.json", {
        **bootstrap, "margins": margins, "lower_bounds": guard.lower_bounds,
        "upper_bounds": guard.upper_bounds, "max_kl": guard.max_kl,
        "baseline_bundle": baseline_bundle, "baseline_utility": baseline_utility,
        "baseline_utility_terms": baseline_terms,
    })

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["optimization"]["learning_rate"]), betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["optimization"]["fp16"]))
    reward_weights = RewardWeights.from_mapping(cfg["reward"])
    update_epochs = int(cfg["policy"]["update_epochs"])
    clip_epsilon = float(cfg["policy"]["clip_epsilon"])
    kl_coefficient = float(cfg["kl"]["coefficient"]) if cfg["kl"]["enabled"] else 0.0
    max_updates = args.max_updates if args.max_updates is not None else (2 if args.smoke else None)
    max_epochs = 1 if args.smoke else int(cfg["optimization"]["max_epochs"])
    log_path = output / "train_log.jsonl"
    best = None; updates = 0; stale = 0; budget_exhausted = False
    critical_safety_failure = False
    max_wall_seconds = float(cfg["optimization"].get("max_wall_seconds", 7920))
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        model.eval()
        for raw_batch in _iterator(task, train_dataset, cfg, epoch, True):
            if time.time() - started >= max_wall_seconds:
                budget_exhausted = True
                break
            if not raw_batch: continue
            ids = raw_batch["id"].cpu().numpy()
            batch = _move(raw_batch, device)
            reference_logits = torch.from_numpy(reference_train[ids]).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])):
                old_output = model(**batch["net_input"])
                old_logits = model.get_logits(old_output, normalize=False).float()
                action = sample_policy(old_logits)
                acs_target, omi_target = _targets(model.get_targets(batch, old_output))
                reward, reward_parts = hard_reward(action.acs_action, action.omi_action,
                    acs_target, omi_target, reward_weights)
                reward = reward * float(args.hard_reward_scale)
                advantage = reward - reward.mean()
                old_log_probability = action.joint_log_probability.detach()
            detail = None
            for _ in range(update_epochs):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])):
                    out = model(**batch["net_input"])
                    logits = model.get_logits(out, normalize=False).float()
                    log_probability = joint_log_probability(logits, action.acs_action, action.omi_action)
                    ratio = torch.exp(log_probability - old_log_probability)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantage
                    policy_objective = torch.minimum(unclipped, clipped).mean()
                    current_acs, current_omi = probabilities(logits)
                    shaping, shaping_parts = soft_f1_shaping(current_acs, current_omi,
                        acs_target, omi_target, reward_weights)
                    kl_values = factorized_kl(logits, reference_logits)
                    loss = -(policy_objective + shaping - kl_coefficient * kl_values["total"].mean())
                if not bool(torch.isfinite(loss)): raise FloatingPointError("non-finite RL loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    float(cfg["optimization"]["max_grad_norm"]))
                scaler.step(optimizer); scaler.update()
                detail = {
                    "loss": float(loss.detach()), "policy_objective": float(policy_objective.detach()),
                    "soft_f1": float(shaping.detach()), "reward_mean": float(reward.mean()),
                    "reward_std": float(reward.std(unbiased=False)), "ratio_mean": float(ratio.mean()),
                    "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                    "clip_fraction": float(((ratio < 1-clip_epsilon) | (ratio > 1+clip_epsilon)).float().mean()),
                    "kl_acs": float(kl_values["acs"].mean()), "kl_omi": float(kl_values["omi"].mean()),
                    "kl_total": float(kl_values["total"].mean()), "grad_norm": float(grad_norm),
                    "acs_entropy": float(action.acs_entropy.mean()), "omi_entropy": float(action.omi_entropy.mean()),
                    "actions": {str(k): int((action.joint_action == k).sum()) for k in range(8)},
                    "reward_components": {name: float(value.mean()) for name, value in reward_parts.items()},
                    "soft_f1_components": {name: float(value) for name, value in shaping_parts.items()},
                }
            updates += 1; detail.update({"event": "update", "epoch": epoch, "update": updates})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
            if max_updates is not None and updates >= max_updates: break

        valid_logits = _cache_logits(model, task, valid_dataset, cfg, device, epoch)
        valid_rows, valid_record_logits = _validation_rows(valid_logits, manifest_dir / "valid_segments.csv", labels)
        bundle = _metric_bundle(valid_rows, valid_record_logits, reference_record_logits)
        safety = guard.evaluate(_guard_view(bundle), bundle["kl"]["total"])
        utility, utility_terms = _utility(bundle, cfg)
        epoch_result = {
            "event": "epoch", "epoch": epoch, "updates": updates,
            "clinical_utility": utility, "clinical_utility_terms": utility_terms,
            "safety": safety, "metrics": bundle,
            "elapsed_seconds": float(time.time()-started),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_result, ensure_ascii=False, allow_nan=False) + "\n")
        _write(output / f"inner_validation_epoch{epoch}.json", epoch_result)
        failed_checks = {name for name, passed in safety["checks"].items() if not passed}
        critical_safety_failure = {
            "acs_brier_guard", "omi_brier_guard", "kl_guard"
        }.issubset(failed_checks)
        if safety["pass"] and (best is None or utility > best["utility"]):
            best = {"epoch": epoch, "utility": utility, "metrics": bundle, "safety": safety,
                    "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
            stale = 0
        else: stale += 1
        if max_updates is not None and updates >= max_updates: break
        if critical_safety_failure: break
        if budget_exhausted: break
        if stale >= int(cfg["optimization"]["patience"]): break

    status = "smoke_pass" if args.smoke else (
        "safety_pass" if best is not None else (
            "critical_safety_gate_stop" if critical_safety_failure else "no_safe_checkpoint"
        )
    )
    final = {
        "status": status, "updates": updates, "epochs": epoch,
        "elapsed_seconds": float(time.time()-started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "baseline_inner_validation": baseline_bundle, "baseline_utility": baseline_utility,
        "best": None if best is None else {key: value for key, value in best.items() if key != "state"},
        "outer_test_loaded": False, "budget_exhausted": budget_exhausted,
        "max_wall_seconds": max_wall_seconds,
        "critical_safety_failure": critical_safety_failure,
    }
    _write(output / "training_summary.json", final)
    if not args.smoke and best is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["model"] = best["state"]
        payload["rl_post_training"] = {
            "algorithm": provenance["algorithm"], "source_checkpoint_sha256": sha256(checkpoint),
            "selected_epoch": best["epoch"], "inner_validation_safety": best["safety"],
            "outer_test_loaded": False, "config": cfg,
        }
        temporary = output / "checkpoint_best.pt.tmp"
        torch.save(payload, temporary); os.replace(temporary, output / "checkpoint_best.pt")
        _write(output / "frozen_config.json", {
            "status": "frozen_before_outer_test", "checkpoint_sha256": sha256(output / "checkpoint_best.pt"),
            "selected_epoch": best["epoch"], "config": cfg, "safety": best["safety"],
            "inner_validation_metrics": best["metrics"], "outer_test_evaluations": 0,
        })
    provenance.update({"status": status, "finished_at_unix": time.time(), "outer_test_loaded": False})
    _write(output / "launch_provenance.json", provenance)
    print(json.dumps({"status": status, "output": str(output), "updates": updates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
