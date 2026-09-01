"""R1-v2 update-budget Head-only stabilization study.

The script starts every candidate from W0, evaluates only Fold0
inner-validation every fixed number of optimizer updates, and never loads
outer-test. Adaptive target-KL control and a hard-KL rollback are explicit in
the run provenance.
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

sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES, acs_metrics, binary_calibration_metrics, binary_metrics,
    multiclass_calibration_metrics, nstemi_vs_ua_metrics, select_omi_threshold,
    sha256,
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
    if raw.ndim == 1: raw = raw.unsqueeze(0)
    if raw.shape[1] != 5 or bool((raw[:, :4].sum(dim=1) != 1).any()):
        raise ValueError(f"R1-v2 requires corrected [B,5] labels, got {tuple(raw.shape)}")
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


def _cache_logits(model, task, dataset, cfg: dict, device: torch.device, epoch: int) -> np.ndarray:
    model.eval(); result = np.full((len(dataset), 5), np.nan, dtype=np.float32)
    with torch.no_grad():
        for batch in _iterator(task, dataset, cfg, epoch, False):
            if not batch: continue
            ids = batch["id"].cpu().numpy(); batch = _move(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch["net_input"])
            result[ids] = model.get_logits(output, normalize=False).float().cpu().numpy()
    if not np.isfinite(result).all(): raise RuntimeError("reference inference left empty logits")
    return result


def _rows(logits: np.ndarray, segment_csv: Path, labels: np.ndarray) -> tuple[list[dict], np.ndarray]:
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


def _bundle(rows: list[dict], record_logits: np.ndarray, reference_record_logits: np.ndarray) -> dict:
    y_acs = np.asarray([row["y_acs"] for row in rows], dtype=np.int64)
    p_acs = np.asarray([[row[f"p_{name}"] for name in ACS_CLASSES] for row in rows])
    y_omi = np.asarray([row["y_omi"] for row in rows], dtype=np.int64)
    p_omi = np.asarray([row["p_omi"] for row in rows])
    threshold, details = select_omi_threshold(y_omi, p_omi, "youden_j")
    if threshold is None: raise RuntimeError("inner-validation Youden threshold infeasible")
    kl = factorized_kl(torch.from_numpy(record_logits), torch.from_numpy(reference_record_logits))
    nste = np.isin(y_acs, (2, 3))
    return {
        "acs": acs_metrics(y_acs, p_acs), "omi": binary_metrics(y_omi, p_omi, threshold),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs": binary_metrics(y_omi[nste], p_omi[nste], threshold),
        "calibration": {"acs": multiclass_calibration_metrics(y_acs, p_acs), "omi": binary_calibration_metrics(y_omi, p_omi)},
        "omi_threshold": float(threshold), "omi_threshold_detail": details,
        "kl": {name: float(value.mean()) for name, value in kl.items()},
    }


def _guard_view(bundle: dict) -> dict[str, float]:
    return {
        "acs_macro_auprc": float(bundle["acs"]["macro_auprc_ovr"]),
        "omi_auprc": float(bundle["omi"]["auprc"]),
        "omi_ppv": float(bundle["omi"]["ppv"]), "omi_specificity": float(bundle["omi"]["specificity"]),
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


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def update_kl_controller(coefficient: float, current_kl: float, target_kl: float, cfg: dict) -> float:
    """Move the KL penalty toward target KL using only current-policy drift."""
    if current_kl > target_kl * float(cfg["kl"]["high_ratio"]):
        coefficient *= float(cfg["kl"]["increase_factor"])
    elif current_kl < target_kl * float(cfg["kl"]["low_ratio"]):
        coefficient *= float(cfg["kl"]["decrease_factor"])
    return float(np.clip(coefficient, float(cfg["kl"]["min_coefficient"]), float(cfg["kl"]["max_coefficient"])))


def _head_only_audit(model) -> dict:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    invalid = [name for name in names if not name.startswith("proj.")]
    if invalid or set(names) != {"proj.weight", "proj.bias"}:
        raise RuntimeError(f"Head-only invariant failed; trainable={names[:20]}")
    return {"trainable_names": names, **parameter_summary(model)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--target_kl", type=float, required=True)
    parser.add_argument("--run_suffix", required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("R1-v2 requires exactly one CUDA_VISIBLE_DEVICES value")
    if not torch.cuda.is_available(): raise RuntimeError("R1-v2 requires CUDA")
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if int(cfg["run"]["fold"]) != 0: raise ValueError("R1-v2 is registered for nested Fold0 only")
    if int(cfg["policy"]["update_epochs"]) != 1: raise ValueError("update-budget runner requires update_epochs=1")
    if float(args.target_kl) not in (0.005, 0.01): raise ValueError("target_KL must be 0.005 or 0.01")
    cfg["optimization"]["learning_rate"] = float(args.learning_rate)
    cfg["kl"]["target_kl"] = float(args.target_kl)
    safe_suffix = str(args.run_suffix)
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in safe_suffix):
        raise ValueError("run_suffix contains unsafe characters")
    run_name = str(cfg["run"]["name"]) + safe_suffix
    seed = int(cfg["run"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    manifest_dir = ROOT / cfg["run"]["manifest_root"] / "fold0"
    checkpoint = ROOT / cfg["run"]["checkpoint"]
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
    # W0 is a LoRA checkpoint and the loader intentionally restores its
    # original LoRA requires_grad flags. R1-v2 explicitly freezes those
    # adapters again: the first round is truly Head-only.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.proj.parameters():
        parameter.requires_grad = True
    trainable = _head_only_audit(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial_trainable_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    task.load_dataset("train", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=True)
    task.load_dataset("valid", combine=False, epoch=1, task_cfg=saved_cfg.task, label=True, shuffle=False)
    train_dataset, valid_dataset = task.dataset("train"), task.dataset("valid")
    labels = np.load(manifest_dir.parent / "y_joint_5.npy", mmap_mode="r")
    reference_train = _cache_logits(model, task, train_dataset, cfg, device, 1)
    reference_valid = _cache_logits(model, task, valid_dataset, cfg, device, 1)
    reference_rows, reference_record_logits = _rows(reference_valid, manifest_dir / "valid_segments.csv", labels)
    margins, bootstrap = patient_bootstrap_margins(
        reference_rows, int(cfg["safety"]["bootstrap_replicates"]), seed,
        float(cfg["safety"]["minimum_absolute_margin"]),
    )
    guard = SafetyGuard.from_reference(
        bootstrap["reference"], margins, max_kl=float(cfg["kl"]["max_inner_validation_mean"])
    )
    baseline_bundle = _bundle(reference_rows, reference_record_logits, reference_record_logits)
    baseline_utility, baseline_terms = _utility(baseline_bundle, cfg)
    provenance = {
        "status": "running", "started_at_unix": started, "run_name": run_name,
        "task": "factorized Categorical(ACS 4-way) x Bernoulli(OMI)",
        "algorithm": "update-budget clipped contextual policy optimization",
        "trainable_policy": "classification head only; LoRA and base encoder frozen",
        "outer_test_loaded": False, "config": cfg, "config_sha256": sha256(cfg_path),
        "w0_checkpoint": str(checkpoint), "w0_checkpoint_sha256": sha256(checkpoint),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "cuda_visible_devices": visible, "trainable": trainable,
        "reference_policy": "exact W0 logits cached before update 1",
        "reference_and_policy_equal_at_init": True, "reward_weights_unchanged": True,
        "update_budget": int(cfg["optimization"]["max_updates"]),
        "validation_interval_updates": int(cfg["optimization"]["eval_interval_updates"]),
        "target_kl": float(args.target_kl), "hard_kl_threshold": float(args.target_kl) * float(cfg["kl"]["hard_stop_multiplier"]),
        "reward_interpretation": cfg["metadata"]["interpretation"],
    }
    _write(output / "launch_provenance.json", provenance)
    _write(output / "reference_safety_guard.json", {
        **bootstrap, "margins": margins, "lower_bounds": guard.lower_bounds,
        "upper_bounds": guard.upper_bounds, "max_kl": guard.max_kl,
        "baseline_bundle": baseline_bundle, "baseline_utility": baseline_utility,
        "baseline_utility_terms": baseline_terms,
    })

    optimizer = torch.optim.Adam([model.proj.weight, model.proj.bias], lr=float(args.learning_rate), betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["optimization"]["fp16"]))
    reward_weights = RewardWeights.from_mapping(cfg["reward"])
    clip_epsilon = float(cfg["policy"]["clip_epsilon"])
    kl_coefficient = float(cfg["kl"]["initial_coefficient"])
    min_coefficient = float(cfg["kl"]["min_coefficient"]); max_coefficient = float(cfg["kl"]["max_coefficient"])
    target_kl = float(args.target_kl); hard_threshold = target_kl * float(cfg["kl"]["hard_stop_multiplier"])
    hard_patience = int(cfg["kl"]["hard_stop_patience"])
    max_updates = int(cfg["optimization"]["max_updates"]); eval_interval = int(cfg["optimization"]["eval_interval_updates"])
    max_wall_seconds = float(cfg["optimization"].get("max_wall_seconds", 7920))
    log_path = output / "train_log.jsonl"
    best = None; updates = 0; hard_kl_streak = 0; status = "completed_update_budget"
    stop_reason = None; epoch = 1; iterator = _iterator(task, train_dataset, cfg, epoch, True)
    torch.cuda.reset_peak_memory_stats(device)

    while updates < max_updates and time.time() - started < max_wall_seconds:
        raw_batch = next(iterator, None)
        if raw_batch is None:
            epoch += 1; iterator = _iterator(task, train_dataset, cfg, epoch, True); continue
        if not raw_batch: continue
        ids = raw_batch["id"].cpu().numpy(); batch = _move(raw_batch, device)
        reference_logits = torch.from_numpy(reference_train[ids]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])):
            old_output = model(**batch["net_input"]); old_logits = model.get_logits(old_output, normalize=False).float()
            action = sample_policy(old_logits)
            acs_target, omi_target = _targets(model.get_targets(batch, old_output))
            reward, reward_parts = hard_reward(action.acs_action, action.omi_action, acs_target, omi_target, reward_weights)
            advantage = reward - reward.mean(); old_log_probability = action.joint_log_probability.detach()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["optimization"]["fp16"])):
            current_output = model(**batch["net_input"]); current_logits = model.get_logits(current_output, normalize=False).float()
            log_probability = joint_log_probability(current_logits, action.acs_action, action.omi_action)
            ratio = torch.exp(log_probability - old_log_probability)
            policy_objective = torch.minimum(ratio * advantage, torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantage).mean()
            current_acs, current_omi = probabilities(current_logits)
            shaping, shaping_parts = soft_f1_shaping(current_acs, current_omi, acs_target, omi_target, reward_weights)
            kl_values = factorized_kl(current_logits, reference_logits); batch_kl = kl_values["total"].mean()
            loss = -(policy_objective + shaping - kl_coefficient * batch_kl)
        if not bool(torch.isfinite(loss)):
            status = "nonfinite_rollback"; stop_reason = "nonfinite_rl_loss"; break
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_([model.proj.weight, model.proj.bias], float(cfg["optimization"]["max_grad_norm"]))
        if not bool(torch.isfinite(grad_norm)):
            status = "nonfinite_rollback"; stop_reason = "nonfinite_gradient"; optimizer.zero_grad(set_to_none=True); break
        scaler.step(optimizer); scaler.update(); updates += 1
        batch_kl_value = float(batch_kl.detach())
        if batch_kl_value > hard_threshold:
            hard_kl_streak += 1
        else:
            hard_kl_streak = 0
        kl_coefficient = update_kl_controller(kl_coefficient, batch_kl_value, target_kl, cfg)
        update_detail = {
            "event": "update", "update": updates, "epoch": epoch, "loss": float(loss.detach()),
            "policy_objective": float(policy_objective.detach()), "soft_f1": float(shaping.detach()),
            "reward_mean": float(reward.mean()), "reward_std": float(reward.std(unbiased=False)),
            "ratio_mean": float(ratio.mean()), "clip_fraction": float(((ratio < 1-clip_epsilon) | (ratio > 1+clip_epsilon)).float().mean()),
            "kl_acs": float(kl_values["acs"].mean()), "kl_omi": float(kl_values["omi"].mean()),
            "kl_total": batch_kl_value, "target_kl": target_kl, "kl_coefficient": kl_coefficient,
            "hard_kl_threshold": hard_threshold, "hard_kl_streak": hard_kl_streak,
            "grad_norm": float(grad_norm), "acs_entropy": float(action.acs_entropy.mean()), "omi_entropy": float(action.omi_entropy.mean()),
            "actions": {str(k): int((action.joint_action == k).sum()) for k in range(8)},
            "reward_components": {name: float(value.mean()) for name, value in reward_parts.items()},
            "soft_f1_components": {name: float(value) for name, value in shaping_parts.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(update_detail, ensure_ascii=False, allow_nan=False) + "\n")
        if hard_kl_streak >= hard_patience:
            status = "hard_kl_stop_rollback"; stop_reason = "hard_kl_threshold_exceeded_consecutively"; break
        if updates % eval_interval == 0:
            valid_logits = _cache_logits(model, task, valid_dataset, cfg, device, epoch)
            valid_rows, valid_record_logits = _rows(valid_logits, manifest_dir / "valid_segments.csv", labels)
            bundle = _bundle(valid_rows, valid_record_logits, reference_record_logits)
            safety = guard.evaluate(_guard_view(bundle), bundle["kl"]["total"]); utility, utility_terms = _utility(bundle, cfg)
            eval_result = {
                "event": "inner_validation", "update": updates, "clinical_utility": utility,
                "clinical_utility_terms": utility_terms, "safety": safety, "metrics": bundle,
                "kl_controller": {"target_kl": target_kl, "coefficient": kl_coefficient, "hard_streak": hard_kl_streak},
                "elapsed_seconds": float(time.time()-started), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(eval_result, ensure_ascii=False, allow_nan=False) + "\n")
            _write(output / f"inner_validation_update{updates}.json", eval_result)
            if safety["pass"] and (best is None or utility > best["utility"]):
                best = {
                    "update": updates, "utility": utility, "utility_terms": utility_terms,
                    "metrics": bundle, "safety": safety,
                    "state": {name: model.state_dict()[name].detach().cpu().clone() for name in trainable_names},
                }
        if updates >= max_updates: break
    if updates >= max_updates and stop_reason is None: status = "completed_update_budget"
    if time.time() - started >= max_wall_seconds and stop_reason is None:
        status = "wall_budget_stop"; stop_reason = "max_wall_seconds"
    if best is None and status == "completed_update_budget": status = "no_safety_passing_candidate"
    rollback_performed = status in {"hard_kl_stop_rollback", "nonfinite_rollback"}
    rollback_state = best["state"] if best is not None else initial_trainable_state
    if rollback_performed:
        for name, value in rollback_state.items():
            dict(model.named_parameters())[name].data.copy_(value.to(device))
    summary = {
        "status": status, "stop_reason": stop_reason, "updates": updates, "outer_test_loaded": False,
        "checkpoint_saved": False, "completed_validation_updates": [int(x) for x in range(eval_interval, updates+1, eval_interval)],
        "elapsed_seconds": float(time.time()-started), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "baseline_inner_validation": baseline_bundle, "baseline_utility": baseline_utility,
        "best": None if best is None else {key: value for key, value in best.items() if key != "state"},
        "rollback": {"performed": rollback_performed, "source": "last_safety_passing_state_or_W0_head", "state_saved": False},
    }
    _write(output / "training_summary.json", summary)
    if best is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name, value in best["state"].items():
            payload["model"][name] = value
        payload["rl_post_training"] = {"study": "R1-v2", "trainable_policy": "head_only", "source_checkpoint_sha256": sha256(checkpoint),
                                        "selected_update": best["update"], "inner_validation_safety": best["safety"], "outer_test_loaded": False, "config": cfg}
        temporary = output / "checkpoint_best.pt.tmp"; torch.save(payload, temporary); os.replace(temporary, output / "checkpoint_best.pt")
        summary["checkpoint_saved"] = True; summary["rollback"]["state_saved"] = True
        _write(output / "frozen_candidate_config.json", {"status": "candidate_frozen_before_outer_test", "outer_test_evaluations": 0,
                                                            "checkpoint_sha256": sha256(output / "checkpoint_best.pt"), "selected_update": best["update"],
                                                            "config": cfg, "safety": best["safety"], "inner_validation_metrics": best["metrics"]})
        _write(output / "training_summary.json", summary)
    provenance.update({"status": status, "finished_at_unix": time.time(), "outer_test_loaded": False})
    _write(output / "launch_provenance.json", provenance)
    print(json.dumps({"status": status, "run_name": run_name, "updates": updates, "best": None if best is None else best["update"]}, ensure_ascii=False))


if __name__ == "__main__": main()
