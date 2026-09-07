"""Evaluate one Joint-6 nested fold with inner-only threshold selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from experiments.acs_joint_task import (
    ACS_CLASSES,
    acs_metrics,
    acs_subtype_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    save_confusion_figures,
    save_reliability_figures,
    save_roc_pr_curves,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint import _aggregate, _read_csv, read_history
from experiments.ecg_lfm.evaluate_joint7_stage_a import generic_calibration
from experiments.ecg_lfm.joint6 import (
    JOINT6_CLASSES,
    encode_source_labels,
    marginalize_joint6_probabilities,
)
from models.ecg_lfm_adapter import load_finetuned_model, parameter_summary


ROOT = Path(__file__).resolve().parents[2]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def infer_joint6(
    checkpoint: Path,
    manifest_dir: Path,
    split: str,
    segment_csv: Path,
    device: torch.device,
    workers: int,
    batch: int,
) -> tuple[list[dict], np.ndarray, dict]:
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    if str(getattr(model, "output_semantics", "")) != "joint6_observed_outcome":
        raise ValueError("checkpoint is not a joint6_observed_outcome model")
    stats = parameter_summary(model)
    stats["trainable_parameters"] = stats["effective_trainable_parameters_at_checkpoint"]
    stats["trainable_parameter_definition"] = "effective after forward-time no_grad guards, not requires_grad alone"
    stats["trainable_percent"] = 100.0 * stats["trainable_parameters"] / stats["total_parameters"]
    model.eval().to(device)
    task.load_dataset(split, combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False)
    dataset = task.dataset(split)
    iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_signals=batch,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=2026,
        num_shards=1,
        shard_id=0,
        num_workers=workers,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=False)
    segment_logits = np.full((len(dataset), 6), np.nan, dtype=np.float32)
    with torch.no_grad():
        for sample in iterator:
            if not sample:
                continue
            ids = sample["id"].cpu().numpy()
            from fairseq_signals.utils import utils

            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**sample["net_input"])
                values = model.get_logits(output, normalize=False).float().cpu().numpy()
            segment_logits[ids] = values
    del model, task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(segment_logits).all():
        raise RuntimeError(f"Joint-6 inference did not fill every {split} row")
    segment_rows = _read_csv(segment_csv)
    if len(segment_rows) != len(segment_logits):
        raise ValueError("segment index and output lengths differ")
    records, record_logits = _aggregate(segment_logits, segment_rows)
    return records, record_logits, stats


def joint6_metrics(y: np.ndarray, q: np.ndarray) -> dict:
    """Return six-class and one-vs-rest metrics without inventing absent-class scores.

    A fold with no positive example for a class cannot support a meaningful OvR
    estimate.  In that case all class metrics are serialized as ``null`` (and as
    an empty/NA cell in CSV exports), rather than the misleading value zero.
    """
    pred = q.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(6))
    per_class = {}
    aurocs, auprcs = [], []
    evaluable_classes = []
    missing_positive_classes = []
    for class_id, name in enumerate(JOINT6_CLASSES):
        target = (y == class_id).astype(int)
        hard = (pred == class_id).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, hard, labels=(0, 1)).ravel()
        support = int(target.sum())
        if support == 0:
            auroc = auprc = f1 = sensitivity = specificity = ppv = None
            missing_positive_classes.append(name)
        else:
            evaluable_classes.append(class_id)
            auroc = float(roc_auc_score(target, q[:, class_id])) if np.unique(target).size == 2 else None
            auprc = float(average_precision_score(target, q[:, class_id])) if np.unique(target).size == 2 else None
            if auroc is not None:
                aurocs.append(auroc)
            if auprc is not None:
                auprcs.append(auprc)
            f1 = float(f1_score(target, hard, zero_division=0))
            sensitivity = float(tp / (tp + fn))
            specificity = float(tn / (tn + fp)) if tn + fp else None
            ppv = float(tp / (tp + fp)) if tp + fp else 0.0
        per_class[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "precision": ppv,
            "ppv": ppv,
            "sensitivity": sensitivity,
            "recall": sensitivity,
            "specificity": specificity,
            "support": support,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    supported_macro_f1 = (
        float(f1_score(y, pred, labels=evaluable_classes, average="macro", zero_division=0))
        if evaluable_classes else None
    )
    return {
        "n": int(len(y)),
        "macro_auroc_ovr": float(np.mean(aurocs)) if len(aurocs) == len(JOINT6_CLASSES) else None,
        "macro_auprc_ovr": float(np.mean(auprcs)) if len(auprcs) == len(JOINT6_CLASSES) else None,
        "macro_f1": supported_macro_f1 if len(evaluable_classes) == len(JOINT6_CLASSES) else None,
        "supported_macro_auroc_ovr": float(np.mean(aurocs)) if aurocs else None,
        "supported_macro_auprc_ovr": float(np.mean(auprcs)) if auprcs else None,
        "supported_macro_f1": supported_macro_f1,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_evaluable_class_count": len(evaluable_classes),
        "missing_positive_classes": missing_positive_classes,
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
    }


def save_joint6_confusion(path: Path, matrix: np.ndarray) -> None:
    """Save the direct six-outcome confusion matrix as a safe aggregate figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(
        np.asarray(matrix, dtype=np.int64), display_labels=JOINT6_CLASSES,
    ).plot(ax=axis, cmap="Blues", xticks_rotation=35, colorbar=False)
    axis.set_title("Joint-6 observed-outcome confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_rows(records: list[dict], logits: np.ndarray, assignments: list[dict], source: np.ndarray) -> list[dict]:
    by_id = {str(row["record_id"]).zfill(5): row for row in assignments}
    if len(by_id) != len(assignments):
        raise ValueError("duplicate assignment record ID")
    assignment_indices = [int(row["idx"]) for row in assignments]
    if len(set(assignment_indices)) != len(assignment_indices):
        raise ValueError("duplicate assignment master index")
    q = softmax(logits)
    marginal = marginalize_joint6_probabilities(q)
    rows = []
    seen = set()
    for position, record in enumerate(records):
        record_id = str(record["record_id"]).zfill(5)
        assignment = by_id.get(record_id)
        if assignment is None:
            raise ValueError(f"prediction missing assignment: {record_id}")
        if record_id in seen:
            raise ValueError(f"duplicate record prediction: {record_id}")
        if record["patient_id"] != assignment["patient_id"]:
            raise ValueError(f"patient mismatch: {record_id}")
        index = int(assignment["idx"])
        if int(record["master_idx"]) != index:
            raise ValueError(f"master index mismatch: {record_id}")
        acs, omi, joint, exception = encode_source_labels(source[index:index + 1])
        if assignment["acs"] != ACS_CLASSES[int(acs[0])] or int(assignment["omi"]) != int(omi[0]):
            raise ValueError(f"assignment/source label mismatch: {record_id}")
        row = {
            "patient_id": assignment["patient_id"],
            "record_id": record_id,
            "fold": int(assignment["fold"]),
            "subset": assignment["subset"],
            "raw_acs_true": int(acs[0]),
            "raw_omi_true": int(omi[0]),
            "joint_true": int(joint[0]),
            "joint_pred": int(q[position].argmax()),
            "joint6_true": int(joint[0]),
            "joint6_pred": int(q[position].argmax()),
            "joint_label_exception": bool(exception[0]),
            **{f"p_{name.lower().replace('+', '_').replace('/', '_')}": float(q[position, k]) for k, name in enumerate(JOINT6_CLASSES)},
            "p_other": float(marginal["acs"][position, 0]),
            "p_stemi": float(marginal["acs"][position, 1]),
            "p_nstemi": float(marginal["acs"][position, 2]),
            "p_ua": float(marginal["acs"][position, 3]),
            "p_omi": float(marginal["omi"][position]),
            "p_nste": float(marginal["nste"][position]),
            "p_nste_omi": float(marginal["nste_omi"][position]),
            "p_omi_given_nste": float(marginal["omi_given_nste"][position]),
        }
        rows.append(row)
        seen.add(record_id)
    if seen != set(by_id):
        raise ValueError("prediction/assignment coverage mismatch")
    if len(rows) != len(assignments):
        raise ValueError("prediction/assignment row-count mismatch")
    return rows


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_joint = np.asarray([row["joint6_true"] for row in rows], dtype=np.int64)
    q = np.asarray([
        [row[f"p_{name.lower().replace('+', '_').replace('/', '_')}"] for name in JOINT6_CLASSES]
        for row in rows
    ], dtype=np.float64)
    y_acs = np.asarray([row["raw_acs_true"] for row in rows], dtype=np.int64)
    p_acs = np.asarray([[row[f"p_{name}"] for name in ("other", "stemi", "nstemi", "ua")] for row in rows])
    y_omi = np.asarray([row["raw_omi_true"] for row in rows], dtype=np.int64)
    return y_joint, q, y_acs, p_acs, y_omi


def metric_bundle(rows: list[dict], thresholds: dict[str, float | None]) -> dict:
    y_joint, q, y_acs, p_acs, y_omi = arrays(rows)
    p_omi = np.asarray([row["p_omi"] for row in rows])
    nste = np.isin(y_acs, (2, 3))
    exceptions = np.asarray([bool(row["joint_label_exception"]) for row in rows])
    youden = thresholds.get("youden_j")
    return json_safe({
        "n": len(rows),
        "joint6": joint6_metrics(y_joint, q),
        "acs": acs_metrics(y_acs, p_acs),
        "conditional_acs": acs_subtype_metrics(y_acs, p_acs),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi": {
            name: None if threshold is None else binary_metrics(y_omi, p_omi, float(threshold))
            for name, threshold in thresholds.items()
        },
        "omi_among_nste_acs": None if youden is None else binary_metrics(y_omi[nste], p_omi[nste], float(youden)),
        "other_omi_sensitivity_analysis": {
            "excluded_records": int(exceptions.sum()),
            "metrics": None if youden is None else binary_metrics(y_omi[~exceptions], p_omi[~exceptions], float(youden)),
        },
        "calibration_uncalibrated": {
            "joint6": generic_calibration(y_joint, q, JOINT6_CLASSES),
            "acs": multiclass_calibration_metrics(y_acs, p_acs),
            "omi": binary_calibration_metrics(y_omi, p_omi),
        },
    })


def json_safe(value):
    """Recursively replace NaN/Inf from undefined subgroup metrics with None."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_artifacts(output: Path, metrics: dict) -> None:
    """Write stable, task-oriented metric files in addition to metrics.json."""
    subsets = ("inner_validation", "outer_test")
    payloads = {
        "joint6_metrics.json": {subset: metrics[subset]["joint6"] for subset in subsets},
        "acs_metrics.json": {subset: metrics[subset]["acs"] for subset in subsets},
        "omi_metrics.json": {
            subset: {
                "operating_points": metrics[subset]["omi"],
                "threshold_source": metrics.get(
                    "threshold_source", "inner-validation only; frozen for outer-test"
                ),
            }
            for subset in subsets
        },
        "nste_metrics.json": {
            subset: {
                "nstemi_vs_ua": metrics[subset]["nstemi_vs_ua"],
                "omi_among_nste_acs": metrics[subset]["omi_among_nste_acs"],
            }
            for subset in subsets
        },
    }
    for filename, payload in payloads.items():
        (output / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    class_rows = []
    for subset in subsets:
        for family in ("joint6", "acs"):
            for class_name, values in metrics[subset][family]["per_class"].items():
                class_rows.append({
                    "subset": subset,
                    "family": family,
                    "class": class_name,
                    **{key: values.get(key) for key in (
                        "support", "auroc", "auprc", "f1", "precision", "ppv",
                        "sensitivity", "recall", "specificity", "tp", "tn", "fp", "fn",
                    )},
                })
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)


def write_completion_receipt(output: Path, identity: dict) -> None:
    """Write a hash-bound success receipt after every required artifact exists."""
    required = (
        "metrics.json", "inner_validation_predictions.csv", "outer_test_predictions.csv",
        "joint6_metrics.json", "per_class_metrics.csv", "acs_metrics.json",
        "omi_metrics.json", "nste_metrics.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"cannot complete evaluation; missing artifacts: {missing}")
    receipt = {
        "status": "pass",
        **identity,
        "artifact_sha256": {name: sha256(output / name) for name in required},
    }
    (output / "evaluation_complete.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("lora", "partial"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--run_root", default="results/joint6_cv/ecg_lfm")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--output_root", default="results/joint6_cv/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    runtime_evaluator_sha = sha256(Path(__file__))

    manifest_root = ROOT / args.manifest_root
    manifest_dir = manifest_root / f"fold{args.fold}"
    checkpoint = ROOT / args.run_root / args.method / f"fold{args.fold}" / "checkpoint_best.pt"
    output = ROOT / args.output_root / args.method / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source = np.load(manifest_root / "y_all.npy", mmap_mode="r")
    assignments_all = _read_csv(manifest_dir / "patient_assignments.csv")
    allowed_subsets = {"inner_train", "inner_validation", "outer_test"}
    unknown_subsets = sorted({row["subset"] for row in assignments_all} - allowed_subsets)
    if unknown_subsets:
        raise ValueError(f"assignment contains unknown subsets: {unknown_subsets}")
    if any(int(row["fold"]) != args.fold for row in assignments_all):
        raise ValueError("assignment contains a different outer fold")
    patient_sets = {
        subset: {row["patient_id"] for row in assignments_all if row["subset"] == subset}
        for subset in ("inner_train", "inner_validation", "outer_test")
    }
    overlaps = {
        "inner_train__inner_validation": len(patient_sets["inner_train"] & patient_sets["inner_validation"]),
        "inner_train__outer_test": len(patient_sets["inner_train"] & patient_sets["outer_test"]),
        "inner_validation__outer_test": len(patient_sets["inner_validation"] & patient_sets["outer_test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"patient leakage: {overlaps}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_rows = {}
    model_stats = None
    for subset, split in (("inner_validation", "valid"), ("outer_test", "outer_test")):
        assignments = [row for row in assignments_all if row["subset"] == subset]
        records, logits, stats = infer_joint6(
            checkpoint, manifest_dir, split, manifest_dir / f"{split}_segments.csv",
            device, args.workers, args.batch,
        )
        model_stats = stats
        all_rows[subset] = build_rows(records, logits, assignments, source)

    _, _, _, _, y_inner_omi = arrays(all_rows["inner_validation"])
    p_inner_omi = np.asarray([row["p_omi"] for row in all_rows["inner_validation"]])
    thresholds, details = {}, {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[method], details[method] = select_omi_threshold(y_inner_omi, p_inner_omi, method)
    metrics = {
        "task": "observed six-class ACS subtype x OMI joint softmax",
        "protocol": "patient-disjoint nested five-fold; record-level evaluation",
        "method": args.method,
        "outer_fold": args.fold,
        "threshold_source": "inner-validation only; frozen for outer-test",
        "thresholds": thresholds,
        "threshold_details_inner_validation": details,
        "model_stats": model_stats,
        "inner_validation": metric_bundle(all_rows["inner_validation"], thresholds),
        "outer_test": metric_bundle(all_rows["outer_test"], thresholds),
    }
    assignment_count = sum(
        row["subset"] in ("inner_validation", "outer_test") for row in assignments_all
    )
    prediction_count = sum(len(rows) for rows in all_rows.values())
    prediction_ids = [row["record_id"] for rows in all_rows.values() for row in rows]
    duplicate_predictions = prediction_count - len(set(prediction_ids))
    missing_predictions = assignment_count - prediction_count
    if duplicate_predictions or missing_predictions:
        raise RuntimeError(
            f"invalid prediction coverage: missing={missing_predictions}, duplicates={duplicate_predictions}"
        )
    provenance = {
        "status": "pass",
        "method": args.method,
        "fold": args.fold,
        "patient_overlap": overlaps,
        "assignment_count": assignment_count,
        "prediction_count": prediction_count,
        "missing_predictions": missing_predictions,
        "duplicate_predictions": duplicate_predictions,
        "checkpoint_sha256": sha256(checkpoint),
        "assignment_sha256": sha256(manifest_dir / "patient_assignments.csv"),
        "evaluator_sha256": runtime_evaluator_sha,
        "elapsed_seconds": float(time.time() - started),
    }
    _, _, y_acs, p_acs, y_omi = arrays(all_rows["outer_test"])
    p_omi = np.asarray([row["p_omi"] for row in all_rows["outer_test"]])
    save_roc_pr_curves(output, y_acs, p_acs, y_omi, p_omi)
    bundle = metrics["outer_test"]
    omi = bundle["omi"]["youden_j"]
    save_confusion_figures(
        output,
        np.asarray(bundle["acs"]["confusion_matrix"]),
        np.asarray(bundle["conditional_acs"]["confusion_matrix"]),
        np.asarray([[omi["tn"], omi["fp"]], [omi["fn"], omi["tp"]]]),
    )
    save_joint6_confusion(output / "joint6_confusion_matrix.png", bundle["joint6"]["confusion_matrix"])
    save_reliability_figures(
        output,
        bundle["calibration_uncalibrated"]["acs"],
        bundle["calibration_uncalibrated"]["omi"],
    )
    histories = read_history(ROOT / args.run_root, args.method, fold=args.fold)
    from experiments.acs_joint_task import save_curves
    save_curves(output, histories, f"Joint-6 ECG-LFM {args.method} fold {args.fold}")

    # The validated provenance is written last and is the completion marker.
    for subset, rows in all_rows.items():
        write_rows(output / f"{subset}_predictions.csv", rows)
    write_metric_artifacts(output, metrics)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    provenance["elapsed_seconds"] = float(time.time() - started)
    (output / "provenance_alignment_report.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_completion_receipt(output, {
        "method": args.method,
        "fold": args.fold,
        "evaluator_sha256": runtime_evaluator_sha,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "assignment_sha256": provenance["assignment_sha256"],
        "provenance_sha256": sha256(output / "provenance_alignment_report.json"),
    })
    print(json.dumps({"status": "pass", "metrics": str(output / "metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
