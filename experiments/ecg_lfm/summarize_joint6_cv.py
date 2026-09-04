"""Summarize Joint-6 nested-CV outputs without changing the experiment protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.acs_joint_task import acs_metrics, binary_metrics, nstemi_vs_ua_metrics, sha256
from experiments.ecg_lfm.evaluate_joint6_fold import joint6_metrics
from experiments.ecg_lfm.joint6 import JOINT6_CLASSES


ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "ECG-LFM LoRA": ("ecg_lfm", "lora"),
    "ECG-LFM Partial FT": ("ecg_lfm", "partial"),
    "InceptionTime": ("traditional", "InceptionTime"),
    "ResNet1D_Wang": ("traditional", "ResNet1D_Wang"),
    "LSTM": ("traditional", "LSTM"),
    "Transformer1D": ("traditional", "Transformer1D"),
    "ViT": ("traditional", "ViT"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_dir(root: Path, kind: str, key: str, fold: int) -> Path:
    if kind == "ecg_lfm":
        return root / "evaluation" / key / f"fold{fold}"
    return root / "traditional" / key / f"fold{fold}"


def validated_provenance(
    directory: Path,
    kind: str,
    expected_method: str | None = None,
    expected_fold: int | None = None,
) -> tuple[bool, str]:
    path = directory / ("provenance_alignment_report.json" if kind == "ecg_lfm" else "provenance.json")
    if not path.exists():
        return False, "missing provenance"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        return False, "provenance status is not pass"
    expected_hash = sha256(ROOT / "experiments/ecg_lfm/evaluate_joint6_fold.py")
    if payload.get("evaluator_sha256") != expected_hash:
        return False, "evaluator hash mismatch"
    if kind == "ecg_lfm":
        if payload.get("missing_predictions") != 0 or payload.get("duplicate_predictions") != 0:
            return False, "prediction coverage failed"
        if any(int(value) for value in payload.get("patient_overlap", {}).values()):
            return False, "patient overlap failed"
    receipt_path = directory / "evaluation_complete.json"
    if not receipt_path.exists():
        return False, "missing hash-bound completion receipt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or receipt.get("evaluator_sha256") != expected_hash:
        return False, "invalid completion receipt"
    if expected_method is not None and str(receipt.get("method")) != expected_method:
        return False, "receipt method mismatch"
    if expected_fold is not None and int(receipt.get("fold", -1)) != expected_fold:
        return False, "receipt fold mismatch"
    expected_artifacts = {
        "metrics.json", "inner_validation_predictions.csv", "outer_test_predictions.csv",
        "joint6_metrics.json", "per_class_metrics.csv", "acs_metrics.json",
        "omi_metrics.json", "nste_metrics.json",
    }
    hashes = receipt.get("artifact_sha256", {})
    if set(hashes) != expected_artifacts:
        return False, "completion receipt has incomplete artifact set"
    for filename, recorded_hash in hashes.items():
        artifact = directory / filename
        if not artifact.exists() or sha256(artifact) != recorded_hash:
            return False, f"artifact hash mismatch: {filename}"
    if receipt.get("provenance_sha256") != sha256(path):
        return False, "provenance hash mismatch"
    fold = int(receipt.get("fold", -1))
    assignment = ROOT / "manifests/ecg_lfm_nested" / f"fold{fold}" / "patient_assignments.csv"
    if not assignment.exists() or receipt.get("assignment_sha256") != sha256(assignment):
        return False, "assignment hash mismatch"
    if kind == "ecg_lfm":
        method = str(receipt.get("method", ""))
        checkpoint = ROOT / "results/joint6_cv/ecg_lfm" / method / f"fold{fold}" / "checkpoint_best.pt"
    else:
        checkpoint = directory / "best_model.pt"
    if not checkpoint.exists() or receipt.get("checkpoint_sha256") != sha256(checkpoint):
        return False, "checkpoint hash mismatch"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    metrics_method = metrics.get("method") if kind == "ecg_lfm" else metrics.get("model")
    metrics_fold = metrics.get("outer_fold") if kind == "ecg_lfm" else metrics.get("fold")
    if expected_method is not None and str(metrics_method) != expected_method:
        return False, "metrics method mismatch"
    if expected_fold is not None and int(metrics_fold) != expected_fold:
        return False, "metrics fold mismatch"
    required_prediction_fields = {
        "patient_id", "record_id", "fold", "joint_true", "joint_pred",
        "p_other", "p_stemi_nomi", "p_stemi_omi", "p_nstemi_nomi",
        "p_nstemi_omi", "p_ua", "p_stemi", "p_nstemi", "p_omi",
        "raw_acs_true", "raw_omi_true",
    }
    for filename in ("inner_validation_predictions.csv", "outer_test_predictions.csv"):
        with (directory / filename).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required_prediction_fields.issubset(set(reader.fieldnames or [])):
                return False, f"prediction schema mismatch: {filename}"
            if expected_fold is not None and any(int(row["fold"]) != expected_fold for row in reader):
                return False, f"prediction fold mismatch: {filename}"
    return True, "pass"


def fold_scalars(metrics: dict) -> dict[str, float | None]:
    outer = metrics["outer_test"]
    omi = outer["omi"]["youden_j"]
    nstemi = outer["acs"]["per_class"]["NSTEMI"]
    nste = outer["omi_among_nste_acs"]
    return {
        "joint6_macro_auroc": outer["joint6"]["macro_auroc_ovr"],
        "joint6_macro_auprc": outer["joint6"]["macro_auprc_ovr"],
        "joint6_macro_f1": outer["joint6"]["macro_f1"],
        "joint6_balanced_accuracy": outer["joint6"]["balanced_accuracy"],
        "joint6_accuracy": outer["joint6"]["accuracy"],
        "acs_macro_auroc": outer["acs"]["macro_auroc_ovr"],
        "acs_macro_auprc": outer["acs"]["macro_auprc_ovr"],
        "acs_macro_f1": outer["acs"]["macro_f1"],
        "acs_balanced_accuracy": outer["acs"]["balanced_accuracy"],
        "acs_accuracy": outer["acs"]["accuracy"],
        "nstemi_auprc": nstemi["auprc"],
        "nstemi_f1": nstemi["f1"],
        "nstemi_sensitivity": nstemi["sensitivity"],
        "nstemi_vs_ua_auroc": outer["nstemi_vs_ua"]["auroc"],
        "nstemi_vs_ua_auprc": outer["nstemi_vs_ua"]["auprc"],
        "omi_auroc": omi["auroc"],
        "omi_auprc": omi["auprc"],
        "omi_f1": omi["f1"],
        "omi_sensitivity": omi["sensitivity"],
        "omi_specificity": omi["specificity"],
        "omi_ppv": omi["ppv"],
        "omi_npv": omi["npv"],
        "omi_mcc": omi["mcc"],
        "omi_accuracy": omi["accuracy"],
        "omi_tp": omi["tp"],
        "omi_fp": omi["fp"],
        "omi_tn": omi["tn"],
        "omi_fn": omi["fn"],
        "omi_threshold": omi["threshold"],
        "nste_omi_auroc": nste["auroc"],
        "nste_omi_auprc": nste["auprc"],
    }


def finite_values(values: list[float | None]) -> np.ndarray:
    output = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=float)


def summarize_values(values: list[float | None]) -> dict[str, float | int | None]:
    array = finite_values(values)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = float(array.std(ddof=1))
    # Two-sided Student-t 0.975 quantiles for df 1..30; five folds use df=4.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }.get(n - 1, 1.96)
    half = critical * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def duration_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("finished_at"):
        return None
    return (datetime.fromisoformat(payload["finished_at"]) - datetime.fromisoformat(payload["started_at"])).total_seconds()


def profiled_peak_memory(path: Path) -> float | None:
    if not path.exists():
        return None
    values = []
    for row in read_csv(path):
        for key, raw in row.items():
            if key.strip() == "memory.used" and raw not in (None, ""):
                values.append(float(raw.strip()))
    return max(values) if values else None


def pooled_metrics(rows: list[dict], fold_thresholds: dict[int, float]) -> dict:
    record_ids = [str(row["record_id"]).zfill(5) for row in rows]
    duplicate_record_ids = len(record_ids) - len(set(record_ids))
    patient_folds: dict[str, set[int]] = {}
    for row in rows:
        patient_folds.setdefault(str(row["patient_id"]), set()).add(int(row["fold"]))
    patient_fold_conflicts = sum(len(folds) > 1 for folds in patient_folds.values())
    if duplicate_record_ids or patient_fold_conflicts:
        raise ValueError(
            f"invalid pooled OOF coverage: duplicate_records={duplicate_record_ids}, "
            f"patient_fold_conflicts={patient_fold_conflicts}"
        )
    y_joint = np.asarray([
        int(row["joint_true"] if "joint_true" in row else row["joint6_true"]) for row in rows
    ])
    q = np.asarray([
        [float(row[f"p_{name.lower().replace('+', '_').replace('/', '_')}"]) for name in JOINT6_CLASSES]
        for row in rows
    ])
    y_acs = np.asarray([int(row["raw_acs_true"]) for row in rows])
    p_acs = np.asarray([[float(row[f"p_{name}"]) for name in ("other", "stemi", "nstemi", "ua")] for row in rows])
    y_omi = np.asarray([int(row["raw_omi_true"]) for row in rows])
    p_omi = np.asarray([float(row["p_omi"]) for row in rows])
    joint_sum_error = float(np.max(np.abs(q.sum(axis=1) - 1.0)))
    acs_sum_error = float(np.max(np.abs(p_acs.sum(axis=1) - 1.0)))
    omi_formula_error = float(np.max(np.abs(p_omi - (q[:, 2] + q[:, 4]))))
    if max(joint_sum_error, acs_sum_error, omi_formula_error) > 1e-5:
        raise ValueError(
            "invalid Joint-6 probability reconstruction: "
            f"joint={joint_sum_error}, acs={acs_sum_error}, omi={omi_formula_error}"
        )
    pred_omi = np.asarray([
        p_omi[index] >= fold_thresholds[int(row["fold"])] for index, row in enumerate(rows)
    ], dtype=int)
    hard = binary_metrics(y_omi, pred_omi.astype(float), 0.5)
    hard["auroc"] = float(roc_auc_score(y_omi, p_omi))
    hard["auprc"] = float(average_precision_score(y_omi, p_omi))
    hard["threshold"] = "fold-specific inner-validation Youden"
    nste = np.isin(y_acs, (2, 3))
    return {
        "n": len(rows),
        "integrity": {
            "status": "pass",
            "unique_record_ids": len(set(record_ids)),
            "unique_patient_ids": len(patient_folds),
            "duplicate_record_ids": duplicate_record_ids,
            "patient_fold_conflicts": patient_fold_conflicts,
            "joint_probability_sum_max_abs_error": joint_sum_error,
            "acs_probability_sum_max_abs_error": acs_sum_error,
            "omi_formula_max_abs_error": omi_formula_error,
        },
        "joint6": joint6_metrics(y_joint, q),
        "acs": acs_metrics(y_acs, p_acs),
        "omi_fold_specific_youden": hard,
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs": {
            "n": int(nste.sum()),
            "auroc": float(roc_auc_score(y_omi[nste], p_omi[nste])),
            "auprc": float(average_precision_score(y_omi[nste], p_omi[nste])),
        },
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def fmt_summary(row: dict, metric: str) -> str:
    mean, std = row.get(metric + "_mean"), row.get(metric + "_std")
    if mean is None:
        return "N/A"
    return fmt(mean) if std is None else f"{fmt(mean)} ± {fmt(std)}"


def fmt_ci(row: dict, metric: str) -> str:
    low, high = row.get(metric + "_ci95_low"), row.get(metric + "_ci95_high")
    return "N/A" if low is None else f"[{fmt(low)}, {fmt(high)}]"


def baseline_metrics(path: Path) -> tuple[dict[str, float | None], str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    omi = payload.get("omi", {}).get("youden_j", {})
    nste_payload = payload.get("omi_among_nste_acs", {})
    nste = nste_payload.get("metrics", nste_payload)
    return ({
        "acs_macro_auprc": payload.get("acs", {}).get("macro_auprc_ovr"),
        "omi_auprc": omi.get("auprc"),
        "omi_sensitivity": omi.get("sensitivity"),
        "nstemi_vs_ua_auprc": payload.get("nstemi_vs_ua", {}).get("auprc"),
        "nste_omi_auprc": nste.get("auprc"),
    }, str(payload.get("evaluation_type", "unknown protocol")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/joint6_cv")
    parser.add_argument("--report", default="docs/ECG_LFM_Joint6_Evaluation_Report.md")
    parser.add_argument("--baseline", default="results/revalidated/lora/metrics.json")
    parser.add_argument("--validate_dir", default=None)
    parser.add_argument("--validate_kind", choices=("ecg_lfm", "traditional"), default=None)
    parser.add_argument("--expected_method", default=None)
    parser.add_argument("--expected_fold", type=int, default=None)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    if args.validate_dir:
        if args.validate_kind is None:
            parser.error("--validate_kind is required with --validate_dir")
        valid, reason = validated_provenance(
            Path(args.validate_dir), args.validate_kind, args.expected_method, args.expected_fold
        )
        print(json.dumps({"valid": valid, "reason": reason}, ensure_ascii=False))
        raise SystemExit(0 if valid else 1)
    root = ROOT / args.root
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    per_class_rows: list[dict] = []
    cost_rows: list[dict] = []
    pooled: dict[str, dict] = {}
    completion: dict[str, list[int]] = {}
    rejected_results: dict[str, str] = {}

    for method, (kind, key) in METHODS.items():
        metrics_by_fold = []
        prediction_rows = []
        thresholds = {}
        present_folds = []
        for fold in range(5):
            directory = result_dir(root, kind, key, fold)
            metrics_path = directory / "metrics.json"
            predictions_path = directory / "outer_test_predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                continue
            provenance_ok, provenance_reason = validated_provenance(directory, kind, key, fold)
            if not provenance_ok:
                rejected_results[f"{method}/fold{fold}"] = provenance_reason
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            values = fold_scalars(metrics)
            fold_rows.append({"method": method, "fold": fold, **values})
            metrics_by_fold.append((fold, metrics, values))
            present_folds.append(fold)
            prediction_rows.extend(read_csv(predictions_path))
            thresholds[fold] = float(metrics["thresholds"]["youden_j"])
            for family in ("joint6", "acs"):
                for class_name, class_metrics in metrics["outer_test"][family]["per_class"].items():
                    support = class_metrics.get("support")
                    if support is None and class_metrics.get("tp") is not None and class_metrics.get("fn") is not None:
                        support = int(class_metrics["tp"]) + int(class_metrics["fn"])
                    per_class_rows.append({
                        "method": method, "fold": fold, "family": family, "class": class_name,
                        **{name: class_metrics.get(name) for name in (
                            "auroc", "auprc", "f1", "sensitivity", "specificity", "ppv"
                        )},
                        "support": support,
                    })
            stats = metrics.get("model_stats", {})
            training_seconds = stats.get("training_seconds")
            peak_memory = stats.get("peak_gpu_memory_mib")
            if kind == "ecg_lfm":
                training_seconds = duration_seconds(root / "ecg_lfm" / key / f"fold{fold}" / "launch_provenance.json")
                peak_memory = profiled_peak_memory(root / "resource" / f"ecg_lfm_{key}_fold{fold}.csv")
            cost_rows.append({
                "method": method, "fold": fold,
                "total_parameters": stats.get("total_parameters"),
                "trainable_parameters": stats.get("trainable_parameters"),
                "trainable_percent": stats.get("trainable_percent"),
                "peak_gpu_memory_mib": peak_memory,
                "training_seconds": training_seconds,
                "best_epoch": stats.get("best_epoch"),
            })
        completion[method] = present_folds
        if not metrics_by_fold:
            continue
        summary = {"method": method, "completed_folds": len(metrics_by_fold), "complete_5fold": len(metrics_by_fold) == 5}
        for metric in metrics_by_fold[0][2]:
            stats = summarize_values([item[2].get(metric) for item in metrics_by_fold])
            for suffix, value in stats.items():
                summary[f"{metric}_{suffix}"] = value
        summary_rows.append(summary)
        if len(metrics_by_fold) == 5:
            pooled[method] = pooled_metrics(prediction_rows, thresholds)

    class_summary_rows = []
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in per_class_rows:
        groups.setdefault((row["method"], row["family"], row["class"]), []).append(row)
    for (method, family, class_name), rows in groups.items():
        aggregate = {
            "method": method, "family": family, "class": class_name,
            "completed_folds": len(rows),
            "support": int(sum(int(row["support"]) for row in rows)),
        }
        for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity", "ppv"):
            stats = summarize_values([row.get(metric) for row in rows])
            for suffix, value in stats.items():
                aggregate[f"{metric}_{suffix}"] = value
        class_summary_rows.append(aggregate)

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "summary_metrics.csv", summary_rows)
    write_csv(output / "per_class_metrics.csv", per_class_rows)
    write_csv(output / "per_class_summary.csv", class_summary_rows)
    write_csv(output / "training_cost.csv", cost_rows)
    (output / "pooled_oof_metrics.json").write_text(
        json.dumps(pooled, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    missing = [f"{method}/fold{fold}" for method, folds in completion.items() for fold in range(5) if fold not in folds]
    status = {
        "complete_all_comparators": not missing,
        "missing": missing,
        "rejected_results": rejected_results,
        "folds_by_method": completion,
        "complete_5fold_methods": [row["method"] for row in summary_rows if row["complete_5fold"]],
    }
    (output / "completion_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    baseline_result = baseline_metrics(ROOT / args.baseline)
    baseline, baseline_protocol = baseline_result if baseline_result else ({}, "not available")
    lora_summary = next((row for row in summary_rows if row["method"] == "ECG-LFM LoRA" and row["complete_5fold"]), None)
    lora_pooled = pooled.get("ECG-LFM LoRA")

    main_rows = []
    definitions = [
        ("Joint-6 Macro AUPRC", None, ("joint6", "macro_auprc_ovr")),
        ("ACS Macro AUPRC", "acs_macro_auprc", ("acs", "macro_auprc_ovr")),
        ("OMI AUPRC", "omi_auprc", ("omi_fold_specific_youden", "auprc")),
        ("OMI Sensitivity", "omi_sensitivity", ("omi_fold_specific_youden", "sensitivity")),
        ("NSTEMI vs UA AUPRC", "nstemi_vs_ua_auprc", ("nstemi_vs_ua", "auprc")),
        ("NSTE-ACS OMI AUPRC", "nste_omi_auprc", ("omi_among_nste_acs", "auprc")),
    ]
    for label, baseline_key, path in definitions:
        base = baseline.get(baseline_key) if baseline_key else None
        joint = lora_pooled.get(path[0], {}).get(path[1]) if lora_pooled else None
        delta = None if base is None or joint is None else float(joint) - float(base)
        main_rows.append(f"| {label} | {fmt(base)} | {fmt(joint)} | {fmt(delta) if delta is not None else '—'} |")

    summary_table = "\n".join(
        f"| {row['method']} | {row['completed_folds']}/5 | {fmt_summary(row, 'joint6_macro_auprc')} | "
        f"{fmt_summary(row, 'joint6_macro_f1')} | {fmt_summary(row, 'acs_macro_auprc')} | "
        f"{fmt_summary(row, 'omi_auprc')} | {fmt_summary(row, 'nstemi_vs_ua_auprc')} | "
        f"{fmt_summary(row, 'nste_omi_auprc')} | {fmt_summary(row, 'omi_sensitivity')} | "
        f"{fmt_summary(row, 'omi_specificity')} |"
        for row in summary_rows
    ) or "| 尚无已完成 Fold | — | — | — | — | — | — | — | — | — |"
    joint_overall_table = "\n".join(
        f"| {row['method']} | {row['completed_folds']}/5 | {fmt_summary(row, 'joint6_macro_auprc')} | "
        f"{fmt_summary(row, 'joint6_macro_auroc')} | {fmt_summary(row, 'joint6_macro_f1')} | "
        f"{fmt_summary(row, 'joint6_balanced_accuracy')} | {fmt_summary(row, 'joint6_accuracy')} |"
        for row in summary_rows
    )
    acs_overall_table = "\n".join(
        f"| {row['method']} | {row['completed_folds']}/5 | {fmt_summary(row, 'acs_macro_auprc')} | "
        f"{fmt_summary(row, 'acs_macro_auroc')} | {fmt_summary(row, 'acs_macro_f1')} | "
        f"{fmt_summary(row, 'acs_balanced_accuracy')} | {fmt_summary(row, 'acs_accuracy')} |"
        for row in summary_rows
    )
    omi_full_table = "\n".join(
        f"| {row['method']} | {row['completed_folds']}/5 | {fmt_summary(row, 'omi_auroc')} | "
        f"{fmt_summary(row, 'omi_auprc')} | {fmt_summary(row, 'omi_f1')} | "
        f"{fmt_summary(row, 'omi_sensitivity')} | {fmt_summary(row, 'omi_specificity')} | "
        f"{fmt_summary(row, 'omi_ppv')} | {fmt_summary(row, 'omi_npv')} | "
        f"{fmt_summary(row, 'omi_mcc')} | {fmt_summary(row, 'omi_accuracy')} | "
        f"{fmt_summary(row, 'omi_threshold')} |"
        for row in summary_rows
    )
    nste_full_table = "\n".join(
        f"| {row['method']} | {row['completed_folds']}/5 | {fmt_summary(row, 'nstemi_vs_ua_auroc')} | "
        f"{fmt_summary(row, 'nstemi_vs_ua_auprc')} | {fmt_summary(row, 'nste_omi_auroc')} | "
        f"{fmt_summary(row, 'nste_omi_auprc')} |"
        for row in summary_rows
    )
    pooled_confusion_table = "\n".join(
        f"| {method} | {values['omi_fold_specific_youden']['tp']} | "
        f"{values['omi_fold_specific_youden']['fp']} | {values['omi_fold_specific_youden']['tn']} | "
        f"{values['omi_fold_specific_youden']['fn']} |"
        for method, values in pooled.items()
    ) or "| 暂无完整五折方法 | N/A | N/A | N/A | N/A |"
    fold_table = "\n".join(
        f"| {row['method']} | {row['fold']} | {fmt(row['joint6_macro_auprc'])} | {fmt(row['joint6_macro_f1'])} | "
        f"{fmt(row['acs_macro_auprc'])} | {fmt(row['omi_auprc'])} | {fmt(row['omi_sensitivity'])} | "
        f"{fmt(row['omi_specificity'])} | {fmt(row['nstemi_vs_ua_auprc'])} | {fmt(row['nste_omi_auprc'])} |"
        for row in fold_rows
    ) or "| 尚无结果 | — | — | — | — | — | — | — | — | — |"
    ci_table = "\n".join(
        f"| {row['method']} | {fmt_ci(row, 'joint6_macro_auprc')} | {fmt_ci(row, 'acs_macro_auprc')} | "
        f"{fmt_ci(row, 'omi_auprc')} | {fmt_ci(row, 'nstemi_vs_ua_auprc')} | {fmt_ci(row, 'nste_omi_auprc')} |"
        for row in summary_rows if row["complete_5fold"]
    ) or "| 暂无完整五折方法 | N/A | N/A | N/A | N/A | N/A |"
    class_table = "\n".join(
        f"| {row['method']} | {row['family']} | "
        f"{'**' + row['class'] + '**' if row['class'] in ('NSTEMI+nOMI', 'NSTEMI+OMI', 'UA') else row['class']} | "
        f"{row['support']} | "
        f"{fmt_summary(row, 'auroc')} | {fmt_summary(row, 'auprc')} | {fmt_summary(row, 'f1')} | "
        f"{fmt_summary(row, 'sensitivity')} | {fmt_summary(row, 'specificity')} | {fmt_summary(row, 'ppv')} |"
        for row in class_summary_rows
    ) or "| 尚无结果 | — | — | — | — | — | — | — | — | — |"

    if not missing:
        missing_text = "无；所有预定对比方法均完成五折。"
    else:
        missing_text = "\n".join(
            f"- `{item}`" + (f"（已拒绝旧产物：{rejected_results[item]}）" if item in rejected_results else "")
            for item in missing
        )
    lora_classes = [row for row in class_summary_rows if row["method"] == "ECG-LFM LoRA" and row["family"] == "joint6"]
    evaluable = [row for row in lora_classes if row.get("auprc_mean") is not None]
    hardest = min(evaluable, key=lambda row: row["auprc_mean"]) if evaluable else None
    nstemi_omi = next((row for row in lora_classes if row["class"] == "NSTEMI+OMI"), None)
    def pooled_value(family: str, metric: str) -> float | None:
        return lora_pooled.get(family, {}).get(metric) if lora_pooled else None
    deltas = {
        key: None if baseline.get(key) is None else pooled_value(family, metric) - baseline[key]
        for key, family, metric in (
            ("acs_macro_auprc", "acs", "macro_auprc_ovr"),
            ("omi_auprc", "omi_fold_specific_youden", "auprc"),
            ("omi_sensitivity", "omi_fold_specific_youden", "sensitivity"),
            ("nstemi_vs_ua_auprc", "nstemi_vs_ua", "auprc"),
            ("nste_omi_auprc", "omi_among_nste_acs", "auprc"),
        ) if lora_pooled
    }
    recommendation = (
        "当前不建议仅凭现有结果把 Joint-6 定为最终主模型：其 ACS 排序性能略升，但 OMI/NSTE 子组未显示一致优势；"
        "还应等待全部对比方法完成，并用同一 nested 协议重算 factorized baseline 后再冻结主模型。"
        if lora_pooled and deltas.get("omi_auprc", 0.0) <= 0 else
        "Joint-6 可作为候选主模型，但仍需等待全部对比方法与严格同协议 baseline 后确认。"
    )

    report = f"""# ECG-LFM ACS–OMI Joint-6 评估报告

## 1. 评估审计结论

- Joint-6 概率顺序为 `OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`。
- ACS 由 `q0, q1+q2, q3+q4, q5` 恢复并使用 `argmax`；没有类别独立阈值。
- OMI 概率为 `q2+q4`；每折阈值只在 inner-validation 用 Youden Index 选择，然后冻结到 outer-test。
- `NSTEMI vs UA` 在真实 NSTEMI/UA 子集内使用 `(q3+q4)/(q3+q4+q5)`。
- `OMI among NSTE-ACS` 沿用真实 NSTEMI+UA 子集和全局 `p_omi=q2+q4`，未重定义 cohort。
- 本次只修改 evaluation、metrics 和 reporting；未修改模型、loss、LoRA、训练配置或划分。
- Joint-6 的某类别若在 Fold 中缺少正样本，其 OvR 指标输出 `NA/null`，不再伪填 0；全六类 Macro 与 supported-class Macro 在机器可读结果中分开保存。

## 2. 完成状态

{missing_text}

已完成五折的方法：{', '.join(status['complete_5fold_methods']) if status['complete_5fold_methods'] else '无'}。

## 3. Level 1：Joint-6 整体与临床核心指标（五折 mean ± std）

| 方法 | 完成 | Joint-6 Macro AUPRC | Joint-6 Macro F1 | ACS Macro AUPRC | OMI AUPRC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUPRC | OMI Sensitivity | OMI Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_table}

Joint-6 的 primary ranking metric 为 Macro AUPRC；不会只凭 Accuracy 选模型。

### Joint-6 六分类完整整体指标

| 方法 | 完成 | Macro AUPRC | Macro AUROC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
{joint_overall_table}

### 恢复后的 ACS 四分类完整整体指标

| 方法 | 完成 | Macro AUPRC | Macro AUROC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
{acs_overall_table}

### 恢复后的 OMI 二分类完整指标

所有阈值依赖指标均使用各 Fold 的 inner-validation Youden threshold，冻结后用于 outer-test。

| 方法 | 完成 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV | MCC | Accuracy | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{omi_full_table}

| 方法 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
{pooled_confusion_table}

### NSTE-ACS 子组完整排序指标

| 方法 | 完成 | NSTEMI vs UA AUROC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUROC | OMI among NSTE-ACS AUPRC |
|---|---:|---:|---:|---:|---:|
{nste_full_table}

## 4. 五折 95% CI

95% CI 为五折均值的双侧 Student-t 区间；它不是 patient-level bootstrap CI。

| 方法 | Joint-6 Macro AUPRC | ACS Macro AUPRC | OMI AUPRC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUPRC |
|---|---:|---:|---:|---:|---:|
{ci_table}

## 5. Level 2：Joint-6 六类与 ACS 四类逐类指标（五折 mean ± std）

粗体类别中 `NSTEMI+nOMI`、`NSTEMI+OMI`、`UA` 是 NSTE-ACS 重点困难类。`support` 是已完成 outer folds 的累计样本数。

| 方法 | 指标族 | 类别 | Support | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{class_table}

## 6. Level 3：恢复临床任务，与 factorized LoRA baseline 的描述性对比

| Metric | Baseline | Joint-6 ECG-LFM LoRA（pooled OOF） | Δ |
|---|---:|---:|---:|
{chr(10).join(main_rows)}

Baseline 来源为 `{args.baseline}`，其协议标记为 `{baseline_protocol}`。它不是完整 nested checkpoint/threshold selection，因此上述 Δ 仅作描述性参考，特别是阈值依赖的 OMI Sensitivity 不能视为严格配对比较，也不能据此宣称统计显著。

## 7. 五折逐折结果

| 方法 | Fold | Joint-6 Macro AUPRC | Joint-6 Macro F1 | ACS Macro AUPRC | OMI AUPRC | OMI Sensitivity | OMI Specificity | NSTEMI vs UA AUPRC | NSTE-ACS OMI AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_table}

## 8. 用户要求的十项结论

1. **Joint-6 Macro AUPRC**：{fmt(pooled_value('joint6', 'macro_auprc_ovr'))}（LoRA pooled OOF）；五折 mean ± std 为 {fmt_summary(lora_summary, 'joint6_macro_auprc') if lora_summary else 'N/A'}。
2. **最难 joint class**：{hardest['class'] if hardest else 'N/A'}，按五折 class AUPRC 最低判断（{fmt_summary(hardest, 'auprc') if hardest else 'N/A'}）。
3. **NSTEMI+OMI**：AUPRC {fmt_summary(nstemi_omi, 'auprc') if nstemi_omi else 'N/A'}；Sensitivity {fmt_summary(nstemi_omi, 'sensitivity') if nstemi_omi else 'N/A'}。
4. **恢复后的 ACS Macro AUPRC 是否优于 baseline**：{'是，描述性 Δ=' + fmt(deltas.get('acs_macro_auprc')) if deltas.get('acs_macro_auprc') is not None and deltas['acs_macro_auprc'] > 0 else '否或尚无严格证据，描述性 Δ=' + fmt(deltas.get('acs_macro_auprc'))}。
5. **OMI AUPRC 是否优于 baseline**：{'是，描述性 Δ=' + fmt(deltas.get('omi_auprc')) if deltas.get('omi_auprc') is not None and deltas['omi_auprc'] > 0 else '否，描述性 Δ=' + fmt(deltas.get('omi_auprc'))}。
6. **OMI Sensitivity 是否下降**：{'是，描述性 Δ=' + fmt(deltas.get('omi_sensitivity')) if deltas.get('omi_sensitivity') is not None and deltas['omi_sensitivity'] < 0 else '未见下降或无法严格比较，描述性 Δ=' + fmt(deltas.get('omi_sensitivity'))}。该项阈值协议不同，只能描述，不能作严格因果结论。
7. **NSTEMI vs UA 是否改善**：{'是，描述性 Δ=' + fmt(deltas.get('nstemi_vs_ua_auprc')) if deltas.get('nstemi_vs_ua_auprc') is not None and deltas['nstemi_vs_ua_auprc'] > 0 else '否，描述性 Δ=' + fmt(deltas.get('nstemi_vs_ua_auprc'))}。
8. **OMI among NSTE-ACS 是否改善**：{'是，描述性 Δ=' + fmt(deltas.get('nste_omi_auprc')) if deltas.get('nste_omi_auprc') is not None and deltas['nste_omi_auprc'] > 0 else '否，描述性 Δ=' + fmt(deltas.get('nste_omi_auprc'))}。
9. **改善是否跨 Fold 稳定**：现有 factorized baseline 缺少同协议逐折 nested 结果，不能合法计算逐折配对改善率；Joint-6 自身的五折波动已完整列出。
10. **是否值得作为最终主模型**：{recommendation}

## 9. 可追溯输出

- `results/joint6_cv/summary/fold_metrics.csv`
- `results/joint6_cv/summary/summary_metrics.csv`
- `results/joint6_cv/summary/per_class_metrics.csv`
- `results/joint6_cv/summary/per_class_summary.csv`
- `results/joint6_cv/summary/pooled_oof_metrics.json`
- `results/joint6_cv/summary/training_cost.csv`
- `results/joint6_cv/summary/completion_status.json`
- 每个 Fold 的 `joint6_metrics.json`、`per_class_metrics.csv`、`acs_metrics.json`、`omi_metrics.json`、`nste_metrics.json`（由新版 evaluator 生成）。

## 10. 解释限制

- 当前结果采用患者互斥的 nested CV 划分，但 `_aggregate` 后按 ECG record 一票计算 outer-test/pooled OOF 指标；它不是 patient-level 聚合性能，也不是 external test。
- baseline 对比协议并不完全一致，因此未计算配对差异或显著性，也不把描述性差异写成稳定提升。
- 传统模型使用项目既有 P2 输入，ECG-LFM 使用原生预处理；完整对比反映工程方案差异，不是单一网络结构消融。
"""

    requested_report = ROOT / args.report
    requested_report.parent.mkdir(parents=True, exist_ok=True)
    requested_report.write_text(report, encoding="utf-8")
    canonical_report = ROOT / "docs/ECG_LFM_Joint6_Evaluation_Report.md"
    if canonical_report.resolve() != requested_report.resolve():
        canonical_report.write_text(report, encoding="utf-8")
    print(json.dumps({**status, "report": str(canonical_report)}, ensure_ascii=False))
    if args.require_complete and (not status["complete_all_comparators"] or rejected_results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
