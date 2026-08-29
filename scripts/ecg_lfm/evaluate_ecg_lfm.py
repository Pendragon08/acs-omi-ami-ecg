"""Evaluate record-level ECG-LFM OOF predictions and select Youden thresholds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "youden_j": float(sensitivity + specificity - 1.0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def select_youden(y: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = [binary_metrics(y, probability, float(threshold)) for threshold in GRID]
    best = max(
        candidates,
        key=lambda item: (
            item["youden_j"], item["f1"], item["sensitivity"], -abs(item["threshold"] - 0.5)
        ),
    )
    return float(best["threshold"]), best


def metric_table(y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, dict]:
    table = {
        label: binary_metrics(y[:, j], probability[:, j], float(thresholds[j]))
        for j, label in enumerate(LABELS)
    }
    table["macro"] = {metric: float(np.mean([table[label][metric] for label in LABELS])) for metric in METRICS}
    return table


def nstemi_vs_ua(y: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    mask = np.logical_xor(y[:, 1] == 1, y[:, 2] == 1)
    target = (y[mask, 1] == 1).astype(np.int64)
    denom = probability[mask, 1] + probability[mask, 2]
    score = np.divide(probability[mask, 1], denom, out=np.full_like(denom, 0.5), where=denom > 0)
    prediction = (probability[mask, 1] >= probability[mask, 2]).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "definition": "records with exactly one of NSTEMI/UA positive; NSTEMI is positive class",
        "n_records": int(mask.sum()),
        "n_nstemi": int(target.sum()),
        "n_ua": int((target == 0).sum()),
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "nstemi_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "ua_sensitivity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion": {"tn_ua": int(tn), "fp_ua_as_nstemi": int(fp), "fn_nstemi_as_ua": int(fn), "tp_nstemi": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_predictions", type=Path, nargs=5, required=True)
    parser.add_argument("--master_csv", type=Path, default=Path("manifests/ecg_lfm/master_records.csv"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    master_folds: dict[int, int] = {}
    with args.master_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["cohort"] == "train":
                master_folds[int(row["idx"])] = int(row["validation_fold"])

    rows: list[dict[str, str]] = []
    seen: set[int] = set()
    for expected_fold, path in enumerate(args.fold_predictions):
        with path.open(encoding="utf-8", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        for row in fold_rows:
            idx = int(row["idx"])
            if master_folds.get(idx) != expected_fold:
                raise ValueError(f"{path}: record {idx} belongs to fold {master_folds.get(idx)}, not {expected_fold}")
            if idx in seen:
                raise ValueError(f"Duplicate OOF prediction for record {idx}")
            seen.add(idx)
            row["fold"] = str(expected_fold)
            rows.append(row)
    if seen != set(master_folds):
        raise ValueError(f"OOF coverage mismatch: predicted={len(seen)}, expected={len(master_folds)}")
    rows.sort(key=lambda row: int(row["idx"]))
    y = np.asarray([[int(row[f"y_{label}"]) for label in LABELS] for row in rows], dtype=np.int64)
    probability = np.asarray([[float(row[f"prob_{label}"]) for label in LABELS] for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)

    thresholds = np.empty(len(LABELS), dtype=np.float64)
    threshold_details = {}
    for j, label in enumerate(LABELS):
        thresholds[j], threshold_details[label] = select_youden(y[:, j], probability[:, j])
    pooled = metric_table(y, probability, thresholds)
    fixed = metric_table(y, probability, np.full(len(LABELS), 0.5))
    fold_metrics = []
    for fold in range(5):
        selected = folds == fold
        fold_metrics.append({
            "fold": fold,
            "n_records": int(selected.sum()),
            "metrics": metric_table(y[selected], probability[selected], thresholds),
            "nstemi_vs_ua": nstemi_vs_ua(y[selected], probability[selected]),
        })
    cv_summary: dict[str, dict[str, dict[str, float]]] = {}
    for label in (*LABELS, "macro"):
        cv_summary[label] = {}
        for metric in METRICS:
            values = np.asarray([fold["metrics"][label][metric] for fold in fold_metrics], dtype=np.float64)
            cv_summary[label][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    pairwise_keys = (
        "auroc", "auprc", "macro_f1", "balanced_accuracy",
        "nstemi_sensitivity", "ua_sensitivity",
    )
    pairwise_cv = {}
    for metric in pairwise_keys:
        values = np.asarray([fold["nstemi_vs_ua"][metric] for fold in fold_metrics], dtype=np.float64)
        pairwise_cv[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    summary = {
        "task": "four independent sigmoid binary labels",
        "label_order": list(LABELS),
        "threshold_selection": "0.01--0.99 pooled OOF Youden J; external labels never used",
        "thresholds": {label: float(thresholds[j]) for j, label in enumerate(LABELS)},
        "threshold_details": threshold_details,
        "oof_youden": pooled,
        "oof_fixed_0_5": fixed,
        "folds_at_global_oof_thresholds": fold_metrics,
        "fivefold_mean_std_at_global_oof_thresholds": cv_summary,
        "nstemi_vs_ua": nstemi_vs_ua(y, probability),
        "nstemi_vs_ua_fivefold_mean_std": pairwise_cv,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof_evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "thresholds.json").write_text(
        json.dumps({label: float(thresholds[j]) for j, label in enumerate(LABELS)}, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "metrics_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scope", "label", "threshold", *METRICS, "tp", "tn", "fp", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in LABELS:
            metric = pooled[label]
            writer.writerow({"scope": "pooled_oof", "label": label, **{key: metric[key] for key in fieldnames if key in metric}})
        for fold in fold_metrics:
            for label in LABELS:
                metric = fold["metrics"][label]
                writer.writerow({"scope": f"fold{fold['fold']}", "label": label, **{key: metric[key] for key in fieldnames if key in metric}})
    with (args.output_dir / "cv_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["label", "metric", "mean", "std", "formatted"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in cv_summary.items():
            for metric, values in metrics.items():
                writer.writerow({
                    "label": label,
                    "metric": metric,
                    **values,
                    "formatted": f"{values['mean']:.4f} ± {values['std']:.4f}",
                })
    with (args.output_dir / "nstemi_vs_ua.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scope", *pairwise_keys]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"scope": "pooled_oof", **{key: summary["nstemi_vs_ua"][key] for key in pairwise_keys}})
        for fold in fold_metrics:
            writer.writerow({"scope": f"fold{fold['fold']}", **{key: fold["nstemi_vs_ua"][key] for key in pairwise_keys}})
        writer.writerow({"scope": "fivefold_mean", **{key: pairwise_cv[key]["mean"] for key in pairwise_keys}})
        writer.writerow({"scope": "fivefold_std", **{key: pairwise_cv[key]["std"] for key in pairwise_keys}})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
