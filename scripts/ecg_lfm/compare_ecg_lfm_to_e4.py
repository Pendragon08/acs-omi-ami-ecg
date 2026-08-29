"""Create a traceable paired OOF comparison between ECG-LFM and historical E4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
RANKING = ("auroc", "auprc")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def threshold_metric(name: str, y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    if name == "f1":
        return float(f1_score(y, prediction, zero_division=0))
    positive = y == 1
    negative = ~positive
    if name == "sensitivity":
        return float(prediction[positive].mean())
    if name == "specificity":
        return float((~prediction[negative]).mean())
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg_summary", type=Path, required=True)
    parser.add_argument("--ecg_oof", type=Path, required=True)
    parser.add_argument(
        "--e4_summary", type=Path,
        default=Path("results/hmoe_kan_stemi_nstemi_ua_omi_e4/e4_4label_summary.json"),
    )
    parser.add_argument(
        "--e4_oof", type=Path,
        default=Path("results/hmoe_kan_stemi_nstemi_ua_omi_e4/oof_predictions.csv"),
    )
    parser.add_argument("--model_name", default="ECG-LFM")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    ecg_summary = json.loads(args.ecg_summary.read_text(encoding="utf-8"))
    e4_summary = json.loads(args.e4_summary.read_text(encoding="utf-8"))
    ecg_rows = load_rows(args.ecg_oof)
    e4_rows = load_rows(args.e4_oof)
    ecg_by_id = {Path(row["record_id"]).stem: row for row in ecg_rows}
    e4_by_id = {Path(row["ecg_row_record"]).stem: row for row in e4_rows}
    if set(ecg_by_id) != set(e4_by_id) or len(ecg_by_id) != 17_960:
        raise ValueError("ECG-LFM and E4 OOF record IDs are not the same 17,960 records")
    ids = sorted(ecg_by_id, key=lambda item: int(ecg_by_id[item]["idx"]))
    patients = np.asarray([ecg_by_id[item]["patient_id"] for item in ids], dtype=object)
    if any(ecg_by_id[item]["patient_id"] != e4_by_id[item]["patient_id"] for item in ids):
        raise ValueError("Patient IDs disagree between paired OOF files")
    unique_patients = np.unique(patients)
    patient_rows = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}

    y = np.asarray([[int(ecg_by_id[item][f"y_{label}"]) for label in LABELS] for item in ids])
    ecg_probability = np.asarray([[float(ecg_by_id[item][f"prob_{label}"]) for label in LABELS] for item in ids])
    e4_probability = np.asarray([[float(e4_by_id[item][f"p_{label}"]) for label in LABELS] for item in ids])
    if not np.array_equal(
        y,
        np.asarray([[int(e4_by_id[item][f"y_{label}"]) for label in LABELS] for item in ids]),
    ):
        raise ValueError("Labels disagree between paired OOF files")
    ecg_thresholds = np.asarray([ecg_summary["thresholds"][label] for label in LABELS], dtype=float)
    e4_thresholds = np.asarray(e4_summary["thresholds"]["youden_j"], dtype=float)

    pooled_rows = []
    for model_name, table in (
        ("E4 HMoE-KAN", e4_summary["oof_metrics_by_threshold_strategy"]["youden_j"]),
        (args.model_name, ecg_summary["oof_youden"]),
    ):
        for label in (*LABELS, "macro"):
            pooled_rows.append({"model": model_name, "label": label, **table[label]})

    rng = np.random.default_rng(args.seed)
    bootstrap_result: dict[str, dict[str, object]] = {}
    for label_idx, label in enumerate(LABELS):
        metrics = (*RANKING, "sensitivity", "specificity", "f1")
        deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(args.bootstrap):
            sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            sampled = np.concatenate([patient_rows[patient] for patient in sampled_patients])
            y_sample = y[sampled, label_idx]
            if len(np.unique(y_sample)) < 2:
                continue
            for metric in metrics:
                if metric == "auroc":
                    ecg_value = roc_auc_score(y_sample, ecg_probability[sampled, label_idx])
                    e4_value = roc_auc_score(y_sample, e4_probability[sampled, label_idx])
                elif metric == "auprc":
                    ecg_value = average_precision_score(y_sample, ecg_probability[sampled, label_idx])
                    e4_value = average_precision_score(y_sample, e4_probability[sampled, label_idx])
                else:
                    ecg_value = threshold_metric(metric, y_sample, ecg_probability[sampled, label_idx], ecg_thresholds[label_idx])
                    e4_value = threshold_metric(metric, y_sample, e4_probability[sampled, label_idx], e4_thresholds[label_idx])
                deltas[metric].append(float(ecg_value - e4_value))
        bootstrap_result[label] = {
            metric: {
                "delta_ecg_lfm_minus_e4": float(
                    ecg_summary["oof_youden"][label][metric]
                    - e4_summary["oof_metrics_by_threshold_strategy"]["youden_j"][label][metric]
                ),
                "paired_patient_cluster_bootstrap_95_ci": interval(values),
                "n_bootstrap": len(values),
            }
            for metric, values in deltas.items()
        }

    # Report the primary macro comparison with a single paired patient-cluster
    # resample shared by all four labels in each bootstrap replicate.
    macro_deltas: dict[str, list[float]] = {
        metric: [] for metric in (*RANKING, "sensitivity", "specificity", "f1")
    }
    macro_rng = np.random.default_rng(args.seed + 1)
    for _ in range(args.bootstrap):
        sampled_patients = macro_rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled = np.concatenate([patient_rows[patient] for patient in sampled_patients])
        if any(len(np.unique(y[sampled, j])) < 2 for j in range(len(LABELS))):
            continue
        for metric in macro_deltas:
            ecg_values = []
            e4_values = []
            for label_idx in range(len(LABELS)):
                y_sample = y[sampled, label_idx]
                if metric == "auroc":
                    ecg_value = roc_auc_score(y_sample, ecg_probability[sampled, label_idx])
                    e4_value = roc_auc_score(y_sample, e4_probability[sampled, label_idx])
                elif metric == "auprc":
                    ecg_value = average_precision_score(y_sample, ecg_probability[sampled, label_idx])
                    e4_value = average_precision_score(y_sample, e4_probability[sampled, label_idx])
                else:
                    ecg_value = threshold_metric(
                        metric, y_sample, ecg_probability[sampled, label_idx], ecg_thresholds[label_idx]
                    )
                    e4_value = threshold_metric(
                        metric, y_sample, e4_probability[sampled, label_idx], e4_thresholds[label_idx]
                    )
                ecg_values.append(float(ecg_value))
                e4_values.append(float(e4_value))
            macro_deltas[metric].append(float(np.mean(ecg_values) - np.mean(e4_values)))
    bootstrap_result["macro"] = {
        metric: {
            "delta_ecg_lfm_minus_e4": float(
                ecg_summary["oof_youden"]["macro"][metric]
                - e4_summary["oof_metrics_by_threshold_strategy"]["youden_j"]["macro"][metric]
            ),
            "paired_patient_cluster_bootstrap_95_ci": interval(values),
            "n_bootstrap": len(values),
        }
        for metric, values in macro_deltas.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in pooled_rows for key in row}, key=lambda key: (key not in ("model", "label"), key))
    with (args.output_dir / "ecg_lfm_vs_e4.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pooled_rows)
    payload = {
        "comparison": f"{args.model_name} minus historical E4 on exact paired OOF records",
        "n_records": len(ids),
        "n_patients": len(unique_patients),
        "thresholds": {
            args.model_name: dict(zip(LABELS, ecg_thresholds.tolist())),
            "E4 HMoE-KAN": dict(zip(LABELS, e4_thresholds.tolist())),
        },
        "paired_patient_cluster_bootstrap": bootstrap_result,
    }
    (args.output_dir / "paired_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
