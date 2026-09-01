"""Patient-cluster paired bootstrap for completed nested outer-fold predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from experiments.acs_joint_task import ACS_CLASSES, acs_metrics


COMPARISONS = (("w6", "w0"), ("w6", "w3"), ("w4", "w1"), ("w5", "w2"))


def read_method(root: Path, method: str) -> list[dict]:
    rows = []
    for fold in range(5):
        path = root / method / f"fold{fold}" / "outer_test_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        if any(int(row["fold"]) != fold or row.get("subset") != "outer_test" for row in fold_rows):
            raise ValueError(f"invalid outer fold metadata: {path}")
        rows.extend(fold_rows)
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate outer record IDs for {method}")
    return sorted(rows, key=lambda row: row["record_id"])


def aligned(a: list[dict], b: list[dict]) -> None:
    for left, right in zip(a, b):
        keys = ("record_id", "patient_id", "y_acs", "y_omi")
        if any(left[key] != right[key] for key in keys):
            raise ValueError(f"paired alignment mismatch: {left['record_id']} / {right['record_id']}")


def arrays(rows: list[dict], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [rows[int(index)] for index in indices]
    y_acs = np.asarray([int(row["y_acs"]) for row in selected])
    p_acs = np.asarray([[float(row[f"p_{name}"]) for name in ACS_CLASSES] for row in selected])
    y_omi = np.asarray([int(row["y_omi"]) for row in selected])
    p_omi = np.asarray([float(row["p_omi"]) for row in selected])
    return y_acs, p_acs, y_omi, p_omi


def endpoints(rows: list[dict], indices: np.ndarray) -> dict[str, float]:
    y_acs, p_acs, y_omi, p_omi = arrays(rows, indices)
    acs = acs_metrics(y_acs, p_acs)
    return {
        "acs_macro_auprc": float(acs["macro_auprc_ovr"]),
        "nstemi_auprc": float(acs["per_class"]["NSTEMI"]["auprc"]),
        "omi_auprc": float(average_precision_score(y_omi, p_omi)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_root", type=Path, default=Path("results/adaptive_weighting_evaluation"))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_weighting/paired_patient_bootstrap.json"))
    args = parser.parse_args()
    methods = {name for comparison in COMPARISONS for name in comparison}
    data = {method: read_method(args.evaluation_root, method) for method in sorted(methods)}
    reference = data["w0"]
    for rows in data.values():
        if len(rows) != len(reference):
            raise ValueError("method outer prediction counts differ")
        aligned(reference, rows)
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        patient_indices[row["patient_id"]].append(index)
    patients = np.asarray(sorted(patient_indices), dtype=object)
    full = np.arange(len(reference), dtype=np.int64)
    point = {method: endpoints(rows, full) for method, rows in data.items()}
    rng = np.random.default_rng(args.seed)
    boot = {f"{left}-{right}": defaultdict(list) for left, right in COMPARISONS}
    for _ in range(args.replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.asarray([index for patient in sampled for index in patient_indices[str(patient)]], dtype=np.int64)
        sampled_metrics = {method: endpoints(rows, indices) for method, rows in data.items()}
        for left, right in COMPARISONS:
            key = f"{left}-{right}"
            for metric in point[left]:
                boot[key][metric].append(sampled_metrics[left][metric] - sampled_metrics[right][metric])
    comparisons = {}
    for left, right in COMPARISONS:
        key = f"{left}-{right}"
        comparisons[key] = {}
        for metric in point[left]:
            values = np.asarray(boot[key][metric], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) < int(0.95 * args.replicates):
                raise RuntimeError(f"too many non-finite bootstrap replicates: {key} {metric}")
            lower, upper = np.percentile(finite, [2.5, 97.5])
            comparisons[key][metric] = {
                "delta": point[left][metric] - point[right][metric],
                "ci95": [float(lower), float(upper)],
                "ci_crosses_zero": bool(lower <= 0.0 <= upper),
                "finite_replicates": int(len(finite)),
            }
    result = {
        "status": "pass",
        "unit": "patient cluster; all ECG records for each sampled patient retained",
        "paired_on": "record_id with patient_id and labels cross-checked",
        "replicates": args.replicates,
        "seed": args.seed,
        "n_patients": int(len(patients)),
        "n_records": int(len(reference)),
        "point_estimates": point,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
