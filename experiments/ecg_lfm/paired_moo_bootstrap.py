"""Paired patient-cluster bootstrap for a frozen MOO candidate vs W0."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from experiments.acs_joint_task import ACS_CLASSES, acs_metrics


def read_rows(root: Path, method: str) -> list[dict[str, str]]:
    rows = []
    for fold in range(5):
        path = root / method / f"fold{fold}" / "outer_test_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        if not fold_rows or any(int(row["fold"]) != fold or row.get("subset") != "outer_test" for row in fold_rows):
            raise ValueError(f"invalid outer-test predictions: {path}")
        rows.extend(fold_rows)
    rows.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate outer record IDs for {method}")
    return rows


def align(reference: list[dict[str, str]], candidate: list[dict[str, str]]) -> None:
    if len(reference) != len(candidate):
        raise ValueError("outer prediction counts differ")
    for left, right in zip(reference, candidate):
        for key in ("record_id", "patient_id", "y_acs", "y_omi"):
            if left[key] != right[key]:
                raise ValueError(f"paired alignment mismatch for {left['record_id']}: {key}")


def endpoints(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, float]:
    selected = [rows[int(i)] for i in indices]
    y_acs = np.asarray([int(row["y_acs"]) for row in selected], dtype=np.int64)
    p_acs = np.asarray([[float(row[f"p_{name}"]) for name in ACS_CLASSES] for row in selected])
    y_omi = np.asarray([int(row["y_omi"]) for row in selected], dtype=np.int64)
    p_omi = np.asarray([float(row["p_omi"]) for row in selected])
    acs = acs_metrics(y_acs, p_acs)
    return {
        "acs_macro_auprc": float(acs["macro_auprc_ovr"]),
        "nstemi_auprc": float(acs["per_class"]["NSTEMI"]["auprc"]),
        "omi_auprc": float(average_precision_score(y_omi, p_omi)),
    }


def compare(reference, candidate, replicates: int, seed: int) -> dict:
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        patient_indices[row["patient_id"]].append(index)
    patients = np.asarray(sorted(patient_indices), dtype=object)
    full = np.arange(len(reference), dtype=np.int64)
    ref_point, cand_point = endpoints(reference, full), endpoints(candidate, full)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.asarray([i for patient in sampled for i in patient_indices[str(patient)]], dtype=np.int64)
        ref, cand = endpoints(reference, indices), endpoints(candidate, indices)
        for metric in cand:
            samples[metric].append(cand[metric] - ref[metric])
    result = {}
    for metric in cand_point:
        values = np.asarray(samples[metric], dtype=float)
        low, high = np.percentile(values, [2.5, 97.5])
        result[metric] = {
            "candidate": cand_point[metric],
            "reference": ref_point[metric],
            "delta": cand_point[metric] - ref_point[metric],
            "ci95": [float(low), float(high)],
            "ci_crosses_zero": bool(low <= 0 <= high),
            "finite_replicates": int(np.isfinite(values).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--candidate_root", type=Path, default=Path("results/multi_objective_evaluation"))
    parser.add_argument("--reference", default="w0")
    parser.add_argument("--reference_root", type=Path, default=Path("results/adaptive_weighting_evaluation"))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/multi_objective/paired_patient_bootstrap.json"))
    args = parser.parse_args()
    reference = read_rows(args.reference_root, args.reference)
    comparisons = {}
    for method in args.candidate:
        candidate = read_rows(args.candidate_root, method)
        align(reference, candidate)
        comparisons[f"{method}-{args.reference}"] = compare(reference, candidate, args.replicates, args.seed)
    patients = {row["patient_id"] for row in reference}
    result = {
        "status": "pass",
        "unit": "patient cluster; all records for each sampled patient retained",
        "paired_on": "record_id with patient_id and labels cross-checked",
        "reference": args.reference,
        "candidates": args.candidate,
        "replicates": args.replicates,
        "seed": args.seed,
        "n_patients": len(patients),
        "n_records": len(reference),
        "comparisons": comparisons,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

