"""Verify a Fold-0 record-level prediction artifact before full five-fold runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--master_csv", type=Path, default=Path("manifests/ecg_lfm/master_records.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_macro_auroc", type=float, default=0.55)
    args = parser.parse_args()

    with args.master_csv.open(encoding="utf-8", newline="") as handle:
        expected = {
            int(row["idx"])
            for row in csv.DictReader(handle)
            if row["cohort"] == "train" and int(row["validation_fold"]) == 0
        }
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = [int(row["idx"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("Fold-0 predictions contain duplicate record indices")
    if set(observed) != expected:
        raise ValueError(
            f"Fold-0 coverage mismatch: observed={len(observed)}, expected={len(expected)}"
        )

    y = np.asarray(
        [[int(row[f"y_{label}"]) for label in LABELS] for row in rows], dtype=np.int64
    )
    probability = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in rows], dtype=np.float64
    )
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Fold-0 probabilities are not finite values in [0,1]")

    metrics: dict[str, dict[str, float | int]] = {}
    for column, label in enumerate(LABELS):
        if len(np.unique(y[:, column])) != 2:
            raise ValueError(f"Fold-0 {label} labels do not contain both classes")
        metrics[label] = {
            "positives": int(y[:, column].sum()),
            "auroc": float(roc_auc_score(y[:, column], probability[:, column])),
            "auprc": float(average_precision_score(y[:, column], probability[:, column])),
            "probability_min": float(probability[:, column].min()),
            "probability_max": float(probability[:, column].max()),
            "probability_std": float(probability[:, column].std()),
        }
    macro_auroc = float(np.mean([metrics[label]["auroc"] for label in LABELS]))
    macro_auprc = float(np.mean([metrics[label]["auprc"] for label in LABELS]))
    nondegenerate = all(metrics[label]["probability_std"] > 1e-6 for label in LABELS)
    passed = bool(nondegenerate and macro_auroc >= args.minimum_macro_auroc)
    result = {
        "gate": "Fold-0 finite/coverage/nondegenerate/performance sanity gate",
        "n_records": len(rows),
        "expected_validation_fold": 0,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "minimum_macro_auroc": args.minimum_macro_auroc,
        "nondegenerate_probabilities": nondegenerate,
        "metrics": metrics,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ECG_LFM_FOLD0_WITNESS " + json.dumps(result, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
