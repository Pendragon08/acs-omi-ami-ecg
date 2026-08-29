"""Evaluate one record-level validation fold for hyperparameter screening."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_ecg_lfm import LABELS, METRICS, metric_table, nstemi_vs_ua, select_youden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Prediction file is empty")
    y = np.asarray([[int(row[f"y_{label}"]) for label in LABELS] for row in rows], dtype=np.int64)
    probability = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in rows], dtype=np.float64
    )
    thresholds = np.empty(len(LABELS), dtype=np.float64)
    threshold_details = {}
    for column, label in enumerate(LABELS):
        thresholds[column], threshold_details[label] = select_youden(y[:, column], probability[:, column])
    youden = metric_table(y, probability, thresholds)
    fixed = metric_table(y, probability, np.full(len(LABELS), 0.5))
    payload = {
        "scope": "single internal validation fold; screening only, not final CV or external test",
        "run_name": args.run_name,
        "fold": args.fold,
        "n_records": len(rows),
        "threshold_selection": "0.01--0.99 Youden J on this validation fold",
        "thresholds": {label: float(thresholds[j]) for j, label in enumerate(LABELS)},
        "threshold_details": threshold_details,
        "youden": youden,
        "fixed_0_5": fixed,
        "nstemi_vs_ua": nstemi_vs_ua(y, probability),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fold_evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["run_name", "fold", "threshold_strategy", "label", "threshold", *METRICS, "tp", "tn", "fp", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, table in (("youden", youden), ("fixed_0_5", fixed)):
            for label in (*LABELS, "macro"):
                metric = table[label]
                writer.writerow(
                    {
                        "run_name": args.run_name,
                        "fold": args.fold,
                        "threshold_strategy": strategy,
                        "label": label,
                        **{key: metric[key] for key in fieldnames if key in metric},
                    }
                )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
