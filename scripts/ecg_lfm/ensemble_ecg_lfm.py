"""Legacy four-independent-sigmoid ECG-LFM ensemble.

Retained for historical artifacts only; it is invalid for mutually exclusive
ACS subtypes. Use ``experiments.ecg_lfm.evaluate_joint`` for corrected runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_predictions", type=Path, nargs=5, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, default=Path("data/CSV/test.csv"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    fold_rows: list[list[dict[str, str]]] = []
    for path in args.fold_predictions:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: int(row["idx"]))
        fold_rows.append(rows)
    reference_ids = [row["record_id"] for row in fold_rows[0]]
    if any([row["record_id"] for row in rows] != reference_ids for rows in fold_rows[1:]):
        raise ValueError("External fold prediction record order differs")
    fold_probabilities = np.asarray(
        [[[float(row[f"prob_{label}"]) for label in LABELS] for row in rows] for rows in fold_rows],
        dtype=np.float64,
    )
    probabilities = fold_probabilities.mean(axis=0)
    threshold_map = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = np.asarray([float(threshold_map[label]) for label in LABELS])
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)

    with args.test_csv.open(encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_by_id = {Path(row["ecg_row_record"]).stem: row["ecg_row_record"] for row in official_rows}
    if set(reference_ids) != set(official_by_id):
        raise ValueError("External predictions do not exactly match official test.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    internal = args.output_dir / "external_ensemble_probabilities.csv"
    with internal.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["ecg_row_record", *[f"prob_{label}" for label in LABELS], *LABELS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, record_id in enumerate(reference_ids):
            writer.writerow({
                "ecg_row_record": official_by_id[record_id],
                **{f"prob_{label}": float(probabilities[i, j]) for j, label in enumerate(LABELS)},
                **{label: int(predictions[i, j]) for j, label in enumerate(LABELS)},
            })
    submission = args.output_dir / "submission_ecg_lfm_4label_youden.csv"
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ecg_row_record", *LABELS])
        writer.writeheader()
        for i, record_id in enumerate(reference_ids):
            writer.writerow({
                "ecg_row_record": official_by_id[record_id],
                **{label: int(predictions[i, j]) for j, label in enumerate(LABELS)},
            })
    summary = {
        "n_external_records": len(reference_ids),
        "ensemble": "arithmetic mean of five fold record-level probabilities",
        "thresholds": threshold_map,
        "external_labels_used": False,
        "positive_predictions": {label: int(predictions[:, j].sum()) for j, label in enumerate(LABELS)},
        "internal_predictions": str(internal),
        "submission": str(submission),
    }
    (args.output_dir / "external_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
