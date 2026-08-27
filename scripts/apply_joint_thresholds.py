"""Apply validation-selected joint thresholds to cached official-test probabilities.

Creates new `_joint` files only.  It never overwrites the original prediction
CSVs and never submits anything to the official evaluation platform.
"""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "final_fusion_focal"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    selection = json.loads((RES / "threshold_selection_summary.json").read_text())
    manifest = {
        "selection_source": "threshold_selection_summary.json",
        "selection_policy": selection["selection_policy"],
        "submitted_to_official_platform": False,
        "tasks": {},
    }

    for task in ("omi", "ami"):
        label = task.upper()
        threshold = float(selection["tasks"][label]["recommended"]["threshold"])
        source = RES / f"official_test_{task}_XceptionTime.csv"
        with source.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        probability_col = f"{label}_probability"
        prediction_col = f"{label}_prediction"
        for row in rows:
            row[prediction_col] = str(int(float(row[probability_col]) >= threshold))

        internal = RES / f"official_test_{task}_XceptionTime_joint.csv"
        with internal.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        submission = RES / f"submission_{task}_XceptionTime_joint.csv"
        with submission.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ecg_row_record", label])
            for row in rows:
                writer.writerow([row["ecg_row_record"], row[prediction_col]])

        manifest["tasks"][label] = {
            "threshold": threshold,
            "n_rows": len(rows),
            "n_positive_predictions": sum(int(row[prediction_col]) for row in rows),
            "source_probability_csv": source.name,
            "source_probability_sha256": sha256(source),
            "internal_output": internal.name,
            "internal_output_sha256": sha256(internal),
            "submission_output": submission.name,
            "submission_output_sha256": sha256(submission),
        }

    (RES / "joint_threshold_predictions.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
