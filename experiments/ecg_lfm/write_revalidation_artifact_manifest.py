"""Hash protected Phase-1 outputs without copying patient-level files to Git."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.acs_joint_task import sha256  # noqa: E402


def csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def main() -> None:
    result_root = ROOT / "results" / "revalidated"
    manifest_root = ROOT / "manifests" / "ecg_lfm"
    methods = {}
    for method in ("baseline", "lora", "dora"):
        output = result_root / method
        oof = output / "oof_predictions.csv"
        row_count, header = csv_shape(oof)
        if row_count != 17960:
            raise ValueError(f"unexpected OOF row count for {method}: {row_count}")
        methods[method] = {
            "oof_predictions": {
                "controlled_server_path": str(oof),
                "sha256": sha256(oof),
                "row_count": row_count,
                "columns": header,
                "contains_patient_identifiers": True,
                "copied_to_git": False,
            },
            "metrics_sha256": sha256(output / "metrics.json"),
            "alignment_sha256": sha256(output / "provenance_alignment_report.json"),
            "fold_manifests": {
                str(fold): {
                    "train_segments_sha256": sha256(manifest_root / f"fold{fold}" / "train_segments.csv"),
                    "valid_segments_sha256": sha256(manifest_root / f"fold{fold}" / "valid_segments.csv"),
                }
                for fold in range(5)
            },
        }
    payload = {
        "status": "pass",
        "phase": "Phase-1 corrected checkpoint re-inference",
        "privacy": "record/patient-level OOF remains on controlled server; only hashes and aggregate JSON are archived",
        "methods": methods,
    }
    (result_root / "output_artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "output": str(result_root / "output_artifact_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
