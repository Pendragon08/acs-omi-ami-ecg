"""Evaluate MOO Stage-A checkpoints on inner validation only.

This script deliberately does not load ``outer_test.tsv``.  It is therefore
safe for method screening and threshold selection before a candidate is
frozen.  The existing nested evaluator remains the only path used for the
post-selection outer-fold evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ecg_lfm.evaluate_nested_fold import (  # noqa: E402
    arrays,
    expected_assignment,
    metric_bundle,
    prediction_rows,
    write_predictions,
)
from experiments.ecg_lfm.evaluate_joint import (  # noqa: E402
    _read_csv,
    infer_split,
    load_record_ground_truth,
)
from experiments.acs_joint_task import select_omi_threshold, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--run_root", default="results/multi_objective")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--label_csv", default="data/CSV/train.csv")
    parser.add_argument("--output_root", default="results/multi_objective_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    manifest_root = ROOT / args.manifest_root
    manifest_dir = manifest_root / f"fold{args.fold}"
    checkpoint = ROOT / args.run_root / args.method / f"fold{args.fold}" / "checkpoint_best.pt"
    output = ROOT / args.output_root / args.method / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    truth, truth_audit = load_record_ground_truth(
        manifest_root / "master_records.csv", ROOT / args.label_csv
    )
    truth_by_record = {row["record_id"]: row for row in truth}
    assignment_path = manifest_dir / "patient_assignments.csv"
    expected = expected_assignment(assignment_path, "inner_validation")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, p_acs, p_omi = infer_split(
        checkpoint,
        manifest_dir,
        "valid",
        manifest_dir / "valid_segments.csv",
        device,
        args.workers,
        args.batch,
    )
    rows = prediction_rows(
        records,
        p_acs,
        p_omi,
        truth_by_record,
        expected,
        args.fold,
        "inner_validation",
    )
    write_predictions(output / "inner_validation_predictions.csv", rows)
    y_acs, p_acs, y_omi, p_omi = arrays(rows)
    thresholds = {}
    details = {}
    for name in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[name], details[name] = select_omi_threshold(y_omi, p_omi, name)
    metrics = {
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + independent OMI binary",
        "protocol": "nested inner-validation only; outer_test not accessed",
        "method": args.method,
        "fold": args.fold,
        "threshold_source": "inner-validation only",
        "thresholds": thresholds,
        "threshold_details": details,
        "inner_validation": metric_bundle(rows, thresholds),
    }
    assignment_rows = _read_csv(assignment_path)
    patient_sets = {
        subset: {row["patient_id"] for row in assignment_rows if row["subset"] == subset}
        for subset in ("inner_train", "inner_validation", "outer_test")
    }
    overlaps = {
        "inner_train__inner_validation": len(patient_sets["inner_train"] & patient_sets["inner_validation"]),
        "inner_train__outer_test": len(patient_sets["inner_train"] & patient_sets["outer_test"]),
        "inner_validation__outer_test": len(patient_sets["inner_validation"] & patient_sets["outer_test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"patient leakage: {overlaps}")
    alignment = {
        **truth_audit,
        "status": "pass",
        "method": args.method,
        "fold": args.fold,
        "join_key": "record_id",
        "outer_test_accessed": False,
        "patient_overlap": overlaps,
        "missing_predictions": 0,
        "duplicate_predictions": 0,
        "inner_validation_records": len(rows),
        "assignment_sha256": sha256(assignment_path),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluator_sha256": sha256(Path(__file__)),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "inner_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "provenance_alignment_report.json").write_text(
        json.dumps(alignment, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "metrics": str(output / "inner_metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

