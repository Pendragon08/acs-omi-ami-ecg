"""Evaluate a nested run without using the outer fold for model decisions."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES,
    acs_metrics,
    acs_subtype_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    save_confusion_figures,
    save_curves,
    save_reliability_figures,
    save_roc_pr_curves,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint import (  # noqa: E402
    _read_csv,
    infer_split,
    load_record_ground_truth,
    read_history,
)


def expected_assignment(path: Path, subset: str) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if row["subset"] == subset]
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate record ID in nested assignment subset {subset}")
    return rows


def prediction_rows(
    records: list[dict],
    p_acs: np.ndarray,
    p_omi: np.ndarray,
    truth_by_record: dict[str, dict],
    expected: list[dict[str, str]],
    fold: int,
    subset: str,
) -> list[dict]:
    expected_ids = [row["record_id"] for row in expected]
    actual_ids = [row["record_id"] for row in records]
    if actual_ids != expected_ids:
        raise ValueError(f"{subset} inference order does not match explicit assignment record IDs")
    rows: list[dict] = []
    for record, probabilities, omi_probability, assignment in zip(records, p_acs, p_omi, expected):
        truth = truth_by_record.get(record["record_id"])
        if truth is None:
            raise ValueError(f"prediction missing joined ground truth: {record['record_id']}")
        if truth["patient_id"] != record["patient_id"] or truth["patient_id"] != assignment["patient_id"]:
            raise ValueError(f"patient cross-check failed: {record['record_id']}")
        rows.append(
            {
                "record_id": record["record_id"],
                "patient_id": truth["patient_id"],
                "fold": fold,
                "subset": subset,
                "y_acs": truth["y_acs"],
                **{f"p_{name}": float(probabilities[index]) for index, name in enumerate(ACS_CLASSES)},
                "pred_acs": int(np.argmax(probabilities)),
                "y_omi": truth["y_omi"],
                "p_omi": float(omi_probability),
            }
        )
    return rows


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_acs = np.asarray([row["y_acs"] for row in rows], dtype=np.int64)
    p_acs = np.asarray(
        [[row[f"p_{name}"] for name in ACS_CLASSES] for row in rows], dtype=np.float64
    )
    y_omi = np.asarray([row["y_omi"] for row in rows], dtype=np.int64)
    p_omi = np.asarray([row["p_omi"] for row in rows], dtype=np.float64)
    return y_acs, p_acs, y_omi, p_omi


def metric_bundle(rows: list[dict], thresholds: dict[str, float | None]) -> dict:
    y_acs, p_acs, y_omi, p_omi = arrays(rows)
    nste = np.isin(y_acs, (2, 3))
    omi_metrics = {
        name: None if threshold is None else binary_metrics(y_omi, p_omi, threshold)
        for name, threshold in thresholds.items()
    }
    youden = thresholds.get("youden_j")
    return {
        "n": len(rows),
        "acs": acs_metrics(y_acs, p_acs),
        "conditional_acs": acs_subtype_metrics(y_acs, p_acs),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi": omi_metrics,
        "omi_among_nste_acs": {
            "n": int(nste.sum()),
            "metrics": None
            if youden is None
            else binary_metrics(y_omi[nste], p_omi[nste], float(youden)),
        },
        "calibration_uncalibrated": {
            "acs": multiclass_calibration_metrics(y_acs, p_acs),
            "omi": binary_calibration_metrics(y_omi, p_omi),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--run_root", default="results/adaptive_weighting")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--label_csv", default="data/CSV/train.csv")
    parser.add_argument("--output_root", default="results/adaptive_weighting_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    manifest_root = ROOT / args.manifest_root
    manifest_dir = manifest_root / f"fold{args.fold}"
    checkpoint = ROOT / args.run_root / args.run_name / f"fold{args.fold}" / "checkpoint_best.pt"
    output = ROOT / args.output_root / args.run_name / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    truth, truth_audit = load_record_ground_truth(manifest_root / "master_records.csv", ROOT / args.label_csv)
    truth_by_record = {row["record_id"]: row for row in truth}
    assignment_path = manifest_dir / "patient_assignments.csv"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_rows: dict[str, list[dict]] = {}
    split_mapping = {"inner_validation": "valid", "outer_test": "outer_test"}
    for subset, fairseq_split in split_mapping.items():
        expected = expected_assignment(assignment_path, subset)
        records, p_acs, p_omi = infer_split(
            checkpoint,
            manifest_dir,
            fairseq_split,
            manifest_dir / f"{fairseq_split}_segments.csv",
            device,
            args.workers,
            args.batch,
        )
        all_rows[subset] = prediction_rows(
            records, p_acs, p_omi, truth_by_record, expected, args.fold, subset
        )
        write_predictions(output / f"{subset}_predictions.csv", all_rows[subset])

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

    y_inner_acs, p_inner_acs, y_inner_omi, p_inner_omi = arrays(all_rows["inner_validation"])
    thresholds: dict[str, float | None] = {}
    threshold_details: dict[str, dict] = {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[method], threshold_details[method] = select_omi_threshold(
            y_inner_omi, p_inner_omi, method
        )
    metrics = {
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + independent OMI binary",
        "protocol": "patient-level nested validation",
        "run_name": args.run_name,
        "outer_fold": args.fold,
        "threshold_source": "inner-validation only; frozen before outer-test evaluation",
        "thresholds": thresholds,
        "threshold_details_inner_validation": threshold_details,
        "inner_validation": metric_bundle(all_rows["inner_validation"], thresholds),
        "outer_test": metric_bundle(all_rows["outer_test"], thresholds),
    }
    alignment = {
        **truth_audit,
        "status": "pass",
        "run_name": args.run_name,
        "outer_fold": args.fold,
        "join_key": "record_id",
        "patient_overlap": overlaps,
        "missing_predictions": 0,
        "duplicate_predictions": 0,
        "inner_validation_records": len(all_rows["inner_validation"]),
        "outer_test_records": len(all_rows["outer_test"]),
        "assignment_sha256": sha256(assignment_path),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluator_sha256": sha256(Path(__file__)),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "provenance_alignment_report.json").write_text(
        json.dumps(alignment, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    y_outer_acs, p_outer_acs, y_outer_omi, p_outer_omi = arrays(all_rows["outer_test"])
    save_roc_pr_curves(output, y_outer_acs, p_outer_acs, y_outer_omi, p_outer_omi)
    outer_bundle = metrics["outer_test"]
    youden_metrics = outer_bundle["omi"]["youden_j"]
    save_confusion_figures(
        output,
        np.asarray(outer_bundle["acs"]["confusion_matrix"]),
        np.asarray(outer_bundle["conditional_acs"]["confusion_matrix"]),
        np.asarray(
            [[youden_metrics["tn"], youden_metrics["fp"]], [youden_metrics["fn"], youden_metrics["tp"]]]
        ),
    )
    save_reliability_figures(
        output,
        outer_bundle["calibration_uncalibrated"]["acs"],
        outer_bundle["calibration_uncalibrated"]["omi"],
    )
    save_curves(output, read_history(ROOT / args.run_root, args.run_name), f"Nested {args.run_name}")
    print(json.dumps({"status": "pass", "metrics": str(output / "metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
