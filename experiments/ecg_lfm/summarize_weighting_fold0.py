"""Build a traceable Fold0 gate table for W0--W6 nested experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = tuple(f"w{index}" for index in range(7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(data: dict, *keys: str):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric_row(method: str, metrics_path: Path, alignment_path: Path) -> dict:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    outer = data["outer_test"]
    inner = data["inner_validation"]
    omi_threshold = nested(data, "thresholds", "youden_j")
    outer_omi = nested(outer, "omi", "youden_j") or {}
    return {
        "method": method,
        "fold": int(data["outer_fold"]),
        "alignment_status": alignment.get("status"),
        "patient_overlap_total": sum(int(value) for value in alignment["patient_overlap"].values()),
        "inner_acs_macro_auprc": nested(inner, "acs", "macro_auprc_ovr"),
        "inner_nstemi_auprc": nested(inner, "acs", "per_class", "NSTEMI", "auprc"),
        "inner_omi_auprc": nested(inner, "omi", "youden_j", "auprc"),
        "outer_acs_macro_auprc": nested(outer, "acs", "macro_auprc_ovr"),
        "outer_acs_macro_f1": nested(outer, "acs", "macro_f1"),
        "outer_acs_balanced_accuracy": nested(outer, "acs", "balanced_accuracy"),
        "outer_nstemi_auprc": nested(outer, "acs", "per_class", "NSTEMI", "auprc"),
        "outer_nstemi_f1": nested(outer, "acs", "per_class", "NSTEMI", "f1"),
        "outer_nstemi_vs_ua_auprc": nested(outer, "nstemi_vs_ua", "auprc"),
        "outer_omi_auroc": outer_omi.get("auroc"),
        "outer_omi_auprc": outer_omi.get("auprc"),
        "outer_omi_f1": outer_omi.get("f1"),
        "outer_omi_sensitivity": outer_omi.get("sensitivity"),
        "outer_omi_specificity": outer_omi.get("specificity"),
        "outer_omi_ppv": outer_omi.get("ppv"),
        "outer_omi_npv": outer_omi.get("npv"),
        "outer_omi_threshold_from_inner": omi_threshold,
        "outer_omi_brier": nested(outer, "calibration_uncalibrated", "omi", "brier"),
        "outer_omi_ece": nested(outer, "calibration_uncalibrated", "omi", "ece"),
        "outer_acs_brier": nested(outer, "calibration_uncalibrated", "acs", "multiclass_brier"),
        "outer_acs_ece": nested(outer, "calibration_uncalibrated", "acs", "top_label_ece"),
        "omi_among_nste_auprc": nested(outer, "omi_among_nste_acs", "metrics", "auprc"),
        "metrics_sha256": sha256(metrics_path),
        "alignment_sha256": sha256(alignment_path),
        "checkpoint_sha256": alignment.get("checkpoint_sha256"),
        "split_provenance_sha256": alignment.get("split_provenance_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_root", type=Path, default=Path("results/adaptive_weighting_evaluation"))
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_weighting/fold0_comparison.csv"))
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    rows = []
    missing = []
    for method in METHODS:
        directory = args.evaluation_root / method / "fold0"
        metrics_path = directory / "metrics.json"
        alignment_path = directory / "provenance_alignment_report.json"
        if not metrics_path.exists() or not alignment_path.exists():
            missing.append(method)
            continue
        row = metric_row(method, metrics_path, alignment_path)
        if row["alignment_status"] != "pass" or row["patient_overlap_total"] != 0:
            raise RuntimeError(f"{method} failed alignment gate: {row}")
        rows.append(row)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing Fold0 evaluations: {missing}")
    if not rows:
        raise RuntimeError("no completed Fold0 evaluations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "rows": len(rows), "missing": missing, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
