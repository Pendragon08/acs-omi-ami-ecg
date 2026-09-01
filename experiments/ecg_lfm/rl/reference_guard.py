"""Build a frozen, patient-bootstrap safety guard from W0 inner validation."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from experiments.acs_joint_task import (
    ACS_CLASSES,
    acs_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    select_omi_threshold,
)


GUARD_METRICS = (
    "acs_macro_auprc",
    "omi_auprc",
    "omi_ppv",
    "omi_specificity",
    "stemi_f1",
    "acs_brier",
    "omi_brier",
)


def _arrays(rows: Iterable[Mapping]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(rows)
    y_acs = np.asarray([int(row["y_acs"]) for row in rows], dtype=np.int64)
    p_acs = np.asarray(
        [[float(row[f"p_{name}"]) for name in ACS_CLASSES] for row in rows],
        dtype=np.float64,
    )
    y_omi = np.asarray([int(row["y_omi"]) for row in rows], dtype=np.int64)
    p_omi = np.asarray([float(row["p_omi"]) for row in rows], dtype=np.float64)
    return y_acs, p_acs, y_omi, p_omi


def guard_metrics(rows: Iterable[Mapping], omi_threshold: float | None = None) -> dict[str, float]:
    y_acs, p_acs, y_omi, p_omi = _arrays(rows)
    if omi_threshold is None:
        omi_threshold, _ = select_omi_threshold(y_omi, p_omi, "youden_j")
    if omi_threshold is None:
        raise ValueError("Youden threshold is infeasible on the inner-validation set")
    acs = acs_metrics(y_acs, p_acs)
    omi = binary_metrics(y_omi, p_omi, float(omi_threshold))
    acs_cal = multiclass_calibration_metrics(y_acs, p_acs)
    omi_cal = binary_calibration_metrics(y_omi, p_omi)
    return {
        "acs_macro_auprc": float(acs["macro_auprc_ovr"]),
        "omi_auprc": float(omi["auprc"]),
        "omi_ppv": float(omi["ppv"]),
        "omi_specificity": float(omi["specificity"]),
        "stemi_f1": float(acs["per_class"]["STEMI"]["f1"]),
        "acs_brier": float(acs_cal["multiclass_brier"]),
        "omi_brier": float(omi_cal["brier"]),
        "omi_threshold": float(omi_threshold),
    }


def patient_bootstrap_margins(
    rows: Iterable[Mapping],
    replicates: int,
    seed: int,
    minimum_absolute_margin: float,
) -> tuple[dict[str, float], dict]:
    """Estimate one-standard-error non-inferiority margins by patient resampling.

    The baseline Youden threshold is frozen before resampling. This estimates
    metric variability without re-tuning a decision threshold inside every
    bootstrap replicate.
    """
    rows = list(rows)
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    by_patient: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    patients = np.asarray(sorted(by_patient), dtype=object)
    if len(patients) < 2:
        raise ValueError("patient-level bootstrap needs at least two patients")
    reference = guard_metrics(rows)
    threshold = reference["omi_threshold"]
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in GUARD_METRICS}
    failures = 0
    for _ in range(replicates):
        selected = rng.choice(patients, size=len(patients), replace=True)
        sampled_rows = [row for patient in selected for row in by_patient[str(patient)]]
        try:
            values = guard_metrics(sampled_rows, omi_threshold=threshold)
        except (ValueError, FloatingPointError):
            failures += 1
            continue
        for name in GUARD_METRICS:
            if np.isfinite(values[name]):
                samples[name].append(values[name])
    margins = {}
    standard_errors = {}
    for name in GUARD_METRICS:
        values = np.asarray(samples[name], dtype=np.float64)
        if len(values) < max(2, int(0.9 * replicates)):
            raise RuntimeError(f"insufficient valid bootstrap values for {name}: {len(values)}")
        se = float(values.std(ddof=1))
        standard_errors[name] = se
        margins[name] = float(max(minimum_absolute_margin, se))
    audit = {
        "method": "patient-level bootstrap standard error",
        "replicates_requested": int(replicates),
        "replicates_failed": int(failures),
        "seed": int(seed),
        "n_records": int(len(rows)),
        "n_patients": int(len(patients)),
        "omi_threshold_frozen_for_bootstrap": float(threshold),
        "minimum_absolute_margin": float(minimum_absolute_margin),
        "standard_errors": standard_errors,
    }
    return margins, {"reference": reference, "audit": audit}
