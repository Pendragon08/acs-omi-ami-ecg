"""Fit calibration on inner-validation and evaluate frozen transforms on outer-test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from experiments.acs_joint_task import (
    ACS_CLASSES,
    acs_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    select_omi_threshold,
)


EPS = 1e-6
THRESHOLD_METHODS = ("fixed_0.5", "max_f1", "youden_j", "sens90", "spec90")


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate-ID predictions: {path}")
    y_acs = np.asarray([int(row["y_acs"]) for row in rows], dtype=np.int64)
    p_acs = np.asarray([[float(row[f"p_{name}"]) for name in ACS_CLASSES] for row in rows])
    y_omi = np.asarray([int(row["y_omi"]) for row in rows], dtype=np.int64)
    p_omi = np.asarray([float(row["p_omi"]) for row in rows], dtype=np.float64)
    if not np.allclose(p_acs.sum(1), 1.0, atol=1e-5):
        raise ValueError("ACS probabilities do not sum to one")
    return y_acs, p_acs, y_omi, p_omi


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_fit(y: np.ndarray, p: np.ndarray) -> float:
    logits = np.log(np.clip(p, EPS, 1.0))

    def objective(temperature: float) -> float:
        scaled = logits / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"temperature optimization failed: {result}")
    return float(result.x)


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(p, EPS, 1.0)) / temperature)


def omi_features(p: np.ndarray, method: str) -> np.ndarray:
    clipped = np.clip(p, EPS, 1.0 - EPS)
    if method == "platt":
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if method == "beta":
        return np.column_stack((np.log(clipped), -np.log1p(-clipped)))
    raise ValueError(method)


def operating_points(y_inner: np.ndarray, p_inner: np.ndarray, y_outer: np.ndarray, p_outer: np.ndarray) -> dict:
    result = {}
    for name in THRESHOLD_METHODS:
        threshold, selection = select_omi_threshold(y_inner, p_inner, name)
        result[name] = {
            "threshold": threshold,
            "inner_selection": selection,
            "outer_metrics": None if threshold is None else binary_metrics(y_outer, p_outer, threshold),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alignment = json.loads((args.evaluation_dir / "provenance_alignment_report.json").read_text(encoding="utf-8"))
    if alignment.get("status") != "pass" or any(alignment.get("patient_overlap", {}).values()):
        raise RuntimeError("calibration refused: evaluator provenance gate did not pass")
    inner = read_predictions(args.evaluation_dir / "inner_validation_predictions.csv")
    outer = read_predictions(args.evaluation_dir / "outer_test_predictions.csv")
    yi_acs, pi_acs, yi_omi, pi_omi = inner
    yo_acs, po_acs, yo_omi, po_omi = outer

    temperature = temperature_fit(yi_acs, pi_acs)
    po_acs_scaled = apply_temperature(po_acs, temperature)
    result: dict = {
        "status": "pass",
        "fit_subset": "inner-validation only",
        "evaluation_subset": "outer-test once with frozen calibrators",
        "outer_test_used_for_fit": False,
        "acs": {
            "temperature": temperature,
            "uncalibrated": {
                "metrics": acs_metrics(yo_acs, po_acs),
                "calibration": multiclass_calibration_metrics(yo_acs, po_acs),
            },
            "temperature_scaled": {
                "metrics": acs_metrics(yo_acs, po_acs_scaled),
                "calibration": multiclass_calibration_metrics(yo_acs, po_acs_scaled),
            },
        },
        "omi": {},
    }
    result["omi"]["uncalibrated"] = {
        "parameters": None,
        "calibration": binary_calibration_metrics(yo_omi, po_omi),
        "operating_points": operating_points(yi_omi, pi_omi, yo_omi, po_omi),
    }
    for method in ("platt", "beta"):
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(omi_features(pi_omi, method), yi_omi)
        calibrated_inner = model.predict_proba(omi_features(pi_omi, method))[:, 1]
        calibrated_outer = model.predict_proba(omi_features(po_omi, method))[:, 1]
        result["omi"][method] = {
            "parameters": {
                "intercept": model.intercept_.astype(float).tolist(),
                "coefficients": model.coef_.astype(float).tolist(),
            },
            "calibration": binary_calibration_metrics(yo_omi, calibrated_outer),
            "operating_points": operating_points(yi_omi, calibrated_inner, yo_omi, calibrated_outer),
        }
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrated_inner = isotonic.fit_transform(pi_omi, yi_omi)
    calibrated_outer = isotonic.predict(po_omi)
    result["omi"]["isotonic"] = {
        "parameters": {
            "x_thresholds": isotonic.X_thresholds_.astype(float).tolist(),
            "y_thresholds": isotonic.y_thresholds_.astype(float).tolist(),
        },
        "calibration": binary_calibration_metrics(yo_omi, calibrated_outer),
        "operating_points": operating_points(yi_omi, calibrated_inner, yo_omi, calibrated_outer),
    }
    output = args.output or args.evaluation_dir / "calibration.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
