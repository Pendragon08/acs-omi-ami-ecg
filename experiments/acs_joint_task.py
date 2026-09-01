"""Shared data, split and evaluation utilities for the corrected ACS task.

The corrected formulation is a joint multi-task problem:

* ACS subtype is *categorical*: ``OTHER`` (the explicit ``[0,0,0]`` state),
  STEMI, NSTEMI or UA.  STEMI/NSTEMI/UA can therefore never be predicted
  simultaneously.
* OMI is a separate binary target and can overlap with an ACS subtype.

The ``OTHER`` sentinel is retained during training so that explicitly
labelled non-ACS records are not silently discarded.  Reports additionally
provide the requested three-way ACS subtype metrics conditional on ACS.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
SUBTYPE_LABELS = ("STEMI", "NSTEMI", "UA")
LABELS = (*SUBTYPE_LABELS, "OMI")
ACS_CLASSES = ("OTHER", *SUBTYPE_LABELS)
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stem(value: str) -> str:
    return Path(str(value)).stem


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load P2 ECGs and audit the four source columns.

    Returns ``x_raw, y_source, patient_ids, record_ids, audit``.  Unknown
    labels are excluded, whereas the explicit all-zero subtype row is kept.
    """
    arr = ROOT / "data" / "arrays"
    meta = np.load(arr / "meta_train.npz", allow_pickle=True)
    p2 = np.load(arr / "P2_train.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    p2_scaler = scalers["P2"]
    raw = p2.astype(np.float32) * (float(p2_scaler["scale"]) + 1e-8) + float(p2_scaler["median"])

    csv_path = ROOT / "data" / "CSV" / "train.csv"
    labels_by_id: dict[str, list[int]] = {}
    csv_row_count = 0
    invalid_rows: list[dict] = []
    duplicate_ids: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            csv_row_count += 1
            rid = stem((row.get("ecg_row_record") or "").strip())
            values = [(row.get(name) or "").strip() for name in LABELS]
            if not rid or any(value not in {"0", "1"} for value in values):
                invalid_rows.append({"record_id": rid, "values": dict(zip(LABELS, values))})
                continue
            if rid in labels_by_id:
                duplicate_ids.append(rid)
                continue
            labels_by_id[rid] = [int(value) for value in values]

    if duplicate_ids:
        raise ValueError(
            "Duplicate record identifiers in training labels are not allowed; "
            f"n={len(duplicate_ids)}, preview={duplicate_ids[:10]}"
        )

    record_ids_all = np.asarray([stem(x) for x in meta["record_ids"]], dtype=object)
    patient_ids_all = np.asarray([str(x) for x in meta["patient_ids"]], dtype=object)
    valid_mask = np.asarray([str(rid) in labels_by_id for rid in record_ids_all], dtype=bool)
    if not valid_mask.any():
        raise RuntimeError("No valid ACS-labelled records remain after label audit")

    y = np.asarray([labels_by_id[str(rid)] for rid in record_ids_all[valid_mask]], dtype=np.int64)
    subtype_sum = y[:, :3].sum(axis=1)
    overlap = np.flatnonzero(subtype_sum > 1)
    if len(overlap):
        preview = y[overlap[:10]].tolist()
        raise ValueError(
            "Mutually-exclusive ACS invariant violated: records have more than one "
            f"subtype (n={len(overlap)}, preview={preview})"
        )

    raw_valid = np.asarray(raw[valid_mask], dtype=np.float32)
    record_ids = record_ids_all[valid_mask]
    patient_ids = patient_ids_all[valid_mask]
    states = np.where(y[:, :3].sum(axis=1) == 0, 0, np.argmax(y[:, :3], axis=1) + 1)
    state_counts = Counter(int(x) for x in states)
    joint_counts = Counter((int(s), int(o)) for s, o in zip(states, y[:, 3]))
    audit = {
        "csv_path": str(csv_path),
        "n_csv_rows": int(csv_row_count),
        "n_meta_records": int(len(record_ids_all)),
        "n_valid_records": int(len(y)),
        "n_excluded_unknown_or_invalid": int((~valid_mask).sum()),
        "n_invalid_csv_rows": int(len(invalid_rows)),
        "invalid_csv_rows_preview": invalid_rows[:20],
        "n_duplicate_csv_record_ids": int(len(duplicate_ids)),
        "duplicate_csv_record_ids_preview": duplicate_ids[:20],
        "record_label_counts": {
            name: {"positive": int(y[:, i].sum()), "negative": int(len(y) - y[:, i].sum())}
            for i, name in enumerate(LABELS)
        },
        "acs_class_counts": {ACS_CLASSES[k]: int(state_counts.get(k, 0)) for k in range(4)},
        "acs_omi_joint_counts": {
            f"{ACS_CLASSES[s]}__OMI_{o}": int(joint_counts.get((s, o), 0))
            for s in range(4) for o in range(2)
        },
        "subtype_overlap_records": int(len(overlap)),
        "explicit_non_acs_000_records": int((states == 0).sum()),
        "non_acs_with_omi_positive": int(((states == 0) & (y[:, 3] == 1)).sum()),
    }
    return raw_valid, y, patient_ids, record_ids, audit


def encode_targets(y_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert source one-hot subtype columns to ACS class and OMI targets."""
    y = np.asarray(y_source, dtype=np.int64)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"expected [N,4] source labels, got {y.shape}")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("source labels must be finite binary values in {0,1}")
    subtype_sum = y[:, :3].sum(axis=1)
    if np.any(subtype_sum > 1):
        raise ValueError("ACS subtype labels are not mutually exclusive")
    acs = np.where(subtype_sum == 0, 0, np.argmax(y[:, :3], axis=1) + 1).astype(np.int64)
    omi = y[:, 3].astype(np.int64)
    return acs, omi


def _patient_targets(patient_ids: np.ndarray, acs: np.ndarray, omi: np.ndarray):
    patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    targets = np.zeros((len(patients), 5), dtype=np.int64)
    for row, pidx in enumerate(inverse):
        targets[pidx, acs[row]] = 1
        targets[pidx, 4] = max(targets[pidx, 4], int(omi[row]))
    return patients, inverse, targets


def _validate_folds(folds, patient_ids: np.ndarray, n_records: int) -> None:
    seen = []
    for fold, (train, val) in enumerate(folds):
        train = np.asarray(train, dtype=np.int64)
        val = np.asarray(val, dtype=np.int64)
        if np.intersect1d(train, val).size:
            raise AssertionError(f"record overlap in fold {fold}")
        if np.intersect1d(np.unique(patient_ids[train]), np.unique(patient_ids[val])).size:
            raise AssertionError(f"patient leakage in fold {fold}")
        seen.extend(val.tolist())
    if not np.array_equal(np.sort(np.asarray(seen)), np.arange(n_records)):
        raise AssertionError("validation folds do not form an exact OOF partition")


def load_or_build_folds(
    patient_ids: np.ndarray,
    acs: np.ndarray,
    omi: np.ndarray,
    n_folds: int = 5,
    seed: int = 2026,
    historical_split: Path | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Reuse the historical patient split when it is valid; otherwise build one.

    Reusing the old patient assignment keeps the corrected reruns comparable to
    previous experiments while the audit below makes the corrected ACS/OMI
    joint-state distribution explicit.
    """
    if historical_split is None:
        candidate = ROOT / "results" / "hmoe_kan_stemi_nstemi_ua_omi_e4" / "split.json"
        historical_split = candidate if candidate.exists() else None
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    source = "new_patient_level_joint_iterative_stratification"
    if historical_split is not None and historical_split.exists():
        payload = json.loads(historical_split.read_text(encoding="utf-8"))
        raw_folds = payload.get("folds_indices")
        if raw_folds and len(raw_folds) == n_folds:
            folds = [
                (np.asarray(item["train"], dtype=np.int64), np.asarray(item["val"], dtype=np.int64))
                for item in raw_folds
            ]
            try:
                _validate_folds(folds, patient_ids, len(acs))
                source = "historical_patient_level_split_reused"
            except (AssertionError, IndexError, ValueError):
                folds = []

    patients, inverse, patient_targets = _patient_targets(patient_ids, acs, omi)
    if not folds:
        if np.any(patient_targets.sum(axis=0) < n_folds):
            raise ValueError("a joint patient target is too sparse for five-fold stratification")
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_p, val_p in splitter.split(np.zeros((len(patients), 1)), patient_targets):
            train = np.flatnonzero(np.isin(inverse, train_p))
            val = np.flatnonzero(np.isin(inverse, val_p))
            folds.append((train, val))
        _validate_folds(folds, patient_ids, len(acs))

    fold_audit = []
    for fold, (train, val) in enumerate(folds):
        fold_audit.append({
            "fold": fold,
            "n_train_records": int(len(train)),
            "n_val_records": int(len(val)),
            "n_train_patients": int(len(np.unique(patient_ids[train]))),
            "n_val_patients": int(len(np.unique(patient_ids[val]))),
            "train_acs_counts": {name: int((acs[train] == i).sum()) for i, name in enumerate(ACS_CLASSES)},
            "val_acs_counts": {name: int((acs[val] == i).sum()) for i, name in enumerate(ACS_CLASSES)},
            "train_omi_positive": int(omi[train].sum()),
            "val_omi_positive": int(omi[val].sum()),
        })
    split_audit = {
        "protocol": "patient_level_five_fold",
        "source": source,
        "historical_split": str(historical_split) if historical_split else None,
        "n_folds": n_folds,
        "fold_seed": seed,
        "n_patients": int(len(patients)),
        "patient_joint_target_definition": "one-hot ACS states OTHER/STEMI/NSTEMI/UA plus OMI presence",
        "folds": fold_audit,
    }
    return folds, split_audit


def _safe_mcc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(matthews_corrcoef(y, pred))
    except ValueError:
        return 0.0


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if tp + fn else 0.0
    spec = float(tn / (tn + fp)) if tn + fp else 0.0
    ppv = float(tp / (tp + fp)) if tp + fp else 0.0
    npv = float(tn / (tn + fn)) if tn + fn else 0.0
    return {
        "auroc": float(roc_auc_score(y, prob)) if np.unique(y).size > 1 else float("nan"),
        "auprc": float(average_precision_score(y, prob)) if np.unique(y).size > 1 else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": _safe_mcc(y, pred),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "youden_j": float(sens + spec - 1.0),
    }


def acs_metrics(y_class: np.ndarray, probs: np.ndarray, prefix: str = "acs") -> dict:
    y_class = np.asarray(y_class, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_class, pred, labels=np.arange(4))
    per_class = {}
    for k, name in enumerate(ACS_CLASSES):
        y = (y_class == k).astype(np.int64)
        p = probs[:, k]
        per_class[name] = binary_metrics(y, p, float("nan"))
        # For categorical prediction, hard one-vs-rest values are the clinically
        # interpretable sensitivity/specificity/PPV/NPV/F1/MCC values.
        hard = binary_metrics(y, (pred == k).astype(float), 0.5)
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy", "balanced_accuracy", "tp", "tn", "fp", "fn"):
            per_class[name][metric] = hard[metric]
        per_class[name]["threshold"] = "argmax_ovr"
        per_class[name]["youden_j"] = float(hard["sensitivity"] + hard["specificity"] - 1.0)
    valid_auc = [v["auroc"] for v in per_class.values() if np.isfinite(v["auroc"])]
    valid_pr = [v["auprc"] for v in per_class.values() if np.isfinite(v["auprc"])]
    return {
        "n": int(len(y_class)),
        "accuracy": float(accuracy_score(y_class, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_class, pred)),
        "macro_f1": float(f1_score(y_class, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_class, pred, average="weighted", zero_division=0)),
        "macro_auroc_ovr": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "macro_auprc_ovr": float(np.mean(valid_pr)) if valid_pr else float("nan"),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
        "prediction_counts": {name: int((pred == k).sum()) for k, name in enumerate(ACS_CLASSES)},
    }


def acs_subtype_metrics(y_class: np.ndarray, probs: np.ndarray) -> dict:
    mask = np.asarray(y_class) > 0
    if not mask.any():
        return {"n_acs": 0, "error": "no ACS-positive records"}
    # Keep only ACS records and remap OTHER-free classes to 0..2.
    y = np.asarray(y_class)[mask] - 1
    p = np.asarray(probs, dtype=np.float64)[mask, 1:4]
    # 条件于 ACS 阳性后重新归一化三种亚型概率。AUROC/AUPRC 必须由连续概率
    # 计算；Sensitivity/Specificity/F1 等工作点指标仍由互斥 argmax 预测计算。
    denom = p.sum(axis=1, keepdims=True)
    if np.any(denom <= 0) or not np.isfinite(p).all():
        raise ValueError("invalid ACS-subtype probabilities")
    p = p / denom
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(3))
    per_class = {}
    for k, name in enumerate(SUBTYPE_LABELS):
        target = (y == k).astype(np.int64)
        metrics = binary_metrics(target, p[:, k], 0.5)
        hard = binary_metrics(target, (pred == k).astype(float), 0.5)
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy", "balanced_accuracy", "tp", "tn", "fp", "fn"):
            metrics[metric] = hard[metric]
        metrics["threshold"] = "argmax_ovr"
        metrics["youden_j"] = float(hard["sensitivity"] + hard["specificity"] - 1.0)
        per_class[name] = metrics
    valid_auc = [v["auroc"] for v in per_class.values() if np.isfinite(v["auroc"])]
    valid_pr = [v["auprc"] for v in per_class.values() if np.isfinite(v["auprc"])]
    return {
        "n_acs": int(mask.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "multiclass_mcc": _safe_mcc(y, pred),
        "macro_auroc_ovr": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "macro_auprc_ovr": float(np.mean(valid_pr)) if valid_pr else float("nan"),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
    }


def nstemi_vs_ua_metrics(y_class: np.ndarray, probs: np.ndarray) -> dict:
    """Evaluate NSTEMI as the positive class among true NSTEMI/UA records."""
    y_class = np.asarray(y_class, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.isin(y_class, (2, 3))
    if not mask.any():
        return {"n": 0, "error": "no NSTEMI/UA records"}
    y = (y_class[mask] == 2).astype(np.int64)
    pair = probs[mask][:, (2, 3)]
    denom = pair.sum(axis=1)
    if np.any(denom <= 0) or not np.isfinite(pair).all():
        raise ValueError("invalid NSTEMI/UA probabilities")
    score = pair[:, 0] / denom
    result = binary_metrics(y, score, 0.5)
    result.update(
        {
            "n": int(mask.sum()),
            "n_nstemi": int(y.sum()),
            "n_ua": int((1 - y).sum()),
            "positive_class": "NSTEMI",
            "probability": "p_NSTEMI/(p_NSTEMI+p_UA)",
            "confusion_matrix": confusion_matrix(
                y, (score >= 0.5).astype(np.int64), labels=[0, 1]
            ).astype(int).tolist(),
        }
    )
    return result


def _binary_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Equal-width expected calibration error with auditable bin statistics."""
    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or not np.isfinite(prob).all():
        raise ValueError("binary calibration inputs are invalid")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (prob >= lower) & (prob < upper if index < n_bins - 1 else prob <= upper)
        count = int(selected.sum())
        if count:
            mean_prob = float(prob[selected].mean())
            fraction_positive = float(y[selected].mean())
            contribution = count / len(y) * abs(mean_prob - fraction_positive)
            ece += contribution
        else:
            mean_prob = None
            fraction_positive = None
            contribution = 0.0
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "n": count,
                "mean_probability": mean_prob,
                "fraction_positive": fraction_positive,
                "ece_contribution": float(contribution),
            }
        )
    return float(ece), bins


def binary_calibration_metrics(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> dict:
    """Brier/ECE and weakly regularized logistic calibration slope/intercept."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    ece, bins = _binary_ece(y, clipped, n_bins=n_bins)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.unique(y).size < 2:
        intercept = slope = float("nan")
    else:
        calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibration.fit(logit, y)
        intercept = float(calibration.intercept_[0])
        slope = float(calibration.coef_[0, 0])
    return {
        "n": int(len(y)),
        "n_bins": int(n_bins),
        "brier": float(brier_score_loss(y, clipped)),
        "ece": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "bins": bins,
    }


def multiclass_calibration_metrics(y_class: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """Multiclass Brier, top-label ECE and one-vs-rest class-wise ECE."""
    y_class = np.asarray(y_class, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 4 or len(y_class) != len(probs):
        raise ValueError("expected ACS labels [N] and probabilities [N,4]")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("ACS probabilities must be finite and sum to one")
    one_hot = np.eye(4, dtype=np.float64)[y_class]
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), pred]
    correct = (pred == y_class).astype(np.int64)
    top_ece, top_bins = _binary_ece(correct, confidence, n_bins=n_bins)
    classwise = {}
    for index, name in enumerate(ACS_CLASSES):
        target = (y_class == index).astype(np.int64)
        ece, bins = _binary_ece(target, probs[:, index], n_bins=n_bins)
        classwise[name] = {"ece": ece, "bins": bins}
    return {
        "n": int(len(y_class)),
        "n_bins": int(n_bins),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "top_label_ece": top_ece,
        "top_label_bins": top_bins,
        "classwise": classwise,
    }


def save_reliability_figures(
    out_dir: Path,
    acs_calibration: dict,
    omi_calibration: dict,
) -> None:
    """Persist reliability diagrams without fitting a calibration model."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        acs_bins = [item for item in acs_calibration["top_label_bins"] if item["n"]]
        omi_bins = [item for item in omi_calibration["bins"] if item["n"]]
        axes[0].plot(
            [item["mean_probability"] for item in acs_bins],
            [item["fraction_positive"] for item in acs_bins],
            marker="o",
            label="ACS top-label",
        )
        axes[1].plot(
            [item["mean_probability"] for item in omi_bins],
            [item["fraction_positive"] for item in omi_bins],
            marker="o",
            label="OMI",
        )
        for ax in axes:
            ax.plot([0, 1], [0, 1], "k:", label="ideal")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability", ylabel="observed frequency")
            ax.legend()
        axes[0].set_title("ACS top-label reliability")
        axes[1].set_title("OMI reliability")
        fig.tight_layout()
        fig.savefig(out_dir / "reliability_diagrams.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "reliability_plot_error.txt").write_text(repr(exc), encoding="utf-8")


def select_omi_threshold(y: np.ndarray, prob: np.ndarray, method: str) -> tuple[float | None, dict]:
    candidates = [(0.5, binary_metrics(y, prob, 0.5))] if method == "fixed_0.5" else [
        (float(t), binary_metrics(y, prob, float(t))) for t in GRID
    ]
    if method == "fixed_0.5":
        return 0.5, candidates[0][1]
    if method == "max_f1":
        best = max(candidates, key=lambda item: (item[1]["f1"], item[1]["sensitivity"], item[1]["specificity"], item[0]))
    elif method == "youden_j":
        best = max(candidates, key=lambda item: (item[1]["youden_j"], item[1]["specificity"], item[1]["f1"], item[0]))
    elif method in {"sens90", "sens95"}:
        target = 0.90 if method == "sens90" else 0.95
        feasible = [item for item in candidates if item[1]["sensitivity"] >= target]
        if not feasible:
            return None, {
                "status": "infeasible",
                "target_sensitivity": target,
                "sensitivity_constraint_satisfied": False,
            }
        chosen = max(feasible, key=lambda item: (item[1]["specificity"], item[1]["f1"], item[0]))
        best = (chosen[0], dict(
            chosen[1],
            status="feasible",
            target_sensitivity=target,
            sensitivity_constraint_satisfied=True,
        ))
    elif method == "spec90":
        target = 0.90
        feasible = [item for item in candidates if item[1]["specificity"] >= target]
        if not feasible:
            return None, {
                "status": "infeasible",
                "target_specificity": target,
                "specificity_constraint_satisfied": False,
            }
        chosen = max(feasible, key=lambda item: (item[1]["sensitivity"], item[1]["f1"], -item[0]))
        best = (chosen[0], dict(
            chosen[1],
            status="feasible",
            target_specificity=target,
            specificity_constraint_satisfied=True,
        ))
    else:
        raise ValueError(method)
    return float(best[0]), best[1]


def save_curves(out_dir: Path, histories: list[list[dict]], title: str) -> None:
    """Save a CSV and a compact loss/LR/validation-score figure."""
    rows = []
    for fold, history in enumerate(histories):
        for item in history:
            rows.append({"fold": fold, **item})
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "training_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for fold, history in enumerate(histories):
            e = [x["epoch"] for x in history]
            axes[0].plot(e, [x.get("train_total_loss", np.nan) for x in history], alpha=0.55, label=f"fold {fold} train")
            axes[0].plot(e, [x.get("val_total_loss", np.nan) for x in history], alpha=0.8, label=f"fold {fold} val")
            axes[1].plot(e, [x.get("val_acs_macro_auroc", np.nan) for x in history], label=f"fold {fold} ACS")
            axes[1].plot(e, [x.get("val_omi_auroc", np.nan) for x in history], linestyle="--", label=f"fold {fold} OMI")
            axes[2].plot(e, [x.get("learning_rate", np.nan) for x in history], label=f"fold {fold}")
        axes[0].set_ylabel("loss"); axes[0].legend(ncol=2, fontsize=8)
        axes[1].set_ylabel("AUROC"); axes[1].legend(ncol=2, fontsize=8)
        axes[2].set_ylabel("learning rate"); axes[2].set_xlabel("epoch"); axes[2].set_yscale("log")
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out_dir / "training_validation_loss_lr.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "curve_plot_error.txt").write_text(repr(exc), encoding="utf-8")


def save_roc_pr_curves(out_dir: Path, y_acs: np.ndarray, p_acs: np.ndarray, y_omi: np.ndarray, p_omi: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for k, name in enumerate(ACS_CLASSES):
            y = (y_acs == k).astype(int)
            if np.unique(y).size > 1:
                fpr, tpr, _ = roc_curve(y, p_acs[:, k]); axes[0].plot(fpr, tpr, label=name)
        if np.unique(y_omi).size > 1:
            fpr, tpr, _ = roc_curve(y_omi, p_omi); axes[0].plot(fpr, tpr, linewidth=2, label="OMI")
            precision, recall, _ = precision_recall_curve(y_omi, p_omi); axes[1].plot(recall, precision, linewidth=2, label="OMI")
        axes[0].plot([0, 1], [0, 1], "k:"); axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC")
        axes[1].set(xlabel="recall", ylabel="precision", title="Precision–recall"); axes[0].legend(); axes[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / "roc_pr_curves.png", dpi=160); plt.close(fig)
    except Exception as exc:
        (out_dir / "roc_pr_plot_error.txt").write_text(repr(exc), encoding="utf-8")


def save_confusion_figures(out_dir: Path, cm_acs: np.ndarray, cm_sub: np.ndarray, cm_omi: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        mats = [(cm_acs, ACS_CLASSES, "ACS (OTHER + 3 subtypes)"), (cm_sub, SUBTYPE_LABELS, "ACS subtype conditional"), (cm_omi, ("NOMI", "OMI"), "OMI")]
        for ax, (cm, names, title) in zip(axes, mats):
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(names)), names, rotation=35, ha="right"); ax.set_yticks(range(len(names)), names)
            ax.set_title(title); ax.set_xlabel("predicted"); ax.set_ylabel("true")
            for (i, j), value in np.ndenumerate(cm): ax.text(j, i, int(value), ha="center", va="center")
        fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices.png", dpi=160); plt.close(fig)
    except Exception as exc:
        (out_dir / "confusion_plot_error.txt").write_text(repr(exc), encoding="utf-8")
