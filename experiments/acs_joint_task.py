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
    invalid_rows: list[dict] = []
    duplicate_ids: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = stem((row.get("ecg_row_record") or "").strip())
            values = [(row.get(name) or "").strip() for name in LABELS]
            if not rid or any(value not in {"0", "1"} for value in values):
                invalid_rows.append({"record_id": rid, "values": dict(zip(LABELS, values))})
                continue
            if rid in labels_by_id:
                duplicate_ids.append(rid)
                continue
            labels_by_id[rid] = [int(value) for value in values]

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
        "n_csv_rows": int(len(labels_by_id) + len(invalid_rows)),
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
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy", "tp", "tn", "fp", "fn"):
            per_class[name][metric] = hard[metric]
    valid_auc = [v["auroc"] for v in per_class.values() if np.isfinite(v["auroc"])]
    valid_pr = [v["auprc"] for v in per_class.values() if np.isfinite(v["auprc"])]
    return {
        "n": int(len(y_class)),
        "accuracy": float(accuracy_score(y_class, pred)),
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
    p = np.asarray(probs)[mask, 1:4]
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(3))
    per_class = {}
    for k, name in enumerate(SUBTYPE_LABELS):
        hard = binary_metrics((y == k).astype(np.int64), (pred == k).astype(float), 0.5)
        per_class[name] = hard
    return {
        "n_acs": int(mask.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
    }


def select_omi_threshold(y: np.ndarray, prob: np.ndarray, method: str) -> tuple[float, dict]:
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
        best = max(feasible or candidates, key=lambda item: (item[1]["specificity"], item[1]["f1"], item[0]))
        best[1] = dict(best[1], sensitivity_constraint_satisfied=bool(feasible))
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
