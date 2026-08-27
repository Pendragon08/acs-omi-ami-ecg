"""Multi-label (OMI + AMI) evaluation metrics.

Per label: AUROC, AUPRC (threshold-free), and at a chosen threshold the clinical
metrics Sensitivity / Specificity / PPV / NPV / Accuracy / F1.  Macro AUPRC /
AUROC / F1 are averaged over the two labels (OMI, AMI).
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

LABELS = ("OMI", "AMI")


def _safe_auc(score_fn, yt, yp):
    if len(np.unique(yt)) < 2:
        return float("nan")
    try:
        return float(score_fn(yt, yp))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_prob, thresholds=None, label_names=None):
    """Args:
        y_true: [N, C] int array (0/1).
        y_prob: [N, C] float probabilities (after sigmoid).
        thresholds: [C] or None (None -> 0.5 for each label).
        label_names: [C] names; defaults to ("OMI", "AMI").  Pass ("OMI",) for
            single-label (e.g. Exp1 OMI benchmark, A5 ablation).
    Returns a nested dict: per_label[name][metric] + macro_*.
    """
    label_names = tuple(label_names) if label_names is not None else LABELS
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_prob.ndim == 1:
        y_prob = y_prob[:, None]
    n, c = y_true.shape
    assert c == len(label_names), f"got {c} columns but {len(label_names)} names"

    per_label = {}
    for j in range(c):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        thr = float(thresholds[j]) if thresholds is not None else 0.5

        pred = (yp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        per_label[label_names[j]] = {
            "auroc": _safe_auc(roc_auc_score, yt, yp),
            "auprc": _safe_auc(average_precision_score, yt, yp),
            "threshold": thr,
            "sens": tp / (tp + fn) if (tp + fn) else 0.0,
            "spec": tn / (tn + fp) if (tn + fp) else 0.0,
            "ppv": tp / (tp + fp) if (tp + fp) else 0.0,
            "npv": tn / (tn + fn) if (tn + fn) else 0.0,
            "acc": (tp + tn) / total if total else 0.0,
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "n_pos": int(yt.sum()),
            "n_neg": int((1 - yt).sum()),
        }

    out = {"per_label": per_label}
    names = [label_names[j] for j in range(c)]
    out["macro_auroc"] = float(np.nanmean([per_label[l]["auroc"] for l in names]))
    out["macro_auprc"] = float(np.nanmean([per_label[l]["auprc"] for l in names]))
    out["macro_f1"] = float(np.nanmean([per_label[l]["f1"] for l in names]))
    return out


def flatten_metrics(metrics):
    """Flatten nested metric dict into a single-level dict for CSV/JSON."""
    flat = {}
    for label, d in metrics["per_label"].items():
        for k, v in d.items():
            flat[f"{label}_{k}"] = v
    flat["macro_auroc"] = metrics["macro_auroc"]
    flat["macro_auprc"] = metrics["macro_auprc"]
    flat["macro_f1"] = metrics["macro_f1"]
    return flat
