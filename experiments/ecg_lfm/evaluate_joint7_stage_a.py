"""Inner-validation-only Stage-A comparison of W0 and Joint-7.

This module never loads outer_test or external manifests.  It exports complete
Joint-7 probabilities so every endpoint can be recomputed without training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES,
    acs_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint import _aggregate, _read_csv  # noqa: E402
from experiments.ecg_lfm.joint7 import (  # noqa: E402
    JOINT_CLASSES,
    encode_source_labels,
    marginalize_joint_probabilities,
)
from models.ecg_lfm_adapter import load_finetuned_model  # noqa: E402


EPS = 1e-7


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def infer_joint7(checkpoint: Path, manifest_dir: Path, device: torch.device, workers: int, batch: int):
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    if str(getattr(model, "output_semantics", "")) != "joint_acs_omi":
        raise ValueError("checkpoint is not a joint_acs_omi model")
    model.eval().to(device)
    task.load_dataset("valid", combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False)
    dataset = task.dataset("valid")
    iterator = task.get_batch_iterator(
        dataset=dataset, max_tokens=None, max_signals=batch,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=2026, num_shards=1, shard_id=0, num_workers=workers,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=False)
    segment_logits = np.full((len(dataset), 7), np.nan, dtype=np.float32)
    with torch.no_grad():
        for sample in iterator:
            if not sample:
                continue
            ids = sample["id"].cpu().numpy()
            from fairseq_signals.utils import utils
            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**sample["net_input"])
                values = model.get_logits(output, normalize=False).float().cpu().numpy()
            segment_logits[ids] = values
    del model, task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(segment_logits).all():
        raise RuntimeError("Joint-7 inference did not fill every validation manifest row")
    segment_rows = _read_csv(manifest_dir / "valid_segments.csv")
    if len(segment_rows) != len(segment_logits):
        raise ValueError("segment index and output lengths differ")
    return _aggregate(segment_logits, segment_rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(y: np.ndarray, logits: np.ndarray) -> float:
    def objective(value: float) -> float:
        scaled = logits / value
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"temperature fit failed: {result}")
    return float(result.x)


def generic_calibration(y: np.ndarray, probs: np.ndarray, names: tuple[str, ...], n_bins: int = 15) -> dict:
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y]
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), pred]
    correct = (pred == y).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def ece(target, probability):
        total = 0.0
        bins = []
        for index in range(n_bins):
            selected = (probability >= edges[index]) & (
                probability < edges[index + 1] if index < n_bins - 1 else probability <= edges[index + 1]
            )
            n = int(selected.sum())
            mp = float(probability[selected].mean()) if n else None
            fp = float(target[selected].mean()) if n else None
            contribution = 0.0 if not n else n / len(target) * abs(mp - fp)
            total += contribution
            bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "n": n, "mean_probability": mp, "fraction_positive": fp})
        return float(total), bins

    top_ece, top_bins = ece(correct, confidence)
    classwise = {}
    for k, name in enumerate(names):
        value, bins = ece((y == k).astype(int), probs[:, k])
        classwise[name] = {"ece": value, "bins": bins}
    nll = float(np.mean(-np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))
    return {
        "n": int(len(y)), "nll": nll,
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "top_label_ece": top_ece, "top_label_bins": top_bins,
        "classwise": classwise,
    }


def joint_metrics(y: np.ndarray, q: np.ndarray) -> dict:
    pred = q.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    per_class = {}
    supported_aurocs, supported_auprcs = [], []
    for k, name in enumerate(JOINT_CLASSES):
        target = (y == k).astype(int)
        prediction = (pred == k).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, prediction, labels=(0, 1)).ravel()
        supported = np.unique(target).size == 2
        auroc = float(roc_auc_score(target, q[:, k])) if supported else None
        auprc = float(average_precision_score(target, q[:, k])) if supported else None
        if auroc is not None:
            supported_aurocs.append(auroc); supported_auprcs.append(auprc)
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision is not None and sensitivity is not None and precision + sensitivity else 0.0 if sensitivity is not None else None
        per_class[name] = {
            "auroc": auroc, "auprc": auprc, "f1": f1,
            "sensitivity": sensitivity, "specificity": specificity,
            "support": int(target.sum()), "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }
    return {
        "macro_auroc_ovr_all_7_classes": None if len(supported_aurocs) < 7 else float(np.mean(supported_aurocs)),
        "macro_auprc_ovr_all_7_classes": None if len(supported_auprcs) < 7 else float(np.mean(supported_auprcs)),
        "n": int(len(y)),
        "macro_auroc_ovr_supported_classes": float(np.mean(supported_aurocs)),
        "macro_auprc_ovr_supported_classes": float(np.mean(supported_auprcs)),
        "macro_f1_all_7_classes": float(f1_score(y, pred, labels=np.arange(7), average="macro", zero_division=0)),
        "balanced_accuracy_supported_classes": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "n_estimable_ovr_classes": len(supported_aurocs),
        "non_estimable_classes": [name for name in JOINT_CLASSES if per_class[name]["auroc"] is None],
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
    }


def select_operating_point(y: np.ndarray, p: np.ndarray, target: float) -> dict:
    candidates = []
    for threshold in np.unique(np.concatenate(([0.0], p, [1.0]))):
        metric = binary_metrics(y, p, float(threshold))
        if metric["sensitivity"] >= target:
            candidates.append((metric["specificity"], metric["ppv"], threshold, metric))
    if not candidates:
        return {"status": "infeasible", "target_sensitivity": target}
    _, _, threshold, metric = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "status": "feasible", "target_sensitivity": target,
        "threshold": float(threshold), "specificity": metric["specificity"],
        "ppv": metric["ppv"], "achieved_sensitivity": metric["sensitivity"],
    }


def make_rows(records, logits, assignments, source):
    if len(records) != len(assignments):
        raise ValueError("validation predictions/assignment lengths differ")
    by_id = {str(row["record_id"]).zfill(5): row for row in assignments}
    if len(by_id) != len(assignments):
        raise ValueError("duplicate assignment record ID")
    q = softmax(logits)
    marginals = marginalize_joint_probabilities(q)
    rows = []
    seen = set()
    for position, record in enumerate(records):
        record_id = str(record["record_id"]).zfill(5)
        assignment = by_id.get(record_id)
        if assignment is None:
            raise ValueError(f"prediction missing explicit assignment: {record_id}")
        if record["patient_id"] != assignment["patient_id"]:
            raise ValueError(f"patient mismatch: {record_id}")
        index = int(assignment["idx"])
        acs, omi, joint, exception = encode_source_labels(source[index:index + 1])
        row = {
            "patient_id": assignment["patient_id"], "record_id": record_id,
            "fold": int(assignment["fold"]), "subset": assignment["subset"],
            "raw_acs_true": int(acs[0]), "raw_omi_true": int(omi[0]),
            "joint_true": int(joint[0]), "joint_pred": int(q[position].argmax()),
            "joint_label_exception": bool(exception[0]),
            "exception_type": "OTHER_OMI" if exception[0] else "",
            "p_other": float(q[position, 0]),
            "p_stemi_nomi": float(q[position, 1]), "p_stemi_omi": float(q[position, 2]),
            "p_nstemi_nomi": float(q[position, 3]), "p_nstemi_omi": float(q[position, 4]),
            "p_ua_nomi": float(q[position, 5]), "p_ua_omi": float(q[position, 6]),
            "p_stemi": float(marginals["acs"][position, 1]),
            "p_nstemi": float(marginals["acs"][position, 2]),
            "p_ua": float(marginals["acs"][position, 3]),
            "p_omi": float(marginals["omi"][position]),
            "p_nste": float(marginals["nste"][position]),
            "p_nste_omi": float(marginals["nste_omi"][position]),
            "p_omi_given_nste": float(marginals["omi_given_nste"][position]),
            "p_nstemi_given_nste": float(marginals["nstemi_given_nste"][position]),
        }
        rows.append(row); seen.add(record_id)
    expected = set(by_id)
    if seen != expected:
        raise ValueError(f"prediction coverage mismatch missing={len(expected-seen)} extra={len(seen-expected)}")
    return rows, q


def arrays_from_rows(rows):
    y_acs = np.asarray([r["raw_acs_true"] for r in rows], dtype=int)
    y_omi = np.asarray([r["raw_omi_true"] for r in rows], dtype=int)
    y_joint = np.asarray([r["joint_true"] for r in rows], dtype=int)
    q = np.asarray([[r["p_other"], r["p_stemi_nomi"], r["p_stemi_omi"], r["p_nstemi_nomi"], r["p_nstemi_omi"], r["p_ua_nomi"], r["p_ua_omi"]] for r in rows])
    marginal = marginalize_joint_probabilities(q)
    return y_acs, y_omi, y_joint, q, marginal


def evaluate(rows, logits):
    y_acs, y_omi, y_joint, q, marginal = arrays_from_rows(rows)
    thresholds, threshold_details = {}, {}
    for name in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[name], threshold_details[name] = select_omi_threshold(y_omi, marginal["omi"], name)
    youden = thresholds["youden_j"]
    nste = np.isin(y_acs, (2, 3))
    nstemi_ua = nste
    exception = np.asarray([bool(r["joint_label_exception"]) for r in rows])
    temperature = fit_temperature(y_joint, logits)
    calibrated_q = softmax(logits / temperature)
    calibrated_marginal = marginalize_joint_probabilities(calibrated_q)
    return {
        "n": len(rows), "threshold_source": "inner-validation only",
        "thresholds": thresholds, "threshold_details": threshold_details,
        "acs": acs_metrics(y_acs, marginal["acs"]),
        "omi": {name: None if threshold is None else binary_metrics(y_omi, marginal["omi"], threshold) for name, threshold in thresholds.items()},
        "joint7": joint_metrics(y_joint, q),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, marginal["acs"]),
        "omi_among_nste_acs_primary": {
            "n": int(nste.sum()), "score": "overall p_omi",
            "auroc": _json_number(roc_auc_score(y_omi[nste], marginal["omi"][nste])),
            "auprc": _json_number(average_precision_score(y_omi[nste], marginal["omi"][nste])),
        },
        "omi_among_nste_acs_exploratory": {
            "n": int(nste.sum()), "score": "p_omi_given_nste",
            "auroc": _json_number(roc_auc_score(y_omi[nste], marginal["omi_given_nste"][nste])),
            "auprc": _json_number(average_precision_score(y_omi[nste], marginal["omi_given_nste"][nste])),
        },
        "fixed_sensitivity_operating_points": {
            "sensitivity_0.90": select_operating_point(y_omi, marginal["omi"], 0.90),
            "sensitivity_0.85": select_operating_point(y_omi, marginal["omi"], 0.85),
        },
        "other_omi_sensitivity_analysis": {
            "excluded_count": int(exception.sum()),
            "primary_includes_exception": True,
            "metrics_excluding_singleton_at_primary_youden": binary_metrics(y_omi[~exception], marginal["omi"][~exception], youden),
        },
        "calibration": {
            "fit_subset": "inner-validation (Stage-A diagnostic; no outer-test accessed)",
            "temperature": temperature,
            "uncalibrated": {
                "joint": generic_calibration(y_joint, q, JOINT_CLASSES),
                "acs": multiclass_calibration_metrics(y_acs, marginal["acs"]),
                "omi": binary_calibration_metrics(y_omi, marginal["omi"]),
            },
            "temperature_scaled": {
                "joint": generic_calibration(y_joint, calibrated_q, JOINT_CLASSES),
                "acs": multiclass_calibration_metrics(y_acs, calibrated_marginal["acs"]),
                "omi": binary_calibration_metrics(y_omi, calibrated_marginal["omi"]),
            },
        },
    }


def read_w0(path: Path):
    rows = _read_csv(path)
    y_acs = np.asarray([int(r["y_acs"]) for r in rows])
    y_omi = np.asarray([int(r["y_omi"]) for r in rows])
    p_acs = np.asarray([[float(r[f"p_{name}"]) for name in ACS_CLASSES] for r in rows])
    p_omi = np.asarray([float(r["p_omi"]) for r in rows])
    threshold, detail = select_omi_threshold(y_omi, p_omi, "youden_j")
    nste = np.isin(y_acs, (2, 3))
    return rows, {
        "n": len(rows), "acs": acs_metrics(y_acs, p_acs),
        "omi_youden_threshold": threshold, "omi_youden_selection": detail,
        "omi": binary_metrics(y_omi, p_omi, threshold),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs_primary": {
            "n": int(nste.sum()), "score": "overall p_omi",
            "auroc": float(roc_auc_score(y_omi[nste], p_omi[nste])),
            "auprc": float(average_precision_score(y_omi[nste], p_omi[nste])),
        },
        "fixed_sensitivity_operating_points": {
            "sensitivity_0.90": select_operating_point(y_omi, p_omi, 0.90),
            "sensitivity_0.85": select_operating_point(y_omi, p_omi, 0.85),
        },
        "calibration_uncalibrated": {
            "acs": multiclass_calibration_metrics(y_acs, p_acs),
            "omi": binary_calibration_metrics(y_omi, p_omi),
        },
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_results(figures: Path, w0_rows, j_rows, j_metrics):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
    figures.mkdir(parents=True, exist_ok=True)
    y_acs, y_omi, y_joint, q, marginal = arrays_from_rows(j_rows)
    w0_y = np.asarray([int(r["y_omi"]) for r in w0_rows]); w0_p = np.asarray([float(r["p_omi"]) for r in w0_rows])

    for filename, cm, labels, title in (
        ("joint7_confusion_matrix.png", np.asarray(j_metrics["joint7"]["confusion_matrix"]), JOINT_CLASSES, "Joint-7 confusion matrix"),
        ("acs4_confusion_matrix.png", np.asarray(j_metrics["acs"]["confusion_matrix"]), ACS_CLASSES, "ACS marginal confusion matrix"),
    ):
        size = 9 if len(labels) == 7 else 6
        fig, ax = plt.subplots(figsize=(size, size * .8)); ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False); ax.set_title(title); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)

    for kind in ("pr", "roc"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, y, p in (("W0", w0_y, w0_p), ("Joint-7", y_omi, marginal["omi"])):
            if kind == "pr":
                precision, recall, _ = precision_recall_curve(y, p); ax.plot(recall, precision, label=name); ax.set(xlabel="Sensitivity", ylabel="PPV", title="OMI PR curve")
            else:
                fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, label=name); ax.set(xlabel="1-Specificity", ylabel="Sensitivity", title="OMI ROC curve")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / f"omi_{kind}_w0_vs_joint7.png", dpi=180); plt.close(fig)

    nste = np.isin(y_acs, (2, 3))
    fig, ax = plt.subplots(figsize=(6, 5)); precision, recall, _ = precision_recall_curve(y_omi[nste], marginal["omi"][nste]); ax.plot(recall, precision); ax.set(xlabel="Sensitivity", ylabel="PPV", title="NSTE-ACS OMI PR curve (overall p_omi)"); fig.tight_layout(); fig.savefig(figures / "nste_acs_omi_pr.png", dpi=180); plt.close(fig)
    for k, filename in ((2, "stemi_omi_pr.png"), (4, "nstemi_omi_pr.png"), (6, "ua_omi_pr.png")):
        fig, ax = plt.subplots(figsize=(6, 5)); target = (y_joint == k).astype(int)
        if target.sum() and target.sum() < len(target):
            precision, recall, _ = precision_recall_curve(target, q[:, k]); ax.plot(recall, precision)
        else:
            ax.text(.5, .5, f"Not estimable: support={target.sum()}", ha="center", va="center")
        ax.set(xlabel="Sensitivity", ylabel="PPV", title=JOINT_CLASSES[k]); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)

    w0_acs = acs_metrics(np.asarray([int(r["y_acs"]) for r in w0_rows]), np.asarray([[float(r[f"p_{name}"]) for name in ACS_CLASSES] for r in w0_rows]))
    labels = ("Macro AUPRC", "Macro AUROC", "Macro F1", "Balanced accuracy", "Accuracy")
    keys = ("macro_auprc_ovr", "macro_auroc_ovr", "macro_f1", "balanced_accuracy", "accuracy")
    x = np.arange(len(keys)); width=.36
    fig, ax=plt.subplots(figsize=(9,5)); ax.bar(x-width/2,[w0_acs[k] for k in keys],width,label="W0"); ax.bar(x+width/2,[j_metrics["acs"][k] for k in keys],width,label="Joint-7"); ax.set_xticks(x,labels,rotation=20,ha="right"); ax.set_ylim(0,1); ax.legend(); ax.set_title("ACS marginal performance"); fig.tight_layout(); fig.savefig(figures/"acs_macro_performance_comparison.png",dpi=180); plt.close(fig)

    for level, cal_key, filename in (("Joint", "joint", "joint_calibration.png"), ("ACS", "acs", "acs_calibration.png"), ("OMI", "omi", "omi_calibration.png")):
        cal = j_metrics["calibration"]["uncalibrated"][cal_key]
        bins = cal.get("top_label_bins", cal.get("bins", [])); bins = [item for item in bins if item["n"]]
        fig, ax=plt.subplots(figsize=(5,5)); ax.plot([b["mean_probability"] for b in bins],[b["fraction_positive"] for b in bins],marker="o"); ax.plot([0,1],[0,1],"k:"); ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted probability",ylabel="Observed frequency",title=f"{level} calibration"); fig.tight_layout(); fig.savefig(figures/filename,dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/joint7_stage_a/joint7/fold0/checkpoint_best.pt")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--w0_predictions", default="results/multi_objective_evaluation/w0/fold0/inner_validation_predictions.csv")
    parser.add_argument("--output_root", default="results/joint7_stage_a")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    output = ROOT / args.output_root; predictions = output / "predictions"; figures = output / "figures"; configs = output / "configs"
    for path in (predictions, figures, configs): path.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / args.checkpoint; manifest_root = ROOT / args.manifest_root; manifest_dir = manifest_root / "fold0"
    audit = json.loads((output / "joint_label_audit/audit_summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass": raise RuntimeError("joint label audit is not PASS")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, logits = infer_joint7(checkpoint, manifest_dir, device, args.workers, args.batch)
    assignments = [row for row in _read_csv(manifest_dir / "patient_assignments.csv") if row["subset"] == "inner_validation"]
    source = np.load(manifest_root / "y_all.npy")
    rows, _ = make_rows(records, logits, assignments, source)
    write_csv(predictions / "joint7_fold0_inner_validation.csv", rows)
    joint7 = evaluate(rows, logits)
    w0_rows, w0 = read_w0(ROOT / args.w0_predictions)
    if [r["record_id"] for r in w0_rows] != [r["record_id"] for r in rows]:
        raise RuntimeError("W0 and Joint-7 validation record order/coverage differ")
    delta_acs = joint7["acs"]["macro_auprc_ovr"] - w0["acs"]["macro_auprc_ovr"]
    delta_omi = joint7["omi"]["youden_j"]["auprc"] - w0["omi"]["auprc"]
    delta_sensitivity = joint7["omi"]["youden_j"]["sensitivity"] - w0["omi"]["sensitivity"]
    go_a = min(delta_acs, delta_omi) >= -0.003 and max(delta_acs, delta_omi) >= 0.005
    go_b = delta_acs >= 0.003 and delta_omi >= 0.003
    decision = "GO" if (go_a or go_b) and delta_sensitivity >= -0.02 else "NO-GO"
    reason = "满足 GO A/B 且 OMI sensitivity 安全边界通过" if decision == "GO" else "未同时满足性能门槛和 OMI sensitivity 安全边界"
    result = {
        "protocol": "nested Fold0 inner-validation only; outer-test/external-test not accessed",
        "w0": w0, "joint7": joint7,
        "delta": {"acs_macro_auprc": delta_acs, "omi_auprc": delta_omi, "omi_sensitivity_youden": delta_sensitivity},
        "gate": {"go_a": go_a, "go_b": go_b, "sensitivity_safety_pass": delta_sensitivity >= -0.02, "decision": decision, "reason": reason},
        "provenance": {
            "checkpoint_sha256": sha256(checkpoint), "joint_predictions_sha256": sha256(predictions / "joint7_fold0_inner_validation.csv"),
            "w0_predictions_sha256": sha256(ROOT / args.w0_predictions), "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
            "outer_test_accessed": False, "external_test_accessed": False,
        },
    }
    (output / "stage_a_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    summary_rows=[]
    for method, metrics in (("W0",w0),("Joint-7",joint7)):
        omi = metrics["omi"] if method=="W0" else metrics["omi"]["youden_j"]
        summary_rows.append({"method":method,"acs_macro_auroc":metrics["acs"]["macro_auroc_ovr"],"acs_macro_auprc":metrics["acs"]["macro_auprc_ovr"],"acs_macro_f1":metrics["acs"]["macro_f1"],"acs_balanced_accuracy":metrics["acs"]["balanced_accuracy"],"acs_accuracy":metrics["acs"]["accuracy"],"omi_auroc":omi["auroc"],"omi_auprc":omi["auprc"],"omi_f1":omi["f1"],"omi_sensitivity":omi["sensitivity"],"omi_specificity":omi["specificity"],"omi_ppv":omi["ppv"],"omi_npv":omi["npv"],"omi_mcc":omi["mcc"],"omi_accuracy":omi["accuracy"],"omi_threshold":omi["threshold"],"nste_omi_auprc":metrics["omi_among_nste_acs_primary"]["auprc"],"nstemi_vs_ua_auprc":metrics["nstemi_vs_ua"]["auprc"]})
    write_csv(output / "joint_stage_a_metrics.csv", summary_rows)
    class_rows=[]
    for name,item in joint7["joint7"]["per_class"].items(): class_rows.append({"class":name,**item})
    write_csv(output / "joint_class_metrics.csv",class_rows)
    shutil.copy2(ROOT / "configs/ecg_lfm/corrected_joint7_lora.yaml", configs / "corrected_joint7_lora.yaml")
    shutil.copy2(ROOT / "configs/ecg_lfm/corrected_joint_lora.yaml", configs / "corrected_joint_lora.yaml")
    plot_results(figures,w0_rows,rows,joint7)
    report = ROOT / "ECG_LFM_joint_acs_omi_stage_a_report.md"
    report.write_text(render_report(result,audit,summary_rows),encoding="utf-8")
    print(json.dumps({"status":"pass","decision":decision,"report":str(report),"outer_test_accessed":False},ensure_ascii=False))


def render_report(result,audit,rows):
    w0,j1=result["w0"],result["joint7"]; d=result["delta"]; gate=result["gate"]
    def f(x): return "NA" if x is None else f"{float(x):.6f}"
    table="\n".join(f"| {r['method']} | {f(r['acs_macro_auprc'])} | {f(r['acs_macro_f1'])} | {f(r['acs_balanced_accuracy'])} | {f(r['acs_accuracy'])} | {f(r['omi_auprc'])} | {f(r['omi_f1'])} | {f(r['omi_sensitivity'])} | {f(r['omi_specificity'])} | {f(r['omi_ppv'])} | {f(r['omi_threshold'])} |" for r in rows)
    return f"""# ECG-LFM ACS×OMI Joint-7 Stage-A 实验报告

## 1. Repository audit

当前 W0 的实际实现是单个 `Linear(1024,5)` 投影层按语义拆分：前四个 logits 使用 ACS softmax/CE，第五个 logit 使用 OMI sigmoid/BCE；不是两个物理独立 Linear Head。本次保留 W0 路径和旧 checkpoint 兼容性，新增 `task.formulation=joint_acs_omi`、`Linear(1024,7)` 与单一 CrossEntropyLoss。

## 2. Joint-label audit

审计状态：**{audit['status']}**。`OTHER+OMI={audit['other_omi_count']}`，按协议映射为 OTHER joint target，同时原始 OMI=1 和异常标志保留。源标签与 master index 逐条不一致数：{audit['source_master_positional_mismatches']}。空联合类：{', '.join(audit['empty_joint_classes_in_outer_cv_pool']) or '无'}。

## 3. 7-class label distribution

完整聚合计数见 `results/joint7_stage_a/joint_label_audit/overall_joint_counts.csv` 和 `fold_joint_counts.csv`。`UA+OMI` 全局 support 为 {audit['ua_omi_global_support']}，因此该类 AUROC/AUPRC/F1/Sensitivity 不可有效估计；Joint macro AUROC/AUPRC只对有正负样本的可估计类别求均值，不能把空类伪装成有效结果。

## 4. W0 reproduction

J0 使用现有、已经过 provenance 核对的 nested Fold0 W0 checkpoint 与内部验证预测；未重新调参。W0 与 Joint-7 的 record_id 顺序和覆盖完全一致。

## 5. Joint-7 implementation

Joint-7 固定类别为 OTHER、STEMI+nOMI、STEMI+OMI、NSTEMI+nOMI、NSTEMI+OMI、UA+nOMI、UA+OMI。训练只使用未加权 joint CE。ACS 和 OMI 概率均由 joint softmax 边缘化得到。

## 6. Training configuration

两者共享 ECG-LFM pretrained initialization、nested Fold0、seed=2026、LoRA r=8/alpha=16/dropout=0.05、Adam、LR=1e-4、batch=64、gradient accumulation=4、Step scheduler、最多100 epoch与 patience=10。Joint-7 checkpoint 仅按 inner-validation joint CE 最小选择。门控前未访问 outer-test 或 external-test。

## 7. ACS 与 OMI 结果

| Method | ACS Macro AUPRC | ACS Macro F1 | ACS BAcc | ACS Acc | OMI AUPRC | OMI F1 | OMI Sens | OMI Spec | OMI PPV | Youden threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 8. NSTE-ACS subgroup results

W0 OMI among NSTE-ACS AUPRC={f(w0['omi_among_nste_acs_primary']['auprc'])}；Joint-7 primary overall-p_omi AUPRC={f(j1['omi_among_nste_acs_primary']['auprc'])}；Joint-7 exploratory conditional-p_omi_given_nste AUPRC={f(j1['omi_among_nste_acs_exploratory']['auprc'])}。W0 NSTEMI-vs-UA AUPRC={f(w0['nstemi_vs_ua']['auprc'])}；Joint-7={f(j1['nstemi_vs_ua']['auprc'])}。

## 9. Joint-class results

Joint-7 supported-class Macro AUROC={f(j1['joint7']['macro_auroc_ovr_supported_classes'])}，Macro AUPRC={f(j1['joint7']['macro_auprc_ovr_supported_classes'])}，7-class Macro F1={f(j1['joint7']['macro_f1_all_7_classes'])}，Balanced Accuracy={f(j1['joint7']['balanced_accuracy_supported_classes'])}，Accuracy={f(j1['joint7']['accuracy'])}。逐类结果见 `joint_class_metrics.csv`。

## 10. Calibration

温度缩放只作用于 7-class joint logits，再边缘化为 ACS/OMI。Stage-A 的温度在 inner-validation 拟合，因此这里只作为拟合集诊断，不声称独立泛化校准性能。Joint NLL由 {f(j1['calibration']['uncalibrated']['joint']['nll'])} 变为 {f(j1['calibration']['temperature_scaled']['joint']['nll'])}。

## 11. Fixed-sensitivity operating points

W0 Specificity@Sensitivity≥0.90={f(w0['fixed_sensitivity_operating_points']['sensitivity_0.90'].get('specificity'))}，PPV={f(w0['fixed_sensitivity_operating_points']['sensitivity_0.90'].get('ppv'))}；Joint-7 分别为 {f(j1['fixed_sensitivity_operating_points']['sensitivity_0.90'].get('specificity'))} 和 {f(j1['fixed_sensitivity_operating_points']['sensitivity_0.90'].get('ppv'))}。

## 12. OTHER+OMI sensitivity analysis

Primary OMI 评价包含 singleton。排除该 singleton 仅作为 sensitivity analysis，结果见 `stage_a_results.json`，不替代主结果。

## 13. Delta vs W0 与 GO gate

ΔACS Macro AUPRC={f(d['acs_macro_auprc'])}；ΔOMI AUPRC={f(d['omi_auprc'])}；ΔOMI Sensitivity={f(d['omi_sensitivity_youden'])}。GO-A={gate['go_a']}，GO-B={gate['go_b']}，safety={gate['sensitivity_safety_pass']}。

## 14. 最终判定

**{gate['decision']}**：{gate['reason']}。

## 15. 是否建议完整 nested 5-fold

{'建议按预注册流程进入完整 5-fold；只冻结当前方案，不增加其它变量。' if gate['decision']=='GO' else '不建议进入完整 5-fold；按协议停止 Joint-7 扩展并保留失败结果。'}

## 16. 可追溯文件

完整概率预测保存在服务器 `results/joint7_stage_a/predictions/`；其中包含 patient_id，按仓库安全规则不提交 Git。聚合指标、配置、审计摘要和图保存在 `results/joint7_stage_a/`。
"""


if __name__ == "__main__": main()
