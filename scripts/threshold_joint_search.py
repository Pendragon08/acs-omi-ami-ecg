"""Joint F1-sensitivity threshold selection for final OMI/AMI classifiers.

Runs validation inference once per task, evaluates thresholds 0.01..0.99,
selects several transparent operating points, and estimates uncertainty with a
patient-cluster bootstrap.  Original training JSON/checkpoints are untouched.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.exp_acs import make_loaders, predict  # noqa: E402
from nets import build_model  # noqa: E402


OUT = ROOT / "results" / "final_fusion_focal"
GRID = np.arange(1, 100, dtype=np.float64) / 100.0
TASKS = {"OMI": 0, "AMI": 1}
N_BOOT = 2000
BOOT_SEED = 20260817


def metric_arrays(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    pred = p[:, None] >= GRID[None, :]
    pos = y[:, None] == 1
    neg = ~pos
    tp = np.sum(pred & pos, axis=0)
    fp = np.sum(pred & neg, axis=0)
    fn = np.sum((~pred) & pos, axis=0)
    tn = np.sum((~pred) & neg, axis=0)

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    sens = div(tp, tp + fn)
    spec = div(tn, tn + fp)
    ppv = div(tp, tp + fp)
    npv = div(tn, tn + fn)
    acc = div(tp + tn, tp + tn + fp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    joint = div(2 * f1 * sens, f1 + sens)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "npv": npv,
        "acc": acc,
        "f1": f1,
        "youden": sens + spec - 1.0,
        "joint_f1_sens_hmean": joint,
        "min_f1_sens": np.minimum(f1, sens),
    }


def select_indices(m: dict[str, np.ndarray]) -> dict[str, int]:
    max_f1 = float(np.max(m["f1"]))
    eligible_95 = np.flatnonzero(m["f1"] >= 0.95 * max_f1)
    eligible_s60 = np.flatnonzero(m["sens"] >= 0.60)
    return {
        # np.argmax returns the lowest threshold on ties, favoring sensitivity.
        "max_f1": int(np.argmax(m["f1"])),
        "max_joint_hmean": int(np.argmax(m["joint_f1_sens_hmean"])),
        "max_min_f1_sens": int(np.argmax(m["min_f1_sens"])),
        "max_sens_within_95pct_max_f1": int(eligible_95[0]),
        "max_f1_with_sens_ge_0.60": (
            int(eligible_s60[np.argmax(m["f1"][eligible_s60])])
            if len(eligible_s60) else int(np.argmax(m["sens"]))
        ),
    }


def row_at(m: dict[str, np.ndarray], idx: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {"threshold": round(float(GRID[idx]), 2)}
    for key, values in m.items():
        value = values[idx]
        row[key] = int(value) if key in {"tp", "fp", "fn", "tn"} else float(value)
    return row


def cluster_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    recommended_idx: int,
) -> dict:
    unique_patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    n_patients = len(unique_patients)
    pred = (p[:, None] >= GRID[None, :]).astype(np.int8)
    y_float = y.astype(float)
    rng = np.random.default_rng(BOOT_SEED)

    selected_f1 = []
    selected_joint = []
    fixed = {k: [] for k in ("f1", "sens", "spec", "ppv", "npv", "acc")}
    valid = 0

    for _ in range(N_BOOT):
        drawn = rng.integers(0, n_patients, size=n_patients)
        patient_counts = np.bincount(drawn, minlength=n_patients)
        weights = patient_counts[inverse].astype(float)
        n_pos = float(weights @ y_float)
        n_neg = float(weights.sum() - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue

        tp = (weights * y_float) @ pred
        fp = (weights * (1.0 - y_float)) @ pred
        fn = n_pos - tp
        tn = n_neg - fp

        def div(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

        sens = div(tp, tp + fn)
        spec = div(tn, tn + fp)
        ppv = div(tp, tp + fp)
        npv = div(tn, tn + fn)
        acc = div(tp + tn, tp + tn + fp + fn)
        f1 = div(2 * tp, 2 * tp + fp + fn)
        joint = div(2 * f1 * sens, f1 + sens)

        selected_f1.append(float(GRID[int(np.argmax(f1))]))
        selected_joint.append(float(GRID[int(np.argmax(joint))]))
        for key, values in {
            "f1": f1,
            "sens": sens,
            "spec": spec,
            "ppv": ppv,
            "npv": npv,
            "acc": acc,
        }.items():
            fixed[key].append(float(values[recommended_idx]))
        valid += 1

    def summarize_threshold(values):
        arr = np.asarray(values)
        counts = Counter(np.round(arr, 2))
        return {
            "median": float(np.median(arr)),
            "ci95": [float(x) for x in np.percentile(arr, [2.5, 97.5])],
            "top5": [
                {"threshold": float(t), "count": int(c), "fraction": float(c / len(arr))}
                for t, c in counts.most_common(5)
            ],
        }

    metric_ci = {
        key: {
            "median": float(np.median(values)),
            "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        }
        for key, values in fixed.items()
    }
    return {
        "method": "patient-cluster bootstrap",
        "requested_replicates": N_BOOT,
        "valid_replicates": valid,
        "n_unique_patients": n_patients,
        "selected_threshold_max_f1": summarize_threshold(selected_f1),
        "selected_threshold_max_joint_hmean": summarize_threshold(selected_joint),
        "recommended_fixed_threshold_metrics": metric_ci,
    }


def write_csv(path: Path, m: dict[str, np.ndarray], selected: dict[str, int]) -> None:
    selected_by_idx: dict[int, list[str]] = {}
    for name, idx in selected.items():
        selected_by_idx.setdefault(idx, []).append(name)
    fields = [
        "threshold", "sens", "spec", "ppv", "npv", "acc", "f1", "youden",
        "joint_f1_sens_hmean", "min_f1_sens", "tp", "fp", "fn", "tn", "selected_by",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(GRID)):
            row = row_at(m, i)
            row["selected_by"] = ";".join(selected_by_idx.get(i, []))
            writer.writerow({key: row.get(key, "") for key in fields})


def calibration_diagnostics(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    """Brier score, equal-width ECE, and logistic calibration intercept/slope."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        mean_prob = float(np.mean(p[mask]))
        event_rate = float(np.mean(y[mask]))
        count = int(np.sum(mask))
        ece += count / len(y) * abs(mean_prob - event_rate)
        bins.append({
            "lower": float(edges[i]), "upper": float(edges[i + 1]),
            "count": count, "mean_probability": mean_prob, "event_rate": event_rate,
        })

    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    cal.fit(logits, y)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "ece_equal_width_15": float(ece),
        "logistic_calibration_intercept": float(cal.intercept_[0]),
        "logistic_calibration_slope": float(cal.coef_[0, 0]),
        "bins": bins,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads((ROOT / "data" / "splits" / "patient_split.json").read_text())
    meta = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
    val_idx = np.asarray(split["val"], dtype=int)
    val_patients = np.asarray(meta["patient_ids"])[val_idx]
    summary = {
        "generated_by": "scripts/ecg/threshold_joint_search.py",
        "grid": {"min": 0.01, "max": 0.99, "step": 0.01, "n": 99},
        "selection_policy": (
            "recommended=max harmonic mean(F1,Sensitivity); ties use the lowest threshold "
            "to favor sensitivity"
        ),
        "bootstrap_seed": BOOT_SEED,
        "tasks": {},
    }

    for task, col in TASKS.items():
        tag = f"XceptionTime_{task}_bce_P3_seed2026"
        state = torch.load(
            OUT / "checkpoints" / f"{tag}.pt", map_location=device, weights_only=True
        )
        model = build_model("XceptionTime", num_class=1, device=device)
        model.load_state_dict(state)
        model.to(device).eval()
        _, val_loader = make_loaders(
            "P3", 256, 2026, num_workers=0, label_cols=(col,)
        )
        y_val, p_val = predict(model, val_loader, device, use_fusion=False)
        y = y_val[:, 0].astype(int)
        p = p_val[:, 0].astype(float)
        if len(y) != len(val_idx):
            raise RuntimeError(f"validation alignment mismatch: {len(y)} != {len(val_idx)}")

        m = metric_arrays(y, p)
        selected = select_indices(m)
        points = {name: row_at(m, idx) for name, idx in selected.items()}
        recommended_idx = selected["max_joint_hmean"]
        bootstrap = cluster_bootstrap(y, p, val_patients, recommended_idx)
        task_summary = {
            "n_validation": int(len(y)),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
            "auroc_sklearn": float(roc_auc_score(y, p)),
            "auprc_sklearn": float(average_precision_score(y, p)),
            "calibration": calibration_diagnostics(y, p),
            "selected_points": points,
            "recommended_criterion": "max_joint_hmean",
            "recommended": points["max_joint_hmean"],
            "bootstrap": bootstrap,
        }
        summary["tasks"][task] = task_summary
        write_csv(OUT / f"threshold_joint_{task}.csv", m, selected)

    (OUT / "threshold_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# OMI/AMI Joint F1–Sensitivity Threshold Selection",
        "",
        "Validation-only grid search over thresholds 0.01–0.99 (step 0.01).",
        "The recommended operating point maximizes the harmonic mean of F1 and Sensitivity.",
        "Uncertainty uses a 2,000-replicate patient-cluster bootstrap.",
        "",
        "| Task | Criterion | Threshold | F1 | Sens | Spec | PPV | Joint H |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, data in summary["tasks"].items():
        for criterion, point in data["selected_points"].items():
            lines.append(
                f"| {task} | {criterion} | {point['threshold']:.2f} | "
                f"{point['f1']:.4f} | {point['sens']:.4f} | {point['spec']:.4f} | "
                f"{point['ppv']:.4f} | {point['joint_f1_sens_hmean']:.4f} |"
            )
        lines.append("")
        rec = data["recommended"]
        ci = data["bootstrap"]["recommended_fixed_threshold_metrics"]
        threshold_boot = data["bootstrap"]["selected_threshold_max_joint_hmean"]
        lines.extend([
            f"## {task} recommended threshold: {rec['threshold']:.2f}",
            "",
            f"- F1: {rec['f1']:.4f} (bootstrap 95% CI "
            f"{ci['f1']['ci95'][0]:.4f}–{ci['f1']['ci95'][1]:.4f})",
            f"- Sensitivity: {rec['sens']:.4f} (bootstrap 95% CI "
            f"{ci['sens']['ci95'][0]:.4f}–{ci['sens']['ci95'][1]:.4f})",
            f"- Specificity: {rec['spec']:.4f} (bootstrap 95% CI "
            f"{ci['spec']['ci95'][0]:.4f}–{ci['spec']['ci95'][1]:.4f})",
            f"- Re-selected joint-threshold bootstrap median: "
            f"{threshold_boot['median']:.2f} (95% interval "
            f"{threshold_boot['ci95'][0]:.2f}–{threshold_boot['ci95'][1]:.2f})",
            "",
        ])
    (OUT / "threshold_selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
