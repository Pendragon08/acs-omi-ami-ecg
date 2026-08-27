"""Final multi-seed OMI loss summary, plots and patient-cluster bootstrap."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exp.exp_acs_ablation import threshold_analysis


SEEDS = [42, 2024, 2025, 2026, 3407]
PRIMARY_OP = "max_f1_sensitivity_hmean_0.01_0.99"


def load_jsons(result_dir):
    out = []
    for path in result_dir.glob("loss_*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("study") == "loss":
            item["_path"] = path
            out.append(item)
    return out


def same_float(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) < 1e-10


def selected_groups(items, selection):
    best_focal = selection["best_focal"]
    best_cb = selection["best_cb_focal"]
    groups = {"Baseline": [], "Weighted BCE": [], "Focal": [], "CB-Focal": []}
    for item in items:
        if item["seed"] not in SEEDS:
            continue
        cfg = item["loss_config"]
        if item["loss"] == "bce":
            groups["Baseline"].append(item)
        elif item["loss"] == "wbce":
            groups["Weighted BCE"].append(item)
        elif item["loss"] == "focal" and same_float(cfg.get("alpha"), best_focal["alpha"]) and same_float(cfg.get("gamma"), best_focal["gamma"]):
            groups["Focal"].append(item)
        elif item["loss"] == "cb_focal" and same_float(cfg.get("beta"), best_cb["beta"]) and same_float(cfg.get("gamma"), best_cb["gamma"]):
            groups["CB-Focal"].append(item)
    for name, runs in groups.items():
        runs.sort(key=lambda x: x["seed"])
        got = [run["seed"] for run in runs]
        if got != sorted(SEEDS):
            raise SystemExit(f"{name} incomplete: got seeds {got}, expected {sorted(SEEDS)}")
    return groups


def seed_row(name, run, op_name=PRIMARY_OP):
    op = run["val"]["operating_points"][op_name]
    return {
        "Loss": name,
        "Seed": run["seed"],
        "AUROC": run["val"]["auroc"],
        "AUPRC": run["val"]["auprc"],
        "Sensitivity": op["sensitivity"],
        "Specificity": op["specificity"],
        "PPV": op["ppv"],
        "NPV": op["npv"],
        "F1": op["f1"],
        "MCC": op["mcc"],
        "Accuracy": op["accuracy"],
        "Threshold": op["threshold"],
        "TP": op["tp"],
        "TN": op["tn"],
        "FP": op["fp"],
        "FN": op["fn"],
        "Best epoch": run["best_epoch"],
    }


def mean_std_text(values, digits=4):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def summary_frame(seed_frame):
    metric_cols = [
        "AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV",
        "F1", "MCC", "Accuracy", "Threshold", "TP", "TN", "FP", "FN",
    ]
    rows = []
    for name, part in seed_frame.groupby("Loss", sort=False):
        row = {"Loss": name}
        for col in metric_cols:
            row[col] = mean_std_text(part[col], digits=2 if col in {"TP", "TN", "FP", "FN"} else 4)
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_path(result_dir, run):
    rel = run["artifacts"]["validation_predictions_csv"]
    return ROOT / rel


def ensemble(group, result_dir):
    frames = [pd.read_csv(prediction_path(result_dir, run)) for run in group]
    base = frames[0][["record_id", "patient_id", "y_true"]].copy()
    probs = []
    for frame in frames:
        if not base[["record_id", "patient_id", "y_true"]].equals(frame[["record_id", "patient_id", "y_true"]]):
            raise ValueError("validation prediction alignment mismatch")
        probs.append(frame["probability"].to_numpy())
    base["probability"] = np.mean(probs, axis=0)
    return base


def point_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "accuracy": (tp + tn) / len(y),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def patient_cluster_bootstrap(base_frame, cb_frame, base_threshold, cb_threshold, n_boot=2000):
    if not base_frame[["record_id", "patient_id", "y_true"]].equals(cb_frame[["record_id", "patient_id", "y_true"]]):
        raise ValueError("paired bootstrap alignment mismatch")
    patient_ids = base_frame["patient_id"].astype(str).to_numpy()
    unique = np.unique(patient_ids)
    lookup = {pid: np.flatnonzero(patient_ids == pid) for pid in unique}
    y = base_frame["y_true"].to_numpy(dtype=int)
    pb = base_frame["probability"].to_numpy(dtype=float)
    pc = cb_frame["probability"].to_numpy(dtype=float)
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "f1"]
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(20260818)
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([lookup[pid] for pid in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        mb = point_metrics(y[idx], pb[idx], base_threshold)
        mc = point_metrics(y[idx], pc[idx], cb_threshold)
        for metric in metrics:
            draws[metric].append(mc[metric] - mb[metric])
    result = {}
    for metric, values in draws.items():
        a = np.asarray(values)
        result[metric] = {
            "delta": float(a.mean()),
            "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "paired_bootstrap_p_two_sided": float(min(1.0, 2.0 * min((a <= 0).mean(), (a >= 0).mean()))),
            "n_bootstrap": int(len(a)),
        }
    return result


def plot_curves(ensembles, thresholds, groups, result_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, frame in ensembles.items():
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["probability"])
        auc = roc_auc_score(frame["y_true"], frame["probability"])
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="OMI validation ROC (5-seed ensembles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = ensembles["Baseline"]["y_true"].mean()
    for name, frame in ensembles.items():
        precision, recall, _ = precision_recall_curve(frame["y_true"], frame["probability"])
        ap = average_precision_score(frame["y_true"], frame["probability"])
        ax.plot(recall, precision, label=f"{name} ({ap:.3f})")
    ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1)
    ax.set(xlabel="Recall", ylabel="Precision", title="OMI validation precision-recall (5-seed ensembles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "precision_recall_curve.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (name, frame) in zip(axes.ravel(), ensembles.items()):
        pred = (frame["probability"].to_numpy() >= thresholds[name]).astype(int)
        cm = confusion_matrix(frame["y_true"], pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set(title=f"{name} (t={thresholds[name]:.2f})", xlabel="Predicted", ylabel="True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
    for ax, (name, runs) in zip(axes.ravel(), groups.items()):
        for run in runs:
            hist = pd.DataFrame(run["history"])
            ax.plot(hist["epoch"], hist["train_loss"], alpha=0.45, label=f"train s{run['seed']}")
            ax.plot(hist["epoch"], hist["val_loss"], alpha=0.45, linestyle="--", label=f"val s{run['seed']}")
        ax.set(title=name, xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(result_dir / "training_validation_loss_curves.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="loss_feature_ablation")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    result_dir = ROOT / "results" / args.exp_name
    selection = json.loads((result_dir / "stage1_selection.json").read_text(encoding="utf-8"))
    groups = selected_groups(load_jsons(result_dir), selection)

    seed_rows = [seed_row(name, run) for name, runs in groups.items() for run in runs]
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(result_dir / "loss_ablation_seeds.csv", index=False)
    summary = summary_frame(seed_frame)
    summary.to_csv(result_dir / "loss_ablation.csv", index=False)

    threshold_rows = []
    for name, runs in groups.items():
        for run in runs:
            for strategy, op in run["val"]["operating_points"].items():
                if op is not None:
                    threshold_rows.append({"Loss": name, "Seed": run["seed"], "Strategy": strategy, **op})
    pd.DataFrame(threshold_rows).to_csv(result_dir / "loss_ablation_threshold_strategies.csv", index=False)

    ensembles = {name: ensemble(runs, result_dir) for name, runs in groups.items()}
    thresholds = {}
    ensemble_metrics = {}
    for name, frame in ensembles.items():
        _, selected = threshold_analysis(frame["y_true"].to_numpy(), frame["probability"].to_numpy())
        thresholds[name] = selected[PRIMARY_OP]["threshold"]
        ensemble_metrics[name] = point_metrics(
            frame["y_true"].to_numpy(), frame["probability"].to_numpy(), thresholds[name]
        )
    (result_dir / "loss_ablation_ensemble.json").write_text(
        json.dumps({"thresholds": thresholds, "metrics": ensemble_metrics}, indent=2), encoding="utf-8"
    )

    bootstrap = patient_cluster_bootstrap(
        ensembles["Baseline"], ensembles["CB-Focal"], thresholds["Baseline"], thresholds["CB-Focal"], args.bootstrap
    )
    (result_dir / "baseline_vs_cb_focal_paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    plot_curves(ensembles, thresholds, groups, result_dir)

    means = summary.set_index("Loss")
    deltas = {
        metric: float(means.loc["CB-Focal", f"{metric}_mean"] - means.loc["Baseline", f"{metric}_mean"])
        for metric in ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "F1"]
    }
    paired_seed_improvements = int(
        sum(
            cb > base
            for cb, base in zip(
                seed_frame[seed_frame["Loss"] == "CB-Focal"].sort_values("Seed")["AUPRC"],
                seed_frame[seed_frame["Loss"] == "Baseline"].sort_values("Seed")["AUPRC"],
            )
        )
    )
    if (
        deltas["AUPRC"] > 0 and deltas["Sensitivity"] > 0 and deltas["F1"] > 0
        and deltas["PPV"] > -0.02 and deltas["Specificity"] > -0.02
        and paired_seed_improvements >= 4
    ):
        verdict = "Strong improvement"
    elif (
        deltas["Sensitivity"] > 0 and (deltas["AUPRC"] > 0 or deltas["F1"] > 0)
        and deltas["Specificity"] > -0.05 and paired_seed_improvements >= 3
    ):
        verdict = "Moderate improvement"
    else:
        verdict = "No convincing improvement"

    best_cb = selection["best_cb_focal"]
    lines = [
        "# OMI loss ablation summary",
        "",
        f"Primary operating point: validation threshold maximizing the harmonic mean of F1 and Sensitivity on the 0.01–0.99 grid. Results are mean ± sample SD over seeds {SEEDS}.",
        "",
        "| Loss | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['Loss']} | {row['AUROC']} | {row['AUPRC']} | {row['Sensitivity']} | "
            f"{row['Specificity']} | {row['PPV']} | {row['NPV']} | {row['F1']} | {row['MCC']} | {row['Accuracy']} |"
        )
    lines += [
        "",
        f"Best CB-Focal: beta={best_cb['beta']}, gamma={best_cb['gamma']}; mean per-seed threshold={means.loc['CB-Focal', 'Threshold_mean']:.4f}.",
        f"Paired seed AUPRC improvements: {paired_seed_improvements}/5.",
        f"Verdict: **{verdict}**.",
        "",
        "## CB-Focal minus Baseline",
        "",
    ]
    for metric, delta in deltas.items():
        boot = bootstrap.get(metric.lower())
        suffix = f"; paired patient bootstrap 95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}], p={boot['paired_bootstrap_p_two_sided']:.4g}" if boot else ""
        lines.append(f"- Δ{metric}={delta:+.4f}{suffix}")
    lines += [
        "",
        "External hidden-label test results were not used to select loss parameters or thresholds. This report is validation evidence; one frozen external evaluation may follow after the method is finalized.",
    ]
    (result_dir / "loss_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "deltas": deltas, "paired_seed_AUPRC_improvements": paired_seed_improvements}, indent=2))


if __name__ == "__main__":
    main()
