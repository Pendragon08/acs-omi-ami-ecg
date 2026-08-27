"""Summarize controlled feature screening and multi-seed confirmations."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 2024, 2025, 2026, 3407]
PRIMARY_OP = "max_f1_sensitivity_hmean_0.01_0.99"


def load_items(result_dir):
    items = []
    for path in result_dir.glob("feature_*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("study") == "feature":
            items.append(item)
    return items


def row(item):
    op = item["val"]["operating_points"][PRIMARY_OP]
    return {
        "Task": item["task"].upper(),
        "Feature mode": item["feature_mode"],
        "Seed": item["seed"],
        "AUROC": item["val"]["auroc"],
        "AUPRC": item["val"]["auprc"],
        "Sensitivity": op["sensitivity"],
        "Specificity": op["specificity"],
        "PPV": op["ppv"],
        "NPV": op["npv"],
        "F1": op["f1"],
        "MCC": op["mcc"],
        "Accuracy": op["accuracy"],
        "Threshold": op["threshold"],
        "TP": op["tp"], "TN": op["tn"], "FP": op["fp"], "FN": op["fn"],
    }


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="loss_feature_ablation")
    args = p.parse_args()
    result_dir = ROOT / "results" / args.exp_name
    selection = json.loads((result_dir / "stage1_selection.json").read_text(encoding="utf-8"))
    items = load_items(result_dir)
    all_rows = pd.DataFrame([row(item) for item in items])
    screen = all_rows[all_rows["Seed"] == 2026].copy()
    if len(screen) != 16:
        raise SystemExit(f"feature screen incomplete: {len(screen)}/16")
    screen["Delta AUPRC vs controlled raw"] = np.nan
    screen["Delta AUROC vs controlled raw"] = np.nan
    for task in ("OMI", "AMI"):
        raw = screen[(screen["Task"] == task) & (screen["Feature mode"] == "raw")].iloc[0]
        mask = screen["Task"] == task
        screen.loc[mask, "Delta AUPRC vs controlled raw"] = screen.loc[mask, "AUPRC"] - raw["AUPRC"]
        screen.loc[mask, "Delta AUROC vs controlled raw"] = screen.loc[mask, "AUROC"] - raw["AUROC"]
    screen.sort_values(["Task", "AUPRC"], ascending=[True, False]).to_csv(
        result_dir / "feature_ablation_stage1.csv", index=False
    )

    confirm_frames = []
    conclusions = {}
    metrics = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Accuracy", "Threshold"]
    summary_rows = []
    for task in ("omi", "ami"):
        best_mode = selection["best_added_feature"][task]["feature_mode"]
        task_rows = all_rows[
            (all_rows["Task"] == task.upper())
            & (all_rows["Feature mode"].isin(["raw", best_mode]))
            & (all_rows["Seed"].isin(SEEDS))
        ].copy()
        for mode in ("raw", best_mode):
            got = sorted(task_rows[task_rows["Feature mode"] == mode]["Seed"].tolist())
            if got != sorted(SEEDS):
                raise SystemExit(f"{task} {mode} confirmation incomplete: {got}")
        confirm_frames.append(task_rows)
        for mode, part in task_rows.groupby("Feature mode", sort=False):
            out = {"Task": task.upper(), "Feature mode": mode}
            for metric in metrics:
                out[metric] = mean_std(part[metric])
                out[f"{metric}_mean"] = float(part[metric].mean())
                out[f"{metric}_std"] = float(part[metric].std(ddof=1))
            summary_rows.append(out)
        pivot = task_rows.pivot(index="Seed", columns="Feature mode", values="AUPRC")
        improvements = int((pivot[best_mode] > pivot["raw"]).sum())
        delta = float(pivot[best_mode].mean() - pivot["raw"].mean())
        conclusions[task] = {
            "best_mode": best_mode,
            "delta_mean_auprc": delta,
            "seeds_improved": improvements,
            "stable_gain": bool(delta > 0 and improvements >= 4),
        }

    confirm = pd.concat(confirm_frames, ignore_index=True)
    confirm.to_csv(result_dir / "feature_ablation_seeds.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(result_dir / "feature_ablation_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, task in zip(axes, ("OMI", "AMI")):
        part = screen[screen["Task"] == task].sort_values("AUPRC", ascending=False)
        ax.bar(part["Feature mode"], part["AUPRC"])
        ax.set(title=f"{task} controlled feature screen (seed 2026)", ylabel="AUPRC")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(result_dir / "feature_ablation_auprc.png", dpi=200)
    plt.close(fig)

    assoc_path = result_dir / "feature_association_summary.md"
    assoc_note = assoc_path.read_text(encoding="utf-8") if assoc_path.exists() else "Association report unavailable."
    lines = [
        "# ECG/age/sex/median-ECG feature ablation summary",
        "",
        "The controlled neural comparison uses the same XceptionTimeFusion architecture and initialization; only input modalities are masked. Results below use the validation threshold maximizing the F1/Sensitivity harmonic mean.",
        "",
        "## Five-seed confirmation",
        "",
        "| Task | Feature mode | AUROC | AUPRC | Sensitivity | Specificity | PPV | F1 | MCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['Task']} | {r['Feature mode']} | {r['AUROC']} | {r['AUPRC']} | {r['Sensitivity']} | "
            f"{r['Specificity']} | {r['PPV']} | {r['F1']} | {r['MCC']} |"
        )
    lines += ["", "## Added-value verdict", ""]
    for task, result in conclusions.items():
        verdict = "stable gain" if result["stable_gain"] else "no stable gain"
        lines.append(
            f"- {task.upper()}: best screened addition `{result['best_mode']}`; ΔAUPRC={result['delta_mean_auprc']:+.4f}; "
            f"improved in {result['seeds_improved']}/5 seeds — **{verdict}**."
        )
    lines += [
        "",
        "Standalone `median` and `clin` rows quantify predictive association, while raw-plus-feature rows test incremental value beyond raw ECG. External test feedback was not used.",
        "",
        "## Clinical association appendix",
        "",
        assoc_note,
    ]
    (result_dir / "feature_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(conclusions, indent=2))


if __name__ == "__main__":
    main()
