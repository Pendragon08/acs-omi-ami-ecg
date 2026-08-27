"""Summarize validation-only stage-1 grids and select confirmation configs."""
import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_results(result_dir):
    results = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if item.get("study") not in {"loss", "feature"}:
            continue
        item["_path"] = str(path.relative_to(ROOT))
        results.append(item)
    return results


def row_from_result(item):
    op = item["val"]["operating_points"]["max_f1_sensitivity_hmean_0.01_0.99"]
    legacy = item["val"]["operating_points"]["legacy_max_f1_0.05_0.94"]
    cfg = item.get("loss_config", {})
    return {
        "study": item["study"],
        "task": item["task"],
        "seed": item["seed"],
        "model": item["model"],
        "feature_mode": item.get("feature_mode", "raw"),
        "loss": item["loss"],
        "alpha": cfg.get("alpha"),
        "beta": cfg.get("beta"),
        "gamma": cfg.get("gamma"),
        "auroc": item["val"]["auroc"],
        "auprc": item["val"]["auprc"],
        "threshold": op["threshold"],
        "sensitivity": op["sensitivity"],
        "specificity": op["specificity"],
        "ppv": op["ppv"],
        "npv": op["npv"],
        "f1": op["f1"],
        "mcc": op["mcc"],
        "accuracy": op["accuracy"],
        "tp": op["tp"],
        "tn": op["tn"],
        "fp": op["fp"],
        "fn": op["fn"],
        "f1_sensitivity_hmean": op["f1_sensitivity_hmean"],
        "legacy_threshold": legacy["threshold"],
        "legacy_f1": legacy["f1"],
        "path": item["_path"],
    }


def select_best(rows):
    return max(
        rows,
        key=lambda row: (
            row["auprc"],
            row["f1_sensitivity_hmean"],
            row["f1"],
            row["sensitivity"],
            row["specificity"],
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="loss_feature_ablation")
    p.add_argument("--require_complete", action="store_true")
    args = p.parse_args()
    result_dir = ROOT / "results" / args.exp_name
    items = load_results(result_dir)
    rows = [row_from_result(item) for item in items]

    loss_rows = [row for row in rows if row["study"] == "loss" and row["seed"] == 2026]
    feature_rows = [row for row in rows if row["study"] == "feature" and row["seed"] == 2026]
    focal_rows = [row for row in loss_rows if row["loss"] == "focal"]
    cb_rows = [row for row in loss_rows if row["loss"] == "cb_focal"]
    if args.require_complete:
        expected = {"focal": 4, "cb_focal": 9, "feature": 16}
        actual = {"focal": len(focal_rows), "cb_focal": len(cb_rows), "feature": len(feature_rows)}
        if actual != expected:
            raise SystemExit(f"incomplete stage 1: expected={expected}, actual={actual}")

    selection = {
        "selection_set": "validation_only",
        "selection_rule": "lexicographic_AUPRC_then_F1-Sensitivity-hmean_then_F1_then_Sensitivity_then_Specificity",
        "best_focal": select_best(focal_rows) if focal_rows else None,
        "best_cb_focal": select_best(cb_rows) if cb_rows else None,
        "best_added_feature": {},
        "counts": {"loss_rows": len(loss_rows), "feature_rows": len(feature_rows)},
    }
    added_modes = {"raw_age", "raw_sex", "raw_clin", "raw_median", "all"}
    for task in ("omi", "ami"):
        candidates = [row for row in feature_rows if row["task"] == task and row["feature_mode"] in added_modes]
        raw = next((row for row in feature_rows if row["task"] == task and row["feature_mode"] == "raw"), None)
        best = select_best(candidates) if candidates else None
        if best and raw:
            best = dict(best)
            best["delta_auprc_vs_controlled_raw"] = best["auprc"] - raw["auprc"]
            best["delta_auroc_vs_controlled_raw"] = best["auroc"] - raw["auroc"]
            best["controlled_raw"] = raw
        selection["best_added_feature"][task] = best

    loss_frame = pd.DataFrame(loss_rows)
    if not loss_frame.empty:
        loss_frame = loss_frame.sort_values(["loss", "auprc"], ascending=[True, False])
    loss_frame.to_csv(result_dir / "stage1_loss_grid.csv", index=False)
    feature_frame = pd.DataFrame(feature_rows)
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values(["task", "auprc"], ascending=[True, False])
    feature_frame.to_csv(result_dir / "stage1_feature_grid.csv", index=False)
    (result_dir / "stage1_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    lines = [
        "# Stage-1 validation selection",
        "",
        f"Selection rule: {selection['selection_rule']}. External test labels/results were not used.",
        "",
    ]
    for key in ("best_focal", "best_cb_focal"):
        row = selection[key]
        if row:
            lines += [
                f"## {key}",
                "",
                f"- alpha={row['alpha']}, beta={row['beta']}, gamma={row['gamma']}",
                f"- AUROC={row['auroc']:.6f}, AUPRC={row['auprc']:.6f}",
                f"- harmonic threshold={row['threshold']:.2f}, Sens={row['sensitivity']:.6f}, Spec={row['specificity']:.6f}, F1={row['f1']:.6f}",
                "",
            ]
    for task, row in selection["best_added_feature"].items():
        if row:
            lines += [
                f"## Best added feature: {task.upper()}",
                "",
                f"- mode={row['feature_mode']}",
                f"- AUPRC={row['auprc']:.6f}; delta vs controlled raw={row['delta_auprc_vs_controlled_raw']:+.6f}",
                f"- AUROC={row['auroc']:.6f}; delta vs controlled raw={row['delta_auroc_vs_controlled_raw']:+.6f}",
                "",
            ]
    (result_dir / "stage1_selection.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
