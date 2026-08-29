"""Build traceable Fold-0 hyperparameter/loss comparison tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")


def load_run(run_dir: Path, run_name: str) -> tuple[dict, list[dict]] | None:
    evaluation_path = run_dir / "evaluation" / "fold_evaluation.json"
    provenance_path = run_dir / "launch_provenance.json"
    audit_path = run_dir / "curves" / "overfitting_audit.json"
    if not evaluation_path.exists() or not provenance_path.exists():
        return None
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    table = evaluation["youden"]
    summary = {
        "run_name": run_name,
        "loss": provenance.get("loss", "unweighted_bce"),
        "fold": evaluation["fold"],
        **{f"macro_{metric}": table["macro"][metric] for metric in METRICS},
        "best_validation_loss_epoch": audit.get("best_validation_loss_epoch"),
        "best_validation_loss": audit.get("best_validation_loss"),
        "possible_overfit": audit.get("possible_overfit"),
        "command": " ".join(str(item) for item in provenance.get("command", [])),
    }
    long_rows = []
    for label in LABELS:
        long_rows.append(
            {
                "run_name": run_name,
                "loss": summary["loss"],
                "label": label,
                **{metric: table[label][metric] for metric in METRICS},
                **{key: table[label][key] for key in ("threshold", "tp", "tn", "fp", "fn")},
            }
        )
    return summary, long_rows


def fmt(value: object) -> str:
    return "Pending" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_root", type=Path, default=Path("results/ecg_lfm_4label_tuning"))
    parser.add_argument("--baseline_dir", type=Path, default=Path("results/ecg_lfm_4label/finetune_paper/fold0"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/ecg_lfm_4label_tuning/summary"))
    args = parser.parse_args()

    summaries: list[dict] = []
    long_rows: list[dict] = []
    baseline = load_run(args.baseline_dir, "finetune_paper")
    if baseline is not None:
        summaries.append(baseline[0])
        long_rows.extend(baseline[1])
    if args.tuning_root.exists():
        for run_dir in sorted(path for path in args.tuning_root.iterdir() if path.is_dir() and path.name != "summary"):
            loaded = load_run(run_dir / "fold0", run_dir.name)
            if loaded is not None:
                summaries.append(loaded[0])
                long_rows.extend(loaded[1])
    if not summaries:
        raise RuntimeError("No completed Fold-0 evaluation artifacts found")
    baseline_auprc = next(
        (float(row["macro_auprc"]) for row in summaries if row["run_name"] == "finetune_paper"),
        None,
    )
    for row in summaries:
        row["delta_macro_auprc_vs_baseline"] = (
            None if baseline_auprc is None else float(row["macro_auprc"]) - baseline_auprc
        )
    summaries.sort(key=lambda row: float(row["macro_auprc"]), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "rank", "run_name", "loss", "fold", *[f"macro_{metric}" for metric in METRICS],
        "delta_macro_auprc_vs_baseline", "best_validation_loss_epoch", "best_validation_loss",
        "possible_overfit", "command",
    ]
    with (args.output_dir / "fold0_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for rank, row in enumerate(summaries, start=1):
            writer.writerow({"rank": rank, **row})
    long_fields = ["run_name", "loss", "label", *METRICS, "threshold", "tp", "tn", "fp", "fn"]
    with (args.output_dir / "fold0_per_label_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)

    lines = [
        "# ECG-LFM Fold 0 hyperparameter/loss screening",
        "",
        "> Screening only. External test labels were not used. Promotion requires multi-fold confirmation.",
        "",
        "| Rank | Run | Loss | Macro AUROC | Macro AUPRC | ΔAUPRC vs baseline | Sensitivity | Specificity | PPV | F1 | Best epoch | Overfit flag |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(summaries, start=1):
        lines.append(
            f"| {rank} | {row['run_name']} | {row['loss']} | {fmt(row['macro_auroc'])} | "
            f"{fmt(row['macro_auprc'])} | {fmt(row['delta_macro_auprc_vs_baseline'])} | "
            f"{fmt(row['macro_sensitivity'])} | {fmt(row['macro_specificity'])} | "
            f"{fmt(row['macro_ppv'])} | {fmt(row['macro_f1'])} | "
            f"{row.get('best_validation_loss_epoch', 'Pending')} | {row.get('possible_overfit', 'Pending')} |"
        )
    lines.extend(["", "## Per-label AUPRC", "", "| Run | STEMI | NSTEMI | UA | OMI |", "|---|---:|---:|---:|---:|"])
    for summary in summaries:
        rows = {row["label"]: row for row in long_rows if row["run_name"] == summary["run_name"]}
        lines.append(
            f"| {summary['run_name']} | " + " | ".join(fmt(rows[label]["auprc"]) for label in LABELS) + " |"
        )
    (args.output_dir / "fold0_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completed_runs": len(summaries), "ranking": [row["run_name"] for row in summaries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
