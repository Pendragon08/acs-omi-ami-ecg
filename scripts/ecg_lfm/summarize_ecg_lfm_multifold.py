"""Summarize completed validation folds for every registered tuning run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_root", type=Path, default=Path("results/ecg_lfm_4label_tuning"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/ecg_lfm_4label_tuning/summary"))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in args.tuning_root.iterdir() if path.is_dir() and path.name != "summary"):
        fold_values: list[tuple[int, dict]] = []
        loss = None
        for fold in range(5):
            evaluation = run_dir / f"fold{fold}" / "evaluation" / "fold_evaluation.json"
            provenance = run_dir / f"fold{fold}" / "launch_provenance.json"
            if not evaluation.exists() or not provenance.exists():
                continue
            ev = json.loads(evaluation.read_text(encoding="utf-8"))
            prov = json.loads(provenance.read_text(encoding="utf-8"))
            if prov.get("returncode") != 0:
                continue
            loss = prov.get("loss")
            fold_values.append((fold, ev["youden"]["macro"]))
        if not fold_values:
            continue
        row: dict[str, object] = {
            "run_name": run_dir.name,
            "loss": loss,
            "n_completed_folds": len(fold_values),
            "folds": ",".join(str(fold) for fold, _ in fold_values),
        }
        for metric in METRICS:
            values = np.asarray([float(table[metric]) for _, table in fold_values])
            row[f"macro_{metric}_mean"] = float(values.mean())
            row[f"macro_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    if not rows:
        raise RuntimeError("No completed tuning-fold evaluations found")
    rows.sort(key=lambda item: (-int(item["n_completed_folds"]), -float(item["macro_auprc_mean"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_name", "loss", "n_completed_folds", "folds"]
    fields += [f"macro_{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with (args.output_dir / "multifold_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ECG-LFM completed-fold comparison",
        "",
        "> Hyperparameters are selected from internal patient-level validation folds only.",
        "",
        "| Run | Loss | Folds | Macro AUROC | Macro AUPRC | Sensitivity | Specificity | PPV | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def f(metric: str) -> str:
            return f"{row[f'macro_{metric}_mean']:.4f} ± {row[f'macro_{metric}_std']:.4f}"
        lines.append(
            f"| {row['run_name']} | {row['loss']} | {row['folds']} | {f('auroc')} | {f('auprc')} | "
            f"{f('sensitivity')} | {f('specificity')} | {f('ppv')} | {f('f1')} |"
        )
    (args.output_dir / "multifold_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(rows), "ranking": [row["run_name"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
