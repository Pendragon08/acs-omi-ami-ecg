"""Apply the preregistered Fold-0 and three-fold ECG-LFM selection rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")


def fold_result(run_name: str, fold_dir: Path, fold: int) -> dict | None:
    evaluation = fold_dir / "evaluation" / "fold_evaluation.json"
    provenance = fold_dir / "launch_provenance.json"
    if not evaluation.exists() or not provenance.exists():
        return None
    ev = json.loads(evaluation.read_text(encoding="utf-8"))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    if prov.get("returncode") != 0:
        return None
    gate_path = fold_dir / "fold0_gate.json"
    if fold == 0 and gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("passed"):
            return None
    return {
        "run_name": run_name,
        "fold": fold,
        "loss": prov.get("loss", "unweighted_bce"),
        "macro": ev["youden"]["macro"],
        "per_label": {label: ev["youden"][label] for label in LABELS},
        "thresholds": ev["thresholds"],
        "command": prov.get("command", []),
    }


def rank_key(row: dict) -> tuple[float, ...]:
    return (
        float(row["macro_auprc_mean"]),
        float(row["minority_auprc_mean"]),
        float(row["macro_auroc_mean"]),
        float(row["macro_f1_mean"]),
        float(row["macro_specificity_mean"]),
        float(row["macro_ppv_mean"]),
    )


def summarize(run_name: str, folds: list[dict]) -> dict:
    result: dict[str, object] = {
        "run_name": run_name,
        "loss": folds[0]["loss"],
        "folds": [item["fold"] for item in folds],
        "commands": [item["command"] for item in folds],
    }
    for metric in METRICS:
        values = np.asarray([float(item["macro"][metric]) for item in folds])
        result[f"macro_{metric}_mean"] = float(values.mean())
        result[f"macro_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    for label in LABELS:
        values = np.asarray([float(item["per_label"][label]["auprc"]) for item in folds])
        result[f"{label.lower()}_auprc_mean"] = float(values.mean())
        result[f"{label.lower()}_auprc_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result["minority_auprc_mean"] = float(
        (result["nstemi_auprc_mean"] + result["omi_auprc_mean"]) / 2
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fold0", "threefold"), required=True)
    parser.add_argument("--tuning_root", type=Path, default=Path("results/ecg_lfm_4label_tuning"))
    parser.add_argument("--baseline_root", type=Path, default=Path("results/ecg_lfm_4label/finetune_paper"))
    parser.add_argument("--stage1_selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "fold0":
        summaries = []
        for run_dir in sorted(path for path in args.tuning_root.iterdir() if path.is_dir() and path.name != "summary"):
            item = fold_result(run_dir.name, run_dir / "fold0", 0)
            if item is not None:
                summaries.append(summarize(run_dir.name, [item]))
        selected = {}
        for loss in ("unweighted_bce", "weighted_bce"):
            group = sorted((item for item in summaries if item["loss"] == loss), key=rank_key, reverse=True)
            if not group:
                raise RuntimeError(f"No completed, passing Fold-0 candidate for {loss}")
            selected[loss] = group[0]["run_name"]
        payload = {
            "stage": "Fold-0 screening",
            "selection_data": "internal patient-level Fold 0 only",
            "ranking_rule": "macro AUPRC, minority NSTEMI/OMI AUPRC, macro AUROC, F1, specificity, PPV",
            "selected": selected,
            "ranking": sorted(summaries, key=rank_key, reverse=True),
        }
    else:
        if args.stage1_selection is None:
            parser.error("--stage1_selection is required for threefold mode")
        stage1 = json.loads(args.stage1_selection.read_text(encoding="utf-8"))
        candidates = ["finetune_paper", *stage1["selected"].values()]
        summaries = []
        for run_name in dict.fromkeys(candidates):
            root = args.baseline_root if run_name == "finetune_paper" else args.tuning_root / run_name
            folds = []
            for fold in (0, 1, 2):
                item = fold_result(run_name, root / f"fold{fold}", fold)
                if item is None:
                    raise RuntimeError(f"Missing completed evaluation: {run_name} fold {fold}")
                folds.append(item)
            summaries.append(summarize(run_name, folds))
        ranking = sorted(summaries, key=rank_key, reverse=True)
        payload = {
            "stage": "three-fold confirmation",
            "selection_data": "internal patient-level folds 0,1,2 only",
            "ranking_rule": "mean macro AUPRC, minority NSTEMI/OMI AUPRC, macro AUROC, F1, specificity, PPV",
            "selected_final": ranking[0]["run_name"],
            "ranking": ranking,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
