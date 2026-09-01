"""Build a safe aggregate Fold0 single-task-versus-joint comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    joint = read_json(ROOT / "results/adaptive_weighting_evaluation/w0/fold0/metrics.json")["outer_test"]
    acs_only = read_json(
        ROOT / "results/single_task_vs_joint_evaluation/acs_only_w0/fold0/metrics.json"
    )["outer_test"]
    omi_only = read_json(
        ROOT / "results/single_task_vs_joint_evaluation/omi_only_w0/fold0/metrics.json"
    )["outer_test"]
    joint_acs = joint["acs"]
    single_acs = acs_only["acs"]
    joint_omi = joint["omi"]["youden_j"]
    single_omi = omi_only["omi"]["youden_j"]

    rows = [
        {
            "task": "ACS", "method": "Joint W0",
            "primary_auprc": joint_acs["macro_auprc_ovr"],
            "auroc": joint_acs["macro_auroc_ovr"],
            "f1": joint_acs["macro_f1"],
            "balanced_accuracy": joint_acs["balanced_accuracy"],
            "sensitivity": None, "specificity": None,
            "threshold": "argmax",
            "nstemi_auprc": joint_acs["per_class"]["NSTEMI"]["auprc"],
            "nstemi_f1": joint_acs["per_class"]["NSTEMI"]["f1"],
            "nstemi_vs_ua_auprc": joint["nstemi_vs_ua"]["auprc"],
        },
        {
            "task": "ACS", "method": "ACS-only W0",
            "primary_auprc": single_acs["macro_auprc_ovr"],
            "auroc": single_acs["macro_auroc_ovr"],
            "f1": single_acs["macro_f1"],
            "balanced_accuracy": single_acs["balanced_accuracy"],
            "sensitivity": None, "specificity": None,
            "threshold": "argmax",
            "nstemi_auprc": single_acs["per_class"]["NSTEMI"]["auprc"],
            "nstemi_f1": single_acs["per_class"]["NSTEMI"]["f1"],
            "nstemi_vs_ua_auprc": acs_only["nstemi_vs_ua"]["auprc"],
        },
        {
            "task": "OMI", "method": "Joint W0",
            "primary_auprc": joint_omi["auprc"], "auroc": joint_omi["auroc"],
            "f1": joint_omi["f1"], "balanced_accuracy": joint_omi["balanced_accuracy"],
            "sensitivity": joint_omi["sensitivity"], "specificity": joint_omi["specificity"],
            "threshold": joint_omi["threshold"],
            "nstemi_auprc": None, "nstemi_f1": None, "nstemi_vs_ua_auprc": None,
        },
        {
            "task": "OMI", "method": "OMI-only W0",
            "primary_auprc": single_omi["auprc"], "auroc": single_omi["auroc"],
            "f1": single_omi["f1"], "balanced_accuracy": single_omi["balanced_accuracy"],
            "sensitivity": single_omi["sensitivity"], "specificity": single_omi["specificity"],
            "threshold": single_omi["threshold"],
            "nstemi_auprc": None, "nstemi_f1": None, "nstemi_vs_ua_auprc": None,
        },
    ]
    summary = {
        "scope": "nested Fold0 screening; not a five-fold generalization claim",
        "fairness": "same pretrained ECG-LFM, LoRA, split, seed, optimizer, scheduler, batch and preprocessing",
        "rows": rows,
        "joint_minus_single": {
            "acs_macro_auprc": joint_acs["macro_auprc_ovr"] - single_acs["macro_auprc_ovr"],
            "acs_nstemi_auprc": joint_acs["per_class"]["NSTEMI"]["auprc"] - single_acs["per_class"]["NSTEMI"]["auprc"],
            "omi_auprc": joint_omi["auprc"] - single_omi["auprc"],
        },
    }
    output = ROOT / "results/single_task_vs_joint_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "fold0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (output / "fold0_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
