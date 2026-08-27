"""Aggregate the 2x2 final comparison (focal vs bce, fusion vs baseline) into a
markdown table, one row per (model, loss, task) run.

Reads results/final_fusion_focal/*.json and prints:
  (1) a full metric table (AUROC/AUPRC/F1/Sens/Spec/PPV/NPV/Acc/thr/epoch/params)
  (2) a compact winner summary comparing bce->focal and raw->fusion per task.

Usage:
    python scripts/ecg/report_final_comparison.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results" / "final_fusion_focal"

MODELS = ["XceptionTime", "XceptionTimeFusion"]
LOSSES = ["bce", "focal"]
TASKS = ["OMI", "AMI"]
METRICS = ["auroc", "auprc", "f1", "sens", "spec", "ppv", "npv", "acc"]


def load(model, loss, task):
    tag = f"{model}_{task}_{loss}_P3_seed2026"
    p = RES / f"{tag}.json"
    if not p.exists():
        return None, tag
    r = json.loads(p.read_text(encoding="utf-8"))
    return r, tag


def main():
    rows = []
    for model in MODELS:
        for loss in LOSSES:
            for task in TASKS:
                r, tag = load(model, loss, task)
                if r is None:
                    rows.append((model, loss, task, None, tag))
                    continue
                rows.append((model, loss, task, r, tag))

    # ---- full table ---------------------------------------------------------
    header = ("| Model | Loss | Task | AUROC | AUPRC | F1 | Sens | Spec | PPV | "
              "NPV | Acc | thr | ep | #param |")
    sep = "|---" * 14 + "|"
    print(header)
    print(sep)
    for model, loss, task, r, tag in rows:
        if r is None:
            print(f"| {model} | {loss} | {task} | MISSING ({tag}) |")
            continue
        v = r["val"]
        m = {k: v.get(f"{task}_{k}", float("nan")) for k in METRICS}
        thr = v.get(f"{task}_threshold", float("nan"))
        print(f"| {model} | {loss} | {task} | {m['auroc']:.4f} | {m['auprc']:.4f} | "
              f"{m['f1']:.4f} | {m['sens']:.4f} | {m['spec']:.4f} | {m['ppv']:.4f} | "
              f"{m['npv']:.4f} | {m['acc']:.4f} | {thr:.3f} | {r['best_epoch']} | "
              f"{r['n_params']} |")
    print()

    # ---- winner summary (AUPRC primary per protocol, F1 shown) ---------------
    print("## 对比结论（AUPRC 为主，F1 参考）")
    for task in TASKS:
        print(f"\n### {task}")
        base = {loss: load("XceptionTime", loss, task)[0] for loss in LOSSES}
        fus = {loss: load("XceptionTimeFusion", loss, task)[0] for loss in LOSSES}
        for loss in LOSSES:
            if base[loss] and fus[loss]:
                d_auprc = fus[loss]["val"][f"{task}_auprc"] - base[loss]["val"][f"{task}_auprc"]
                d_f1 = fus[loss]["val"][f"{task}_f1"] - base[loss]["val"][f"{task}_f1"]
                print(f"  {loss}: fusion vs raw -> AUPRC {d_auprc:+.4f}, F1 {d_f1:+.4f}")
        for model in MODELS:
            b = load(model, "bce", task)[0]
            f = load(model, "focal", task)[0]
            if b and f:
                d_auprc = f["val"][f"{task}_auprc"] - b["val"][f"{task}_auprc"]
                d_f1 = f["val"][f"{task}_f1"] - b["val"][f"{task}_f1"]
                print(f"  {model}: focal vs bce -> AUPRC {d_auprc:+.4f}, F1 {d_f1:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
