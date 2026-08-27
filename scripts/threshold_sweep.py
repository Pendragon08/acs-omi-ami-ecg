"""Fine threshold sweep (0.01-0.99) for the final scheme (XceptionTime+bce+P3).

For each task (OMI / AMI): load the best checkpoint, run inference on the
validation split ONCE, then sweep the decision threshold from 0.01 to 0.99
(step 0.01).  Report, for every point metric (Sens/Spec/PPV/NPV/Acc/F1/Youden),
the threshold that maximises it and the full tradeoff at that point.

AUROC / AUPRC are threshold-independent ranking metrics and are reported once
per task (they never change with the threshold).

Saves the full per-threshold table to
    results/final_fusion_focal/threshold_sweep_{task}.csv

Usage:
    python scripts/ecg/threshold_sweep.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from nets import build_model                                 # noqa: E402
from exp.exp_acs import make_loaders, predict                # noqa: E402

RES = ROOT / "results" / "final_fusion_focal"
TASKS = {"OMI": (0,), "AMI": (1,)}


def _load_model(task, device):
    tag = f"XceptionTime_{task}_bce_P3_seed2026"
    state = torch.load(
        RES / "checkpoints" / f"{tag}.pt", map_location=device, weights_only=True
    )
    model = build_model("XceptionTime", num_class=1, device=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _metrics(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    youden = sens + spec - 1.0
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "acc": acc,
            "f1": f1, "youden": youden, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _auc(y_true, y_prob):
    """Return standard AUROC and average precision.

    The previous hand-written AUROC rank calculation indexed a permutation as
    though it were inverse ranks, which produced values near 0.5 for the final
    models despite their true AUROC being about 0.88.
    """
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, y_prob)), float(average_precision_score(y, y_prob))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = np.arange(0.01, 1.00, 0.01)   # 99 points: 0.01 .. 0.99

    for task, (col,) in TASKS.items():
        model = _load_model(task, device)
        _, val_loader = make_loaders("P3", 256, 2026, label_cols=(col,))
        y_val, p_val = predict(model, val_loader, device, use_fusion=False)
        y = y_val[:, 0].astype(int)
        p = p_val[:, 0]
        n_pos = int(y.sum())
        auroc, auprc = _auc(y, p)

        # full sweep
        rows = []
        for t in grid:
            m = _metrics(y, p, float(t))
            m["thr"] = round(float(t), 2)
            rows.append(m)

        # save full table
        csv_path = RES / f"threshold_sweep_{task}.csv"
        keys = ["thr", "sens", "spec", "ppv", "npv", "acc", "f1", "youden",
                "tp", "fp", "fn", "tn"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for m in rows:
                w.writerow({k: m[k] for k in keys})

        print(f"\n========== {task}  (val N={len(y)}, n_pos={n_pos}, "
              f"正类率={n_pos/len(y):.3f}) ==========")
        print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"(阈值无关，不随阈值变化)")

        # argmax per metric over the 0.01..0.99 sweep
        argmaxes = {}
        for metric in ("sens", "spec", "ppv", "npv", "acc", "f1", "youden"):
            best = max(rows, key=lambda m: m[metric])
            argmaxes[metric] = best

        print(f"\n各指标在 0.01–0.99 全扫描下的最优阈值：")
        print(f"{'指标':<8}{'最优阈值':>8}{'该指标值':>10}"
              f"{'Sens':>8}{'Spec':>8}{'PPV':>8}{'Acc':>8}{'F1':>8}{'Youden':>8}"
              f"{'TP/FP/FN':>12}")
        for metric in ("f1", "youden", "sens", "spec", "ppv", "npv", "acc"):
            b = argmaxes[metric]
            print(f"{metric:<8}{b['thr']:>8.2f}{b[metric]:>10.3f}"
                  f"{b['sens']:>8.3f}{b['spec']:>8.3f}{b['ppv']:>8.3f}"
                  f"{b['acc']:>8.3f}{b['f1']:>8.3f}{b['youden']:>8.3f}"
                  f"  {b['tp']}/{b['fp']}/{b['fn']}")

        # readable fine table (step 0.02) so the tradeoff curve is visible
        print(f"\n  {'阈值':>6}{'Sens':>8}{'Spec':>8}{'PPV':>8}{'NPV':>8}"
              f"{'Acc':>8}{'F1':>8}{'Youden':>8}")
        for m in rows:
            if round(m['thr'] * 100) % 2 == 0:
                print(f"  {m['thr']:>6.2f}{m['sens']:>8.3f}{m['spec']:>8.3f}"
                      f"{m['ppv']:>8.3f}{m['npv']:>8.3f}{m['acc']:>8.3f}"
                      f"{m['f1']:>8.3f}{m['youden']:>8.3f}")

        print(f"\n[SAVED] {csv_path}")

    print("\nALL_DONE_SWEEP")


if __name__ == "__main__":
    main()
