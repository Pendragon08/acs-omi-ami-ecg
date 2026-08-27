"""Threshold-selection analysis for the final scheme (XceptionTime + bce + P3).

For each task (OMI / AMI), load the best checkpoint, run inference on the
validation split, and report Sens/Spec/PPV/NPV/Acc/F1/Youden at several
threshold-selection criteria:
  - argmax F1   (current default)
  - Youden index (max Sens + Spec - 1)
  - fixed 0.5 / 0.3 / 0.1

Also prints a coarse full sweep so the Sens/Spec/F1/Youden tradeoff is visible.
This is validation-only (threshold choice is frozen before official test).

Usage:
    python scripts/ecg/threshold_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from nets import build_model                                 # noqa: E402
from exp.exp_acs import make_loaders, predict                # noqa: E402

RES = ROOT / "results" / "final_fusion_focal"
TASKS = {"OMI": (0,), "AMI": (1,)}


def _load_model(task, device):
    tag = f"XceptionTime_{task}_bce_P3_seed2026"
    state = torch.load(RES / "checkpoints" / f"{tag}.pt", map_location=device)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = np.arange(0.05, 0.951, 0.01)

    for task, (col,) in TASKS.items():
        model = _load_model(task, device)
        _, val_loader = make_loaders("P3", 256, 2026, label_cols=(col,))
        y_val, p_val = predict(model, val_loader, device, use_fusion=False)
        y = y_val[:, 0].astype(int)
        p = p_val[:, 0]
        n_pos = int(y.sum())

        print(f"\n========== {task}  (val N={len(y)}, n_pos={n_pos}, "
              f"正类率={n_pos/len(y):.3f}) ==========")

        # argmax over criteria
        f1_best = max(((_metrics(y, p, t)["f1"], t) for t in grid), key=lambda x: x[0])
        youd_best = max(((_metrics(y, p, t)["youden"], t) for t in grid), key=lambda x: x[0])

        hdr = (f"{'准则':<20}{'阈值':>6}{'Sens':>8}{'Spec':>8}{'PPV':>8}{'NPV':>8}"
               f"{'Acc':>8}{'F1':>8}{'Youden':>8}{'TP/FP/FN':>12}")
        print(hdr)
        print("-" * len(hdr))
        rows = [
            ("argmax F1 (当前)", f1_best[1]),
            ("Youden 最优", youd_best[1]),
            ("固定 0.5", 0.5),
            ("固定 0.3", 0.3),
            ("固定 0.1", 0.1),
        ]
        for name, thr in rows:
            m = _metrics(y, p, thr)
            print(f"{name:<20}{thr:>6.2f}{m['sens']:>8.3f}{m['spec']:>8.3f}"
                  f"{m['ppv']:>8.3f}{m['npv']:>8.3f}{m['acc']:>8.3f}"
                  f"{m['f1']:>8.3f}{m['youden']:>8.3f}"
                  f"  {m['tp']}/{m['fp']}/{m['fn']}")

        # coarse full sweep (step 0.05) so the tradeoff curve is visible
        print(f"\n  {'阈值':>6}{'Sens':>8}{'Spec':>8}{'F1':>8}{'Youden':>8}")
        for t in np.arange(0.05, 0.951, 0.05):
            m = _metrics(y, p, float(t))
            print(f"  {t:>6.2f}{m['sens']:>8.3f}{m['spec']:>8.3f}"
                  f"{m['f1']:>8.3f}{m['youden']:>8.3f}")


if __name__ == "__main__":
    main()
