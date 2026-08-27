"""Re-threshold the frozen three-label E4 ensemble without retraining.

The frozen E4 checkpoints are evaluated on their original patient-level five
fold validation partitions.  Thresholds are selected from aggregated OOF
validation probabilities by Youden's J (sensitivity + specificity - 1), then
applied unchanged to the hidden test inputs to create a new submission CSV.
No hidden test labels are read.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import hmoe_kan_github as h  # noqa: E402

h.LABELS = ("STEMI", "NSTEMI", "UA")


def recover_array(path, scalers):
    p = np.load(path, mmap_mode="r")
    s = scalers["P2"]
    return p.astype(np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])


def predict(model, x, y, device, batch_size=256):
    ds = h.ECGDataset(x, y, augment_flag=False, seed=2026)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                    pin_memory=True)
    rows = []
    with torch.no_grad():
        for xb, _ in dl:
            o = model(xb.to(device, non_blocking=True))
            rows.append(torch.sigmoid(o["logits"]).cpu().numpy())
    return np.concatenate(rows, axis=0)


def youden_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y.astype(int), p.astype(float))
    j = tpr - fpr
    t = float(thresholds[int(np.nanargmax(j))])
    return float(np.clip(t, 0.01, 0.99))


def metric(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(thr), "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def main():
    out = ROOT / "results" / "hmoe_kan_stemi_nstemi_ua_e4_youden"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arr = ROOT / "data" / "arrays"
    scalers = json.loads((arr / "scalers.json").read_text())
    meta, x_raw, labels = h._load_raw_ecg_and_labels(h.LABELS)
    x_test = recover_array(arr / "P2_test.npy", scalers)
    meta_test = np.load(arr / "meta_test.npz", allow_pickle=True)
    folds, joint_counts = h.build_joint_patient_folds(
        meta["patient_ids"], labels, n_folds=5, seed=2026)
    oof = np.zeros_like(labels, dtype=np.float32)
    test_fold = []
    for fi, (tr, va) in enumerate(folds):
        scaler = h.fit_robust_scaler(x_raw[tr])
        x_va = h.apply_robust_scaler(x_raw[va], scaler)
        x_te = h.apply_robust_scaler(x_test, scaler)
        model = h.HMoEKAN("e4", n_tasks=3, head_kind="kan", device=str(device)).to(device)
        state = torch.load(ROOT / "results" / "hmoe_kan_stemi_nstemi_ua_e4" /
                           f"e4_fold{fi}_seed2026.pt", map_location=device)
        model.load_state_dict(state); model.eval()
        oof[va] = predict(model, x_va, labels[va], device)
        test_fold.append(predict(model, x_te, np.zeros((len(x_te), 3), dtype=np.float32), device))
        print(f"FOLD {fi} DONE", flush=True)
    test_prob = np.mean(np.stack(test_fold, axis=0), axis=0)
    thresholds = [youden_threshold(labels[:, j], oof[:, j]) for j in range(3)]
    metrics = {name: metric(labels[:, j], oof[:, j], thresholds[j])
               for j, name in enumerate(h.LABELS)}
    np.savez_compressed(out / "e4_oof_test_probabilities.npz",
                        y=labels, oof_prob=oof, test_prob=test_prob)
    (out / "thresholds_youden.json").write_text(json.dumps({"strategy": "Youden J / balanced accuracy",
        "thresholds": dict(zip(h.LABELS, thresholds)), "metrics_oof": metrics,
        "fold_seed": 2026, "note": "validation OOF only; hidden test labels not used"}, indent=2))
    with (out / "oof_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["row_index", *[f"y_{n}" for n in h.LABELS], *[f"p_{n}" for n in h.LABELS]])
        for i in range(len(labels)): w.writerow([i, *labels[i].astype(int).tolist(), *oof[i].tolist()])
    test_ids = {}
    for i, rid in enumerate(meta_test["record_ids"]):
        test_ids[str(rid)] = i
    ordered = []
    with (ROOT / "data" / "CSV" / "test.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            ordered.append(row["ecg_row_record"].split(".")[0])
    order_idx = [test_ids[x] for x in ordered]
    test_order = test_prob[order_idx]
    submission = out / "submission_stemi_nstemi_ua_hmoe_e4_youden.csv"
    with submission.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["ecg_row_record", *h.LABELS])
        for rid, p in zip([x + ".dat" for x in ordered], test_order):
            w.writerow([rid, *[(p[j] >= thresholds[j]).astype(int) for j in range(3)]])
    print(json.dumps({"thresholds": dict(zip(h.LABELS, thresholds)), "metrics_oof": metrics,
                      "test_positive_counts": dict(zip(h.LABELS, (test_order >= np.asarray(thresholds)).sum(axis=0).tolist())),
                      "submission": str(submission)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
