"""Official-test inference for the GPU1 E4 three-label HMoE model.

The five fold checkpoints are ensembled.  Each fold uses its own train-only
robust scaler; thresholds are selected once from out-of-fold validation
probabilities and then fixed for the external test prediction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.hmoe_kan_github import (  # noqa: E402
    HMoEKAN,
    _load_raw_ecg_and_labels,
    choose_thresholds,
)
from preprocessing.robust_scaler import apply_robust_scaler, fit_robust_scaler  # noqa: E402


LABELS = ("STEMI", "NSTEMI", "UA")


def _csv_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [(row.get("ecg_row_record") or "").strip()
                for row in csv.DictReader(f)]


def _stem(value):
    return Path(str(value)).stem


def _raw_from_p2(path, scaler_path):
    p2 = np.load(path, mmap_mode="r")
    scalers = json.loads(scaler_path.read_text(encoding="utf-8"))
    s = scalers["P2"]
    return p2.astype(np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])


def _predict(model, x, device, batch_size):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(
                device, non_blocking=True)
            out.append(torch.sigmoid(model(xb)["logits"]).cpu().numpy())
    return np.concatenate(out, axis=0)


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/hmoe_kan_stemi_nstemi_ua_e4")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fold_seed", type=int, default=2026)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / args.run_dir
    split = json.loads((run_dir / "e4_split.json").read_text(encoding="utf-8"))
    folds = [(np.asarray(f["train"], dtype=np.int64),
              np.asarray(f["val"], dtype=np.int64)) for f in split["folds"]]

    meta, x_raw_train, labels = _load_raw_ecg_and_labels(LABELS)
    test_meta = np.load(ROOT / "data" / "arrays" / "meta_test.npz", allow_pickle=True)
    x_raw_test = _raw_from_p2(ROOT / "data" / "arrays" / "P2_test.npy",
                              ROOT / "data" / "arrays" / "scalers.json")

    oof_prob = np.zeros_like(labels, dtype=np.float32)
    test_prob = []
    fold_thresholds = []
    fold_param_counts = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        scaler = fit_robust_scaler(x_raw_train[train_idx])
        x_val = apply_robust_scaler(x_raw_train[val_idx], scaler)
        x_test = apply_robust_scaler(x_raw_test, scaler)
        model = HMoEKAN("e4", n_tasks=len(LABELS), head_kind="kan",
                        device=str(device)).to(device)
        checkpoint = run_dir / f"e4_fold{fold_idx}_seed{args.seed}.pt"
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        p_val = _predict(model, x_val, device, args.batch_size)
        p_test = _predict(model, x_test, device, args.batch_size)
        oof_prob[val_idx] = p_val
        test_prob.append(p_test)
        fold_thresholds.append(choose_thresholds(labels[val_idx], p_val).tolist())
        fold_param_counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(json.dumps({"fold": fold_idx, "val_rows": len(val_idx),
                          "thresholds": fold_thresholds[-1]}, ensure_ascii=False), flush=True)

    thresholds = choose_thresholds(labels, oof_prob)
    probs_meta = np.mean(np.stack(test_prob, axis=0), axis=0)
    preds_meta = (probs_meta >= thresholds[None, :]).astype(np.int64)

    ids_meta_raw = [str(x) for x in test_meta["record_ids"]]
    ids_meta = [_stem(x) for x in ids_meta_raw]
    ids_out_raw = _csv_ids(ROOT / "data" / "CSV" / "test.csv")
    ids_out = [_stem(x) for x in ids_out_raw]
    index = {rid: i for i, rid in enumerate(ids_meta)}
    if len(ids_out) != len(ids_meta) or set(ids_out) != set(ids_meta):
        raise ValueError("test.csv identifiers do not match meta_test identifiers")
    order = np.asarray([index[rid] for rid in ids_out], dtype=np.int64)
    probs = probs_meta[order]
    preds = preds_meta[order]

    internal = run_dir / "official_test_hmoe_e4_stemi_nstemi_ua.csv"
    with internal.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecg_row_record"] + [v for label in LABELS
                                          for v in (f"{label}_probability", f"{label}_prediction")])
        for raw_id, pv, pd in zip(ids_out_raw, probs, preds):
            row = [raw_id]
            for j in range(len(LABELS)):
                row.extend([f"{pv[j]:.8f}", int(pd[j])])
            w.writerow(row)

    for j, label in enumerate(LABELS):
        path = run_dir / f"submission_{label.lower()}_hmoe_e4.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ecg_row_record", label])
            for raw_id, pd in zip(ids_out_raw, preds[:, j]):
                w.writerow([raw_id, int(pd)])

    combined = run_dir / "submission_stemi_nstemi_ua_hmoe_e4.csv"
    with combined.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecg_row_record", *LABELS])
        for raw_id, row in zip(ids_out_raw, preds):
            w.writerow([raw_id, *[int(v) for v in row]])

    payload = {
        "task": "stemi_nstemi_ua",
        "variant": "e4",
        "model": "HMoE_WavelNet_KAN",
        "labels": list(LABELS),
        "seed": args.seed,
        "fold_seed": args.fold_seed,
        "threshold_method": "OOF validation F1 maximization",
        "thresholds": thresholds.tolist(),
        "fold_thresholds": fold_thresholds,
        "n_test": int(len(preds)),
        "n_positive_predictions": {label: int(preds[:, j].sum())
                                    for j, label in enumerate(LABELS)},
        "n_multi_positive": int((preds.sum(axis=1) >= 2).sum()),
        "n_none_positive": int((preds.sum(axis=1) == 0).sum()),
        "submission_sha256": _sha256(combined),
        "checkpoint_param_counts": fold_param_counts,
        "checkpoint_dir": str(run_dir),
    }
    (run_dir / "official_test_hmoe_e4_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
