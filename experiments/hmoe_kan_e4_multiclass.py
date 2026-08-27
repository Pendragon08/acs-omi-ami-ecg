"""Four-class softmax E4 experiment for the mutually exclusive ACS labels.

The historical E4 run has three independent sigmoid heads.  The training CSV,
however, contains only the mutually exclusive states 000/001/010/100.  This
script keeps the E4 experts, patient-level split, scaler, augmentation,
optimizer and stopping protocol, but changes the task head/loss to four-class
softmax: OTHER, STEMI, NSTEMI, UA.  The historical three-label artifacts are
not overwritten.

The implementation reuses the existing GitHub-sourced E4 components through
``experiments.hmoe_kan_github``.  It writes fold checkpoints, loss histories,
OOF probabilities, per-class one-vs-rest metrics, confusion matrices, and a
three-column one-hot submission for compatibility with the existing platform.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import hmoe_kan_github as h  # noqa: E402


CLASS_NAMES = ("OTHER", "STEMI", "NSTEMI", "UA")
POSITIVE_NAMES = ("STEMI", "NSTEMI", "UA")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MulticlassECGDataset(Dataset):
    def __init__(self, x, y, augment_flag=False, seed=2026):
        self.x = x
        self.y = np.asarray(y, dtype=np.int64)
        self.augment_flag = augment_flag
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        if self.augment_flag:
            x = h.augment(x, self.rng)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.tensor(int(self.y[idx]), dtype=torch.long))


def load_data():
    meta, x_raw, y3 = h._load_raw_ecg_and_labels(POSITIVE_NAMES)
    y3 = (np.asarray(y3) > 0.5).astype(np.int64)
    overlap = y3.sum(axis=1)
    if np.any(overlap > 1):
        raise ValueError("Four-class conversion requires mutually exclusive labels")
    y_class = np.zeros(len(y3), dtype=np.int64)
    y_class[y3[:, 0] == 1] = 1  # STEMI
    y_class[y3[:, 1] == 1] = 2  # NSTEMI
    y_class[y3[:, 2] == 1] = 3  # UA
    return meta, x_raw, y3, y_class


def metric_table(y, probs, pred):
    rows = {}
    for ci, name in enumerate(CLASS_NAMES):
        yt = (y == ci).astype(np.int64)
        yp = (pred == ci).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows[name] = {
            "auroc": float(roc_auc_score(yt, probs[:, ci])),
            "auprc": float(average_precision_score(yt, probs[:, ci])),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
            "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, yp)),
            "accuracy": float(accuracy_score(yt, yp)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    macro_keys = ("auroc", "auprc", "sensitivity", "specificity", "ppv",
                  "npv", "f1", "mcc", "accuracy")
    rows["macro"] = {k: float(np.mean([rows[n][k] for n in CLASS_NAMES]))
                      for k in macro_keys}
    rows["overall_accuracy"] = float(accuracy_score(y, pred))
    rows["macro_f1_sklearn"] = float(f1_score(y, pred, average="macro"))
    rows["weighted_f1"] = float(f1_score(y, pred, average="weighted"))
    rows["confusion_matrix"] = confusion_matrix(y, pred, labels=list(range(4))).tolist()
    return rows


def predict(model, loader, device):
    ys, ps, gs = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device, non_blocking=True))
            ys.append(yb.numpy())
            ps.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
            if out.get("gates") is not None:
                gs.append(out["gates"].cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(gs) if gs else None)


def run_fold(args, fold_idx, train_idx, val_idx, x_raw, y_class, device):
    set_seed(args.seed + fold_idx)
    scaler = h.fit_robust_scaler(x_raw[train_idx])
    x_tr = h.apply_robust_scaler(x_raw[train_idx], scaler)
    x_va = h.apply_robust_scaler(x_raw[val_idx], scaler)
    train_ds = MulticlassECGDataset(x_tr, y_class[train_idx], True, args.seed + fold_idx)
    val_ds = MulticlassECGDataset(x_va, y_class[val_idx], False, args.seed + fold_idx)
    generator = torch.Generator().manual_seed(args.seed + fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers,
                              generator=generator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = h.HMoEKAN("e4", n_tasks=4, head_kind="kan", device=str(device)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_score, best_epoch, wait = None, -np.inf, -1, 0
    history = []
    gate_rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out["logits"], yb)
            if out.get("gates") is not None:
                mean_gate = out["gates"].mean(dim=(0, 1))
                loss = loss + args.lambda_balance * ((mean_gate - 1.0 / 3.0) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()); nb += 1
        yv, pv, gv = predict(model, val_loader, device)
        score = float(np.mean([roc_auc_score((yv == ci).astype(int), pv[:, ci])
                               for ci in range(4)]))
        val_loss = float(-np.mean(np.log(np.maximum(pv[np.arange(len(yv)), yv], 1e-8))))
        history.append({"epoch": epoch, "train_loss": running / max(1, nb),
                        "val_nll": val_loss, "val_macro_auroc": score})
        print(json.dumps({"variant": "e4_multiclass", "fold": fold_idx,
                          "epoch": epoch, "train_loss": running / max(1, nb),
                          "val_macro_auroc": score}), flush=True)
        if gv is not None:
            gate_rows.append(gv)
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state); model.to(device); model.eval()
    yv, pv, gv = predict(model, val_loader, device)
    pred = pv.argmax(axis=1)
    result = metric_table(yv, pv, pred)
    result.update({"fold": fold_idx, "best_epoch": best_epoch,
                   "train_sec": round(time.time() - t0, 2),
                   "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                   "n_params": int(sum(q.numel() for q in model.parameters()
                                        if q.requires_grad)),
                   "gate_mean": np.concatenate(gate_rows).mean(axis=0).tolist()
                   if gate_rows else None,
                   "history": history})
    return result, model.state_dict(), pv, gv


def recover_array(path, scalers):
    p = np.load(path, mmap_mode="r")
    s = scalers["P2"]
    return p.astype(np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])


def stem(value):
    return Path(str(value)).stem


def sha256(path):
    hsh = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hsh.update(block)
    return hsh.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/hmoe_kan_e4_multiclass")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_balance", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta, x_raw, y3, y_class = load_data()
    folds, joint_counts = h.build_joint_patient_folds(
        meta["patient_ids"], y3, args.n_folds, args.fold_seed)
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({
        "protocol": "patient_level_joint_stratified_5fold",
        "task": "four_class_softmax", "classes": CLASS_NAMES,
        "fold_seed": args.fold_seed, "joint_patient_state_counts": joint_counts,
        "folds": [{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]
    }, indent=2), encoding="utf-8")
    (out / "class_counts.json").write_text(json.dumps({
        name: int((y_class == i).sum()) for i, name in enumerate(CLASS_NAMES)
    }, indent=2), encoding="utf-8")
    fold_results, oof_prob = [], np.zeros((len(y_class), 4), dtype=np.float32)
    for fi, (tr, va) in enumerate(folds):
        result, state, pv, _ = run_fold(args, fi, tr, va, x_raw, y_class, device)
        fold_results.append(result); oof_prob[va] = pv
        torch.save(state, out / f"e4_multiclass_fold{fi}_seed{args.seed}.pt")
        (out / f"history_fold{fi}.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
        print("FOLD_RESULT " + json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    oof_pred = oof_prob.argmax(axis=1)
    oof_metrics = metric_table(y_class, oof_prob, oof_pred)
    np.savez_compressed(out / "oof_probabilities.npz", y_class=y_class,
                        oof_prob=oof_prob, oof_pred=oof_pred)
    with (out / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_index", "y_class", "y_name", "pred_class", "pred_name"] +
                                     [f"p_{n}" for n in CLASS_NAMES])
        for i in range(len(y_class)):
            w.writerow([i, int(y_class[i]), CLASS_NAMES[y_class[i]], int(oof_pred[i]),
                        CLASS_NAMES[oof_pred[i]], *oof_prob[i].tolist()])

    # Test inference is performed only after all fold checkpoints are frozen.
    # The three-column file is one-hot compatible with the previous platform,
    # while OTHER is represented by all three disease columns being zero.
    arr = ROOT / "data" / "arrays"
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    x_test_raw = recover_array(arr / "P2_test.npy", scalers)
    test_meta = np.load(arr / "meta_test.npz", allow_pickle=True)
    test_probs_by_fold = []
    for fi, (tr, _) in enumerate(folds):
        scaler = h.fit_robust_scaler(x_raw[tr])
        x_test = h.apply_robust_scaler(x_test_raw, scaler)
        test_ds = MulticlassECGDataset(x_test, np.zeros(len(x_test), dtype=np.int64), False, args.seed)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        model = h.HMoEKAN("e4", n_tasks=4, head_kind="kan", device=str(device)).to(device)
        state = torch.load(out / f"e4_multiclass_fold{fi}_seed{args.seed}.pt", map_location=device)
        model.load_state_dict(state); model.eval()
        _, p_test, _ = predict(model, test_loader, device)
        test_probs_by_fold.append(p_test)
    test_prob = np.mean(np.stack(test_probs_by_fold, axis=0), axis=0)
    test_pred = test_prob.argmax(axis=1)
    ids_meta_raw = [str(x) for x in test_meta["record_ids"]]
    id_to_i = {stem(x): i for i, x in enumerate(ids_meta_raw)}
    with (ROOT / "data" / "CSV" / "test.csv").open(encoding="utf-8-sig", newline="") as f:
        ids_out_raw = [(row.get("ecg_row_record") or "").strip() for row in csv.DictReader(f)]
    order = np.asarray([id_to_i[stem(x)] for x in ids_out_raw], dtype=np.int64)
    test_prob = test_prob[order]; test_pred = test_pred[order]
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ecg_row_record", "pred_class", "pred_name"] + [f"p_{n}" for n in CLASS_NAMES])
        for rid, cls, pp in zip(ids_out_raw, test_pred, test_prob):
            w.writerow([rid, int(cls), CLASS_NAMES[int(cls)], *pp.tolist()])
    submission = out / "submission_stemi_nstemi_ua_e4_multiclass.csv"
    with submission.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ecg_row_record", "STEMI", "NSTEMI", "UA"])
        for rid, cls in zip(ids_out_raw, test_pred):
            w.writerow([rid, int(cls == 1), int(cls == 2), int(cls == 3)])
    summary = {
        "task": "four_class_softmax", "classes": CLASS_NAMES,
        "model": "HMoE_WavelNet_KAN_E4 with four KAN heads",
        "loss": "CrossEntropyLoss (unweighted)", "seed": args.seed,
        "fold_seed": args.fold_seed, "folds": fold_results,
        "class_counts": {n: int((y_class == i).sum()) for i, n in enumerate(CLASS_NAMES)},
        "oof_metrics": oof_metrics,
        "test_n": int(len(test_pred)),
        "test_predicted_class_counts": {n: int((test_pred == i).sum())
                                         for i, n in enumerate(CLASS_NAMES)},
        "test_multi_positive_count": 0,
        "test_none_positive_count": int((test_pred == 0).sum()),
        "submission": str(submission),
        "submission_sha256": sha256(submission),
        "mean_std_macro": {
            k: {"mean": float(np.mean([r["macro"][k] for r in fold_results])),
                "std": float(np.std([r["macro"][k] for r in fold_results]))}
            for k in ("auroc", "auprc", "sensitivity", "specificity", "ppv",
                      "npv", "f1", "mcc", "accuracy")
        },
        "note": "This is a new four-class experiment; historical three-label sigmoid results are preserved.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
