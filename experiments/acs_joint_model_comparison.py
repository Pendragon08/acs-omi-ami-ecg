"""Corrected architecture comparison for the joint ACS/OMI task.

All models consume the same patient-level folds, fold-fitted robust scaling,
augmentation and optimizer budget.  Each emits five logits: a four-way ACS
categorical head (OTHER/STEMI/NSTEMI/UA) and an independent OMI logit.  Thus
the comparison never trains mutually exclusive subtype labels as independent
sigmoids.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from experiments import hmoe_kan_github as h  # noqa: E402
from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES, LABELS, SUBTYPE_LABELS, acs_metrics, acs_subtype_metrics,
    binary_metrics, encode_targets, load_data, load_or_build_folds,
    save_confusion_figures, save_curves, save_roc_pr_curves, select_omi_threshold,
)
from nets.traditional import InceptionTime, LSTM, ResNet1D_Wang, ViT  # noqa: E402


MODEL_NAMES = ("InceptionTime", "ResNet1D_Wang", "LSTM", "ViT", "Transformer1D")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class Transformer1D(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, sequence_length=1000, patch_size=25,
                 d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.patch = nn.Conv1d(input_channels, d_model, patch_size, patch_size)
        n_tokens = sequence_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02); nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        z = self.patch(x.transpose(1, 2)).transpose(1, 2)
        z = torch.cat((self.cls_token.expand(z.shape[0], -1, -1), z), dim=1)
        return self.head(self.norm(self.encoder(z + self.positional_embedding)[:, 0]))


def build_model(name: str) -> nn.Module:
    if name == "InceptionTime": return InceptionTime(num_classes=5)
    if name == "ResNet1D_Wang": return ResNet1D_Wang(num_classes=5)
    if name == "LSTM": return LSTM(num_classes=5)
    if name == "ViT": return ViT(num_classes=5)
    if name == "Transformer1D": return Transformer1D(num_classes=5)
    raise ValueError(name)


def logits(model, x):
    out = model(x); out = out["logits"] if isinstance(out, dict) else out
    if out.ndim != 2 or out.shape[1] != 5: raise RuntimeError(f"expected [B,5], got {tuple(out.shape)}")
    return out


def loss_parts(out, y):
    acs = F.cross_entropy(out[:, :4], y[:, 0].long())
    omi = F.binary_cross_entropy_with_logits(out[:, 4], y[:, 1].float())
    return acs + omi, acs, omi


def predict(model, loader, device):
    ya, yo, pa, po = [], [], [], []; model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = logits(model, xb.to(device, non_blocking=True))
            ya.append(yb[:, 0].numpy().astype(np.int64)); yo.append(yb[:, 1].numpy().astype(np.int64))
            pa.append(torch.softmax(out[:, :4].float(), 1).cpu().numpy()); po.append(torch.sigmoid(out[:, 4].float()).cpu().numpy())
    return np.concatenate(ya), np.concatenate(yo), np.concatenate(pa), np.concatenate(po)


def run_fold(args, name, fold, train_idx, val_idx, x_raw, acs, omi, patient_ids, device, out_dir):
    seed = args.seed + fold; set_seed(seed)
    scaler = h.fit_robust_scaler(x_raw[train_idx]); xtr = h.apply_robust_scaler(x_raw[train_idx], scaler); xv = h.apply_robust_scaler(x_raw[val_idx], scaler)
    ytr = np.column_stack([acs[train_idx], omi[train_idx]]).astype(np.float32); yv = np.column_stack([acs[val_idx], omi[val_idx]]).astype(np.float32)
    tr_loader = DataLoader(h.ECGDataset(xtr, ytr, True, seed), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(seed), pin_memory=True)
    va_loader = DataLoader(h.ECGDataset(xv, yv, False, seed), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = build_model(name).to(device); opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=4, min_lr=1e-7) if args.scheduler else None
    best_state = None; best = -np.inf; best_epoch = -1; wait = 0; history = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = acs_sum = omi_sum = 0.; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
            l, la, lo = loss_parts(logits(model, xb), yb); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm); opt.step()
            total += l.item(); acs_sum += la.item(); omi_sum += lo.item(); n += 1
        ya, yo, pa, po = predict(model, va_loader, device); am = acs_metrics(ya, pa); om = binary_metrics(yo, po, .5); score = float(np.nanmean([am["macro_auroc_ovr"], om["auroc"]]))
        if scheduler: scheduler.step(score)
        rec = {"epoch": epoch, "train_total_loss": total / max(1, n), "train_acs_ce": acs_sum / max(1, n), "train_omi_bce": omi_sum / max(1, n), "val_acs_macro_auroc": am["macro_auroc_ovr"], "val_omi_auroc": om["auroc"], "val_omi_auprc": om["auprc"], "val_acs_accuracy": am["accuracy"], "val_joint_score": score, "learning_rate": opt.param_groups[0]["lr"]}; history.append(rec)
        print(json.dumps({"experiment": "acs_joint_model_comparison", "model": name, "fold": fold, **rec}), flush=True)
        if score > best + args.min_delta:
            best, best_epoch, wait = score, epoch, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience: break
    model.load_state_dict(best_state); ya, yo, pa, po = predict(model, va_loader, device)
    model_dir = out_dir / name; model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / f"fold{fold}_seed{args.seed}.pt")
    fold_dir = model_dir / f"fold{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, fold_dir / "best_model.pt")
    (fold_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    (model_dir / f"history_fold{fold}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": name, "fold": fold, "best_epoch": best_epoch, "best_val_joint_score": best, "train_sec": round(time.time() - started, 2), "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "history": history}, (ya, yo, pa, po)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out_dir", default="results/corrected_joint_model_comparison"); p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES); p.add_argument("--n_folds", type=int, default=5); p.add_argument("--fold_seed", type=int, default=2026); p.add_argument("--seed", type=int, default=2026); p.add_argument("--epochs", type=int, default=50); p.add_argument("--patience", type=int, default=8); p.add_argument("--min_delta", type=float, default=1e-4); p.add_argument("--batch_size", type=int, default=128); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight_decay", type=float, default=0.0); p.add_argument("--scheduler", action="store_true"); p.add_argument("--gradient_clip_norm", type=float, default=1.0); p.add_argument("--num_workers", type=int, default=4); p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    if args.n_folds != 5: raise ValueError("registered protocol is fixed to five folds")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); out_dir = ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    x, ysrc, patients, records, label_audit = load_data(); acs, omi = encode_targets(ysrc); folds, split_audit = load_or_build_folds(patients, acs, omi, args.n_folds, args.fold_seed)
    summary = {"experiment": "ACS-JOINT-ARCHITECTURE-COMPARISON-CORRECTED", "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + OMI binary", "models": {}, "label_audit": label_audit, "split_audit": split_audit, "training": vars(args)}
    for name in args.models:
        oof_a = np.zeros((len(acs), 4), np.float32); oof_o = np.zeros(len(omi), np.float32); results = []; histories = []
        for fold, (tr, va) in enumerate(folds):
            res, pred = run_fold(args, name, fold, tr, va, x, acs, omi, patients, device, out_dir); ya, yo, pa, po = pred; oof_a[va] = pa; oof_o[va] = po; histories.append(res.pop("history")); results.append(res)
        thresholds = {}; details = {}
        for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95"): thresholds[method], details[method] = select_omi_threshold(omi, oof_o, method)
        pooled = {m: {"acs": acs_metrics(acs, oof_a), "acs_subtype": acs_subtype_metrics(acs, oof_a), "omi": None if t is None else binary_metrics(omi, oof_o, t)} for m, t in thresholds.items()}
        model_dir = out_dir / name; save_curves(model_dir, histories, f"Corrected joint {name}"); save_roc_pr_curves(model_dir, acs, oof_a, omi, oof_o)
        cm_omi = np.asarray([[int(((omi == 0) & (oof_o < thresholds["youden_j"])).sum()), int(((omi == 0) & (oof_o >= thresholds["youden_j"])).sum())], [int(((omi == 1) & (oof_o < thresholds["youden_j"])).sum()), int(((omi == 1) & (oof_o >= thresholds["youden_j"])).sum())]])
        save_confusion_figures(model_dir, np.asarray(pooled["youden_j"]["acs"]["confusion_matrix"]), np.asarray(pooled["youden_j"]["acs_subtype"]["confusion_matrix"]), cm_omi)
        (model_dir / "oof_predictions.npz").write_bytes(b"") if False else np.savez_compressed(model_dir / "oof_predictions.npz", acs=acs, omi=omi, p_acs=oof_a, p_omi=oof_o)
        summary["models"][name] = {"folds": results, "thresholds": thresholds, "threshold_details": details, "pooled_oof_metrics": pooled, "recommended_threshold": "youden_j"}
        (model_dir / "summary.json").write_text(json.dumps({"model": name, **summary["models"][name]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"); print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
