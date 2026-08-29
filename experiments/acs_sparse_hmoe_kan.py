"""Legacy feasibility run for the revised ACS sparse HMoE--KAN plan.

This historical runner predates the task-definition correction and must not be
used for current ACS subtype claims.  It is retained for audit history; use
``experiments.acs_joint_e4`` for the corrected categorical ACS + OMI task.

This is an independent experiment and does not alter the historical baseline
or the earlier three-expert HMoE run.  It uses the existing ACS arrays and
preprocessing, a patient-level joint-stratified five-fold CV protocol, and the
revised four-expert architecture from
``ACS_STEMI_NSTEMI_UA_Sparse_HMoE_KAN_README.md``:

* Xception morphology expert;
* InceptionTime multi-scale expert;
* PyG Lead-GAT spatial expert;
* a fixed Morlet CWT magnitude followed by a trainable CNN (no MVMD/VMD);
* three class-specific noisy Top-2 routers and KAN heads.

The script deliberately keeps all statistics and thresholds validation-only.
It writes fold checkpoints, OOF predictions, threshold comparisons, pairwise
NSTEMI-vs-UA metrics, router usage, calibration, figures, and JSON/CSV/Markdown
friendly artifacts under the requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIRD_PARTY / "pykan_tmp"))
sys.path.insert(0, str(THIRD_PARTY / "pytorch_geometric"))

from kan import KAN  # noqa: E402  (GitHub: KindXiaoming/pykan)
from torch_geometric.nn import GATConv  # noqa: E402 (GitHub: pyg-team/PyG)

from nets.cnn_variants import XceptionTime  # noqa: E402
from nets.traditional import InceptionTime  # noqa: E402
from preprocessing.augment import augment  # noqa: E402
from preprocessing.robust_scaler import apply_robust_scaler, fit_robust_scaler  # noqa: E402


LABELS = ("STEMI", "NSTEMI", "UA")
EXPERTS = ("xception", "inception", "lead_gat", "morlet_cwt")
METRICS = ("auroc", "auprc", "sens", "spec", "ppv", "npv", "f1", "mcc", "acc")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load the historical P2 arrays and align labels by record id."""
    arr = ROOT / "data" / "arrays"
    meta = np.load(arr / "meta_train.npz", allow_pickle=True)
    p2 = np.load(arr / "P2_train.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    s = scalers["P2"]
    # P2 is the historical globally scaled 0.5--40 Hz array.  Recover the
    # band-passed signal before fitting a fold-specific train-only scaler.
    x = p2.astype(np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])
    csv_by_id = {}
    with (ROOT / "data" / "CSV" / "train.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rid = row["ecg_row_record"].split(".")[0]
            csv_by_id[rid] = [int(row[k]) for k in LABELS]
    labels = np.asarray([csv_by_id[str(rid)] for rid in meta["record_ids"]], dtype=np.float32)
    return meta, x, labels


def build_joint_patient_folds(patient_ids, labels, n_folds=5, seed=2026):
    """Patient-level folds stratified by the observed joint label state."""
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    patient_state = np.zeros(len(patients), dtype=np.int64)
    for pi in range(len(patients)):
        rows = np.where(inverse == pi)[0]
        for j in range(labels.shape[1]):
            if np.any(labels[rows, j] > 0.5):
                patient_state[pi] |= 1 << j
    counts = np.bincount(patient_state, minlength=2 ** labels.shape[1])
    nonzero = counts[counts > 0]
    if len(nonzero) == 0 or int(nonzero.min()) < n_folds:
        raise ValueError(f"joint patient states too sparse for {n_folds}-fold: {counts.tolist()}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_pat, val_pat in skf.split(patients, patient_state):
        tr = np.where(np.isin(inverse, train_pat))[0]
        va = np.where(np.isin(inverse, val_pat))[0]
        folds.append((tr, va))
    return folds, counts.tolist(), patients, inverse, patient_state


class ECGDataset(Dataset):
    def __init__(self, x, y, augment_flag=False, seed=2026):
        self.x = x
        self.y = np.asarray(y, dtype=np.float32)
        self.augment_flag = bool(augment_flag)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        if self.augment_flag:
            x = augment(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.y[idx].copy())


class XceptionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = XceptionTime(num_classes=3).inner

    def forward(self, x):
        return self.backbone.forward_features(x.transpose(1, 2))


class InceptionExpert(nn.Module):
    """Original InceptionTime body, exposed at its 128-D feature layer."""
    def __init__(self):
        super().__init__()
        self.backbone = InceptionTime(num_classes=3).inner

    def forward(self, x):
        b = self.backbone
        x = x.transpose(1, 2)
        shortcut1 = b.bn1(b.conv1(x))
        o1 = b.BaseBlock3(b.BaseBlock2(b.BaseBlock1(x)))
        o1 = b.relu(o1 + shortcut1)
        shortcut2 = b.bn2(b.conv2(o1))
        o2 = b.BaseBlock6(b.BaseBlock5(b.BaseBlock4(o1)))
        o2 = b.relu(o2 + shortcut2)
        return b.Avgpool(o2).view(o2.size(0), -1)


class LeadGATExpert(nn.Module):
    """Lead-wise encoder followed by a complete 12-node PyG GAT graph."""
    def __init__(self, out_dim=128, n_leads=12):
        super().__init__()
        self.n_leads = n_leads
        self.lead_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 9, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gat = GATConv(64, out_dim, heads=2, concat=False, dropout=0.1,
                           add_self_loops=True)
        edges = torch.cartesian_prod(torch.arange(n_leads), torch.arange(n_leads)).t().contiguous()
        self.register_buffer("edge_index", edges, persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        if c != self.n_leads:
            raise ValueError(f"LeadGAT expects {self.n_leads} leads, got {c}")
        nodes = x.transpose(1, 2).reshape(b * c, 1, t)
        h = self.lead_encoder(nodes).squeeze(-1)
        e = self.edge_index.shape[1]
        offsets = torch.arange(b, device=x.device).repeat_interleave(e) * c
        edge = self.edge_index.to(x.device).repeat(1, b) + offsets.repeat(2, 1)
        h = self.gat(h, edge).reshape(b, c, -1)
        return h.mean(dim=1)


def _morlet_kernels(fs=100.0, n_scales=8, kernel_size=65, f_min=1.0, f_max=40.0):
    """Return fixed real/imaginary Morlet kernels for a differentiable CWT."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    t = (np.arange(kernel_size, dtype=np.float64) - kernel_size // 2) / fs
    freqs = np.geomspace(f_min, f_max, n_scales)
    real, imag = [], []
    for freq in freqs:
        sigma = max(0.035, 6.0 / (2.0 * np.pi * freq))
        env = np.exp(-0.5 * (t / sigma) ** 2)
        phase = 2.0 * np.pi * freq * t
        re = env * np.cos(phase)
        im = env * np.sin(phase)
        re -= re.mean(); im -= im.mean()
        norm = np.sqrt(np.sum(re * re + im * im)) + 1e-12
        real.append((re / norm).astype(np.float32))
        imag.append((im / norm).astype(np.float32))
    return np.stack(real)[:, None, :], np.stack(imag)[:, None, :], freqs.tolist()


class MorletCWT(nn.Module):
    """Fixed Morlet CWT bank; its magnitude is learned by the CNN after it."""
    def __init__(self, fs=100, n_scales=8, kernel_size=65, stride=4):
        super().__init__()
        re, im, freqs = _morlet_kernels(fs, n_scales, kernel_size)
        self.register_buffer("real", torch.from_numpy(re), persistent=False)
        self.register_buffer("imag", torch.from_numpy(im), persistent=False)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.fs = int(fs)
        self.freqs = freqs

    def forward(self, x):
        # x: [B, 12, T] -> [B, 12, scales, time/stride]
        b, c, t = x.shape
        z = x.reshape(b * c, 1, t)
        pad = self.kernel_size // 2
        re = F.conv1d(z, self.real, padding=pad, stride=self.stride)
        im = F.conv1d(z, self.imag, padding=pad, stride=self.stride)
        mag = torch.sqrt(re.square() + im.square() + 1e-6)
        return mag.reshape(b, c, mag.shape[1], mag.shape[2])


class MorletCWTExpert(nn.Module):
    """CWT-CNN frequency expert, explicitly avoiding MVMD/VMD preprocessing."""
    def __init__(self, out_dim=128, n_leads=12, fs=100, n_scales=8,
                 kernel_size=65, stride=4):
        super().__init__()
        self.cwt = MorletCWT(fs, n_scales, kernel_size, stride)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_leads, 32, (3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        h = self.cwt(x.transpose(1, 2))
        h = self.encoder(h).flatten(1)
        return self.proj(h)


class Projection(nn.Module):
    def __init__(self, in_dim=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class RouterStem(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.fc(self.net(x.transpose(1, 2)).flatten(1))


class SparseTop2Router(nn.Module):
    def __init__(self, context_dim=128, n_experts=4, noise_std=0.1):
        super().__init__()
        self.proj = nn.Linear(context_dim, n_experts)
        self.n_experts = n_experts
        self.noise_std = float(noise_std)

    def forward(self, context):
        logits = self.proj(context)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = torch.softmax(logits, dim=-1)
        top_scores, top_idx = torch.topk(probs, k=2, dim=-1)
        return probs, top_idx, top_scores


class SelectedCombiner(nn.Module):
    def __init__(self, dim=256, n_experts=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2 + 2),
            nn.Linear(dim * 2 + 2, dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )

    def forward(self, expert_stack, idx, scores):
        # expert_stack [B, E, D], idx/scores [B, 2]
        b, _, d = expert_stack.shape
        gather_idx = idx.unsqueeze(-1).expand(b, 2, d)
        selected = expert_stack.gather(1, gather_idx).reshape(b, 2 * d)
        return self.net(torch.cat([selected, scores], dim=-1))


class KANHead(nn.Module):
    def __init__(self, dim=256, device="cpu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            self.kan = KAN(width=[dim, 64, 1], grid=3, k=3,
                           symbolic_enabled=False, save_act=False, auto_save=False,
                           seed=2026, device=device)
        except TypeError:
            self.kan = KAN(width=[dim, 64, 1], grid=3, k=3, device=device)

    def forward(self, x):
        return self.kan(self.norm(x)).reshape(-1)


class ACS_Sparse_HMoE_KAN(nn.Module):
    def __init__(self, device="cpu", n_tasks=3, n_experts=4, d_model=256,
                 noise_std=0.1, cwt_scales=8, cwt_kernel=65, cwt_stride=4):
        super().__init__()
        self.d_model = d_model
        self.xception = XceptionExpert()
        self.inception = InceptionExpert()
        self.lead_gat = LeadGATExpert(128)
        self.morlet_cwt = MorletCWTExpert(128, n_scales=cwt_scales,
                                          kernel_size=cwt_kernel, stride=cwt_stride)
        self.projections = nn.ModuleList([Projection(128, d_model) for _ in EXPERTS])
        self.router_stem = RouterStem(128)
        self.routers = nn.ModuleList([
            SparseTop2Router(128, n_experts, noise_std) for _ in range(n_tasks)
        ])
        self.combiners = nn.ModuleList([SelectedCombiner(d_model, n_experts) for _ in range(n_tasks)])
        self.heads = nn.ModuleList([KANHead(d_model, str(device)) for _ in range(n_tasks)])

    def forward(self, x):
        z128 = [self.xception(x), self.inception(x), self.lead_gat(x), self.morlet_cwt(x)]
        z = torch.stack([p(h) for p, h in zip(self.projections, z128)], dim=1)
        context = self.router_stem(x)
        all_probs, all_idx, all_scores, task_z = [], [], [], []
        for router, combiner in zip(self.routers, self.combiners):
            probs, idx, scores = router(context)
            all_probs.append(probs); all_idx.append(idx); all_scores.append(scores)
            task_z.append(combiner(z, idx, scores))
        logits = torch.stack([head(h) for head, h in zip(self.heads, task_z)], dim=1)
        return {
            "logits": logits,
            "router_probs": torch.stack(all_probs, dim=1),
            "top_idx": torch.stack(all_idx, dim=1),
            "top_scores": torch.stack(all_scores, dim=1),
        }


def _safe_auc(fn, y, p):
    return float(fn(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _safe_mcc(y, pred):
    try:
        return float(matthews_corrcoef(y, pred))
    except Exception:
        return 0.0


def resolve_multilabel(p, thresholds):
    """Apply per-class thresholds and normalized-margin conflict resolution."""
    p = np.asarray(p)
    thresholds = np.asarray(thresholds)
    raw = p >= thresholds[None, :]
    pred = raw.astype(np.int64)
    margins = (p - thresholds[None, :]) / np.maximum(1.0 - thresholds[None, :], 1e-8)
    for i in range(len(pred)):
        pos = np.flatnonzero(raw[i])
        if len(pos) > 1:
            keep = pos[int(np.argmax(margins[i, pos]))]
            pred[i, :] = 0
            pred[i, keep] = 1
    return pred


def choose_thresholds(y, p, kind):
    out = []
    grid = np.arange(0.01, 1.0, 0.01)
    for j in range(y.shape[1]):
        yt, pj = y[:, j].astype(int), p[:, j]
        if kind == "fixed05":
            out.append(0.5); continue
        if kind == "youden":
            fpr, tpr, ts = roc_curve(yt, pj)
            score = tpr - fpr
            t = float(ts[int(np.nanargmax(score))])
            out.append(float(np.clip(t, 0.01, 0.99))); continue
        best_t, best = 0.5, -np.inf
        for t in grid:
            pred = (pj >= t).astype(int)
            if kind == "max_f1":
                value = f1_score(yt, pred, zero_division=0)
            elif kind == "balanced_accuracy":
                value = balanced_accuracy_score(yt, pred)
            else:
                raise ValueError(kind)
            if value > best:
                best, best_t = value, float(t)
        out.append(best_t)
    return np.asarray(out, dtype=np.float64)


def choose_joint_thresholds(y, p, base_kind="max_f1"):
    """T4: independent STEMI threshold + validation-only NSTEMI/UA search."""
    thr = choose_thresholds(y, p, base_kind)
    mask = (y[:, 0] < 0.5) & ((y[:, 1] + y[:, 2]) == 1)
    if int(mask.sum()) < 20:
        return thr
    best, best_pair = -np.inf, (thr[1], thr[2])
    pair_y = y[mask, 1].astype(int)
    for tn in np.arange(0.05, 0.96, 0.01):
        for tu in np.arange(0.05, 0.96, 0.01):
            pair_pred = resolve_pairwise(p[mask, 1], p[mask, 2], tn, tu)
            value = f1_score(pair_y, pair_pred, average="macro", zero_division=0)
            if value > best:
                best, best_pair = float(value), (float(tn), float(tu))
    thr[1], thr[2] = best_pair
    return thr


def resolve_pairwise(pn, pu, tn, tu):
    """Binary NSTEMI-vs-UA decision with threshold-aware conflict handling."""
    pn, pu = np.asarray(pn), np.asarray(pu)
    n, u = pn >= tn, pu >= tu
    margins_n = (pn - tn) / max(1.0 - tn, 1e-8)
    margins_u = (pu - tu) / max(1.0 - tu, 1e-8)
    out = np.where(n & ~u, 1, np.where(u & ~n, 0, (pn / np.maximum(pn + pu, 1e-8) >= 0.5).astype(int)))
    both = n & u
    out[both] = (margins_n[both] >= margins_u[both]).astype(int)
    return out.astype(int)


def metric_table(y, p, thresholds):
    pred = resolve_multilabel(p, thresholds)
    out = {}
    for j, name in enumerate(LABELS):
        yt, yp = y[:, j].astype(int), pred[:, j].astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        out[name] = {
            "auroc": _safe_auc(roc_auc_score, yt, p[:, j]),
            "auprc": _safe_auc(average_precision_score, yt, p[:, j]),
            "sens": float(recall_score(yt, yp, zero_division=0)),
            "spec": float(tn / max(1, tn + fp)),
            "ppv": float(precision_score(yt, yp, zero_division=0)),
            "npv": float(tn / max(1, tn + fn)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "mcc": _safe_mcc(yt, yp),
            "acc": float(accuracy_score(yt, yp)),
            "threshold": float(thresholds[j]),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    out["macro_auroc"] = float(np.nanmean([out[n]["auroc"] for n in LABELS]))
    out["macro_auprc"] = float(np.nanmean([out[n]["auprc"] for n in LABELS]))
    out["macro_f1"] = float(np.mean([out[n]["f1"] for n in LABELS]))
    out["micro_f1"] = float(f1_score(y.reshape(-1), pred.reshape(-1), zero_division=0))
    out["hamming_loss"] = float(np.mean(y.astype(int) != pred))
    return out, pred


def pairwise_metrics(y, p, thresholds):
    mask = (y[:, 0] < 0.5) & ((y[:, 1] + y[:, 2]) == 1)
    if int(mask.sum()) == 0:
        return {"n": 0}
    yy = y[mask, 1].astype(int)  # 1 NSTEMI, 0 UA
    pn, pu = p[mask, 1], p[mask, 2]
    score = pn / np.maximum(pn + pu, 1e-8)
    yp = resolve_pairwise(pn, pu, thresholds[1], thresholds[2])
    cm = confusion_matrix(yy, yp, labels=[0, 1])
    return {
        "n": int(mask.sum()),
        "auroc_nstemi_vs_ua": _safe_auc(roc_auc_score, yy, score),
        "auprc_nstemi_vs_ua": _safe_auc(average_precision_score, yy, score),
        "f1_macro": float(f1_score(yy, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yy, yp)),
        "sensitivity_nstemi": float(recall_score(yy, yp, pos_label=1, zero_division=0)),
        "sensitivity_ua": float(recall_score(1 - yy, 1 - yp, pos_label=1, zero_division=0)),
        "specificity_macro": float(np.mean([
            cm[0, 0] / max(1, cm[0].sum()), cm[1, 1] / max(1, cm[1].sum())
        ])),
        "confusion_matrix_true_ua_nstemi": cm.tolist(),
        "threshold_nstemi": float(thresholds[1]), "threshold_ua": float(thresholds[2]),
    }


def calibration_metrics(y, p, n_bins=10):
    rows = []
    for j, name in enumerate(LABELS):
        yt, pj = y[:, j].astype(int), p[:, j]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0; rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pj >= lo) & (pj <= hi if hi == 1 else pj < hi)
            if not np.any(m): continue
            conf, acc = float(pj[m].mean()), float(yt[m].mean())
            ece += float(m.mean()) * abs(conf - acc)
            rel.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                        "confidence": conf, "accuracy": acc})
        rows.append({"label": name, "brier": float(brier_score_loss(yt, pj)),
                     "ece": float(ece), "reliability": rel})
    return rows


def js_divergence(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    kl1 = np.sum(np.where(a > 0, a * np.log(a / np.maximum(m, 1e-12)), 0.0))
    kl2 = np.sum(np.where(b > 0, b * np.log(b / np.maximum(m, 1e-12)), 0.0))
    return float(0.5 * (kl1 + kl2))


def train_fold(args, fold_idx, train_idx, val_idx, x_raw, labels, device, out):
    set_seed(args.seed + fold_idx)
    scaler = fit_robust_scaler(x_raw[train_idx])
    x_tr = apply_robust_scaler(x_raw[train_idx], scaler)
    x_va = apply_robust_scaler(x_raw[val_idx], scaler)
    train_ds = ECGDataset(x_tr, labels[train_idx], True, args.seed + fold_idx)
    val_ds = ECGDataset(x_va, labels[val_idx], False, args.seed + fold_idx)
    gen = torch.Generator().manual_seed(args.seed + fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=args.num_workers,
                              generator=gen, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    pos = labels[train_idx].sum(axis=0)
    neg = len(train_idx) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device)
    model = ACS_Sparse_HMoE_KAN(device=device, cwt_scales=args.cwt_scales,
                                cwt_kernel=args.cwt_kernel, cwt_stride=args.cwt_stride).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_score, best_epoch, wait = None, -np.inf, 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train(); sums = {"total": 0.0, "bce": 0.0, "excl": 0.0, "balance": 0.0, "entropy": 0.0}; nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            o = model(xb); logits = o["logits"]
            bce = criterion(logits, yb)
            prob = torch.sigmoid(logits)
            excl = (prob[:, 0] * prob[:, 1] + prob[:, 0] * prob[:, 2] + prob[:, 1] * prob[:, 2]).mean()
            mean_route = o["router_probs"].mean(dim=(0, 1))
            balance = ((mean_route - 1.0 / len(EXPERTS)) ** 2).mean()
            route = o["router_probs"].clamp_min(1e-8)
            entropy = -(route * route.log()).sum(dim=-1).mean()
            ent_weight = args.lambda_entropy * min(1.0, epoch / max(1, args.entropy_warmup))
            loss = bce + args.lambda_excl * excl + args.lambda_balance * balance - ent_weight * entropy
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for k, v in (("total", loss), ("bce", bce), ("excl", excl), ("balance", balance), ("entropy", entropy)):
                sums[k] += float(v.detach().item())
            nb += 1
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                o = model(xb.to(device, non_blocking=True))
                ys.append(yb.numpy()); ps.append(torch.sigmoid(o["logits"]).cpu().numpy())
        yv, pv = np.concatenate(ys), np.concatenate(ps)
        score = float(np.nanmean([_safe_auc(roc_auc_score, yv[:, j], pv[:, j]) for j in range(3)]))
        row = {"epoch": epoch, "train_total": sums["total"] / max(1, nb),
               "train_bce": sums["bce"] / max(1, nb), "train_excl": sums["excl"] / max(1, nb),
               "train_balance": sums["balance"] / max(1, nb), "train_entropy": sums["entropy"] / max(1, nb),
               "val_macro_auroc": score}
        history.append(row)
        print(json.dumps({"fold": fold_idx, **row}), flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("no best state recorded")
    model.load_state_dict(best_state); model.to(device); model.eval()
    ys, ps, routes, idxs, scores = [], [], [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            o = model(xb.to(device, non_blocking=True))
            ys.append(yb.numpy()); ps.append(torch.sigmoid(o["logits"]).cpu().numpy())
            routes.append(o["router_probs"].cpu().numpy())
            idxs.append(o["top_idx"].cpu().numpy()); scores.append(o["top_scores"].cpu().numpy())
    yv, pv = np.concatenate(ys), np.concatenate(ps)
    routev, idxv, scorev = np.concatenate(routes), np.concatenate(idxs), np.concatenate(scores)
    state = {"model": best_state, "pos_weight": pos_weight.detach().cpu().tolist(),
             "fold": fold_idx, "best_epoch": best_epoch, "val_macro_auroc": best_score}
    torch.save(state, out / f"fold_{fold_idx}_checkpoint.pt")
    np.savez_compressed(out / f"fold_{fold_idx}_predictions.npz", y=yv, p=pv,
                        router_probs=routev, top_idx=idxv, top_scores=scorev,
                        indices=val_idx)
    with (out / f"history_fold_{fold_idx}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    fold_info = {
        "fold": fold_idx, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "best_epoch": best_epoch, "val_macro_auroc": best_score,
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "train_seconds": round(time.time() - t0, 2),
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    print("FOLD_DONE " + json.dumps(fold_info), flush=True)
    return fold_info, yv, pv, routev, idxv, scorev


def write_figures(out, y, p, pred, router_mean):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for j, name in enumerate(LABELS):
            cm = confusion_matrix(y[:, j].astype(int), pred[:, j].astype(int), labels=[0, 1])
            ax = axes[j]; im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            for (r, c), v in np.ndenumerate(cm): ax.text(c, r, str(int(v)), ha="center", va="center")
        fig.tight_layout(); fig.savefig(out / "confusion_matrix_t4.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        for j, name in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y[:, j], p[:, j]); ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); fig.tight_layout(); fig.savefig(out / "roc_oof.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        for j, name in enumerate(LABELS):
            frac, mean = calibration_curve(y[:, j], p[:, j], n_bins=10, strategy="uniform"); ax.plot(mean, frac, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency"); ax.legend(); fig.tight_layout(); fig.savefig(out / "reliability_oof.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3)); im = ax.imshow(router_mean, aspect="auto", cmap="viridis"); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS); ax.set_xticks(range(4)); ax.set_xticklabels(EXPERTS, rotation=20, ha="right"); ax.set_title("Mean router probability"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "router_probability_heatmap.png", dpi=160); plt.close(fig)
    except Exception as exc:
        (out / "figure_warning.txt").write_text(f"figure generation failed: {exc}\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/acs_sparse_hmoe_kan_fullcohort_gpu1_20260826")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_excl", type=float, default=0.05)
    p.add_argument("--lambda_balance", type=float, default=0.01)
    p.add_argument("--lambda_entropy", type=float, default=0.001)
    p.add_argument("--entropy_warmup", type=int, default=10)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--cwt_scales", type=int, default=8)
    p.add_argument("--cwt_kernel", type=int, default=65)
    p.add_argument("--cwt_stride", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta, x_raw, labels = load_data()
    folds, joint_counts, patients, inverse, patient_state = build_joint_patient_folds(
        meta["patient_ids"], labels, args.n_folds, args.fold_seed)
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    # Required README audit artifacts.
    positives = labels.sum(axis=0).astype(int).tolist()
    pattern_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(labels.astype(int).dot(1 << np.arange(3)), return_counts=True))}
    pair_counts = {}
    for a in range(3):
        for b in range(a + 1, 3):
            pair_counts[f"{LABELS[a]}+{LABELS[b]}"] = int(np.sum((labels[:, a] > .5) & (labels[:, b] > .5)))
    audit = {
        "n_records": int(len(labels)), "n_patients": int(len(patients)),
        "n_positive": dict(zip(LABELS, positives)),
        "n_negative": dict(zip(LABELS, (len(labels) - np.asarray(positives)).tolist())),
        "joint_record_pattern_counts_bitmask_1_stemi_2_nstemi_4_ua": pattern_counts,
        "joint_patient_state_counts": joint_counts,
        "pairwise_cooccurrence": pair_counts,
        "triple_positive": int(np.sum(labels.sum(axis=1) == 3)),
        "all_zero": int(np.sum(labels.sum(axis=1) == 0)),
        "mutually_exclusive_observed": bool(max(pair_counts.values() or [0]) == 0),
        "note": "Audit is on visible train labels; external platform labels remain hidden.",
    }
    (out / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (out / "label_cooccurrence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["label_a", "label_b", "cooccurrence"])
        for k, v in pair_counts.items(): w.writerow([*k.split("+"), v])
    split = {"protocol": "patient_level_joint_stratified_5fold", "fold_seed": args.fold_seed,
             "n_records": len(labels), "n_patients": len(patients),
             "folds": [{"train_indices": tr.tolist(), "val_indices": va.tolist(),
                         "train_patients": int(len(np.unique(meta["patient_ids"][tr]))),
                         "val_patients": int(len(np.unique(meta["patient_ids"][va])))} for tr, va in folds]}
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    config = vars(args).copy(); config.update({"labels": LABELS, "experts": EXPERTS,
        "device_resolved": str(device), "loss": "weighted_bce + exclusivity + router_balance - entropy",
        "pos_weight_source": "training fold only", "cwt": "fixed Morlet CWT magnitude + trainable CNN; no MVMD/VMD"})
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("DATA_AUDIT " + json.dumps(audit), flush=True)
    print("CONFIG " + json.dumps(config), flush=True)

    all_y = np.zeros_like(labels); all_p = np.zeros_like(labels); seen = np.zeros(len(labels), dtype=bool)
    route_sum = np.zeros((3, 4), dtype=np.float64); route_n = 0
    router_rows = []
    fold_infos = []
    for fi, (tr, va) in enumerate(folds):
        info, yv, pv, routes, top_idx, top_scores = train_fold(args, fi, tr, va, x_raw, labels, device, out)
        fold_infos.append(info); all_y[va] = yv; all_p[va] = pv; seen[va] = True
        route_sum += routes.sum(axis=0); route_n += routes.shape[0]
        for cls in range(3):
            for ex in range(4):
                router_rows.append({"fold": fi, "class": LABELS[cls], "expert": EXPERTS[ex],
                    "mean_soft_probability": float(routes[:, cls, ex].mean()),
                    "top1_fraction": float(np.mean(top_idx[:, cls, 0] == ex)),
                    "top2_fraction": float(np.mean(np.any(top_idx[:, cls, :] == ex, axis=1))),
                    "mean_selected_score": float(np.mean(top_scores[:, cls, :][top_idx[:, cls, :] == ex])) if np.any(top_idx[:, cls, :] == ex) else 0.0})
    if not np.all(seen): raise RuntimeError("OOF predictions do not cover every row")
    np.savez_compressed(out / "oof_predictions.npz", y=all_y, p=all_p)
    with (out / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_index", *[f"y_{n}" for n in LABELS], *[f"p_{n}" for n in LABELS]])
        for i in range(len(all_y)): w.writerow([i, *all_y[i].astype(int).tolist(), *all_p[i].tolist()])
    with (out / "router_usage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(router_rows[0])); w.writeheader(); w.writerows(router_rows)

    thresholds = {}
    metric_rows = []
    pred_by_kind = {}
    for kind, name in (("fixed05", "T0_fixed_0.5"), ("youden", "T1_youden"),
                       ("max_f1", "T2_max_f1"), ("balanced_accuracy", "T3_balanced_accuracy")):
        thresholds[name] = choose_thresholds(all_y, all_p, kind)
    thresholds["T4_joint_nstemi_ua"] = choose_joint_thresholds(all_y, all_p, "max_f1")
    for name, thr in thresholds.items():
        mm, pred = metric_table(all_y, all_p, thr); pp = pairwise_metrics(all_y, all_p, thr)
        pred_by_kind[name] = pred; metric_rows.append({"threshold": name, "thresholds": thr.tolist(), "metrics": mm, "pairwise": pp})
    (out / "thresholds.json").write_text(json.dumps({k: v.tolist() for k, v in thresholds.items()}, indent=2), encoding="utf-8")
    (out / "metrics_by_threshold.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    cal = calibration_metrics(all_y, all_p)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    router_mean = route_sum / max(1, route_n)
    router_class = router_mean.tolist()
    js = js_divergence(router_mean[1], router_mean[2])
    cosine = float(np.dot(router_mean[1], router_mean[2]) / (np.linalg.norm(router_mean[1]) * np.linalg.norm(router_mean[2]) + 1e-12))
    router_summary = {"mean_probability_by_class_expert": dict(zip(LABELS, router_class)),
                      "nstemi_ua_js_divergence": js, "nstemi_ua_cosine_similarity": cosine}
    (out / "router_summary.json").write_text(json.dumps(router_summary, indent=2), encoding="utf-8")
    write_figures(out, all_y, all_p, pred_by_kind["T4_joint_nstemi_ua"], router_mean)
    summary = {"task": "STEMI/NSTEMI/UA multilabel full cohort", "protocol": "patient-level joint-stratified 5-fold OOF",
               "folds": fold_infos, "thresholds": {k: v.tolist() for k, v in thresholds.items()},
               "metrics_by_threshold": metric_rows, "calibration": cal, "router": router_summary,
               "feasibility": "to be judged from OOF macro AUPRC/F1, pairwise NSTEMI/UA, and router usage"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
