"""GitHub-component feasibility experiment for the ACS HMoE/KAN plan.

The integration deliberately keeps the project data loader and XceptionTime
backbone, while importing the proposed external components from pinned GitHub
source trees:

* KindXiaoming/pykan -> KAN heads
* pyg-team/pytorch_geometric -> GATConv lead-spatial expert
* NamhoKim-Kor/WavelNet -> WaveletConv frequency expert

The frequency branch is end-to-end: WaveletConv is applied directly to each
lead and no MVMD/VMD decomposition or cached frequency array is used.  E0--E4
share the patient-level five-fold split and training loop; only the named
architectural block changes between rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
sys.path.insert(0, str(THIRD_PARTY / "pykan_tmp"))
sys.path.insert(0, str(THIRD_PARTY / "pytorch_geometric"))
sys.path.insert(0, str(THIRD_PARTY / "WavelNet"))

from kan import KAN  # noqa: E402  (GitHub: KindXiaoming/pykan)
from torch_geometric.nn import GATConv  # noqa: E402  (GitHub: pyg-team/PyG)
from WaveletConv import WaveletConv  # noqa: E402  (GitHub: NamhoKim-Kor/WavelNet)
import pywt  # noqa: E402  (GitHub: PyWavelets; Symlet-4 mother wavelet)

from nets.cnn_variants import XceptionTime  # noqa: E402
from preprocessing.augment import augment  # noqa: E402
from preprocessing.robust_scaler import apply_robust_scaler, fit_robust_scaler  # noqa: E402


TASK_LABELS = {
    "ami_omi": ("AMI", "OMI"),
    "stemi_nstemi_ua": ("STEMI", "NSTEMI", "UA"),
}
LABELS = TASK_LABELS["ami_omi"]
METRICS = ("auroc", "auprc", "f1", "sens", "spec", "ppv", "npv", "mcc", "acc")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_raw_ecg_and_labels(label_names):
    arr = ROOT / "data" / "arrays"
    meta = np.load(arr / "meta_train.npz", allow_pickle=True)
    p2 = np.load(arr / "P2_train.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    # P2 is the historical globally scaled 0.5--40 Hz array. Recover the
    # band-passed values so every CV fold can fit its own train-only scaler.
    s = scalers["P2"]
    x = p2.astype(np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])
    csv_by_id = {}
    with (ROOT / "data" / "CSV" / "train.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rid = row["ecg_row_record"].split(".")[0]
            csv_by_id[rid] = [int(row[k]) for k in label_names]
    labels = np.asarray([csv_by_id[str(rid)] for rid in meta["record_ids"]], dtype=np.float32)
    return meta, x, labels


def build_joint_patient_folds(patient_ids, labels, n_folds=5, seed=2026):
    """Patient-level folds stratified by the observed joint label state."""
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    patient_state = np.zeros(len(patients), dtype=np.int64)
    for pi in range(len(patients)):
        rows = np.where(inverse == pi)[0]
        state = sum(int(np.any(labels[rows, j] > 0.5)) << j
                    for j in range(labels.shape[1]))
        patient_state[pi] = state
    counts = np.bincount(patient_state, minlength=2 ** labels.shape[1])
    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) == 0 or nonzero_counts.min() < n_folds:
        raise ValueError(f"non-empty joint patient states too sparse for {n_folds}-fold: {counts.tolist()}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_pat, val_pat in skf.split(patients, patient_state):
        tr = np.where(np.isin(inverse, train_pat))[0]
        va = np.where(np.isin(inverse, val_pat))[0]
        folds.append((tr, va))
    return folds, counts.tolist()


class ECGDataset(Dataset):
    def __init__(self, x, y, augment_flag=False, seed=2026):
        self.x, self.y = x, np.asarray(y, dtype=np.float32)
        self.augment_flag = augment_flag
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        if self.augment_flag:
            x = augment(x, self.rng)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.y[idx].copy()))


class MorphologyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = XceptionTime(num_classes=2).inner

    def forward(self, x):
        return self.backbone.forward_features(x.transpose(1, 2))


class LeadSpatialExpert(nn.Module):
    """Shared lead encoder followed by GitHub PyG GATConv on a complete graph."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.lead_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 9, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gat = GATConv(64, out_dim, heads=2, concat=False, dropout=0.1,
                           add_self_loops=True)
        edges = torch.cartesian_prod(torch.arange(12), torch.arange(12)).t().contiguous()
        self.register_buffer("edge_index", edges, persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        nodes = x.transpose(1, 2).reshape(b * c, 1, t)
        h = self.lead_encoder(nodes).squeeze(-1)
        offsets = torch.arange(b, device=x.device).repeat_interleave(self.edge_index.shape[1]) * c
        edge = self.edge_index.to(x.device).repeat(1, b) + offsets.repeat(2, 1)
        h = self.gat(h, edge).reshape(b, c, -1)
        return h.mean(dim=1)


def _sym4_mother_wavelet(kernel_size=103):
    """Create the Symlet-4 mother wavelet used by the WavelNet paper.

    The public WavelNet repository supplies the WaveletConv implementation;
    PyWavelets supplies the canonical Symlet-4 prototype.  Resampling is only
    needed because ACS has 1000 samples per record rather than the paper's
    256-sample beat input.
    """
    _, psi, _ = pywt.Wavelet("sym4").wavefun(level=8)
    source_grid = np.linspace(0.0, 1.0, len(psi), dtype=np.float64)
    target_grid = np.linspace(0.0, 1.0, kernel_size, dtype=np.float64)
    mother = np.interp(target_grid, source_grid, np.asarray(psi, dtype=np.float64))
    mother -= mother.mean()
    mother /= np.sqrt(np.sum(mother ** 2)) + 1e-12
    return mother.astype(np.float32)


class FrequencyExpert(nn.Module):
    """End-to-end WavelNet frequency expert (no MVMD/VMD preprocessing).

    The official WaveletConv layer only accepts one input channel.  We therefore
    apply the same 16-filter Symlet-4 WaveletConv independently to all 12 leads,
    mix the resulting 192 channels with a pointwise convolution, and retain the
    paper's Conv--pool hierarchy.  The scale factors in WaveletConv remain
    trainable and are available for post-hoc pseudo-frequency analysis.
    """
    def __init__(self, out_dim=128, n_leads=12, fs=100, kernel_size=103):
        super().__init__()
        mother = _sym4_mother_wavelet(kernel_size)
        self.n_leads = n_leads
        self.n_wavelet_filters = 16
        self.wavelet = WaveletConv(
            in_channels=1,
            out_channels=self.n_wavelet_filters,
            kernel_size=kernel_size,
            fs=fs,
            mother_wavelet=mother,
            fs_wavelet=fs,
            fc_wavelet=40.0,
            a_min=0.2,
            bias=None,
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(n_leads * self.n_wavelet_filters, 16, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        b, t, c = x.shape
        if c != self.n_leads:
            raise ValueError(f"WavelNet frequency expert expects {self.n_leads} leads, got {c}")
        # Official WaveletConv is shared across leads; flatten B*C for one call.
        h = self.wavelet(x.transpose(1, 2).reshape(b * c, 1, t))
        h = h.reshape(b, c * self.n_wavelet_filters, t)
        h = self.encoder(h).squeeze(-1)
        return self.proj(h)


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class KANHead(nn.Module):
    def __init__(self, dim=128, device="cpu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # KAN is used only on the latent task feature, following the plan.
        self.kan = KAN(width=[dim, 16, 1], grid=3, k=3,
                       symbolic_enabled=False, save_act=False, auto_save=False,
                       seed=2026, device=device)

    def forward(self, x):
        return self.kan(self.norm(x)).reshape(-1)


class HMoEKAN(nn.Module):
    def __init__(self, variant, n_tasks=2, head_kind="mlp", device="cpu"):
        super().__init__()
        self.variant = variant
        self.n_tasks = int(n_tasks)
        self.head_kind = head_kind
        self.morph = MorphologyExpert()
        if variant in ("e2",):
            self.moe_experts = nn.ModuleList([
                nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))
                for _ in range(3)
            ])
            self.lead = self.freq = None
        elif variant in ("e3", "e4"):
            self.lead = LeadSpatialExpert(128)
            self.freq = FrequencyExpert(128)
            self.proj_morph = Projection(128, 128)
            self.moe_experts = None
        else:
            self.lead = self.freq = self.proj_morph = self.moe_experts = None

        if variant in ("e2", "e3", "e4"):
            self.routers = nn.ModuleList([
                nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 3))
                for _ in range(self.n_tasks)
            ])
        if variant in ("e0",):
            self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.n_tasks)])
        elif variant in ("e1",):
            self.heads = nn.ModuleList([KANHead(128, device) for _ in range(self.n_tasks)])
        elif variant in ("e2", "e3"):
            self.heads = nn.ModuleList([MLPHead(128) for _ in range(self.n_tasks)])
        elif variant == "e4":
            self.heads = nn.ModuleList([KANHead(128, device) for _ in range(self.n_tasks)])
        else:
            raise ValueError(f"unknown variant {variant}")

    def forward(self, x):
        z_m = self.morph(x)
        gates = None
        if self.variant in ("e0", "e1"):
            z = z_m
        elif self.variant == "e2":
            z_all = torch.stack([e(z_m) for e in self.moe_experts], dim=1)
            all_feat = torch.cat([z_all[:, i] for i in range(3)], dim=1)
            gates = torch.stack([torch.softmax(r(all_feat), dim=-1)
                                 for r in self.routers], dim=1)
            z_tasks = [(gates[:, j].unsqueeze(-1) * z_all).sum(dim=1)
                       for j in range(self.n_tasks)]
        elif self.variant in ("e3", "e4"):
            z_stack = torch.stack([self.proj_morph(z_m), self.lead(x), self.freq(x)], dim=1)
            all_feat = torch.cat([z_stack[:, i] for i in range(3)], dim=1)
            gates = torch.stack([torch.softmax(r(all_feat), dim=-1)
                                 for r in self.routers], dim=1)
            z_tasks = [(gates[:, j].unsqueeze(-1) * z_stack).sum(dim=1)
                       for j in range(self.n_tasks)]
        else:
            raise AssertionError(self.variant)

        if gates is None:
            logits = torch.cat([head(z_m).reshape(-1, 1) for head in self.heads], dim=1)
            return {"logits": logits, "gates": None, "expert_features": {"morph": z_m}}
        logits = torch.cat([head(z_tasks[j]).reshape(-1, 1)
                            for j, head in enumerate(self.heads)], dim=1)
        return {"logits": logits, "gates": gates, "expert_features": {"morph": z_m}}


def choose_thresholds(y, p):
    out = []
    for j in range(y.shape[1]):
        best_t, best = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.01):
            score = f1_score(y[:, j], p[:, j] >= t, zero_division=0)
            if score > best:
                best, best_t = score, float(t)
        out.append(best_t)
    return np.asarray(out)


def metrics(y, p, thresholds):
    result = {}
    for j, name in enumerate(LABELS):
        yt = y[:, j].astype(int)
        pred = (p[:, j] >= thresholds[j]).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        result[name] = {
            "auroc": float(roc_auc_score(yt, p[:, j])) if len(np.unique(yt)) > 1 else float("nan"),
            "auprc": float(average_precision_score(yt, p[:, j])) if len(np.unique(yt)) > 1 else float("nan"),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "sens": float(tp / (tp + fn)) if tp + fn else 0.0,
            "spec": float(tn / (tn + fp)) if tn + fp else 0.0,
            "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
            "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
            "mcc": float(matthews_corrcoef(yt, pred)),
            "acc": float((tp + tn) / max(1, tp + tn + fp + fn)),
            "threshold": float(thresholds[j]),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    result["macro_auroc"] = float(np.nanmean([result[n]["auroc"] for n in LABELS]))
    result["macro_auprc"] = float(np.nanmean([result[n]["auprc"] for n in LABELS]))
    result["macro_f1"] = float(np.nanmean([result[n]["f1"] for n in LABELS]))
    return result


def run_fold(args, variant, fold_idx, train_idx, val_idx, x_raw, labels, device):
    set_seed(args.seed + fold_idx)
    scaler = fit_robust_scaler(x_raw[train_idx])
    x_tr = apply_robust_scaler(x_raw[train_idx], scaler)
    x_va = apply_robust_scaler(x_raw[val_idx], scaler)
    train_ds = ECGDataset(x_tr, labels[train_idx], True, args.seed + fold_idx)
    val_ds = ECGDataset(x_va, labels[val_idx], False, args.seed + fold_idx)
    generator = torch.Generator().manual_seed(args.seed + fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers,
                              generator=generator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = HMoEKAN(variant, n_tasks=len(LABELS),
                    head_kind="kan" if variant in ("e1", "e4") else "mlp",
                    device=str(device)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_score, best_epoch, wait = None, -np.inf, -1, 0
    best_thr = np.full(len(LABELS), 0.5)
    gate_rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out["logits"], yb)
            if out["gates"] is not None:
                mean_gate = out["gates"].mean(dim=(0, 1))
                loss = loss + args.lambda_balance * ((mean_gate - 1.0 / 3.0) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); running += float(loss.item()); nb += 1
        model.eval(); ys, ps, gs = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device, non_blocking=True))
                ys.append(yb.numpy()); ps.append(torch.sigmoid(out["logits"]).cpu().numpy())
                if out["gates"] is not None: gs.append(out["gates"].cpu().numpy())
        yv, pv = np.concatenate(ys), np.concatenate(ps)
        score = float(np.nanmean([roc_auc_score(yv[:, j], pv[:, j])
                                  for j in range(len(LABELS))]))
        thr = choose_thresholds(yv, pv)
        print(json.dumps({"variant": variant, "fold": fold_idx, "epoch": epoch,
                          "train_loss": running / max(1, nb), "val_macro_auroc": score}), flush=True)
        if gs: gate_rows.append(np.concatenate(gs, axis=0))
        if score > best_score:
            best_score, best_epoch, wait, best_thr = score, epoch, 0, thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience: break
    model.load_state_dict(best_state); model.to(device); model.eval()
    ys, ps, gs = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb.to(device, non_blocking=True))
            ys.append(yb.numpy()); ps.append(torch.sigmoid(out["logits"]).cpu().numpy())
            if out["gates"] is not None: gs.append(out["gates"].cpu().numpy())
    yv, pv = np.concatenate(ys), np.concatenate(ps)
    result = metrics(yv, pv, best_thr)
    result["fold"] = fold_idx; result["best_epoch"] = best_epoch
    result["train_sec"] = round(time.time() - t0, 2)
    result["n_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    result["gate_mean"] = np.concatenate(gs).mean(axis=0).tolist() if gs else None
    result["gate_top1"] = np.bincount(np.argmax(np.concatenate(gs), axis=-1).ravel(), minlength=3).tolist() if gs else None
    if variant in ("e3", "e4"):
        scales = model.freq.wavelet.a.detach().cpu().numpy().reshape(-1)
        result["wavelnet_scale"] = {
            "min": float(scales.min()), "max": float(scales.max()),
            "mean": float(scales.mean()), "std": float(scales.std()),
            "n_filters": int(scales.size), "mother_wavelet": "sym4",
            "kernel_size": int(model.freq.wavelet.size_kernel), "fs": 100,
        }
    return result, model.state_dict(), result.get("gate_mean")


def main():
    global LABELS
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["e0", "e1", "e2", "e3", "e4"], required=True)
    p.add_argument("--task", choices=list(TASK_LABELS), default="ami_omi")
    p.add_argument("--out_dir", default="results/hmoe_kan_github")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_balance", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    LABELS = TASK_LABELS[args.task]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta, x_raw, labels = _load_raw_ecg_and_labels(LABELS)
    folds, joint_counts = build_joint_patient_folds(meta["patient_ids"], labels, args.n_folds, args.fold_seed)
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.variant}_split.json").write_text(json.dumps({
        "protocol": "patient_level_joint_stratified_5fold", "variant": args.variant,
        "fold_seed": args.fold_seed, "joint_patient_state_counts": joint_counts,
        "folds": [{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]
    }, indent=2), encoding="utf-8")
    fold_results = []
    for fi, (tr, va) in enumerate(folds):
        result, state, gate = run_fold(args, args.variant, fi, tr, va, x_raw, labels, device)
        fold_results.append(result)
        torch.save(state, out / f"{args.variant}_fold{fi}_seed{args.seed}.pt")
        print("FOLD_RESULT " + json.dumps(result), flush=True)
    summary = {
        "variant": args.variant,
        "task": args.task,
        "labels": LABELS,
        "source_repositories": {
            "wavelnet": "https://github.com/NamhoKim-Kor/WavelNet",
            "pykan": "https://github.com/KindXiaoming/pykan",
            "pytorch_geometric": "https://github.com/pyg-team/pytorch_geometric",
        },
        "folds": fold_results,
        "mean_std": {
            n: {
                m: {
                    "mean": float(np.mean([r[n][m] for r in fold_results])),
                    "std": float(np.std([r[n][m] for r in fold_results])),
                }
                for m in METRICS
            }
            for n in LABELS
        },
    }
    if args.variant in ("e3", "e4"):
        gates = np.asarray([r["gate_mean"] for r in fold_results], dtype=float)
        summary["gate_mean_over_folds"] = gates.mean(axis=0).tolist()
        summary["gate_std_over_folds"] = gates.std(axis=0).tolist()
    (out / f"{args.variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
