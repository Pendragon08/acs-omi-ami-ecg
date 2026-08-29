"""Four-label architecture comparison on the registered ACS cohort.

This runner deliberately reuses the data loader, patient-level folds,
preprocessing and metric/threshold code from the historical E4 experiment.
Only the model is changed.  The four outputs are independent sigmoid logits
for STEMI, NSTEMI, UA and OMI; this is not a mutually-exclusive softmax task.

The ECG-LFM fine-tune is trained by its own official adapter and is included
in the comparison report from its saved evaluation artifacts.  This runner
trains the requested conventional baselines:

* InceptionTime (multi-scale CNN)
* ResNet1D_Wang (residual CNN)
* LSTM (two-layer recurrent baseline)
* ViT-1D (patch vision transformer ported from ACS-Baseline-main)
* Transformer1D (compact PyTorch TransformerEncoder baseline)

All baselines use exactly the same P2 data, historical patient-level
iterative-multilabel five-fold split, fold-fitted robust scaling, train-only
augmentation, optimizer, batch size, epoch budget, early stopping and
threshold selection.  The loss is the same unweighted BCEWithLogitsLoss used
by E4 so that this is an architecture comparison rather than a loss ablation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from experiments import hmoe_kan_github as h  # noqa: E402
from experiments.hmoe_kan_e4_4label_multilabel import (  # noqa: E402
    LABELS,
    METRIC_KEYS,
    build_patient_multilabel_folds,
    load_data,
    metric_table,
    select_thresholds,
    set_seed,
)
from nets.traditional import InceptionTime, LSTM, ResNet1D_Wang, ViT  # noqa: E402


BASELINE_MODELS = ("InceptionTime", "ResNet1D_Wang", "LSTM", "ViT", "Transformer1D")


class Transformer1D(nn.Module):
    """A small sequence Transformer using non-overlapping ECG patches.

    The implementation uses the official PyTorch ``TransformerEncoder``
    building block.  It accepts the project canonical [B, T, C] layout and
    emits four independent logits.
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 12,
        sequence_length: int = 1000,
        patch_size: int = 25,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sequence_length % patch_size:
            raise ValueError("sequence_length must be divisible by patch_size")
        self.patch = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        n_tokens = sequence_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project canonical [B,T,C] to patches [B,N,d_model].
        z = self.patch(x.transpose(1, 2)).transpose(1, 2)
        cls = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls, z), dim=1)
        z = self.encoder(z + self.positional_embedding)
        return self.head(self.norm(z[:, 0]))


def build_baseline_model(name: str) -> nn.Module:
    if name == "InceptionTime":
        return InceptionTime(num_classes=len(LABELS))
    if name == "ResNet1D_Wang":
        return ResNet1D_Wang(num_classes=len(LABELS))
    if name == "LSTM":
        return LSTM(num_classes=len(LABELS))
    if name == "ViT":
        return ViT(num_classes=len(LABELS))
    if name == "Transformer1D":
        return Transformer1D(num_classes=len(LABELS))
    raise ValueError("unknown comparison model: %s" % name)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, dict):
        output = output["logits"]
    if output.ndim != 2 or output.shape[1] != len(LABELS):
        raise RuntimeError("model output must be [batch, 4], got %r" % (tuple(output.shape),))
    return output


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    y_rows: List[np.ndarray] = []
    p_rows: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = _logits(model, xb.to(device, non_blocking=True))
            p_rows.append(torch.sigmoid(logits).cpu().numpy())
            y_rows.append(yb.numpy())
    return np.concatenate(y_rows, axis=0), np.concatenate(p_rows, axis=0)


def _ranking_score(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    auroc = [roc_auc_score(y[:, j], p[:, j]) for j in range(len(LABELS))]
    auprc = [average_precision_score(y[:, j], p[:, j]) for j in range(len(LABELS))]
    return float(np.mean(auroc)), float(np.mean(auprc))


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _source_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_history_csv(path: Path, history: Sequence[dict]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_macro_auroc",
        "val_macro_auprc",
        "learning_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in history)


def run_fold(
    args: argparse.Namespace,
    model_name: str,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_raw: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    device: torch.device,
    out_model: Path,
) -> Tuple[dict, np.ndarray]:
    # The fold seed is common to all architectures.  This fixes augmentation,
    # DataLoader order and deterministic model initialization.
    set_seed(args.seed + fold_idx)
    scaler = h.fit_robust_scaler(x_raw[train_idx])
    x_train = h.apply_robust_scaler(x_raw[train_idx], scaler)
    x_val = h.apply_robust_scaler(x_raw[val_idx], scaler)
    train_dataset = h.ECGDataset(x_train, labels[train_idx], True, args.seed + fold_idx)
    val_dataset = h.ECGDataset(x_val, labels[val_idx], False, args.seed + fold_idx)
    generator = torch.Generator().manual_seed(args.seed + fold_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    fold_dir = out_model / ("fold%d" % fold_idx)
    if args.resume and (fold_dir / "best_model.pt").exists() and (fold_dir / "history.json").exists():
        # A previous worker may have completed training but failed while
        # writing the aggregate summary. Reconstruct its validation
        # probabilities without retraining the fold.
        model = build_baseline_model(model_name).to(device)
        state = torch.load(fold_dir / "best_model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        y_val, p_val = _predict(model, val_loader, device)
        history = json.loads((fold_dir / "history.json").read_text(encoding="utf-8"))
        score_key = "val_macro_auroc" if args.selection_metric == "val_macro_auroc" else "val_macro_auprc"
        best_record = max(history, key=lambda item: float(item[score_key]))
        result = {
            "model": model_name,
            "fold": int(fold_idx),
            "best_epoch": int(best_record["epoch"]),
            "best_val_score": float(best_record[score_key]),
            "selection_metric": args.selection_metric,
            "train_sec": 0.0,
            "resumed_from_checkpoint": True,
            "n_train_records": int(len(train_idx)),
            "n_val_records": int(len(val_idx)),
            "n_train_patients": int(len(np.unique(patient_ids[train_idx]))),
            "n_val_patients": int(len(np.unique(patient_ids[val_idx]))),
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "history": history,
            "ranking_at_best_checkpoint": metric_table(
                y_val, p_val, np.full(len(LABELS), 0.5, dtype=np.float32)
            ),
        }
        return result, p_val
    model = build_baseline_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state = None
    best_score = -np.inf
    best_epoch = -1
    patience_counter = 0
    history: List[dict] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_logits(model, xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        y_val, p_val = _predict(model, val_loader, device)
        val_loss = float(
            nn.functional.binary_cross_entropy(
                torch.from_numpy(np.clip(p_val, 1e-7, 1.0 - 1e-7)),
                torch.from_numpy(y_val),
            ).item()
        )
        val_auroc, val_auprc = _ranking_score(y_val, p_val)
        record = {
            "epoch": int(epoch),
            "train_loss": float(running / max(1, batches)),
            "val_loss": val_loss,
            "val_macro_auroc": val_auroc,
            "val_macro_auprc": val_auprc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "experiment": "model_comparison_4label",
                    "model": model_name,
                    "fold": fold_idx,
                    **record,
                }
            ),
            flush=True,
        )
        score = val_auroc if args.selection_metric == "val_macro_auroc" else val_auprc
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("no checkpoint selected for %s fold %d" % (model_name, fold_idx))
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_val, p_val = _predict(model, val_loader, device)
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, fold_dir / "best_model.pt")
    (fold_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    (fold_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_history_csv(fold_dir / "history.csv", history)
    result = {
        "model": model_name,
        "fold": int(fold_idx),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "selection_metric": args.selection_metric,
        "resume": bool(args.resume),
        "train_sec": round(time.time() - started, 2),
        "n_train_records": int(len(train_idx)),
        "n_val_records": int(len(val_idx)),
        "n_train_patients": int(len(np.unique(patient_ids[train_idx]))),
        "n_val_patients": int(len(np.unique(patient_ids[val_idx]))),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "history": history,
        "ranking_at_best_checkpoint": metric_table(
            y_val, p_val, np.full(len(LABELS), 0.5, dtype=np.float32)
        ),
    }
    return result, p_val


def _mean_std(fold_results: Sequence[dict], key: str) -> dict:
    values = [float(row[key]) for row in fold_results]
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _summarize_model(
    model_name: str,
    out_model: Path,
    labels: np.ndarray,
    record_ids: np.ndarray,
    patient_ids: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    fold_results: List[dict],
    oof_prob: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    threshold_methods = ("fixed_0.5", "youden_j", "max_f1", "sens90", "sens95")
    thresholds: Dict[str, np.ndarray] = {}
    details: Dict[str, dict] = {}
    for method in threshold_methods:
        thresholds[method], details[method] = select_thresholds(labels, oof_prob, method)
    metrics_by_threshold = {
        method: metric_table(labels, oof_prob, thresholds[method])
        for method in threshold_methods
    }
    recommended = "youden_j"
    recommended_thresholds = thresholds[recommended]
    for fold_result, (_, val_idx) in zip(fold_results, folds):
        fold_result["metrics_at_pooled_oof_youden"] = metric_table(
            labels[val_idx], oof_prob[val_idx], recommended_thresholds
        )
    mean_std = {}
    for name in list(LABELS) + ["macro"]:
        mean_std[name] = {}
        for key in METRIC_KEYS:
            values = [
                row["metrics_at_pooled_oof_youden"][name][key]
                for row in fold_results
            ]
            mean_std[name][key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

    oof_pred = (oof_prob >= recommended_thresholds.reshape(1, -1)).astype(np.int8)
    np.savez_compressed(
        out_model / "oof_probabilities.npz",
        labels=labels.astype(np.int8),
        probabilities=oof_prob.astype(np.float32),
        predictions_youden=oof_pred,
        thresholds_youden=recommended_thresholds.astype(np.float32),
    )
    with (out_model / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["row_index", "ecg_row_record", "patient_id"]
            + ["y_" + name for name in LABELS]
            + ["p_" + name for name in LABELS]
            + ["pred_youden_" + name for name in LABELS]
        )
        for index, (rid, pid, y, p, pred) in enumerate(
            zip(record_ids, patient_ids, labels, oof_prob, oof_pred)
        ):
            writer.writerow(
                [index, rid, pid]
                + y.astype(int).tolist()
                + p.tolist()
                + pred.astype(int).tolist()
            )

    summary = {
        "experiment": "ACS-four-label-model-comparison-20260828",
        "model": model_name,
        "task": "four independent multilabel binary outputs",
        "labels": list(LABELS),
        "loss": "BCEWithLogitsLoss (unweighted)",
        "training": {
            "optimizer": "Adam",
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "epochs_max": int(args.epochs),
            "early_stopping_patience": int(args.patience),
            "scheduler": "none",
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "selection_metric": args.selection_metric,
            "augmentation": "existing train-only ECGDataset augment",
        },
        "seed": int(args.seed),
        "fold_seed": int(args.fold_seed),
        "n_folds": int(args.n_folds),
        "device": str(args.device),
        "run_tag": str(args.run_tag),
        "recommended_threshold_method": recommended,
        "thresholds": {method: thresholds[method].tolist() for method in threshold_methods},
        "threshold_selection_details": details,
        "oof_metrics_by_threshold_strategy": metrics_by_threshold,
        "mean_std_youden": mean_std,
        "folds": fold_results,
        "oof_prediction_pattern": metrics_by_threshold[recommended]["prediction_pattern"],
        "source_code_git_revision": _git_revision(),
        "source_code_sha256": _source_sha256(),
        "comparison_source": {
            "traditional_baselines": "ACS-Baseline-main port in nets/traditional.py",
            "transformer": "PyTorch nn.TransformerEncoder",
        },
    }
    (out_model / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_model / "threshold_analysis.json").write_text(
        json.dumps(
            {
                "recommended": recommended,
                "thresholds": {method: thresholds[method].tolist() for method in threshold_methods},
                "details": details,
                "metrics": metrics_by_threshold,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (out_model / "fold_metrics_youden.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "label"] + list(METRIC_KEYS))
        for row in fold_results:
            for label in list(LABELS) + ["macro"]:
                writer.writerow(
                    [row["fold"], label]
                    + [row["metrics_at_pooled_oof_youden"][label][key] for key in METRIC_KEYS]
                )
    _plot_histories(out_model, fold_results, model_name)
    return summary


def _plot_histories(out_model: Path, fold_results: Sequence[dict], model_name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_model / "plot_unavailable.txt").write_text(str(exc), encoding="utf-8")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for row in fold_results:
        history = row["history"]
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["train_loss"] for item in history], alpha=0.7, label="fold%d train" % row["fold"])
        axes[0].plot(epochs, [item["val_loss"] for item in history], "--", alpha=0.7, label="fold%d val" % row["fold"])
        axes[1].plot(epochs, [item["val_macro_auroc"] for item in history], alpha=0.8)
        axes[2].plot(epochs, [item["val_macro_auprc"] for item in history], alpha=0.8)
    axes[0].set_title("Loss")
    axes[1].set_title("Validation macro AUROC")
    axes[2].set_title("Validation macro AUPRC")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(model_name + " four-label training curves")
    fig.tight_layout()
    fig.savefig(out_model / "training_validation_curves.png", dpi=160)
    plt.close(fig)


def run_model(
    args: argparse.Namespace,
    model_name: str,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    x_raw: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    record_ids: np.ndarray,
    device: torch.device,
    out_dir: Path,
) -> dict:
    out_model = out_dir / model_name
    out_model.mkdir(parents=True, exist_ok=True)
    oof_prob = np.zeros_like(labels, dtype=np.float32)
    fold_results: List[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        result, val_prob = run_fold(
            args,
            model_name,
            fold_idx,
            train_idx,
            val_idx,
            x_raw,
            labels,
            patient_ids,
            device,
            out_model,
        )
        oof_prob[val_idx] = val_prob
        fold_results.append(result)
        print(
            "FOLD_RESULT "
            + json.dumps(
                {key: value for key, value in result.items() if key != "history"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary = _summarize_model(
        model_name,
        out_model,
        labels,
        record_ids,
        patient_ids,
        folds,
        fold_results,
        oof_prob,
        args,
    )
    print("MODEL_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(BASELINE_MODELS),
        help="comma-separated subset of " + ",".join(BASELINE_MODELS),
    )
    parser.add_argument("--out_dir", default="results/model_comparison_4label")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument(
        "--selection_metric",
        choices=("val_macro_auroc", "val_macro_auprc"),
        default="val_macro_auroc",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_tag", default="", help="unique tag for per-worker protocol provenance")
    parser.add_argument("--resume", action="store_true", help="reuse completed fold checkpoints")
    args = parser.parse_args()
    if args.n_folds != 5:
        raise ValueError("registered comparison protocol is fixed to five folds")
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in BASELINE_MODELS]
    if unknown:
        raise ValueError("unknown models: " + ",".join(unknown))
    if not model_names:
        raise ValueError("at least one model is required")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, labels, patient_ids, record_ids, label_audit = load_data()
    folds, split_audit = build_patient_multilabel_folds(
        patient_ids, labels, args.n_folds, args.fold_seed
    )
    (out_dir / "label_audit.json").write_text(
        json.dumps(label_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "split.json").write_text(
        json.dumps(
            {
                **split_audit,
                "folds_indices": [
                    {"train": train.tolist(), "val": val.tolist()}
                    for train, val in folds
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    protocol = {
        "experiment": "ACS-four-label-model-comparison-20260828",
        "labels": list(LABELS),
        "task": "four independent multilabel binary outputs",
        "data_loader": "experiments.hmoe_kan_e4_4label_multilabel.load_data",
        "split": "historical patient-level iterative multilabel five-fold, seed 2026",
        "loss": "BCEWithLogitsLoss (unweighted)",
        "optimizer": "Adam",
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs_max": int(args.epochs),
        "early_stopping_patience": int(args.patience),
        "scheduler": "none",
        "selection_metric": args.selection_metric,
        "thresholds": "pooled OOF validation-only grid 0.01-0.99; fixed 0.5 retained",
        "git_revision": _git_revision(),
        "source_code_sha256": _source_sha256(),
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_tag = args.run_tag or "_".join(model_names)
    run_protocol = dict(protocol, models=model_names, device=str(device), run_tag=run_tag)
    (out_dir / ("protocol_%s.json" % run_tag)).write_text(
        json.dumps(run_protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    all_summaries = {}
    for model_name in model_names:
        all_summaries[model_name] = run_model(
            args,
            model_name,
            folds,
            x_raw,
            labels,
            patient_ids,
            record_ids,
            device,
            out_dir,
        )
    # Keep a worker-specific aggregate as well as the per-model summaries.
    # Two GPU workers may finish concurrently, so a single shared JSON file
    # would otherwise be vulnerable to last-writer-wins truncation.
    run_tag = args.run_tag or "_".join(model_names)
    (out_dir / ("all_model_summaries_%s.json" % run_tag)).write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    merged_path = out_dir / "all_model_summaries.json"
    merged = {}
    if merged_path.exists():
        try:
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(all_summaries)
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
