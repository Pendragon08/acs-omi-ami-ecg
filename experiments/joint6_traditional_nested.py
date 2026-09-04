"""Nested patient-level Joint-6 comparison for established traditional models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.acs_joint_task import (
    save_confusion_figures,
    save_curves,
    save_reliability_figures,
    save_roc_pr_curves,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint6_fold import (
    metric_bundle,
    save_joint6_confusion,
    write_completion_receipt,
    write_metric_artifacts,
)
from experiments.ecg_lfm.joint6 import JOINT6_CLASSES, encode_source_labels, marginalize_joint6_probabilities
from experiments.ecg_lfm.evaluate_joint import _read_csv
from nets.traditional import InceptionTime, LSTM, ResNet1D_Wang, ViT
from preprocessing.augment import augment
from preprocessing.robust_scaler import apply_robust_scaler, fit_robust_scaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("InceptionTime", "ResNet1D_Wang", "LSTM", "Transformer1D", "ViT")


class ECGDataset(Dataset):
    """Minimal comparison dataset without importing unused HMoE/PyG code."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment_flag: bool, seed: int):
        self.x = x
        self.y = np.asarray(y, dtype=np.int64)
        self.augment_flag = augment_flag
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.array(self.x[index], dtype=np.float32, copy=True)
        if self.augment_flag:
            x = augment(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.y[index], dtype=torch.long)


class Transformer1D(nn.Module):
    """The exact compact Transformer architecture used in the prior comparison."""

    def __init__(self, num_classes: int = 6, input_channels: int = 12,
                 sequence_length: int = 1000, patch_size: int = 25,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch = nn.Conv1d(input_channels, d_model, patch_size, patch_size)
        n_tokens = sequence_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x.transpose(1, 2)).transpose(1, 2)
        z = torch.cat((self.cls_token.expand(z.shape[0], -1, -1), z), dim=1)
        return self.head(self.norm(self.encoder(z + self.positional_embedding)[:, 0]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str) -> nn.Module:
    if name == "InceptionTime":
        return InceptionTime(num_classes=6)
    if name == "ResNet1D_Wang":
        return ResNet1D_Wang(num_classes=6)
    if name == "LSTM":
        return LSTM(num_classes=6)
    if name == "Transformer1D":
        return Transformer1D(num_classes=6)
    if name == "ViT":
        return ViT(num_classes=6)
    raise ValueError(name)


def model_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    output = output["logits"] if isinstance(output, dict) else output
    if output.ndim != 2 or output.shape[1] != 6:
        raise RuntimeError(f"expected [B,6] logits, got {tuple(output.shape)}")
    return output


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    targets, probabilities = [], []
    total_loss = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            output = model_logits(model, xb.to(device, non_blocking=True)).float()
            target = yb.long().to(device, non_blocking=True)
            total_loss += float(F.cross_entropy(output, target).item())
            batches += 1
            targets.append(target.cpu().numpy())
            probabilities.append(torch.softmax(output, dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities), total_loss / max(1, batches)


def assignment_indices(
    assignment_path: Path,
    record_to_index: dict[str, int],
    patient_ids: np.ndarray,
    acs: np.ndarray,
    omi: np.ndarray,
    expected_fold: int,
) -> dict[str, np.ndarray]:
    rows = _read_csv(assignment_path)
    output: dict[str, list[int]] = {"inner_train": [], "inner_validation": [], "outer_test": []}
    seen = set()
    for row in rows:
        if row["subset"] not in output:
            raise ValueError(f"unknown assignment subset: {row['subset']}")
        if int(row["fold"]) != expected_fold:
            raise ValueError(f"assignment fold mismatch: expected {expected_fold}, got {row['fold']}")
        record_id = str(row["record_id"]).zfill(5)
        index = record_to_index.get(record_id)
        if index is None:
            raise ValueError(f"assignment record not found in P2 cache: {record_id}")
        if record_id in seen:
            raise ValueError(f"duplicate assignment record: {record_id}")
        seen.add(record_id)
        if str(patient_ids[index]) != row["patient_id"]:
            raise ValueError(f"patient mismatch: {record_id}")
        expected_acs = ("OTHER", "STEMI", "NSTEMI", "UA")[int(acs[index])]
        if row["acs"] != expected_acs or int(row["omi"]) != int(omi[index]):
            raise ValueError(f"label mismatch: {record_id}")
        output[row["subset"]].append(index)
    if len(seen) != len(record_to_index):
        raise ValueError("assignment does not cover the complete training cohort")
    subsets = {key: np.asarray(value, dtype=np.int64) for key, value in output.items()}
    for left, right in (("inner_train", "inner_validation"), ("inner_train", "outer_test"), ("inner_validation", "outer_test")):
        if np.intersect1d(np.unique(patient_ids[subsets[left]]), np.unique(patient_ids[subsets[right]])).size:
            raise ValueError(f"patient leakage: {left}/{right}")
    return subsets


def prediction_rows(
    indices: np.ndarray,
    q: np.ndarray,
    records: np.ndarray,
    patients: np.ndarray,
    acs: np.ndarray,
    omi: np.ndarray,
    joint: np.ndarray,
    exception: np.ndarray,
    fold: int,
    subset: str,
) -> list[dict]:
    marginal = marginalize_joint6_probabilities(q)
    rows = []
    for position, index in enumerate(indices):
        rows.append({
            "patient_id": str(patients[index]),
            "record_id": str(records[index]).zfill(5),
            "fold": fold,
            "subset": subset,
            "raw_acs_true": int(acs[index]),
            "raw_omi_true": int(omi[index]),
            "joint_true": int(joint[index]),
            "joint_pred": int(q[position].argmax()),
            "joint6_true": int(joint[index]),
            "joint6_pred": int(q[position].argmax()),
            "joint_label_exception": bool(exception[index]),
            **{f"p_{name.lower().replace('+', '_').replace('/', '_')}": float(q[position, k]) for k, name in enumerate(JOINT6_CLASSES)},
            "p_other": float(marginal["acs"][position, 0]),
            "p_stemi": float(marginal["acs"][position, 1]),
            "p_nstemi": float(marginal["acs"][position, 2]),
            "p_ua": float(marginal["acs"][position, 3]),
            "p_omi": float(marginal["omi"][position]),
            "p_nste": float(marginal["nste"][position]),
            "p_nste_omi": float(marginal["nste_omi"][position]),
            "p_omi_given_nste": float(marginal["omi_given_nste"][position]),
        })
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--output_root", default="results/joint6_cv/traditional")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_train_batches", type=int, default=0, help="smoke-only cap; 0 uses every batch")
    args = parser.parse_args()
    runtime_evaluator_sha = sha256(Path(__import__(
        "experiments.ecg_lfm.evaluate_joint6_fold", fromlist=["__file__"]
    ).__file__))
    runtime_runner_sha = sha256(Path(__file__))

    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("traditional Joint-6 runs require exactly one CUDA_VISIBLE_DEVICES value")
    set_seed(args.seed + args.fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # Initialise the selected logical device before resetting its counters;
        # PyTorch 2.5 otherwise raises ``Invalid device argument`` here.
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    x_raw, source, patients, records, label_audit = __import__(
        "experiments.acs_joint_task", fromlist=["load_data"]
    ).load_data()
    acs, omi, joint, exception = encode_source_labels(source)
    record_to_index = {str(record).zfill(5): index for index, record in enumerate(records)}
    if len(record_to_index) != len(records):
        raise ValueError("duplicate P2 record IDs")
    manifest_root = ROOT / args.manifest_root
    fold_dir = manifest_root / f"fold{args.fold}"
    subsets = assignment_indices(
        fold_dir / "patient_assignments.csv", record_to_index, patients, acs, omi, args.fold
    )

    scaler = fit_robust_scaler(x_raw[subsets["inner_train"]])
    arrays = {
        subset: apply_robust_scaler(x_raw[indices], scaler)
        for subset, indices in subsets.items()
    }
    loaders = {
        "inner_train": DataLoader(
            ECGDataset(arrays["inner_train"], joint[subsets["inner_train"]], True, args.seed + args.fold),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed + args.fold), pin_memory=True,
        ),
        "inner_validation": DataLoader(
            ECGDataset(arrays["inner_validation"], joint[subsets["inner_validation"]], False, args.seed + args.fold),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        ),
        "outer_test": DataLoader(
            ECGDataset(arrays["outer_test"], joint[subsets["outer_test"]], False, args.seed + args.fold),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        ),
    }

    model = build_model(args.model).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_loss = math.inf
    best_epoch = -1
    wait = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch_index, (xb, yb) in enumerate(loaders["inner_train"]):
            xb = xb.to(device, non_blocking=True)
            yb = yb.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model_logits(model, xb).float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if args.max_train_batches and batch_index + 1 >= args.max_train_batches:
                break
        _, q_val, val_loss = predict(model, loaders["inner_validation"], device)
        record = {
            "epoch": epoch,
            "train_total_loss": train_loss / max(1, batches),
            "val_total_loss": val_loss,
            "val_joint6_accuracy": float((q_val.argmax(axis=1) == joint[subsets["inner_validation"]]).mean()),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps({"model": args.model, "fold": args.fold, **record}), flush=True)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    _, q_inner, _ = predict(model, loaders["inner_validation"], device)
    _, q_outer, _ = predict(model, loaders["outer_test"], device)

    output = ROOT / args.output_root / args.model / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "best_model.pt")
    (output / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    inner_rows = prediction_rows(subsets["inner_validation"], q_inner, records, patients, acs, omi, joint, exception, args.fold, "inner_validation")
    outer_rows = prediction_rows(subsets["outer_test"], q_outer, records, patients, acs, omi, joint, exception, args.fold, "outer_test")
    write_rows(output / "inner_validation_predictions.csv", inner_rows)
    write_rows(output / "outer_test_predictions.csv", outer_rows)
    y_inner_omi = omi[subsets["inner_validation"]]
    p_inner_omi = np.asarray([row["p_omi"] for row in inner_rows])
    thresholds, details = {}, {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[method], details[method] = select_omi_threshold(y_inner_omi, p_inner_omi, method)
    metrics = {
        "task": "observed six-class ACS subtype x OMI joint softmax",
        "protocol": "patient-disjoint nested five-fold; record-level evaluation",
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed + args.fold,
        "checkpoint_criterion": "minimum inner-validation Joint-6 CE",
        "threshold_source": "inner-validation only; frozen for outer-test",
        "thresholds": thresholds,
        "threshold_details_inner_validation": details,
        "model_stats": {
            "total_parameters": int(total_params),
            "trainable_parameters": int(total_params),
            "trainable_percent": 100.0,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
            "training_seconds": float(time.time() - started),
            "best_epoch": best_epoch,
        },
        "training_config": vars(args),
        "label_audit": label_audit,
        "inner_validation": metric_bundle(inner_rows, thresholds),
        "outer_test": metric_bundle(outer_rows, thresholds),
    }
    bundle = metrics["outer_test"]
    youden = bundle["omi"]["youden_j"]
    save_confusion_figures(
        output,
        np.asarray(bundle["acs"]["confusion_matrix"]),
        np.asarray(bundle["conditional_acs"]["confusion_matrix"]),
        np.asarray([[youden["tn"], youden["fp"]], [youden["fn"], youden["tp"]]]),
    )
    save_joint6_confusion(output / "joint6_confusion_matrix.png", bundle["joint6"]["confusion_matrix"])
    save_roc_pr_curves(output, acs[subsets["outer_test"]], np.asarray([[row[f"p_{name}"] for name in ("other", "stemi", "nstemi", "ua")] for row in outer_rows]), omi[subsets["outer_test"]], np.asarray([row["p_omi"] for row in outer_rows]))
    save_reliability_figures(output, bundle["calibration_uncalibrated"]["acs"], bundle["calibration_uncalibrated"]["omi"])
    save_curves(output, [history], f"Joint-6 {args.model} fold {args.fold}")
    provenance = {
        "status": "pass",
        "cuda_visible_devices": visible,
        "patient_overlap": 0,
        "outer_test_used_for_selection": False,
        "external_test_accessed": False,
        "evaluator_sha256": runtime_evaluator_sha,
        "runner_sha256": runtime_runner_sha,
    }
    write_metric_artifacts(output, metrics)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    # Provenance is the validated completion marker and must be written last.
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_completion_receipt(output, {
        "method": args.model,
        "fold": args.fold,
        "evaluator_sha256": runtime_evaluator_sha,
        "runner_sha256": runtime_runner_sha,
        "checkpoint_sha256": sha256(output / "best_model.pt"),
        "assignment_sha256": sha256(fold_dir / "patient_assignments.csv"),
        "provenance_sha256": sha256(output / "provenance.json"),
    })
    print(json.dumps({"status": "pass", "metrics": str(output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
