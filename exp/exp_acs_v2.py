"""Leakage-controlled ACS OMI/AMI training protocol v2.

Stages are patient-disjoint:
  fit          -> optimizer updates and all learned preprocessing parameters
  selection    -> early stopping by AUPRC only
  calibration  -> Platt calibration and threshold selection only
  evaluation   -> one final evaluation after every choice is frozen

This module deliberately writes to results/protocol_v2 and never overwrites
historical experiment artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

from acs_utils.seed import set_seed  # noqa: E402
from nets import build_model  # noqa: E402
from preprocessing.augment import augment as augment_ecg  # noqa: E402


GRID = np.arange(1, 100, dtype=np.float64) / 100.0
TASKS = {"omi": (0, "OMI"), "ami": (1, "AMI")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reserve_evaluation_once(task, split_sha256, run_id, checkpoint_sha256):
    """Atomically prevent a second access to the same task/split evaluation set."""
    registry_dir = ROOT / "results" / "evaluation_access_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    marker = registry_dir / f"ACS_PROTOCOL_V2_{split_sha256}_{task}.json"
    payload = {
        "protocol": "ACS_PROTOCOL_V2",
        "task": task,
        "split_sha256": split_sha256,
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "immutable exclusive marker; existence forbids another evaluation access",
    }
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(
            f"evaluation access refused: immutable marker already exists: {marker}"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return marker


def assert_evaluation_available(task, split_sha256):
    marker = (
        ROOT / "results" / "evaluation_access_registry" /
        f"ACS_PROTOCOL_V2_{split_sha256}_{task}.json"
    )
    if marker.exists():
        raise RuntimeError(
            f"evaluation access refused before training: immutable marker exists: {marker}"
        )


class ProtocolV2Dataset(Dataset):
    def __init__(
        self,
        indices,
        label_col,
        median,
        scale,
        clip_abs,
        augment,
        seed,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.array = np.load(ROOT / "data" / "arrays" / "P2_train.npy", mmap_mode="r")
        meta = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
        self.labels = np.asarray(meta["labels"][:, [label_col]], dtype=np.float32)
        self.median = float(median)
        self.scale = float(scale)
        self.clip_abs = float(clip_abs)
        self.augment_flag = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        global_idx = int(self.indices[local_idx])
        x = np.asarray(self.array[global_idx], dtype=np.float32).copy()
        x = (x - self.median) / (self.scale + 1e-8)
        x = np.clip(x, -self.clip_abs, self.clip_abs).astype(np.float32, copy=False)
        if self.augment_flag:
            # Record/epoch-derived RNG is independent of worker scheduling and
            # changes every epoch.  No copied mutable Generator exists.
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, global_idx])
            )
            x = augment_ecg(x, rng=rng)
        y = self.labels[global_idx]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(y.copy()),
            global_idx,
        )


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        pin_memory=True,
        persistent_workers=False,
    )


def predict(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            logits = model(x.to(device, non_blocking=True))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ids.append(idx.numpy())
    return (
        np.concatenate(ys, axis=0)[:, 0].astype(int),
        np.concatenate(ps, axis=0)[:, 0].astype(float),
        np.concatenate(ids, axis=0).astype(int),
    )


def safe_ranking_metrics(y, p):
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def point_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    joint = 2 * f1 * sens / (f1 + sens) if f1 + sens else 0.0
    return {
        "threshold": float(threshold),
        "sens": float(sens),
        "spec": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "acc": float(acc),
        "f1": float(f1),
        "joint_f1_sens_hmean": float(joint),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_positive": int(np.sum(y)), "n_negative": int(len(y) - np.sum(y)),
    }


def threshold_table(y, p):
    rows = [point_metrics(y, p, float(t)) for t in GRID]
    # Ascending grid plus np.argmax means lower threshold wins exact ties.
    joint_idx = int(np.argmax([r["joint_f1_sens_hmean"] for r in rows]))
    f1_idx = int(np.argmax([r["f1"] for r in rows]))
    return rows, {
        "criterion": "max_harmonic_mean_f1_sensitivity",
        "selected": rows[joint_idx],
        "max_f1_reference": rows[f1_idx],
    }


def fit_platt(y, p):
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def apply_platt(p, params):
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return expit(params["intercept"] + params["slope"] * logits)


def calibration_metrics(y, p, n_bins=15):
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bins == i
        if np.any(mask):
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"brier": brier, "ece_15_equal_width": float(ece)}


def write_predictions(path, ids, y, p_raw, p_cal, threshold, label, meta):
    record_ids = np.asarray(meta["record_ids"]).astype(str)
    patient_ids = np.asarray(meta["patient_ids"]).astype(str)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_index", "record_id", "patient_id", label,
            "probability_raw", "probability_calibrated", "prediction",
        ])
        for idx, yt, pr, pc in zip(ids, y, p_raw, p_cal):
            writer.writerow([
                int(idx), record_ids[idx], patient_ids[idx], int(yt),
                f"{pr:.9f}", f"{pc:.9f}", int(pc >= threshold),
            ])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", default="XceptionTime")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exp_name", default="protocol_v2")
    parser.add_argument(
        "--skip_evaluation", action="store_true",
        help="Do not construct or access evaluation data (required for smoke/development runs).",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    col, label = TASKS[args.task]
    protocol_dir = ROOT / "data" / "protocol_v2"
    split_path = protocol_dir / "protocol_v2_split.json"
    prep_path = protocol_dir / "protocol_v2_preprocessing.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    split_sha256 = sha256_file(split_path)
    if not args.skip_evaluation:
        assert_evaluation_available(label, split_sha256)
    affine = prep["fit_only_affine_in_P2_space"]
    clip_abs = prep["fit_only_clip"]["clip_abs"]

    datasets = {
        "fit": ProtocolV2Dataset(
            split["fit"], col, affine["median"], affine["scale_1p4826_mad"],
            clip_abs, True, args.seed,
        ),
        "selection": ProtocolV2Dataset(
            split["selection"], col, affine["median"], affine["scale_1p4826_mad"],
            clip_abs, False, args.seed,
        ),
        "calibration": ProtocolV2Dataset(
            split["calibration"], col, affine["median"], affine["scale_1p4826_mad"],
            clip_abs, False, args.seed,
        ),
    }
    if not args.skip_evaluation:
        datasets["evaluation"] = ProtocolV2Dataset(
            split["evaluation"], col, affine["median"], affine["scale_1p4826_mad"],
            clip_abs, False, args.seed,
        )
    loaders = {
        "fit": make_loader(datasets["fit"], args.batch_size, True, args.seed, args.num_workers),
        "selection": make_loader(datasets["selection"], args.batch_size, False, args.seed, args.num_workers),
        "calibration": make_loader(datasets["calibration"], args.batch_size, False, args.seed, args.num_workers),
    }
    if not args.skip_evaluation:
        loaders["evaluation"] = make_loader(
            datasets["evaluation"], args.batch_size, False, args.seed, args.num_workers
        )

    model = build_model(args.model, num_class=1, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    patience_count = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        datasets["fit"].set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, y, _ in loaders["fit"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            loss_sum += float(loss.item()) * len(x)
            n_seen += len(x)

        y_selection, p_selection_raw, _ = predict(model, loaders["selection"], device)
        ranking = safe_ranking_metrics(y_selection, p_selection_raw)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "selection_auroc": ranking["auroc"],
            "selection_auprc": ranking["auprc"],
        }
        history.append(row)
        print(
            f"[{label}] epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"selection_AUROC={ranking['auroc']:.6f} selection_AUPRC={ranking['auprc']:.6f}",
            flush=True,
        )
        score = ranking["auprc"]
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[{label}] early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                break

    train_time = time.time() - start
    if best_state is None:
        raise RuntimeError("no best checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device).eval()

    result_dir = ROOT / "results" / args.exp_name
    checkpoint_dir = result_dir / "checkpoints"
    prediction_dir = result_dir / "predictions"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{label}_bce_P4_{args.exp_name}_seed{args.seed}"

    checkpoint_path = checkpoint_dir / f"{tag}.pt"
    torch.save(best_state, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    y_selection, p_selection_raw, ids_selection = predict(model, loaders["selection"], device)
    y_cal, p_cal_raw, ids_cal = predict(model, loaders["calibration"], device)
    platt = fit_platt(y_cal, p_cal_raw)
    p_cal = apply_platt(p_cal_raw, platt)
    sweep_rows, threshold_selection = threshold_table(y_cal, p_cal)
    threshold = threshold_selection["selected"]["threshold"]

    evaluation_marker = None
    if args.skip_evaluation:
        y_eval = p_eval_raw = p_eval = ids_eval = None
    else:
        # Calibration and operating point are frozen before this exclusive access.
        evaluation_marker = reserve_evaluation_once(
            label, split_sha256, tag, checkpoint_sha256
        )
        y_eval, p_eval_raw, ids_eval = predict(model, loaders["evaluation"], device)
        p_eval = apply_platt(p_eval_raw, platt)

    history_path = result_dir / f"{tag}_epochs.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    sweep_path = result_dir / f"{tag}_threshold_grid.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    meta = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
    selection_pred_path = prediction_dir / f"{tag}_selection.csv"
    cal_pred_path = prediction_dir / f"{tag}_calibration.csv"
    eval_pred_path = prediction_dir / f"{tag}_evaluation.csv"
    write_predictions(selection_pred_path, ids_selection, y_selection, p_selection_raw,
                      apply_platt(p_selection_raw, platt), threshold, label, meta)
    write_predictions(cal_pred_path, ids_cal, y_cal, p_cal_raw, p_cal, threshold, label, meta)
    if not args.skip_evaluation:
        write_predictions(eval_pred_path, ids_eval, y_eval, p_eval_raw, p_eval, threshold, label, meta)

    peak_mem = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if device.startswith("cuda") else 0.0
    )
    result = {
        "protocol": "ACS_PROTOCOL_V2",
        "run_id": tag,
        "task": label,
        "model": args.model,
        "loss": "BCEWithLogitsLoss",
        "variant": "P4",
        "selection_contract": {
            "fit": "optimizer updates and learned preprocessing",
            "selection": "AUPRC early stopping only",
            "calibration": "Platt calibration and threshold selection only",
            "evaluation": (
                "skipped by development/smoke contract" if args.skip_evaluation
                else "one-shot evaluation after choices frozen"
            ),
            "early_stopping_metric": "selection_AUPRC",
            "threshold_metric": "harmonic_mean(F1,Sensitivity)",
            "historical_caveat": (
                "evaluation patients were used by pre-v2 historical experiments; "
                "this run is protocol-correct prospectively but not a pristine confirmation"
            ),
        },
        "config": {
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "patience": args.patience,
            "clip_grad_norm": args.clip_grad,
            "num_workers": args.num_workers,
            "device_argument": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "skip_evaluation": args.skip_evaluation,
        },
        "data": {
            "split_stats": split["stats"],
            "split_sha256": split_sha256,
            "preprocessing_sha256": sha256_file(prep_path),
            "preprocessing": prep,
        },
        "model_info": {
            "n_trainable_parameters": int(n_params),
            "best_epoch": int(best_epoch),
            "train_time_seconds": float(train_time),
            "peak_gpu_memory_mb": peak_mem,
        },
        "calibration_model": {"method": "Platt_logistic", **platt},
        "threshold_selection": threshold_selection,
        "metrics": {
            "selection": {
                "ranking_raw": safe_ranking_metrics(y_selection, p_selection_raw),
                "ranking_calibrated_for_reference": safe_ranking_metrics(
                    y_selection, apply_platt(p_selection_raw, platt)
                ),
            },
            "calibration": {
                "ranking_raw": safe_ranking_metrics(y_cal, p_cal_raw),
                "ranking_calibrated": safe_ranking_metrics(y_cal, p_cal),
                "calibration_raw": calibration_metrics(y_cal, p_cal_raw),
                "calibration_after_platt": calibration_metrics(y_cal, p_cal),
                "operating_point": point_metrics(y_cal, p_cal, threshold),
            },
            "evaluation": None if args.skip_evaluation else {
                "ranking_raw": safe_ranking_metrics(y_eval, p_eval_raw),
                "ranking_calibrated": safe_ranking_metrics(y_eval, p_eval),
                "calibration_raw": calibration_metrics(y_eval, p_eval_raw),
                "calibration_after_platt": calibration_metrics(y_eval, p_eval),
                "operating_point": point_metrics(y_eval, p_eval, threshold),
            },
        },
        "history": history,
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
            "epoch_csv": str(history_path.relative_to(ROOT)),
            "epoch_csv_sha256": sha256_file(history_path),
            "threshold_grid_csv": str(sweep_path.relative_to(ROOT)),
            "threshold_grid_csv_sha256": sha256_file(sweep_path),
            "selection_predictions": str(selection_pred_path.relative_to(ROOT)),
            "selection_predictions_sha256": sha256_file(selection_pred_path),
            "calibration_predictions": str(cal_pred_path.relative_to(ROOT)),
            "calibration_predictions_sha256": sha256_file(cal_pred_path),
        },
        "code_hashes": {
            "exp/exp_acs_v2.py": sha256_file(Path(__file__)),
            "preprocessing/build_protocol_v2.py": sha256_file(ROOT / "preprocessing" / "build_protocol_v2.py"),
            "nets/cnn_variants.py": sha256_file(ROOT / "nets" / "cnn_variants.py"),
            "preprocessing/augment.py": sha256_file(ROOT / "preprocessing" / "augment.py"),
        },
    }
    if not args.skip_evaluation:
        result["artifacts"].update({
            "evaluation_predictions": str(eval_pred_path.relative_to(ROOT)),
            "evaluation_predictions_sha256": sha256_file(eval_pred_path),
            "evaluation_access_marker": str(evaluation_marker.relative_to(ROOT)),
            "evaluation_access_marker_sha256": sha256_file(evaluation_marker),
        })
    result_path = result_dir / f"{tag}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "result": str(result_path),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "calibration": result["metrics"]["calibration"],
        "evaluation": result["metrics"]["evaluation"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
