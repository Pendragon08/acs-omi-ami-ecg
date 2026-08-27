"""Frozen historical-protocol loss and feature ablation runner.

Loss study: XceptionTime + P3, with only the criterion changed.
Feature study: one fixed XceptionTimeFusion architecture, with modalities
masked at the input so model capacity and initialization remain identical.
"""
import argparse
import csv
import hashlib
import json
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent

from acs_utils.seed import set_seed  # noqa: E402
from data_provider.acs_dataset import ACSDataset  # noqa: E402
from losses.focal import ClassBalancedFocalLoss, FocalLoss  # noqa: E402
from nets import build_model  # noqa: E402


TASK_MAP = {"omi": (0, "OMI"), "ami": (1, "AMI")}
FEATURE_MODES = (
    "raw",
    "median",
    "clin",
    "raw_age",
    "raw_sex",
    "raw_clin",
    "raw_median",
    "all",
)
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
LEGACY_GRID = np.arange(0.05, 0.95, 0.01)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(task):
    label_col, label_name = TASK_MAP[task]
    split_path = ROOT / "data/splits/patient_split.json"
    meta_path = ROOT / "data/arrays/meta_train.npz"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    meta = np.load(meta_path, allow_pickle=True)
    labels = meta["labels"][:, label_col].astype(np.float32)[:, None]
    tr = np.asarray(split["train"], dtype=int)
    va = np.asarray(split["val"], dtype=int)
    n_pos = int(labels[tr].sum())
    n_neg = int(len(tr) - n_pos)
    return label_name, split, meta, labels, tr, va, n_neg, n_pos, split_path


def make_loaders(args, split, labels):
    arr_path = ROOT / "data/arrays/P2_train.npy"
    fusion = args.study == "feature"
    kwargs = {}
    if fusion:
        kwargs = {
            "med_path": ROOT / "data/arrays/med_P2_train.npy",
            "clin_path": ROOT / "data/arrays/clin_train.npy",
        }
    train_ds = ACSDataset(
        arr_path,
        labels=labels,
        augment=True,
        rng_seed=args.seed,
        **kwargs,
    )
    val_ds = ACSDataset(arr_path, labels=labels, augment=False, **kwargs)
    train_ds = torch.utils.data.Subset(train_ds, split["train"])
    val_ds = torch.utils.data.Subset(val_ds, split["val"])
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_criterion(args, n_neg, n_pos, device):
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss(), {"name": "BCEWithLogitsLoss"}
    if args.loss == "wbce":
        pos_weight = n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )
        return criterion, {
            "name": "BCEWithLogitsLoss",
            "pos_weight": pos_weight,
            "weight_source": "training_set_n_negative/n_positive",
        }
    if args.loss == "focal":
        alpha = args.focal_alpha
        alpha_source = "explicit"
        if args.focal_alpha_mode == "ratio":
            alpha = n_neg / (n_neg + n_pos)
            alpha_source = "training_set_negative_fraction"
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha)
        return criterion, {
            "name": "FocalLoss",
            "gamma": args.gamma,
            "alpha": alpha,
            "alpha_source": alpha_source,
        }
    if args.loss == "cb_focal":
        criterion = ClassBalancedFocalLoss(
            n_negative=n_neg,
            n_positive=n_pos,
            beta=args.beta,
            gamma=args.gamma,
        ).to(device)
        return criterion, {
            "name": "ClassBalancedFocalLoss",
            "beta": args.beta,
            "gamma": args.gamma,
            "class_weights_negative_positive": criterion.class_weights.detach().cpu().tolist(),
            "weight_source": "effective_number_training_set_only",
            "weight_normalization": "empirical_expected_weight_equals_one",
        }
    raise ValueError(args.loss)


def prepare_batch(batch, args, device):
    if args.study == "loss":
        x, y = batch
        return (x.to(device),), y.to(device)

    x, med, clin, y = batch
    x = x.to(device)
    med = med.to(device)
    clin = clin.to(device)
    y = y.to(device)
    mode = args.feature_mode

    raw_enabled = mode in {"raw", "raw_age", "raw_sex", "raw_clin", "raw_median", "all"}
    median_enabled = mode in {"median", "raw_median", "all"}
    if not raw_enabled:
        x = torch.zeros_like(x)
    if not median_enabled:
        med = None

    if mode in {"clin", "raw_clin", "all"}:
        pass
    elif mode == "raw_age":
        clin = torch.stack((torch.zeros_like(clin[:, 0]), clin[:, 1]), dim=1)
    elif mode == "raw_sex":
        clin = torch.stack((clin[:, 0], torch.zeros_like(clin[:, 1])), dim=1)
    else:
        clin = torch.zeros_like(clin)
    return (x, med, clin), y


def forward_model(model, inputs, study):
    if study == "loss":
        return model(inputs[0])
    return model(inputs[0], med=inputs[1], clin=inputs[2])


def predict(model, loader, criterion, args):
    model.eval()
    ys, ps = [], []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            inputs, y = prepare_batch(batch, args, args.device)
            out = forward_model(model, inputs, args.study)
            loss = criterion(out, y)
            batch_n = y.shape[0]
            loss_sum += float(loss.item()) * batch_n
            n += batch_n
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(ys)[:, 0], np.concatenate(ps)[:, 0], loss_sum / max(n, 1)


def threshold_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float((tp + tn) / len(y)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def choose_threshold(rows, key, maximize=True):
    values = np.asarray([row[key] for row in rows], dtype=float)
    idx = int(np.nanargmax(values) if maximize else np.nanargmin(values))
    return rows[idx]


def threshold_analysis(y, p):
    rows = []
    for threshold in THRESHOLD_GRID:
        row = threshold_metrics(y, p, threshold)
        row["youden"] = row["sensitivity"] + row["specificity"] - 1.0
        denom = row["f1"] + row["sensitivity"]
        row["f1_sensitivity_hmean"] = (
            2.0 * row["f1"] * row["sensitivity"] / denom if denom else 0.0
        )
        rows.append(row)

    def sensitivity_target(target):
        eligible = [row for row in rows if row["sensitivity"] >= target]
        if not eligible:
            return None
        return max(eligible, key=lambda row: (row["specificity"], row["f1"], row["threshold"]))

    legacy_rows = [threshold_metrics(y, p, float(t)) for t in LEGACY_GRID]
    selected = {
        "fixed_0.5": threshold_metrics(y, p, 0.5),
        "legacy_max_f1_0.05_0.94": choose_threshold(legacy_rows, "f1"),
        "max_f1_0.01_0.99": choose_threshold(rows, "f1"),
        "max_youden_0.01_0.99": choose_threshold(rows, "youden"),
        "max_f1_sensitivity_hmean_0.01_0.99": choose_threshold(rows, "f1_sensitivity_hmean"),
        "sensitivity_at_least_0.90": sensitivity_target(0.90),
        "sensitivity_at_least_0.95": sensitivity_target(0.95),
    }
    return rows, selected


def train(args):
    set_seed(args.seed)
    label_name, split, meta, labels, tr, va, n_neg, n_pos, split_path = load_protocol(args.task)
    train_loader, val_loader = make_loaders(args, split, labels)
    model_name = "XceptionTime" if args.study == "loss" else "XceptionTimeFusion"
    model = build_model(model_name, num_class=1, device=args.device).to(args.device)
    criterion, loss_config = make_criterion(args, n_neg, n_pos, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auroc = -np.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            inputs, y = prepare_batch(batch, args, args.device)
            optimizer.zero_grad()
            out = forward_model(model, inputs, args.study)
            loss = criterion(out, y)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        y_val, p_val, val_loss = predict(model, val_loader, criterion, args)
        auroc = float(roc_auc_score(y_val, p_val))
        auprc = float(average_precision_score(y_val, p_val))
        f1_at_half = float(f1_score(y_val, p_val >= 0.5, zero_division=0))
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_loss": val_loss,
            "val_auroc": auroc,
            "val_auprc": auprc,
            "val_f1_at_0.5": f1_at_half,
        }
        history.append(row)
        print(
            f"[{args.study} {args.task} {args.loss} seed {args.seed}] "
            f"epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} auroc={auroc:.4f} auprc={auprc:.4f}",
            flush=True,
        )
        if auroc > best_auroc:
            best_auroc = auroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stop epoch={epoch} best={best_epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    model.to(args.device)
    y_val, p_val, val_loss = predict(model, val_loader, criterion, args)
    grid_rows, operating_points = threshold_analysis(y_val, p_val)
    result = {
        "protocol": "historical_patient_split_90_10_external_test_separate",
        "study": args.study,
        "task": args.task,
        "label": label_name,
        "model": model_name,
        "feature_mode": args.feature_mode if args.study == "feature" else "raw",
        "loss": args.loss,
        "loss_config": loss_config,
        "seed": args.seed,
        "variant": "P3",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "clip": args.clip,
        "optimizer": "Adam",
        "scheduler": None,
        "sampling_strategy": "shuffle_no_oversampling",
        "augmentation": "historical_P3_train_only",
        "early_stopping_metric": "validation_AUROC",
        "best_epoch": best_epoch,
        "stop_epoch": history[-1]["epoch"],
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_time_sec": round(time.time() - t0, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(args.device) / 2**20, 2),
        "train_counts": {"negative": n_neg, "positive": n_pos},
        "validation_counts": {
            "negative": int(len(va) - labels[va].sum()),
            "positive": int(labels[va].sum()),
        },
        "split_sha256": sha256_file(split_path),
        "threshold_grid": {"minimum": 0.01, "maximum": 0.99, "step": 0.01},
        "threshold_selection_uses": "validation_only",
        "val": {
            "auroc": float(roc_auc_score(y_val, p_val)),
            "auprc": float(average_precision_score(y_val, p_val)),
            "loss": val_loss,
            "operating_points": operating_points,
        },
        "history": history,
    }
    val_meta = {
        "record_ids": meta["record_ids"][va],
        "patient_ids": meta["patient_ids"][va],
        "y": y_val,
        "p": p_val,
    }
    return result, best_state, grid_rows, val_meta


def tag_for(args):
    if args.study == "loss":
        if args.loss == "focal":
            alpha = "ratio" if args.focal_alpha_mode == "ratio" else str(args.focal_alpha).replace(".", "p")
            detail = f"g{args.gamma:g}_a{alpha}"
        elif args.loss == "cb_focal":
            detail = f"b{args.beta:g}_g{args.gamma:g}"
        else:
            detail = "auto" if args.loss == "wbce" else "plain"
        return f"loss_{args.loss}_{detail}_{args.task}_seed{args.seed}"
    return f"feature_{args.feature_mode}_{args.task}_seed{args.seed}"


def write_outputs(args, result, state, grid_rows, val_meta):
    out_dir = ROOT / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    tag = tag_for(args)
    ckpt_path = out_dir / "checkpoints" / f"{tag}.pt"
    torch.save(state, ckpt_path)
    result["checkpoint_sha256"] = sha256_file(ckpt_path)
    result["runner_sha256"] = sha256_file(Path(__file__))
    result["artifacts"] = {
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "epochs_csv": f"results/{args.exp_name}/{tag}_epochs.csv",
        "threshold_grid_csv": f"results/{args.exp_name}/{tag}_threshold_grid.csv",
        "validation_predictions_csv": f"results/{args.exp_name}/{tag}_val_predictions.csv",
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with open(out_dir / f"{tag}_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result["history"][0]))
        w.writeheader()
        w.writerows(result["history"])
    with open(out_dir / f"{tag}_threshold_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
        w.writeheader()
        w.writerows(grid_rows)
    with open(out_dir / f"{tag}_val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "patient_id", "y_true", "probability"])
        for row in zip(
            val_meta["record_ids"],
            val_meta["patient_ids"],
            val_meta["y"],
            val_meta["p"],
        ):
            w.writerow([str(row[0]), str(row[1]), int(row[2]), f"{float(row[3]):.10f}"])
    print(json.dumps({"tag": tag, "best_epoch": result["best_epoch"], "val": result["val"]}, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=["loss", "feature"])
    p.add_argument("--task", required=True, choices=list(TASK_MAP))
    p.add_argument("--loss", default="bce", choices=["bce", "wbce", "focal", "cb_focal"])
    p.add_argument("--feature_mode", default="raw", choices=list(FEATURE_MODES))
    p.add_argument("--focal_alpha", type=float, default=0.5)
    p.add_argument("--focal_alpha_mode", choices=["explicit", "ratio"], default="explicit")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--exp_name", default="omi_loss_feature_ablation")
    args = p.parse_args(argv)
    if args.study == "feature" and args.loss != "bce":
        p.error("feature study is frozen to BCE; do not mix feature and loss changes")
    if args.study == "loss" and args.task != "omi":
        p.error("the requested CB-Focal loss study is OMI-only")
    return args


def main(argv=None):
    args = parse_args(argv)
    result, state, grid_rows, val_meta = train(args)
    write_outputs(args, result, state, grid_rows, val_meta)


if __name__ == "__main__":
    main()
