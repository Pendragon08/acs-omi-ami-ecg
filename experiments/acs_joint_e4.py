"""Corrected E4 joint ACS-subtype + OMI experiment.

This replaces the historical four-independent-sigmoid runner.  The model has
five scalar KAN heads, interpreted as a four-way categorical ACS head
(``OTHER``, STEMI, NSTEMI, UA) and one independent OMI binary head.  The
explicit all-zero subtype state is therefore retained as ``OTHER`` while the
three ACS subtypes remain mutually exclusive by construction.
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
    ACS_CLASSES,
    LABELS,
    SUBTYPE_LABELS,
    acs_metrics,
    acs_subtype_metrics,
    binary_metrics,
    encode_targets,
    load_data,
    load_or_build_folds,
    save_confusion_figures,
    save_curves,
    save_roc_pr_curves,
    select_omi_threshold,
    sha256,
    stem,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _joint_loss(logits: torch.Tensor, target: torch.Tensor, omi_pos_weight: torch.Tensor | None, omi_weight: float):
    acs_target = target[:, 0].long()
    omi_target = target[:, 1].float()
    acs_loss = F.cross_entropy(logits[:, :4], acs_target)
    omi_loss = F.binary_cross_entropy_with_logits(
        logits[:, 4], omi_target, pos_weight=omi_pos_weight
    )
    return acs_loss + float(omi_weight) * omi_loss, acs_loss, omi_loss


def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    y_acs, y_omi, acs_rows, omi_rows = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device, non_blocking=True))["logits"]
            y_acs.append(yb[:, 0].numpy().astype(np.int64))
            y_omi.append(yb[:, 1].numpy().astype(np.int64))
            acs_rows.append(torch.softmax(out[:, :4].float(), dim=1).cpu().numpy())
            omi_rows.append(torch.sigmoid(out[:, 4].float()).cpu().numpy())
    return (
        np.concatenate(y_acs), np.concatenate(y_omi),
        np.concatenate(acs_rows), np.concatenate(omi_rows),
    )


def _predict_array(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int):
    acs_rows, omi_rows = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
            logits = model(xb)["logits"]
            acs_rows.append(torch.softmax(logits[:, :4].float(), dim=1).cpu().numpy())
            omi_rows.append(torch.sigmoid(logits[:, 4].float()).cpu().numpy())
    return np.concatenate(acs_rows), np.concatenate(omi_rows)


def _recover_test_raw() -> tuple[np.ndarray, np.ndarray]:
    arr = ROOT / "data" / "arrays"
    p2 = np.load(arr / "P2_test.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    scaler = scalers["P2"]
    raw = p2.astype(np.float32) * (float(scaler["scale"]) + 1e-8) + float(scaler["median"])
    meta = np.load(arr / "meta_test.npz", allow_pickle=True)
    return raw, np.asarray([stem(x) for x in meta["record_ids"]], dtype=object)


def _test_csv_ids() -> list[str]:
    with (ROOT / "data" / "CSV" / "test.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [(row.get("ecg_row_record") or "").strip() for row in csv.DictReader(handle)]


def run_fold(args, fold_idx, train_idx, val_idx, x_raw, acs, omi, patient_ids, device):
    seed = int(args.seed + fold_idx)
    set_seed(seed)
    scaler = h.fit_robust_scaler(x_raw[train_idx])
    x_train = h.apply_robust_scaler(x_raw[train_idx], scaler)
    x_val = h.apply_robust_scaler(x_raw[val_idx], scaler)
    y_train = np.column_stack([acs[train_idx], omi[train_idx]]).astype(np.float32)
    y_val = np.column_stack([acs[val_idx], omi[val_idx]]).astype(np.float32)
    train_dataset = h.ECGDataset(x_train, y_train, True, seed)
    val_dataset = h.ECGDataset(x_val, y_val, False, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, generator=generator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    model = h.HMoEKAN("e4", n_tasks=5, head_kind="kan", device=str(device)).to(device)
    pos = max(1, int(omi[train_idx].sum()))
    neg = max(1, int(len(train_idx) - pos))
    omi_pos_weight = None if args.omi_weighting == "none" else torch.tensor(neg / pos, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr,
    ) if args.scheduler == "plateau" else None
    best_state, best_score, best_epoch, wait = None, -np.inf, -1, 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = acs_loss_sum = omi_loss_sum = 0.0; batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, acs_loss, omi_loss = _joint_loss(model(xb)["logits"], yb, omi_pos_weight, args.omi_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            total += float(loss.item()); acs_loss_sum += float(acs_loss.item()); omi_loss_sum += float(omi_loss.item()); batches += 1
        yv_acs, yv_omi, pv_acs, pv_omi = _predict(model, val_loader, device)
        val_acs_loss = float(F.cross_entropy(torch.from_numpy(np.log(np.clip(pv_acs, 1e-7, 1.0))), torch.from_numpy(yv_acs)).item())
        val_omi_loss = float(F.binary_cross_entropy(torch.from_numpy(np.clip(pv_omi, 1e-7, 1 - 1e-7)), torch.from_numpy(yv_omi).float()).item())
        am = acs_metrics(yv_acs, pv_acs)
        om = binary_metrics(yv_omi, pv_omi, 0.5)
        score = float(np.nanmean([am["macro_auroc_ovr"], om["auroc"]]))
        if scheduler is not None:
            scheduler.step(score)
        lr_now = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "train_total_loss": total / max(1, batches),
            "train_acs_ce": acs_loss_sum / max(1, batches),
            "train_omi_bce": omi_loss_sum / max(1, batches),
            "val_total_loss": val_acs_loss + args.omi_loss_weight * val_omi_loss,
            "val_acs_ce": val_acs_loss,
            "val_omi_bce": val_omi_loss,
            "val_acs_macro_auroc": am["macro_auroc_ovr"],
            "val_omi_auroc": om["auroc"],
            "val_omi_auprc": om["auprc"],
            "val_acs_accuracy": am["accuracy"],
            "val_joint_score": score,
            "learning_rate": lr_now,
        }
        history.append(record)
        print(json.dumps({"experiment": "acs_joint_e4", "fold": fold_idx, **record}), flush=True)
        if score > best_score + args.min_delta:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for fold {fold_idx}")
    model.load_state_dict(best_state); model.to(device).eval()
    yv_acs, yv_omi, pv_acs, pv_omi = _predict(model, val_loader, device)
    return {
        "fold": fold_idx, "best_epoch": best_epoch, "best_val_joint_score": best_score,
        "train_sec": round(time.time() - started, 2), "history": history,
        "n_train_records": len(train_idx), "n_val_records": len(val_idx),
        "n_train_patients": len(np.unique(patient_ids[train_idx])),
        "n_val_patients": len(np.unique(patient_ids[val_idx])),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "omi_training_counts": {"positive": pos, "negative": neg, "pos_weight": neg / pos if args.omi_weighting != "none" else None},
    }, best_state, (yv_acs, yv_omi, pv_acs, pv_omi)


def _write_submission(out_dir: Path, x_raw: np.ndarray, folds, fold_results, args, device):
    x_test_raw, test_meta_ids = _recover_test_raw()
    acs_by_fold, omi_by_fold = [], []
    for fold_idx, (train_idx, _) in enumerate(folds):
        scaler = h.fit_robust_scaler(x_raw[train_idx])
        x_test = h.apply_robust_scaler(x_test_raw, scaler)
        model = h.HMoEKAN("e4", n_tasks=5, head_kind="kan", device=str(device)).to(device)
        state = torch.load(out_dir / f"acs_joint_e4_fold{fold_idx}_seed{args.seed}.pt", map_location=device)
        model.load_state_dict(state)
        pa, po = _predict_array(model, x_test, device, args.batch_size)
        acs_by_fold.append(pa); omi_by_fold.append(po)
    p_acs_meta = np.mean(np.stack(acs_by_fold), axis=0)
    p_omi_meta = np.mean(np.stack(omi_by_fold), axis=0)
    test_ids = _test_csv_ids()
    by_id = {str(rid): i for i, rid in enumerate(test_meta_ids)}
    requested = [stem(x) for x in test_ids]
    if set(requested) != set(map(str, test_meta_ids)):
        raise ValueError("test.csv identifiers do not match meta_test identifiers")
    order = np.asarray([by_id[x] for x in requested])
    p_acs, p_omi = p_acs_meta[order], p_omi_meta[order]
    omi_threshold = float(fold_results["thresholds"]["youden_j"])
    acs_pred = p_acs.argmax(axis=1)
    source_pred = np.zeros((len(test_ids), 4), dtype=np.int64)
    for k in (1, 2, 3): source_pred[:, k - 1] = (acs_pred == k).astype(np.int64)
    source_pred[:, 3] = (p_omi >= omi_threshold).astype(np.int64)
    internal = out_dir / "test_predictions_corrected_joint.csv"
    with internal.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["ecg_row_record", *[f"p_acs_{x}" for x in ACS_CLASSES], "p_omi", "acs_pred", "omi_pred"])
        for rid, pa, po, ac, op in zip(test_ids, p_acs, p_omi, acs_pred, source_pred[:, 3]): writer.writerow([rid, *pa.tolist(), float(po), ACS_CLASSES[int(ac)], int(op)])
    submission = out_dir / "submission_corrected_acs_subtype_omi.csv"
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["ecg_row_record", *LABELS])
        for rid, pred in zip(test_ids, source_pred): writer.writerow([rid, *pred.tolist()])
    return {"n_records": len(test_ids), "omi_threshold": omi_threshold, "submission": str(submission), "submission_sha256": sha256(submission), "predicted_acs_counts": {name: int((acs_pred == i).sum()) for i, name in enumerate(ACS_CLASSES)}, "predicted_omi_positive": int(source_pred[:, 3].sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/corrected_joint_acs_omi_e4")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--omi_loss_weight", type=float, default=1.0)
    parser.add_argument("--omi_weighting", choices=("none", "train_pos_weight"), default="none")
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.n_folds != 5: raise ValueError("registered protocol is fixed to five folds")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    x_raw, y_source, patient_ids, record_ids, label_audit = load_data()
    acs, omi = encode_targets(y_source)
    folds, split_audit = load_or_build_folds(patient_ids, acs, omi, args.n_folds, args.fold_seed)
    (out_dir / "label_audit.json").write_text(json.dumps(label_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps({**split_audit, "folds_indices": [{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]}, indent=2, ensure_ascii=False), encoding="utf-8")
    oof_acs = np.zeros((len(acs), 4), dtype=np.float32); oof_omi = np.zeros(len(omi), dtype=np.float32); fold_results = []; histories = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        result, state, pred = run_fold(args, fold_idx, train_idx, val_idx, x_raw, acs, omi, patient_ids, device)
        yv_acs, yv_omi, pv_acs, pv_omi = pred
        oof_acs[val_idx] = pv_acs; oof_omi[val_idx] = pv_omi; histories.append(result.pop("history"))
        torch.save(state, out_dir / f"acs_joint_e4_fold{fold_idx}_seed{args.seed}.pt")
        (out_dir / f"history_fold{fold_idx}.json").write_text(json.dumps(histories[-1], indent=2), encoding="utf-8")
        fold_results.append(result)
    thresholds = {}; threshold_details = {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95"):
        thresholds[method], threshold_details[method] = select_omi_threshold(omi, oof_omi, method)
    # Per-fold descriptive metrics at one pooled validation threshold.
    for result, (_, val_idx) in zip(fold_results, folds):
        result["val_metrics_at_pooled_youden"] = {"acs": acs_metrics(acs[val_idx], oof_acs[val_idx]), "acs_subtype": acs_subtype_metrics(acs[val_idx], oof_acs[val_idx]), "omi": binary_metrics(omi[val_idx], oof_omi[val_idx], thresholds["youden_j"])}
    pooled = {}
    for name, threshold in thresholds.items():
        pooled[name] = {"acs": acs_metrics(acs, oof_acs), "acs_subtype": acs_subtype_metrics(acs, oof_acs), "omi": binary_metrics(omi, oof_omi, threshold)}
    save_curves(out_dir, histories, "Corrected joint ACS categorical + OMI binary E4")
    save_roc_pr_curves(out_dir, acs, oof_acs, omi, oof_omi)
    y_pred_acs = oof_acs.argmax(axis=1); omi_pred = (oof_omi >= thresholds["youden_j"]).astype(int)
    save_confusion_figures(out_dir, np.asarray(pooled["youden_j"]["acs"]["confusion_matrix"]), np.asarray(pooled["youden_j"]["acs_subtype"]["confusion_matrix"]), np.asarray([[int(((omi == 0) & (omi_pred == 0)).sum()), int(((omi == 0) & (omi_pred == 1)).sum())], [int(((omi == 1) & (omi_pred == 0)).sum()), int(((omi == 1) & (omi_pred == 1)).sum())]]))
    with (out_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["row_index", "record_id", "patient_id", "acs_true", "omi_true", *[f"p_acs_{x}" for x in ACS_CLASSES], "p_omi", "acs_pred", "omi_pred"])
        for i in range(len(acs)): writer.writerow([i, record_ids[i], patient_ids[i], int(acs[i]), int(omi[i]), *oof_acs[i].tolist(), float(oof_omi[i]), int(y_pred_acs[i]), int(omi_pred[i])])
    (out_dir / "threshold_analysis.json").write_text(json.dumps({"thresholds": thresholds, "details": threshold_details, "pooled_metrics": pooled, "recommended": "youden_j"}, indent=2, ensure_ascii=False), encoding="utf-8")
    test_summary = _write_submission(out_dir, x_raw, folds, {"thresholds": thresholds}, args, device)
    summary = {
        "experiment": "ACS-JOINT-ACS-CATEGORICAL-OMI-BINARY-E4-CORRECTED",
        "task": "joint multi-task: ACS categorical OTHER/STEMI/NSTEMI/UA + independent OMI binary",
        "source_labels": list(LABELS), "acs_classes": list(ACS_CLASSES),
        "model": "HMoE-WavelNet-KAN E4; five scalar task heads (first four are ACS logits, fifth OMI logit)",
        "loss": "CrossEntropyLoss(ACS) + omi_loss_weight * BCEWithLogitsLoss(OMI)",
        "omi_weighting": args.omi_weighting, "omi_loss_weight": args.omi_loss_weight,
        "seed": args.seed, "fold_seed": args.fold_seed, "n_folds": args.n_folds,
        "training": {"optimizer": "Adam", "learning_rate": args.lr, "batch_size": args.batch_size, "epochs_max": args.epochs, "early_stopping_patience": args.patience, "scheduler": args.scheduler, "selection_metric": "mean(ACS macro OVR AUROC, OMI AUROC)"},
        "label_audit": label_audit, "split_audit": split_audit,
        "thresholds": thresholds, "threshold_details": threshold_details,
        "pooled_oof_metrics": pooled, "folds": fold_results,
        "test": test_summary,
        "supersedes": "hmoe_kan_stemi_nstemi_ua_omi_e4 independent sigmoid results (invalid task formulation)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
