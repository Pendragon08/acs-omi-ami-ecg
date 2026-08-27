"""Patient-level five-fold E4 experiment for four independent ACS labels.

Task definition
---------------
The model has four independent binary outputs: STEMI, NSTEMI, UA and OMI.
It uses sigmoid activation at inference and BCEWithLogitsLoss at training.
Rows with the explicit subtype state [STEMI, NSTEMI, UA] = [0, 0, 0] are
retained as valid non-ACS negatives.  Only a row with a missing/non-binary
value in one of the four target columns is excluded.

The three subtype labels are mutually exclusive per record in the current
dataset, but OMI overlaps with STEMI/NSTEMI.  Consequently this is correctly a
multi-label, multi-output binary task rather than a softmax multiclass task.

The E4 backbone is reused unchanged from ``experiments.hmoe_kan_github``:
Xception morphology, LeadSpatial GAT, WavelNet frequency expert, task routers,
and one KAN head per label.  This file deliberately uses a new output directory
and does not overwrite historical three-label or four-class results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import hmoe_kan_github as h  # noqa: E402


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRIC_KEYS = (
    "auroc", "auprc", "sensitivity", "specificity", "ppv", "npv",
    "f1", "mcc", "accuracy",
)
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stem(value: str) -> str:
    return Path(str(value)).stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_counts(labels: np.ndarray) -> dict[str, int]:
    states = Counter("".join(map(str, row.astype(int).tolist())) for row in labels)
    return {key: int(states[key]) for key in sorted(states)}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load P2 ECGs plus valid four-label targets.

    The P2 array is historically globally scaled.  Recover its band-passed
    values so that the train-fold robust scaler is fitted afresh for each fold.
    """
    arr = ROOT / "data" / "arrays"
    meta = np.load(arr / "meta_train.npz", allow_pickle=True)
    p2 = np.load(arr / "P2_train.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    p2_scaler = scalers["P2"]
    raw = p2.astype(np.float32) * (float(p2_scaler["scale"]) + 1e-8) + float(p2_scaler["median"])

    csv_path = ROOT / "data" / "CSV" / "train.csv"
    labels_by_id: dict[str, list[int]] = {}
    invalid_csv_rows: list[dict] = []
    duplicate_ids: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = stem((row.get("ecg_row_record") or "").strip())
            raw_values = [(row.get(name) or "").strip() for name in LABELS]
            if not rid or any(value not in {"0", "1"} for value in raw_values):
                invalid_csv_rows.append({"record_id": rid, "values": dict(zip(LABELS, raw_values))})
                continue
            if rid in labels_by_id:
                duplicate_ids.append(rid)
                continue
            labels_by_id[rid] = [int(value) for value in raw_values]

    record_ids_all = np.asarray([stem(x) for x in meta["record_ids"]], dtype=object)
    patient_ids_all = np.asarray([str(x) for x in meta["patient_ids"]], dtype=object)
    missing_meta_ids = [str(rid) for rid in record_ids_all if str(rid) not in labels_by_id]
    valid_mask = np.asarray([str(rid) in labels_by_id for rid in record_ids_all], dtype=bool)
    if not valid_mask.any():
        raise RuntimeError("No valid labelled training ECGs remain after label audit")
    labels = np.asarray([labels_by_id[str(rid)] for rid in record_ids_all[valid_mask]], dtype=np.float32)
    raw_valid = raw if valid_mask.all() else np.asarray(raw[valid_mask], dtype=np.float32)
    record_ids = record_ids_all[valid_mask]
    patient_ids = patient_ids_all[valid_mask]

    pairwise = {}
    for i, left in enumerate(LABELS):
        for j in range(i + 1, len(LABELS)):
            pairwise[f"{left}&{LABELS[j]}"] = int(np.logical_and(labels[:, i] == 1, labels[:, j] == 1).sum())
    subtype_sum = labels[:, :3].sum(axis=1)
    audit = {
        "csv_path": str(csv_path),
        "n_csv_rows": int(len(labels_by_id) + len(invalid_csv_rows)),
        "n_meta_records": int(len(record_ids_all)),
        "n_valid_records": int(len(labels)),
        "n_excluded_unknown_or_invalid": int((~valid_mask).sum()),
        "n_invalid_csv_rows": int(len(invalid_csv_rows)),
        "invalid_csv_rows_preview": invalid_csv_rows[:20],
        "n_duplicate_csv_record_ids": int(len(duplicate_ids)),
        "duplicate_csv_record_ids_preview": duplicate_ids[:20],
        "n_meta_records_without_valid_label": int(len(missing_meta_ids)),
        "missing_meta_record_ids_preview": missing_meta_ids[:20],
        "record_label_counts": {
            name: {"positive": int(labels[:, i].sum()), "negative": int(len(labels) - labels[:, i].sum())}
            for i, name in enumerate(LABELS)
        },
        "record_joint_state_counts_bit_order_STEMI_NSTEMI_UA_OMI": _state_counts(labels),
        "pairwise_copositives": pairwise,
        "subtype_label_overlap_gt1": int((subtype_sum > 1).sum()),
        "explicit_non_acs_subtype_negative_000": int((subtype_sum == 0).sum()),
        "explicit_non_acs_subtype_negative_with_omi_positive": int(
            np.logical_and(subtype_sum == 0, labels[:, 3] == 1).sum()
        ),
    }
    return raw_valid, labels, patient_ids, record_ids, audit


def build_patient_multilabel_folds(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Make no-leakage folds via iterative multilabel stratification of patients."""
    patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    patient_targets = np.zeros((len(patients), labels.shape[1]), dtype=np.int64)
    for row_idx, patient_idx in enumerate(inverse):
        patient_targets[patient_idx] = np.maximum(patient_targets[patient_idx], labels[row_idx].astype(np.int64))
    if np.any(patient_targets.sum(axis=0) < n_folds):
        bad = {LABELS[i]: int(patient_targets[:, i].sum()) for i in range(labels.shape[1])
               if patient_targets[:, i].sum() < n_folds}
        raise ValueError(f"At least one label has fewer than {n_folds} positive patients: {bad}")

    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_audit = []
    for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(
        splitter.split(np.zeros((len(patients), 1), dtype=np.float32), patient_targets)
    ):
        train_mask = np.isin(inverse, train_patient_idx)
        val_mask = np.isin(inverse, val_patient_idx)
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if np.intersect1d(patient_ids[train_idx], patient_ids[val_idx]).size:
            raise AssertionError(f"patient leakage in fold {fold_idx}")
        folds.append((train_idx, val_idx))
        fold_audit.append({
            "fold": int(fold_idx),
            "n_train_records": int(len(train_idx)),
            "n_val_records": int(len(val_idx)),
            "n_train_patients": int(len(train_patient_idx)),
            "n_val_patients": int(len(val_patient_idx)),
            "train_positive_records": {name: int(labels[train_idx, j].sum()) for j, name in enumerate(LABELS)},
            "val_positive_records": {name: int(labels[val_idx, j].sum()) for j, name in enumerate(LABELS)},
            "train_positive_patients": {name: int(patient_targets[train_patient_idx, j].sum()) for j, name in enumerate(LABELS)},
            "val_positive_patients": {name: int(patient_targets[val_patient_idx, j].sum()) for j, name in enumerate(LABELS)},
        })
    all_val = np.sort(np.concatenate([val for _, val in folds]))
    if not np.array_equal(all_val, np.arange(len(labels))):
        raise AssertionError("validation folds do not form an exact OOF partition")
    audit = {
        "protocol": "patient_level_multilabel_iterative_stratified_5fold",
        "splitter": "iterative-stratification.MultilabelStratifiedKFold",
        "n_folds": int(n_folds),
        "fold_seed": int(seed),
        "n_patients": int(len(patients)),
        "patient_positive_counts": {name: int(patient_targets[:, j].sum()) for j, name in enumerate(LABELS)},
        "patient_joint_state_counts_bit_order_STEMI_NSTEMI_UA_OMI": _state_counts(patient_targets),
        "folds": fold_audit,
    }
    return folds, audit


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = _binary_counts(y_true, pred)
    sens = float(tp / (tp + fn)) if tp + fn else 0.0
    spec = float(tn / (tn + fp)) if tn + fp else 0.0
    ppv = float(tp / (tp + fp)) if tp + fp else 0.0
    npv = float(tn / (tn + fn)) if tn + fn else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, prob)) if np.unique(y_true).size > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, prob)) if np.unique(y_true).size > 1 else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "youden_j": float(sens + spec - 1.0),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metric_table(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(LABELS):
        out[name] = _binary_metrics(y[:, j].astype(np.int64), prob[:, j], float(thresholds[j]))
    out["macro"] = {
        key: float(np.nanmean([out[name][key] for name in LABELS]))
        for key in METRIC_KEYS
    }
    pred = (prob >= thresholds.reshape(1, -1)).astype(np.int64)
    out["prediction_pattern"] = {
        "n_records": int(len(pred)),
        "n_none_positive": int((pred.sum(axis=1) == 0).sum()),
        "n_multi_positive": int((pred.sum(axis=1) >= 2).sum()),
        "n_positive_by_label": {name: int(pred[:, j].sum()) for j, name in enumerate(LABELS)},
    }
    return out


def select_thresholds(y: np.ndarray, prob: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Choose independent validation thresholds strictly from supplied labels/probabilities."""
    if method == "fixed_0.5":
        threshold = np.full(y.shape[1], 0.5, dtype=np.float32)
        return threshold, {name: {"threshold": 0.5, "method": method} for name in LABELS}
    thresholds, detail = [], {}
    for j, name in enumerate(LABELS):
        candidates = []
        for value in GRID:
            metric = _binary_metrics(y[:, j].astype(np.int64), prob[:, j], float(value))
            candidates.append(metric)
        if method == "max_f1":
            best = max(candidates, key=lambda r: (r["f1"], r["sensitivity"], r["specificity"], r["threshold"]))
        elif method == "youden_j":
            best = max(candidates, key=lambda r: (r["youden_j"], r["specificity"], r["f1"], r["threshold"]))
        elif method in {"sens90", "sens95"}:
            target = 0.90 if method == "sens90" else 0.95
            feasible = [r for r in candidates if r["sensitivity"] >= target]
            if feasible:
                best = max(feasible, key=lambda r: (r["specificity"], r["f1"], r["threshold"]))
                best = dict(best)
                best["sensitivity_constraint_satisfied"] = True
            else:
                best = max(candidates, key=lambda r: (r["sensitivity"], r["specificity"], -r["threshold"]))
                best = dict(best)
                best["sensitivity_constraint_satisfied"] = False
        else:
            raise ValueError(f"unknown threshold method: {method}")
        thresholds.append(float(best["threshold"]))
        detail[name] = best
    return np.asarray(thresholds, dtype=np.float32), detail


def predict_loader(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ys, logits_rows, gate_rows = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device, non_blocking=True))
            ys.append(yb.numpy())
            logits_rows.append(out["logits"].cpu().numpy())
            if out.get("gates") is not None:
                gate_rows.append(out["gates"].cpu().numpy())
    y = np.concatenate(ys, axis=0)
    logits = np.concatenate(logits_rows, axis=0)
    gates = np.concatenate(gate_rows, axis=0) if gate_rows else None
    return y, logits, gates


def predict_array(model, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(
                device, non_blocking=True
            )
            chunks.append(torch.sigmoid(model(xb)["logits"]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_fold(
    args: argparse.Namespace,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_raw: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    device: torch.device,
) -> tuple[dict, dict, np.ndarray]:
    set_seed(args.seed + fold_idx)
    scaler = h.fit_robust_scaler(x_raw[train_idx])
    x_train = h.apply_robust_scaler(x_raw[train_idx], scaler)
    x_val = h.apply_robust_scaler(x_raw[val_idx], scaler)
    train_dataset = h.ECGDataset(x_train, labels[train_idx], True, args.seed + fold_idx)
    val_dataset = h.ECGDataset(x_val, labels[val_idx], False, args.seed + fold_idx)
    generator = torch.Generator().manual_seed(args.seed + fold_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, generator=generator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    model = h.HMoEKAN("e4", n_tasks=len(LABELS), head_kind="kan", device=str(device)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_score, best_epoch, patience_counter = None, -np.inf, -1, 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, batches = 0.0, 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1

        y_val, logits_val, _ = predict_loader(model, val_loader, device)
        prob_val = 1.0 / (1.0 + np.exp(-logits_val))
        val_bce = float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logits_val), torch.from_numpy(y_val)
        ).item())
        aurocs = [roc_auc_score(y_val[:, j], prob_val[:, j]) for j in range(len(LABELS))]
        auprcs = [average_precision_score(y_val[:, j], prob_val[:, j]) for j in range(len(LABELS))]
        score = float(np.mean(aurocs))
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss / max(1, batches)),
            "val_bce": val_bce,
            "val_macro_auroc": score,
            "val_macro_auprc": float(np.mean(auprcs)),
        }
        history.append(record)
        print(json.dumps({"experiment": "e4_4label_multilabel", "fold": fold_idx, **record}), flush=True)
        if score > best_score:
            best_score, best_epoch, patience_counter = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected in fold {fold_idx}")
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_val, logits_val, gates = predict_loader(model, val_loader, device)
    prob_val = 1.0 / (1.0 + np.exp(-logits_val))
    result = {
        "fold": int(fold_idx),
        "best_epoch": int(best_epoch),
        "best_val_macro_auroc": float(best_score),
        "train_sec": round(time.time() - started, 2),
        "n_train_records": int(len(train_idx)),
        "n_val_records": int(len(val_idx)),
        "n_train_patients": int(len(np.unique(patient_ids[train_idx]))),
        "n_val_patients": int(len(np.unique(patient_ids[val_idx]))),
        "n_params": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "history": history,
        "ranking_at_best_checkpoint": metric_table(y_val, prob_val, np.full(len(LABELS), 0.5)),
        "gate_mean": gates.mean(axis=0).tolist() if gates is not None else None,
    }
    scales = model.freq.wavelet.a.detach().cpu().numpy().reshape(-1)
    result["wavelnet_scale"] = {
        "min": float(scales.min()), "max": float(scales.max()),
        "mean": float(scales.mean()), "std": float(scales.std()),
        "n_filters": int(scales.size), "mother_wavelet": "sym4",
        "kernel_size": int(model.freq.wavelet.size_kernel), "fs": 100,
    }
    return result, best_state, prob_val


def recover_test_raw() -> tuple[np.ndarray, np.ndarray]:
    arr = ROOT / "data" / "arrays"
    p2 = np.load(arr / "P2_test.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    scaler = scalers["P2"]
    raw = p2.astype(np.float32) * (float(scaler["scale"]) + 1e-8) + float(scaler["median"])
    meta = np.load(arr / "meta_test.npz", allow_pickle=True)
    return raw, np.asarray([stem(x) for x in meta["record_ids"]], dtype=object)


def test_csv_ids() -> list[str]:
    with (ROOT / "data" / "CSV" / "test.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [(row.get("ecg_row_record") or "").strip() for row in csv.DictReader(handle)]


def write_outputs(
    out_dir: Path,
    labels: np.ndarray,
    record_ids: np.ndarray,
    patient_ids: np.ndarray,
    oof_prob: np.ndarray,
    thresholds: dict[str, np.ndarray],
    threshold_details: dict,
    fold_results: list[dict],
    folds: list[tuple[np.ndarray, np.ndarray]],
    x_raw: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    recommended_name = "youden_j"
    recommended_thresholds = thresholds[recommended_name]
    all_threshold_metrics = {name: metric_table(labels, oof_prob, value) for name, value in thresholds.items()}
    for result, (_, val_idx) in zip(fold_results, folds):
        result["metrics_at_pooled_oof_youden_thresholds"] = metric_table(
            labels[val_idx], oof_prob[val_idx], recommended_thresholds
        )

    oof_pred = (oof_prob >= recommended_thresholds.reshape(1, -1)).astype(np.int64)
    np.savez_compressed(
        out_dir / "oof_probabilities.npz",
        labels=labels.astype(np.int8),
        probabilities=oof_prob.astype(np.float32),
        predictions_youden=oof_pred.astype(np.int8),
        thresholds_youden=recommended_thresholds.astype(np.float32),
    )
    with (out_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "ecg_row_record", "patient_id"] +
                        [f"y_{name}" for name in LABELS] +
                        [f"p_{name}" for name in LABELS] +
                        [f"pred_youden_{name}" for name in LABELS])
        for index, (record_id, patient_id, y, prob, pred) in enumerate(
            zip(record_ids, patient_ids, labels, oof_prob, oof_pred)
        ):
            writer.writerow([index, record_id, patient_id, *y.astype(int).tolist(),
                             *prob.tolist(), *pred.astype(int).tolist()])

    # Test inference is only performed after all five checkpoints are frozen.
    x_test_raw, test_meta_ids = recover_test_raw()
    test_probs_by_fold = []
    for fold_idx, (train_idx, _) in enumerate(folds):
        scaler = h.fit_robust_scaler(x_raw[train_idx])
        x_test = h.apply_robust_scaler(x_test_raw, scaler)
        model = h.HMoEKAN("e4", n_tasks=len(LABELS), head_kind="kan", device=str(device)).to(device)
        state = torch.load(out_dir / f"e4_4label_fold{fold_idx}_seed{args.seed}.pt", map_location=device)
        model.load_state_dict(state)
        test_probs_by_fold.append(predict_array(model, x_test, device, args.batch_size))
    test_prob_meta = np.mean(np.stack(test_probs_by_fold, axis=0), axis=0)
    test_ids_raw = test_csv_ids()
    id_to_meta_index = {str(rid): idx for idx, rid in enumerate(test_meta_ids)}
    requested_stems = [stem(value) for value in test_ids_raw]
    if len(requested_stems) != len(test_meta_ids) or set(requested_stems) != set(map(str, test_meta_ids)):
        raise ValueError("test.csv ECG identifiers do not exactly match meta_test identifiers")
    order = np.asarray([id_to_meta_index[value] for value in requested_stems], dtype=np.int64)
    test_prob = test_prob_meta[order]
    test_pred = (test_prob >= recommended_thresholds.reshape(1, -1)).astype(np.int64)
    internal = out_dir / "test_predictions_stemi_nstemi_ua_omi_e4.csv"
    with internal.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_row_record"] +
                        [f"p_{name}" for name in LABELS] +
                        [f"pred_youden_{name}" for name in LABELS])
        for record_id, prob, pred in zip(test_ids_raw, test_prob, test_pred):
            writer.writerow([record_id, *prob.tolist(), *pred.astype(int).tolist()])
    submission = out_dir / "submission_stemi_nstemi_ua_omi_e4_youden.csv"
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_row_record", *LABELS])
        for record_id, pred in zip(test_ids_raw, test_pred):
            writer.writerow([record_id, *pred.astype(int).tolist()])

    mean_std = {}
    for name in LABELS:
        mean_std[name] = {}
        for key in METRIC_KEYS:
            values = [fold["metrics_at_pooled_oof_youden_thresholds"][name][key] for fold in fold_results]
            mean_std[name][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    mean_std["macro"] = {}
    for key in METRIC_KEYS:
        values = [fold["metrics_at_pooled_oof_youden_thresholds"]["macro"][key] for fold in fold_results]
        mean_std["macro"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary = {
        "experiment": "ACS-HMOE-WAVELNET-E4-4LABEL-MULTILABEL-20260827",
        "task": "four_independent_multilabel_binary_classification",
        "labels": list(LABELS),
        "model": "HMoE_WavelNet_KAN_E4 with four independent KAN heads",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss (unweighted)",
        "seed": int(args.seed),
        "fold_seed": int(args.fold_seed),
        "n_folds": int(args.n_folds),
        "training": {
            "optimizer": "Adam", "learning_rate": float(args.lr), "batch_size": int(args.batch_size),
            "epochs_max": int(args.epochs), "early_stopping_patience": int(args.patience),
            "scheduler": "none", "gradient_clip_norm": float(args.gradient_clip_norm),
            "lambda_gate_balance": float(args.lambda_balance),
            "selection_metric": "validation macro AUROC",
        },
        "recommended_threshold_method": "validation_pooled_oof_youden_j",
        "thresholds": {name: value.tolist() for name, value in thresholds.items()},
        "threshold_selection_details": threshold_details,
        "oof_metrics_by_threshold_strategy": all_threshold_metrics,
        "fold_metric_note": (
            "Per-fold hard metrics use one fixed threshold vector selected from all pooled OOF predictions; "
            "these fold-wise threshold metrics are descriptive and not nested-CV estimates."
        ),
        "folds": fold_results,
        "mean_std_youden": mean_std,
        "test": {
            "n_records": int(len(test_pred)),
            "n_positive_predictions": {name: int(test_pred[:, j].sum()) for j, name in enumerate(LABELS)},
            "n_none_positive": int((test_pred.sum(axis=1) == 0).sum()),
            "n_multi_positive": int((test_pred.sum(axis=1) >= 2).sum()),
            "test_labels_used": False,
            "internal_predictions": str(internal),
            "submission": str(submission),
            "submission_sha256": sha256(submission),
        },
        "source_repositories": {
            "wavelnet": "https://github.com/NamhoKim-Kor/WavelNet",
            "pykan": "https://github.com/KindXiaoming/pykan",
            "pytorch_geometric": "https://github.com/pyg-team/pytorch_geometric",
            "iterative_stratification": "https://github.com/trent-b/iterative-stratification",
        },
    }
    (out_dir / "threshold_analysis.json").write_text(
        json.dumps({
            "thresholds": {name: value.tolist() for name, value in thresholds.items()},
            "selection_details": threshold_details,
            "oof_metrics_by_strategy": all_threshold_metrics,
            "recommended": recommended_name,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "e4_4label_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/hmoe_kan_stemi_nstemi_ua_omi_e4")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_balance", type=float, default=0.01)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.n_folds != 5:
        raise ValueError("The registered protocol is fixed to patient-level five-fold CV")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, labels, patient_ids, record_ids, label_audit = load_data()
    folds, split_audit = build_patient_multilabel_folds(patient_ids, labels, args.n_folds, args.fold_seed)
    (out_dir / "label_audit.json").write_text(json.dumps(label_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps({
        **split_audit,
        "folds_indices": [{"train": train.tolist(), "val": val.tolist()} for train, val in folds],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    oof_prob = np.zeros_like(labels, dtype=np.float32)
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        result, state, val_prob = run_fold(
            args, fold_idx, train_idx, val_idx, x_raw, labels, patient_ids, device
        )
        oof_prob[val_idx] = val_prob
        torch.save(state, out_dir / f"e4_4label_fold{fold_idx}_seed{args.seed}.pt")
        (out_dir / f"history_fold{fold_idx}.json").write_text(
            json.dumps(result["history"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        fold_results.append(result)
        print("FOLD_RESULT " + json.dumps({key: value for key, value in result.items() if key != "history"}), flush=True)

    thresholds, threshold_details = {}, {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95"):
        thresholds[method], threshold_details[method] = select_thresholds(labels, oof_prob, method)
    summary = write_outputs(
        out_dir, labels, record_ids, patient_ids, oof_prob, thresholds, threshold_details,
        fold_results, folds, x_raw, args, device,
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
