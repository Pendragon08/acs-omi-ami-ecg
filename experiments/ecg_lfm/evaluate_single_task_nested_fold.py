"""Record-ID aligned nested evaluation for ACS-only or OMI-only controls."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES,
    acs_metrics,
    acs_subtype_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    save_curves,
    select_omi_threshold,
    sha256,
)
from experiments.ecg_lfm.evaluate_joint import (  # noqa: E402
    _aggregate,
    _read_csv,
    load_record_ground_truth,
    read_history,
)
from models.ecg_lfm_adapter import load_finetuned_model  # noqa: E402


def expected_assignment(path: Path, subset: str) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if row["subset"] == subset]
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate record ID in {subset}")
    return rows


def infer_single_split(
    checkpoint: Path,
    manifest_dir: Path,
    split: str,
    segment_csv: Path,
    task_mode: str,
    device: torch.device,
    workers: int,
    batch: int,
) -> tuple[list[dict], np.ndarray]:
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.eval().to(device)
    task.load_dataset(split, combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False)
    dataset = task.dataset(split)
    iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_signals=batch,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=2026,
        num_shards=1,
        shard_id=0,
        num_workers=workers,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=False)
    width = 4 if task_mode == "acs" else 1
    logits = np.full((len(dataset), width), np.nan, dtype=np.float32)
    with torch.no_grad():
        for sample in iterator:
            if not sample:
                continue
            ids = sample["id"].cpu().numpy()
            from fairseq_signals.utils import utils

            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**sample["net_input"])
                batch_logits = model.get_logits(output, normalize=False).float().cpu().numpy()
            if batch_logits.ndim == 1:
                batch_logits = batch_logits[:, None]
            if batch_logits.shape[1] != width:
                raise ValueError(f"{task_mode} checkpoint emitted {batch_logits.shape[1]} outputs")
            logits[ids] = batch_logits
    del model, task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(logits).all():
        raise RuntimeError(f"inference did not fill every {split} row")
    segment_rows = _read_csv(segment_csv)
    if len(segment_rows) != len(logits):
        raise ValueError("segment index and model output lengths differ")
    records, record_logits = _aggregate(logits, segment_rows)
    probabilities = (
        torch.softmax(torch.from_numpy(record_logits), dim=1).numpy()
        if task_mode == "acs"
        else torch.sigmoid(torch.from_numpy(record_logits[:, 0])).numpy()
    )
    if task_mode == "acs" and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("ACS probabilities do not sum to one")
    return records, probabilities


def prediction_rows(
    records: list[dict],
    probabilities: np.ndarray,
    truth_by_record: dict[str, dict],
    expected: list[dict[str, str]],
    task_mode: str,
    fold: int,
    subset: str,
) -> list[dict]:
    if [row["record_id"] for row in records] != [row["record_id"] for row in expected]:
        raise ValueError(f"{subset} inference order differs from explicit record-ID assignment")
    rows: list[dict] = []
    for record, probability, assignment in zip(records, probabilities, expected):
        truth = truth_by_record.get(record["record_id"])
        if truth is None:
            raise ValueError(f"prediction missing joined truth: {record['record_id']}")
        if truth["patient_id"] != record["patient_id"] or truth["patient_id"] != assignment["patient_id"]:
            raise ValueError(f"patient cross-check failed: {record['record_id']}")
        base = {
            "record_id": record["record_id"], "patient_id": truth["patient_id"],
            "fold": fold, "subset": subset,
        }
        if task_mode == "acs":
            base.update({"y_acs": truth["y_acs"]})
            base.update({f"p_{name}": float(probability[index]) for index, name in enumerate(ACS_CLASSES)})
            base["pred_acs"] = int(np.argmax(probability))
        else:
            base.update({"y_omi": truth["y_omi"], "p_omi": float(probability)})
            # Retain y_acs only for the pre-specified OMI-among-NSTE subgroup.
            base["y_acs_for_subgroup"] = truth["y_acs"]
        rows.append(base)
    return rows


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_acs(output: Path, y: np.ndarray, probability: np.ndarray, metrics: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, name in enumerate(ACS_CLASSES):
        target = (y == index).astype(int)
        fpr, tpr, _ = roc_curve(target, probability[:, index])
        recall, precision, _ = precision_recall_curve(target, probability[:, index])
        axes[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(target, probability[:, index]):.3f}")
        axes[1].plot(recall, precision, label=f"{name} AP={average_precision_score(target, probability[:, index]):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="1-Specificity", ylabel="Sensitivity", title="ACS-only ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="ACS-only PR")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "roc_pr_curves.png", dpi=180)
    plt.close(figure)
    display = ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=ACS_CLASSES)
    display.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()


def plot_omi(output: Path, y: np.ndarray, probability: np.ndarray, metrics: dict) -> None:
    fpr, tpr, _ = roc_curve(y, probability)
    recall, precision, _ = precision_recall_curve(y, probability)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, probability):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(y, probability):.3f}")
    axes[0].set(xlabel="1-Specificity", ylabel="Sensitivity", title="OMI-only ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="OMI-only PR")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "roc_pr_curves.png", dpi=180)
    plt.close(figure)
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    ConfusionMatrixDisplay(matrix, display_labels=("NOMI", "OMI")).plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_mode", choices=("acs", "omi"), required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--run_root", default="results/single_task_vs_joint")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--label_csv", default="data/CSV/train.csv")
    parser.add_argument("--output_root", default="results/single_task_vs_joint_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    manifest_root = ROOT / args.manifest_root
    manifest_dir = manifest_root / f"fold{args.fold}"
    checkpoint = ROOT / args.run_root / args.run_name / f"fold{args.fold}" / "checkpoint_best.pt"
    output = ROOT / args.output_root / args.run_name / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    truth, truth_audit = load_record_ground_truth(manifest_root / "master_records.csv", ROOT / args.label_csv)
    truth_by_record = {row["record_id"]: row for row in truth}
    assignment_path = manifest_dir / "patient_assignments.csv"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_rows: dict[str, list[dict]] = {}
    for subset, fairseq_split in {"inner_validation": "valid", "outer_test": "outer_test"}.items():
        expected = expected_assignment(assignment_path, subset)
        records, probabilities = infer_single_split(
            checkpoint, manifest_dir, fairseq_split, manifest_dir / f"{fairseq_split}_segments.csv",
            args.task_mode, device, args.workers, args.batch,
        )
        all_rows[subset] = prediction_rows(
            records, probabilities, truth_by_record, expected, args.task_mode, args.fold, subset
        )
        write_predictions(output / f"{subset}_predictions.csv", all_rows[subset])

    assignment_rows = _read_csv(assignment_path)
    patient_sets = {
        subset: {row["patient_id"] for row in assignment_rows if row["subset"] == subset}
        for subset in ("inner_train", "inner_validation", "outer_test")
    }
    overlaps = {
        "inner_train__inner_validation": len(patient_sets["inner_train"] & patient_sets["inner_validation"]),
        "inner_train__outer_test": len(patient_sets["inner_train"] & patient_sets["outer_test"]),
        "inner_validation__outer_test": len(patient_sets["inner_validation"] & patient_sets["outer_test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"patient leakage: {overlaps}")

    if args.task_mode == "acs":
        def acs_bundle(rows):
            y = np.asarray([row["y_acs"] for row in rows], dtype=np.int64)
            probability = np.asarray([[row[f"p_{name}"] for name in ACS_CLASSES] for row in rows])
            return {
                "n": len(rows), "acs": acs_metrics(y, probability),
                "conditional_acs": acs_subtype_metrics(y, probability),
                "nstemi_vs_ua": nstemi_vs_ua_metrics(y, probability),
                "calibration_uncalibrated": multiclass_calibration_metrics(y, probability),
            }
        inner_metrics = acs_bundle(all_rows["inner_validation"])
        outer_metrics = acs_bundle(all_rows["outer_test"])
        thresholds = None
        y_outer = np.asarray([row["y_acs"] for row in all_rows["outer_test"]], dtype=np.int64)
        p_outer = np.asarray([[row[f"p_{name}"] for name in ACS_CLASSES] for row in all_rows["outer_test"]])
        plot_acs(output, y_outer, p_outer, outer_metrics["acs"])
    else:
        y_inner = np.asarray([row["y_omi"] for row in all_rows["inner_validation"]], dtype=np.int64)
        p_inner = np.asarray([row["p_omi"] for row in all_rows["inner_validation"]], dtype=np.float64)
        thresholds = {}
        threshold_details = {}
        for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
            thresholds[method], threshold_details[method] = select_omi_threshold(y_inner, p_inner, method)

        def omi_bundle(rows):
            y = np.asarray([row["y_omi"] for row in rows], dtype=np.int64)
            probability = np.asarray([row["p_omi"] for row in rows], dtype=np.float64)
            y_acs = np.asarray([row["y_acs_for_subgroup"] for row in rows], dtype=np.int64)
            nste = np.isin(y_acs, (2, 3))
            operating_points = {
                name: None if threshold is None else binary_metrics(y, probability, threshold)
                for name, threshold in thresholds.items()
            }
            youden = thresholds.get("youden_j")
            return {
                "n": len(rows), "omi": operating_points,
                "omi_among_nste_acs": {
                    "n": int(nste.sum()),
                    "metrics": None if youden is None else binary_metrics(y[nste], probability[nste], youden),
                },
                "calibration_uncalibrated": binary_calibration_metrics(y, probability),
            }
        inner_metrics = omi_bundle(all_rows["inner_validation"])
        inner_metrics["threshold_details"] = threshold_details
        outer_metrics = omi_bundle(all_rows["outer_test"])
        y_outer = np.asarray([row["y_omi"] for row in all_rows["outer_test"]], dtype=np.int64)
        p_outer = np.asarray([row["p_omi"] for row in all_rows["outer_test"]], dtype=np.float64)
        plot_omi(output, y_outer, p_outer, outer_metrics["omi"]["youden_j"])

    metrics = {
        "task_mode": args.task_mode,
        "protocol": "patient-level nested validation; inner validation selects checkpoint/OMI threshold",
        "run_name": args.run_name,
        "outer_fold": args.fold,
        "thresholds": thresholds,
        "inner_validation": inner_metrics,
        "outer_test": outer_metrics,
    }
    alignment = {
        **truth_audit,
        "status": "pass", "join_key": "record_id", "run_name": args.run_name,
        "task_mode": args.task_mode, "outer_fold": args.fold, "patient_overlap": overlaps,
        "missing_predictions": 0, "duplicate_predictions": 0,
        "inner_validation_records": len(all_rows["inner_validation"]),
        "outer_test_records": len(all_rows["outer_test"]),
        "assignment_sha256": sha256(assignment_path),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluator_sha256": sha256(Path(__file__)),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "provenance_alignment_report.json").write_text(json.dumps(alignment, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_curves(output, read_history(ROOT / args.run_root, args.run_name), f"Nested {args.run_name}")
    print(json.dumps({"status": "pass", "metrics": str(output / "metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
