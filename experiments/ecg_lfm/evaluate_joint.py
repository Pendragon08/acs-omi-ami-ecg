"""Record-level evaluation for corrected ECG-LFM joint checkpoints.

The evaluator deliberately joins predictions to ground truth by ``record_id``.
The numeric master index is checked as provenance, but is never the sole label
lookup key.  This prevents a silently reordered ``y_all.npy`` from changing
the reported result.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
from models.ecg_lfm_adapter import load_finetuned_model  # noqa: E402
from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES,
    LABELS,
    acs_metrics,
    acs_subtype_metrics,
    binary_calibration_metrics,
    binary_metrics,
    multiclass_calibration_metrics,
    nstemi_vs_ua_metrics,
    save_confusion_figures,
    save_curves,
    save_reliability_figures,
    save_roc_pr_curves,
    select_omi_threshold,
    sha256,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _record_id(value: str) -> str:
    return Path(str(value).strip()).stem


def _target_from_source(source: list[int]) -> tuple[int, int]:
    if len(source) != 4 or any(value not in (0, 1) for value in source):
        raise ValueError(f"invalid source target: {source}")
    subtype_sum = sum(source[:3])
    if subtype_sum > 1:
        raise ValueError(f"overlapping STEMI/NSTEMI/UA target: {source}")
    acs = 0 if subtype_sum == 0 else int(np.argmax(source[:3])) + 1
    return acs, int(source[3])


def load_record_ground_truth(master_csv: Path, label_csv: Path) -> tuple[list[dict], dict]:
    """Load and cross-check master and source labels using explicit record IDs."""
    master_rows = _read_csv(master_csv)
    labels_rows = _read_csv(label_csv)
    master_by_id: dict[str, dict] = {}
    master_indices: set[int] = set()
    duplicate_master_ids: list[str] = []
    duplicate_master_indices: list[int] = []
    for row in master_rows:
        record_id = _record_id(row.get("record_id", ""))
        if not record_id:
            raise ValueError("master contains an empty record_id")
        index = int(row["idx"])
        if record_id in master_by_id:
            duplicate_master_ids.append(record_id)
        if index in master_indices:
            duplicate_master_indices.append(index)
        master_indices.add(index)
        master_by_id[record_id] = row
    if duplicate_master_ids or duplicate_master_indices:
        raise ValueError(
            f"duplicate master identity: record_ids={duplicate_master_ids[:10]}, "
            f"indices={duplicate_master_indices[:10]}"
        )

    labels_by_id: dict[str, dict] = {}
    duplicate_label_ids: list[str] = []
    for row in labels_rows:
        record_id = _record_id(row.get("ecg_row_record", ""))
        values = [(row.get(name) or "").strip() for name in LABELS]
        if not record_id or any(value not in {"0", "1"} for value in values):
            continue
        if record_id in labels_by_id:
            duplicate_label_ids.append(record_id)
            continue
        labels_by_id[record_id] = row
    if duplicate_label_ids:
        raise ValueError(f"duplicate source-label record IDs: {duplicate_label_ids[:10]}")

    joined: list[dict] = []
    missing_labels: list[str] = []
    label_mismatches: list[str] = []
    patient_mismatches: list[str] = []
    for record_id, master in master_by_id.items():
        if master.get("cohort") != "train":
            continue
        label = labels_by_id.get(record_id)
        if label is None:
            missing_labels.append(record_id)
            continue
        source = [int((label.get(name) or "").strip()) for name in LABELS]
        master_source = [int((master.get(name) or "").strip()) for name in LABELS]
        if source != master_source:
            label_mismatches.append(record_id)
        source_patient = str(label.get("Patient_id", "")).strip()
        if source_patient and source_patient != str(master.get("patient_id", "")).strip():
            patient_mismatches.append(record_id)
        y_acs, y_omi = _target_from_source(source)
        joined.append(
            {
                "master_idx": int(master["idx"]),
                "record_id": record_id,
                "patient_id": str(master["patient_id"]),
                "validation_fold": int(master["validation_fold"]),
                "y_acs": y_acs,
                "y_omi": y_omi,
            }
        )
    if missing_labels or label_mismatches or patient_mismatches:
        raise ValueError(
            "record-ID ground-truth join failed: "
            f"missing={len(missing_labels)}, label_mismatch={len(label_mismatches)}, "
            f"patient_mismatch={len(patient_mismatches)}"
        )
    joined.sort(key=lambda row: row["master_idx"])
    indices = np.asarray([row["master_idx"] for row in joined], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(joined))):
        raise ValueError("training master indices must be contiguous 0..N-1")
    audit = {
        "master_rows": int(len(master_rows)),
        "source_label_rows": int(len(labels_rows)),
        "joined_training_records": int(len(joined)),
        "duplicate_master_record_ids": int(len(duplicate_master_ids)),
        "duplicate_master_indices": int(len(duplicate_master_indices)),
        "duplicate_source_label_record_ids": int(len(duplicate_label_ids)),
        "missing_labels": int(len(missing_labels)),
        "label_mismatches": int(len(label_mismatches)),
        "patient_mismatches": int(len(patient_mismatches)),
        "master_sha256": sha256(master_csv),
        "source_label_csv_sha256": sha256(label_csv),
        "join_key": "record_id",
        "master_index_role": "cross-checked provenance only; not sole ground-truth lookup key",
    }
    return joined, audit


def _aggregate(
    logits: np.ndarray,
    segment_rows: list[dict[str, str]],
) -> tuple[list[dict], np.ndarray]:
    by_index: dict[int, list[tuple[np.ndarray, dict[str, str]]]] = {}
    for position, row in enumerate(segment_rows):
        if int(row["manifest_position"]) != position:
            raise ValueError("segment CSV is not in manifest order")
        by_index.setdefault(int(row["idx"]), []).append((logits[position], row))
    records: list[dict] = []
    record_logits: list[np.ndarray] = []
    for index in sorted(by_index):
        selected = by_index[index]
        if len(selected) != 2:
            raise ValueError(f"record {index} has {len(selected)} segments, expected 2")
        rows = [item[1] for item in selected]
        record_ids = {_record_id(row["record_id"]) for row in rows}
        patient_ids = {str(row["patient_id"]) for row in rows}
        segments = {int(row["segment"]) for row in rows}
        if len(record_ids) != 1 or len(patient_ids) != 1 or segments != {0, 1}:
            raise ValueError(f"inconsistent segment identity for master idx {index}")
        records.append(
            {
                "master_idx": index,
                "record_id": next(iter(record_ids)),
                "patient_id": next(iter(patient_ids)),
            }
        )
        record_logits.append(np.asarray([item[0] for item in selected]).mean(axis=0))
    return records, np.asarray(record_logits, dtype=np.float32)


def infer_split(
    checkpoint: Path,
    manifest_dir: Path,
    split: str,
    segment_csv: Path,
    device: torch.device,
    workers: int,
    batch: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.eval().to(device)
    task.load_dataset(split, combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False)
    dataset = task.dataset(split)
    itr = task.get_batch_iterator(
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
    logits = np.full((len(dataset), 5), np.nan, np.float32)
    with torch.no_grad():
        for sample in itr:
            if not sample:
                continue
            ids = sample["id"].cpu().numpy()
            from fairseq_signals.utils import utils

            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(**sample["net_input"])
                batch_logits = model.get_logits(output, normalize=False).float().cpu().numpy()
            logits[ids] = batch_logits
    del model, task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(logits).all():
        raise RuntimeError(f"inference did not fill every {split} manifest row")
    segment_rows = _read_csv(segment_csv)
    if len(segment_rows) != len(logits):
        raise ValueError("segment index and model output lengths differ")
    records, record_logits = _aggregate(logits, segment_rows)
    p_acs = torch.softmax(torch.from_numpy(record_logits[:, :4]), dim=1).numpy()
    p_omi = torch.sigmoid(torch.from_numpy(record_logits[:, 4])).numpy()
    if not np.allclose(p_acs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("ACS probabilities do not sum to one")
    return records, p_acs, p_omi


def read_history(run_root: Path, run_name: str, fold: int | None = None) -> list[list[dict]]:
    from experiments.ecg_lfm.training_reporting import read_fold_history

    if fold is not None and fold not in range(5):
        raise ValueError("fold must be in 0..4")
    return [
        read_fold_history(run_root / run_name / f"fold{k}" / "fairseq_train.jsonl", k)
        if fold is None or k == fold else []
        for k in range(5)
    ]


def _write_oof(path: Path, rows: list[dict]) -> None:
    fields = [
        "record_id",
        "patient_id",
        "fold",
        "master_idx",
        "y_acs",
        "p_OTHER",
        "p_STEMI",
        "p_NSTEMI",
        "p_UA",
        "pred_acs",
        "y_omi",
        "p_omi",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _update_alignment_index(root: Path, run_name: str, alignment: dict) -> None:
    path = root / "provenance_alignment_report.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"task": "corrected ACS categorical + independent OMI", "methods": {}}
    payload["methods"][run_name] = alignment
    payload["all_completed_methods_pass"] = all(
        item.get("status") == "pass" for item in payload["methods"].values()
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--run_root", default="results/corrected_ecg_lfm_joint")
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm")
    parser.add_argument("--label_csv", default="data/CSV/train.csv")
    parser.add_argument("--revalidation_root", default="results/revalidated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--skip_external", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_root = ROOT / args.run_root
    manifests = ROOT / args.manifest_root
    revalidation_root = ROOT / args.revalidation_root
    output = revalidation_root / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    joined_gt, gt_audit = load_record_ground_truth(
        manifests / "master_records.csv", ROOT / args.label_csv
    )
    gt_by_record = {row["record_id"]: row for row in joined_gt}
    expected_all_ids = set(gt_by_record)
    oof_rows: list[dict] = []
    fold_info: list[dict] = []
    seen_ids: set[str] = set()
    started = time.time()
    patient_overlap_total = 0

    for fold in range(5):
        checkpoint = run_root / args.run_name / f"fold{fold}" / "checkpoint_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"best checkpoint missing: {checkpoint}")
        records, p_acs, p_omi = infer_split(
            checkpoint,
            manifests / f"fold{fold}",
            "valid",
            manifests / f"fold{fold}" / "valid_segments.csv",
            device,
            args.workers,
            args.batch,
        )
        expected = sorted(
            (row for row in joined_gt if row["validation_fold"] == fold),
            key=lambda row: row["master_idx"],
        )
        if [row["record_id"] for row in records] != [row["record_id"] for row in expected]:
            raise ValueError(f"fold {fold} predictions do not match master record-ID order")
        train_patients = {row["patient_id"] for row in _read_csv(manifests / f"fold{fold}" / "train_segments.csv")}
        valid_patients = {row["patient_id"] for row in _read_csv(manifests / f"fold{fold}" / "valid_segments.csv")}
        overlap = train_patients.intersection(valid_patients)
        patient_overlap_total += len(overlap)
        if overlap:
            raise ValueError(f"patient leakage in fold {fold}")
        for record, probabilities, omi_probability in zip(records, p_acs, p_omi):
            record_id = record["record_id"]
            if record_id in seen_ids:
                raise ValueError(f"duplicate OOF record ID: {record_id}")
            truth = gt_by_record.get(record_id)
            if truth is None:
                raise ValueError(f"prediction has no record-ID ground truth: {record_id}")
            if record["master_idx"] != truth["master_idx"] or record["patient_id"] != truth["patient_id"]:
                raise ValueError(f"master-index/patient cross-check failed: {record_id}")
            seen_ids.add(record_id)
            oof_rows.append(
                {
                    "record_id": record_id,
                    "patient_id": truth["patient_id"],
                    "fold": fold,
                    "master_idx": truth["master_idx"],
                    "y_acs": truth["y_acs"],
                    "p_OTHER": float(probabilities[0]),
                    "p_STEMI": float(probabilities[1]),
                    "p_NSTEMI": float(probabilities[2]),
                    "p_UA": float(probabilities[3]),
                    "pred_acs": int(np.argmax(probabilities)),
                    "y_omi": truth["y_omi"],
                    "p_omi": float(omi_probability),
                }
            )
        fold_info.append(
            {
                "fold": fold,
                "n_records": int(len(records)),
                "n_train_patients": int(len(train_patients)),
                "n_valid_patients": int(len(valid_patients)),
                "patient_overlap": int(len(overlap)),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "valid_segments_sha256": sha256(manifests / f"fold{fold}" / "valid_segments.csv"),
            }
        )

    missing_prediction_ids = expected_all_ids - seen_ids
    extra_prediction_ids = seen_ids - expected_all_ids
    if missing_prediction_ids or extra_prediction_ids:
        raise RuntimeError(
            f"OOF coverage failure: missing={len(missing_prediction_ids)}, extra={len(extra_prediction_ids)}"
        )
    oof_rows.sort(key=lambda row: int(row["master_idx"]))
    _write_oof(output / "oof_predictions.csv", oof_rows)
    y_acs = np.asarray([row["y_acs"] for row in oof_rows], dtype=np.int64)
    p_acs = np.asarray(
        [[row[f"p_{name}"] for name in ACS_CLASSES] for row in oof_rows], dtype=np.float64
    )
    y_omi = np.asarray([row["y_omi"] for row in oof_rows], dtype=np.int64)
    p_omi = np.asarray([row["p_omi"] for row in oof_rows], dtype=np.float64)

    thresholds: dict[str, float | None] = {}
    threshold_details: dict[str, dict] = {}
    for method in ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90"):
        thresholds[method], threshold_details[method] = select_omi_threshold(y_omi, p_omi, method)
    omi_metrics = {
        method: None if threshold is None else binary_metrics(y_omi, p_omi, threshold)
        for method, threshold in thresholds.items()
    }
    nste_mask = np.isin(y_acs, (2, 3))
    global_youden = thresholds["youden_j"]
    if global_youden is None:
        raise RuntimeError("global OMI Youden threshold is unexpectedly infeasible")
    acs_calibration = multiclass_calibration_metrics(y_acs, p_acs)
    omi_calibration = binary_calibration_metrics(y_omi, p_omi)
    metrics = {
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + OMI binary",
        "run_name": args.run_name,
        "evaluation_type": "descriptive pooled OOF; no nested checkpoint/threshold selection",
        "acs": acs_metrics(y_acs, p_acs),
        "conditional_acs": acs_subtype_metrics(y_acs, p_acs),
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi": omi_metrics,
        "omi_among_nste_acs": {
            "n": int(nste_mask.sum()),
            "global_oof_youden_threshold": float(global_youden),
            "metrics": binary_metrics(y_omi[nste_mask], p_omi[nste_mask], float(global_youden)),
        },
        "calibration": {"acs": acs_calibration, "omi": omi_calibration},
        "thresholds": thresholds,
        "threshold_details": threshold_details,
    }
    alignment = {
        **gt_audit,
        "run_name": args.run_name,
        "status": "pass",
        "duplicate_prediction_record_ids": int(len(oof_rows) - len(seen_ids)),
        "missing_predictions": int(len(missing_prediction_ids)),
        "extra_predictions": int(len(extra_prediction_ids)),
        "missing_labels": int(gt_audit["missing_labels"]),
        "patient_overlap": int(patient_overlap_total),
        "fold_coverage_fraction": float(len(seen_ids) / len(expected_all_ids)),
        "folds": fold_info,
        "evaluator_sha256": sha256(Path(__file__)),
        "task_metrics_sha256": sha256(ROOT / "experiments" / "acs_joint_task.py"),
        "model_source_sha256": sha256(
            ROOT
            / "third_party"
            / "ECG-LFM"
            / "fairseq_signals"
            / "models"
            / "classification"
            / "ecg_transformer_classifier.py"
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "provenance_alignment_report.json").write_text(
        json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_alignment_index(revalidation_root, args.run_name, alignment)
    save_roc_pr_curves(output, y_acs, p_acs, y_omi, p_omi)
    omi_cm = omi_metrics["youden_j"]
    save_confusion_figures(
        output,
        np.asarray(metrics["acs"]["confusion_matrix"]),
        np.asarray(metrics["conditional_acs"]["confusion_matrix"]),
        np.asarray([[omi_cm["tn"], omi_cm["fp"]], [omi_cm["fn"], omi_cm["tp"]]]),
    )
    save_reliability_figures(output, acs_calibration, omi_calibration)
    save_curves(output, read_history(run_root, args.run_name), f"Revalidated ECG-LFM {args.run_name}")

    if not args.skip_external:
        external_probabilities_acs = []
        external_probabilities_omi = []
        external_records: list[dict] | None = None
        for fold in range(5):
            checkpoint = run_root / args.run_name / f"fold{fold}" / "checkpoint_best.pt"
            records, probabilities, omi_probability = infer_split(
                checkpoint,
                manifests / f"fold{fold}",
                "external",
                manifests / f"fold{fold}" / "external_segments.csv",
                device,
                args.workers,
                args.batch,
            )
            if external_records is None:
                external_records = records
            elif [row["record_id"] for row in records] != [row["record_id"] for row in external_records]:
                raise ValueError("external manifest record IDs differ across folds")
            external_probabilities_acs.append(probabilities)
            external_probabilities_omi.append(omi_probability)
        ensemble_acs = np.mean(external_probabilities_acs, axis=0)
        ensemble_omi = np.mean(external_probabilities_omi, axis=0)
        test_ids = [_record_id(row.get("ecg_row_record", "")) for row in _read_csv(ROOT / "data" / "CSV" / "test.csv")]
        external_by_id = {row["record_id"]: index for index, row in enumerate(external_records or [])}
        if len(external_by_id) != len(external_records or []) or set(test_ids) != set(external_by_id):
            raise ValueError("external record IDs do not uniquely match test.csv")
        order = np.asarray([external_by_id[record_id] for record_id in test_ids])
        ensemble_acs = ensemble_acs[order]
        ensemble_omi = ensemble_omi[order]
        predictions = np.zeros((len(test_ids), 4), dtype=np.int8)
        state = ensemble_acs.argmax(axis=1)
        for class_index in (1, 2, 3):
            predictions[:, class_index - 1] = (state == class_index).astype(np.int8)
        predictions[:, 3] = (ensemble_omi >= float(global_youden)).astype(np.int8)
        submission = output / "submission_corrected.csv"
        with submission.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ecg_row_record", *LABELS])
            for record_id, row in zip(test_ids, predictions):
                writer.writerow([f"{record_id}.dat", *row.tolist()])
        (output / "external_summary.json").write_text(
            json.dumps(
                {
                    "external_labels_used": False,
                    "omi_threshold": float(global_youden),
                    "submission": str(submission),
                    "submission_sha256": sha256(submission),
                    "predicted_counts": predictions.sum(axis=0).astype(int).tolist(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "status": "pass",
                "metrics": str(output / "metrics.json"),
                "alignment": str(output / "provenance_alignment_report.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
