"""Freeze one patient-level inner split inside each existing outer fold.

The current outer folds are treated as immutable.  Only their training
patients are split into inner-train/inner-validation.  The historical outer
validation subset is copied verbatim as ``outer_test`` and is never used for
checkpoint, threshold, calibration, or loss-hyperparameter selection.

Patient/record assignments are execution artifacts and must not enter Git.
The aggregate summary and canonical assignment hash are safe provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


ACS_NAMES = ("OTHER", "STEMI", "NSTEMI", "UA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_records(segment_rows: list[dict[str, str]], labels: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in segment_rows:
        record_id = row["record_id"]
        if record_id in seen:
            continue
        seen.add(record_id)
        index = int(row["idx"])
        target = labels[index]
        if target.shape != (5,) or int(target[:4].sum()) != 1:
            raise ValueError(f"invalid joint target for record {record_id}")
        acs = int(np.argmax(target[:4]))
        omi = int(target[4])
        records.append(
            {
                "record_id": record_id,
                "patient_id": row["patient_id"],
                "idx": index,
                "acs": acs,
                "omi": omi,
                "stratum": acs * 2 + omi,
            }
        )
    return records


def subset_segments(rows: list[dict[str, str]], record_ids: set[str]) -> list[dict[str, object]]:
    selected = [dict(row) for row in rows if row["record_id"] in record_ids]
    for position, row in enumerate(selected):
        row["manifest_position"] = position
    return selected


def write_tsv(path: Path, signal_root: str, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(signal_root.rstrip("\n") + "\n")
        for row in rows:
            handle.write(f"{row['mat_file']}\t2500\n")


def label_counts(records: list[dict[str, object]]) -> dict[str, object]:
    acs = Counter(int(row["acs"]) for row in records)
    omi = Counter(int(row["omi"]) for row in records)
    return {
        "records": len(records),
        "patients": len({str(row["patient_id"]) for row in records}),
        "acs": {name: int(acs[index]) for index, name in enumerate(ACS_NAMES)},
        "omi": {"NOMI": int(omi[0]), "OMI": int(omi[1])},
    }


def assignment_hash(rows: list[dict[str, object]]) -> str:
    canonical = "\n".join(
        f"{row['fold']}\t{row['subset']}\t{row['record_id']}\t{row['patient_id']}"
        for row in sorted(rows, key=lambda item: (int(item["fold"]), str(item["subset"]), str(item["record_id"])))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve() and sha256(destination) != sha256(source):
            raise ValueError(f"existing label artifact differs: {destination}")
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def build_fold(source_root: Path, destination_root: Path, fold: int, labels: np.ndarray, seed: int) -> dict:
    source = source_root / f"fold{fold}"
    destination = destination_root / f"fold{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    train_segments = read_csv(source / "train_segments.csv")
    outer_segments = read_csv(source / "valid_segments.csv")
    records = unique_records(train_segments, labels)
    y = np.asarray([row["stratum"] for row in records], dtype=np.int64)
    groups = np.asarray([row["patient_id"] for row in records], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_train_index, inner_valid_index = next(splitter.split(np.zeros(len(records)), y, groups))
    train_ids = {str(records[index]["record_id"]) for index in inner_train_index}
    valid_ids = {str(records[index]["record_id"]) for index in inner_valid_index}
    outer_records = unique_records(outer_segments, labels)
    outer_ids = {str(row["record_id"]) for row in outer_records}
    inner_train_records = [row for row in records if str(row["record_id"]) in train_ids]
    inner_valid_records = [row for row in records if str(row["record_id"]) in valid_ids]
    patient_sets = {
        "inner_train": {str(row["patient_id"]) for row in inner_train_records},
        "inner_validation": {str(row["patient_id"]) for row in inner_valid_records},
        "outer_test": {str(row["patient_id"]) for row in outer_records},
    }
    overlaps = {
        "inner_train__inner_validation": len(patient_sets["inner_train"] & patient_sets["inner_validation"]),
        "inner_train__outer_test": len(patient_sets["inner_train"] & patient_sets["outer_test"]),
        "inner_validation__outer_test": len(patient_sets["inner_validation"] & patient_sets["outer_test"]),
    }
    if train_ids & valid_ids or train_ids & outer_ids or valid_ids & outer_ids or any(overlaps.values()):
        raise RuntimeError(f"nested split leakage in outer fold {fold}: {overlaps}")
    if train_ids | valid_ids != {str(row["record_id"]) for row in records}:
        raise RuntimeError(f"inner split does not cover outer-train in fold {fold}")

    signal_root = (source / "train.tsv").read_text(encoding="utf-8").splitlines()[0]
    segment_fields = list(train_segments[0])
    subsets = {
        "train": subset_segments(train_segments, train_ids),
        "valid": subset_segments(train_segments, valid_ids),
        "outer_test": subset_segments(outer_segments, outer_ids),
    }
    for name, rows in subsets.items():
        write_csv(destination / f"{name}_segments.csv", rows, segment_fields)
        write_tsv(destination / f"{name}.tsv", signal_root, rows)

    assignment_rows: list[dict[str, object]] = []
    for subset, subset_records in (
        ("inner_train", inner_train_records),
        ("inner_validation", inner_valid_records),
        ("outer_test", outer_records),
    ):
        for row in subset_records:
            assignment_rows.append(
                {
                    "fold": fold,
                    "subset": subset,
                    "record_id": row["record_id"],
                    "patient_id": row["patient_id"],
                    "idx": row["idx"],
                    "acs": ACS_NAMES[int(row["acs"])],
                    "omi": int(row["omi"]),
                }
            )
    write_csv(
        destination / "patient_assignments.csv",
        assignment_rows,
        ["fold", "subset", "record_id", "patient_id", "idx", "acs", "omi"],
    )
    summary = {
        "outer_fold": fold,
        "seed": seed,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "stratification_target": "8-state ACS_class * 2 + OMI",
        "selected_inner_fold": 0,
        "assignment_sha256": assignment_hash(assignment_rows),
        "source_train_segments_sha256": sha256(source / "train_segments.csv"),
        "source_outer_test_segments_sha256": sha256(source / "valid_segments.csv"),
        "subsets": {
            "inner_train": label_counts(inner_train_records),
            "inner_validation": label_counts(inner_valid_records),
            "outer_test": label_counts(outer_records),
        },
        "patient_overlap": overlaps,
        "coverage": {
            "outer_train_records": len(records),
            "inner_union_records": len(train_ids | valid_ids),
            "outer_test_records": len(outer_ids),
        },
        "files": {
            name: sha256(destination / name)
            for name in (
                "train.tsv",
                "valid.tsv",
                "outer_test.tsv",
                "train_segments.csv",
                "valid_segments.csv",
                "outer_test_segments.csv",
                "patient_assignments.csv",
            )
        },
    }
    (destination / "split_provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, default=Path("manifests/ecg_lfm"))
    parser.add_argument("--destination_root", type=Path, default=Path("manifests/ecg_lfm_nested"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    labels = np.load(source_root / "y_joint_5.npy", mmap_mode="r")
    for name in ("y_all.npy", "y_joint_5.npy", "master_records.csv"):
        link_or_copy(source_root / name, destination_root / name)
    summaries = [build_fold(source_root, destination_root, fold, labels, args.seed) for fold in args.folds]
    aggregate = {
        "status": "pass" if all(not any(item["patient_overlap"].values()) for item in summaries) else "fail",
        "source_outer_folds": str(source_root),
        "destination_nested_folds": str(destination_root),
        "seed": args.seed,
        "folds": summaries,
    }
    (destination_root / "nested_split_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": aggregate["status"], "folds": args.folds}, ensure_ascii=False))
    if aggregate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
