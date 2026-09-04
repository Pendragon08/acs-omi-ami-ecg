"""Fail-closed audit and materialization of the observed Joint-6 labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.ecg_lfm.joint6 import JOINT6_CLASSES, encode_source_labels, one_hot_joint6


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_assignment_hash(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{row['fold']}\t{row['subset']}\t{row['record_id']}\t{row['patient_id']}"
        for row in sorted(rows, key=lambda item: (int(item["fold"]), item["subset"], item["record_id"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--source_labels", default="manifests/ecg_lfm_nested/y_all.npy")
    parser.add_argument("--master_csv", default="manifests/ecg_lfm/master_records.csv")
    parser.add_argument("--output", default="results/joint6_cv/label_audit")
    args = parser.parse_args()

    manifest_root = ROOT / args.manifest_root
    source_path = ROOT / args.source_labels
    master_path = ROOT / args.master_csv
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(f"source labels must be [N,4], got {source.shape}")
    if not np.isfinite(source).all() or not np.isin(source, (0, 1)).all():
        raise ValueError("source labels must be finite exact binary values")
    acs, omi, joint, exception = encode_source_labels(source)
    other_omi = int(exception.sum())
    ua_omi = int(((acs == 3) & (omi == 1)).sum())
    if other_omi != 1 or ua_omi != 0:
        raise RuntimeError(f"Joint-6 hard guard failed: OTHER+OMI={other_omi}, UA+OMI={ua_omi}")

    label_path = manifest_root / "y_joint_6.npy"
    materialized = one_hot_joint6(joint)
    if label_path.exists():
        existing = np.load(label_path, mmap_mode="r")
        if existing.shape != materialized.shape or not np.array_equal(existing, materialized):
            raise RuntimeError("existing y_joint_6.npy differs from audited mapping")
    else:
        np.save(label_path, materialized)

    master = read_csv(master_path)
    if len(master) != len(source):
        raise RuntimeError(f"master/source length mismatch: {len(master)} != {len(source)}")
    by_record: dict[str, dict] = {}
    train_indices: list[int] = []
    for row in master:
        idx = int(row["idx"])
        if row.get("cohort") != "train":
            continue
        train_indices.append(idx)
        record_id = str(row["record_id"]).zfill(5)
        if record_id in by_record:
            raise RuntimeError(f"duplicate master record_id: {record_id}")
        by_record[record_id] = {
            "idx": idx,
            "patient_id": row["patient_id"],
            "acs": ("OTHER", "STEMI", "NSTEMI", "UA")[int(acs[idx])],
            "omi": int(omi[idx]),
            "joint": int(joint[idx]),
        }

    train_indices_array = np.asarray(train_indices, dtype=np.int64)
    class_rows = []
    for class_id, name in enumerate(JOINT6_CLASSES):
        selected_indices = train_indices_array[joint[train_indices_array] == class_id]
        patient_count = len({master[index]["patient_id"] for index in selected_indices})
        class_rows.append({
            "joint6_id": class_id,
            "joint6_class": name,
            "ecg_count": int(len(selected_indices)),
            "unique_patient_count": patient_count,
            "prevalence": float(len(selected_indices) / len(train_indices_array)),
        })
    write_csv(output / "joint6_class_counts.csv", class_rows)

    fold_rows: list[dict] = []
    cv_rows: list[dict] = []
    outer_records: list[set[str]] = []
    outer_patients: list[set[str]] = []
    for fold in range(5):
        fold_dir = manifest_root / f"fold{fold}"
        assignments = read_csv(fold_dir / "patient_assignments.csv")
        provenance = json.loads((fold_dir / "split_provenance.json").read_text(encoding="utf-8"))
        if canonical_assignment_hash(assignments) != provenance["assignment_sha256"]:
            raise RuntimeError(f"fold{fold} canonical assignment hash mismatch")
        if len(assignments) != len(by_record):
            raise RuntimeError(f"fold{fold} assignment coverage mismatch")
        subset_records: dict[str, set[str]] = defaultdict(set)
        subset_patients: dict[str, set[str]] = defaultdict(set)
        buckets: dict[tuple[str, int], dict[str, object]] = defaultdict(
            lambda: {"records": 0, "patients": set()}
        )
        for assignment in assignments:
            record_id = str(assignment["record_id"]).zfill(5)
            expected = by_record.get(record_id)
            if expected is None:
                raise RuntimeError(f"fold{fold} unknown record: {record_id}")
            if int(assignment["fold"]) != fold or int(assignment["idx"]) != expected["idx"]:
                raise RuntimeError(f"fold{fold} assignment fold/idx mismatch: {record_id}")
            if assignment["patient_id"] != expected["patient_id"]:
                raise RuntimeError(f"fold{fold} patient mismatch: {record_id}")
            if assignment["acs"] != expected["acs"] or int(assignment["omi"]) != expected["omi"]:
                raise RuntimeError(f"fold{fold} label mismatch: {record_id}")
            subset = assignment["subset"]
            if subset not in {"inner_train", "inner_validation", "outer_test"}:
                raise RuntimeError(f"fold{fold} invalid subset: {subset}")
            subset_records[subset].add(record_id)
            subset_patients[subset].add(assignment["patient_id"])
            bucket = buckets[(subset, expected["joint"])]
            bucket["records"] = int(bucket["records"]) + 1
            bucket["patients"].add(assignment["patient_id"])
        for left, right in (("inner_train", "inner_validation"), ("inner_train", "outer_test"), ("inner_validation", "outer_test")):
            if subset_patients[left] & subset_patients[right]:
                raise RuntimeError(f"fold{fold} patient overlap: {left}/{right}")
        for subset in ("inner_train", "inner_validation", "outer_test"):
            total = len(subset_records[subset])
            for class_id, name in enumerate(JOINT6_CLASSES):
                item = buckets[(subset, class_id)]
                fold_rows.append({
                    "fold": fold,
                    "subset": subset,
                    "joint6_id": class_id,
                    "joint6_class": name,
                    "ecg_count": int(item["records"]),
                    "unique_patient_count": len(item["patients"]),
                    "prevalence": int(item["records"]) / total,
                })
            cv_rows.append({
                "fold": fold,
                "subset": subset,
                "record_count": total,
                "patient_count": len(subset_patients[subset]),
                "record_ids_sha256": sha256_values(subset_records[subset]),
                "patient_ids_sha256": sha256_values(subset_patients[subset]),
                "canonical_assignment_sha256": provenance["assignment_sha256"],
                "split_seed": int(provenance["seed"]),
            })
        outer_records.append(subset_records["outer_test"])
        outer_patients.append(subset_patients["outer_test"])

    if len(set().union(*outer_records)) != len(by_record) or sum(map(len, outer_records)) != len(by_record):
        raise RuntimeError("outer-test records are not a disjoint complete five-fold partition")
    all_patients = {row["patient_id"] for row in master if row.get("cohort") == "train"}
    if set().union(*outer_patients) != all_patients or sum(map(len, outer_patients)) != len(all_patients):
        raise RuntimeError("outer-test patients are not a disjoint complete five-fold partition")
    write_csv(output / "fold_joint6_counts.csv", fold_rows)
    write_csv(output / "cv_fingerprint.csv", cv_rows)

    summary = {
        "status": "pass",
        "formulation": "joint6_observed_outcome",
        "mapping": {str(index): name for index, name in enumerate(JOINT6_CLASSES)},
        "known_label_training_records": len(by_record),
        "external_unknown_placeholder_records": len(source) - len(by_record),
        "unique_patients": len(all_patients),
        "other_omi_exception_count": other_omi,
        "ua_omi_count": ua_omi,
        "patient_overlap": 0,
        "outer_record_coverage": len(set().union(*outer_records)),
        "outer_patient_coverage": len(set().union(*outer_patients)),
        "source_label_sha256": sha256_file(source_path),
        "joint6_label_sha256": sha256_file(label_path),
        "master_sha256": sha256_file(master_path),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
