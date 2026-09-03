"""Audit whether the frozen ACS/OMI cohort can support Nature-style Joint-10.

This module is deliberately a pre-training gate.  It never modifies labels,
manifests, folds, checkpoints, or ECG waveforms.  Patient-level detail is
written only to the controlled server result directory and must not enter Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ACS_CLASSES = ("OTHER", "STEMI", "NSTEMI", "UA")
JOINT10_CLASSES = (
    "OTHER",
    "UA",
    "STEMI+nOMI",
    "NSTEMI+nOMI",
    "STEMI+OMI+LM_LAD",
    "STEMI+OMI+LCX",
    "STEMI+OMI+RCA",
    "NSTEMI+OMI+LM_LAD",
    "NSTEMI+OMI+LCX",
    "NSTEMI+OMI+RCA",
)
CORONARY_GROUPS = {
    "LM_LAD": ("LM", "PLAD", "MLAD", "DLAD", "DB"),
    "LCX": ("PLCX", "MLCX", "DLCX", "OM"),
    "RCA": ("PRCA", "MRCA", "DRCA"),
}
CORONARY_FIELDS = tuple(field for fields in CORONARY_GROUPS.values() for field in fields)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_assignment_hash(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{row['fold']}\t{row['subset']}\t{row['record_id']}\t{row['patient_id']}"
        for row in sorted(rows, key=lambda item: (int(item["fold"]), item["subset"], item["record_id"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_binary(row: dict[str, str], field: str) -> int:
    value = int(row[field])
    if value not in (0, 1):
        raise ValueError(f"{field} must be binary, got {value}")
    return value


def acs_label(row: dict[str, str]) -> str:
    active = [name for name in ("STEMI", "NSTEMI", "UA") if parse_binary(row, name)]
    if len(active) > 1:
        raise ValueError(f"non-mutually-exclusive ACS labels: {active}")
    return active[0] if active else "OTHER"


def acute_and_cto(row: dict[str, str]) -> tuple[dict[str, list[str]], list[str]]:
    acute = {
        group: [field for field in fields if int(row[field]) == 1]
        for group, fields in CORONARY_GROUPS.items()
    }
    cto_fields = [field for field in CORONARY_FIELDS if int(row[field]) == 2]
    return acute, cto_fields


def resolve_culprit_group(row: dict[str, str]) -> dict[str, object]:
    """Resolve only an unambiguous large-vessel acute group.

    Multiple positive segments inside one large-vessel group are allowed.
    Multiple positive large-vessel groups are never guessed.
    Value 2 is CTO and is excluded from acute evidence.
    """
    acute, cto_fields = acute_and_cto(row)
    groups = [group for group, fields in acute.items() if fields]
    if len(groups) == 1:
        status = "UNIQUE_GROUP"
        group = groups[0]
    elif not groups:
        status = "NO_GROUP"
        group = None
    else:
        status = "MULTI_GROUP"
        group = None
    return {
        "status": status,
        "culprit_group": group,
        "acute_groups": groups,
        "acute_fields": [field for fields in acute.values() for field in fields],
        "cto_fields": cto_fields,
    }


def candidate_joint10(row: dict[str, str]) -> dict[str, object]:
    acs = acs_label(row)
    omi = parse_binary(row, "OMI")
    resolution = resolve_culprit_group(row)
    result: dict[str, object] = {
        "acs": acs,
        "omi": omi,
        "joint10_id": None,
        "joint10_class": None,
        "exception": None,
        **resolution,
    }
    if acs == "UA" and omi:
        result["status"] = "UA_OMI_INVALID"
        return result
    if acs == "OTHER":
        result["joint10_id"] = 0
        result["joint10_class"] = JOINT10_CLASSES[0]
        if omi:
            result["exception"] = "OTHER_OMI"
            result["status"] = "OTHER_OMI_EXCEPTION"
        return result
    if acs == "UA":
        result["joint10_id"] = 1
        result["joint10_class"] = JOINT10_CLASSES[1]
        return result
    if not omi:
        joint_id = 2 if acs == "STEMI" else 3
        result["joint10_id"] = joint_id
        result["joint10_class"] = JOINT10_CLASSES[joint_id]
        return result
    group = resolution["culprit_group"]
    if group is None:
        return result
    offset = {"LM_LAD": 0, "LCX": 1, "RCA": 2}[str(group)]
    joint_id = (4 if acs == "STEMI" else 7) + offset
    result["joint10_id"] = joint_id
    result["joint10_class"] = JOINT10_CLASSES[joint_id]
    return result


def aggregate(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for row in rows:
        name = str(row[key])
        item = buckets.setdefault(name, {"ecg_count": 0, "patients": set()})
        item["ecg_count"] = int(item["ecg_count"]) + 1
        cast_patients = item["patients"]
        assert isinstance(cast_patients, set)
        cast_patients.add(str(row["patient_id"]))
    return [
        {key: name, "ecg_count": int(item["ecg_count"]), "unique_patient_count": len(item["patients"])}
        for name, item in sorted(buckets.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_csv", type=Path, default=Path("data/CSV/train.csv"))
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm_nested"))
    parser.add_argument("--master_csv", type=Path, default=Path("manifests/ecg_lfm/master_records.csv"))
    parser.add_argument("--source_labels", type=Path, default=Path("manifests/ecg_lfm_nested/y_all.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/joint10_label_audit"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()

    label_csv = (ROOT / args.label_csv).resolve()
    manifest_root = (ROOT / args.manifest_root).resolve()
    master_csv = (ROOT / args.master_csv).resolve()
    source_labels_path = (ROOT / args.source_labels).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv(label_csv)
    required = {"Patient_id", "ecg_row_record", "AMI", "STEMI", "NSTEMI", "UA", "OMI", "CTO", *CORONARY_FIELDS}
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise RuntimeError(f"label CSV is missing required columns: {missing}")
    record_ids = [Path(row["ecg_row_record"]).stem.zfill(5) for row in raw_rows]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("duplicate record_id in label CSV")

    master = read_csv(master_csv)
    source_labels = np.load(source_labels_path, mmap_mode="r")
    if len(master) != len(source_labels):
        raise RuntimeError(f"master/source array length mismatch: {len(master)} != {len(source_labels)}")
    if source_labels.ndim != 2 or source_labels.shape[1] != 4:
        raise RuntimeError(f"source labels must have exact shape [N,4], got {source_labels.shape}")
    if not np.isfinite(source_labels).all() or not np.isin(source_labels, (0, 1)).all():
        raise RuntimeError("source labels must be finite exact binary values")
    label_by_record = {record_id: row for record_id, row in zip(record_ids, raw_rows)}
    known_master = [row for row in master if row["cohort"] == "train"]
    known_record_ids = {str(row["record_id"]).zfill(5) for row in known_master}
    master_by_record = {str(row["record_id"]).zfill(5): row for row in known_master}
    if known_record_ids != set(record_ids):
        raise RuntimeError(
            f"master/CSV record coverage mismatch: missing={len(set(record_ids) - known_record_ids)}, "
            f"extra={len(known_record_ids - set(record_ids))}"
        )
    provenance_errors: list[dict[str, object]] = []
    for row in known_master:
        idx = int(row["idx"])
        record_id = str(row["record_id"]).zfill(5)
        raw = label_by_record.get(record_id)
        expected = None if raw is None else np.asarray([int(raw[name]) for name in ("STEMI", "NSTEMI", "UA", "OMI")])
        observed = np.asarray(source_labels[idx])
        if raw is None or row["patient_id"] != raw["Patient_id"] or not np.array_equal(expected, observed):
            provenance_errors.append({"idx": idx, "record_id": record_id, "reason": "master/source/CSV mismatch"})
    if provenance_errors:
        write_csv(output / "provenance_errors.csv", provenance_errors)
        raise RuntimeError(f"label provenance failed for {len(provenance_errors)} records")

    audited: list[dict[str, object]] = []
    invalid_acs = 0
    for record_id, row in zip(record_ids, raw_rows):
        try:
            mapped = candidate_joint10(row)
        except ValueError:
            invalid_acs += 1
            continue
        audited.append({
            "idx": int(master_by_record[record_id]["idx"]),
            "record_id": record_id,
            "patient_id": row["Patient_id"],
            "ami": parse_binary(row, "AMI"),
            "cto_flag": int(row["CTO"]),
            **mapped,
        })
    if invalid_acs:
        raise RuntimeError(f"found {invalid_acs} invalid ACS label rows")

    # Aggregate-only field dictionary and value audit are Git-safe.
    field_rows = []
    value_rows = []
    field_group = {field: group for group, fields in CORONARY_GROUPS.items() for field in fields}
    for field in CORONARY_FIELDS:
        counts = Counter(int(row[field]) for row in raw_rows)
        unexpected = sorted(value for value in counts if value not in (0, 1, 2))
        if unexpected:
            raise RuntimeError(f"unexpected values in {field}: {unexpected}")
        field_rows.append({
            "field": field,
            "large_vessel_group": field_group[field],
            "value_0": "not encoded as acute value 1 or CTO value 2 at this segment",
            "value_1": "significant stenosis with acute occlusion in OMI cases",
            "value_2": "chronic total occlusion (CTO), not acute culprit evidence",
        })
        for value in (0, 1, 2):
            value_rows.append({"field": field, "value": value, "ecg_count": counts[value]})
    write_csv(output / "raw_coronary_fields.csv", field_rows)
    write_csv(output / "coronary_field_values.csv", value_rows)
    write_csv(output / "metadata_columns.csv", [
        {"column_position": index, "column_name": name}
        for index, name in enumerate(raw_rows[0].keys())
    ])

    omi_rows = [row for row in audited if int(row["omi"]) == 1]
    ami_without_subtype = [row for row in audited if int(row["ami"]) == 1 and row["acs"] == "OTHER"]
    subtype_without_ami = [row for row in audited if int(row["ami"]) == 0 and row["acs"] in ("STEMI", "NSTEMI")]
    other_omi = [row for row in omi_rows if row["acs"] == "OTHER"]
    ua_omi = [row for row in omi_rows if row["acs"] == "UA"]
    standard_omi = [row for row in omi_rows if row["acs"] in ("STEMI", "NSTEMI")]
    unique = [row for row in standard_omi if row["status"] == "UNIQUE_GROUP"]
    multigroup = [row for row in standard_omi if row["status"] == "MULTI_GROUP"]
    no_group = [row for row in standard_omi if row["status"] == "NO_GROUP"]
    unresolved = multigroup + no_group
    with_cto = [row for row in omi_rows if int(row["cto_flag"]) == 1 or bool(row["cto_fields"])]
    all_with_cto_flag = [row for row in audited if int(row["cto_flag"]) == 1]
    all_with_segment_cto = [row for row in audited if bool(row["cto_fields"])]
    omi_with_cto_flag = [row for row in omi_rows if int(row["cto_flag"]) == 1]
    omi_with_segment_cto = [row for row in omi_rows if bool(row["cto_fields"])]
    omi_with_acute_and_cto = [row for row in omi_rows if bool(row["acute_fields"]) and bool(row["cto_fields"])]

    location_rows = []
    for row in omi_rows:
        location = str(row["culprit_group"] or row["status"])
        location_rows.append({**row, "location_or_status": location})
    write_csv(output / "omi_location_counts.csv", aggregate(location_rows, "location_or_status"))
    write_csv(output / "omi_by_acs.csv", aggregate(omi_rows, "acs"))
    write_csv(output / "ami_subtype_audit.csv", [
        {"status": "AMI_WITHOUT_STEMI_OR_NSTEMI", "ecg_count": len(ami_without_subtype),
         "unique_patient_count": len({str(row["patient_id"]) for row in ami_without_subtype})},
        {"status": "STEMI_OR_NSTEMI_WITHOUT_AMI", "ecg_count": len(subtype_without_ami),
         "unique_patient_count": len({str(row["patient_id"]) for row in subtype_without_ami})},
    ])
    by_acs_location: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"records": 0, "patients": set()})
    for row in location_rows:
        key = (str(row["acs"]), str(row["location_or_status"]))
        by_acs_location[key]["records"] = int(by_acs_location[key]["records"]) + 1
        patients = by_acs_location[key]["patients"]
        assert isinstance(patients, set)
        patients.add(str(row["patient_id"]))
    write_csv(output / "omi_location_by_acs.csv", [
        {"acs": acs, "location_or_status": location, "ecg_count": int(item["records"]),
         "unique_patient_count": len(item["patients"])}
        for (acs, location), item in sorted(by_acs_location.items())
    ])
    combination_rows = []
    multigroup_buckets: dict[str, dict[str, object]] = defaultdict(lambda: {"records": 0, "patients": set()})
    for row in multigroup:
        combination = "+".join(str(value) for value in row["acute_groups"])
        multigroup_buckets[combination]["records"] = int(multigroup_buckets[combination]["records"]) + 1
        patients = multigroup_buckets[combination]["patients"]
        assert isinstance(patients, set)
        patients.add(str(row["patient_id"]))
    for combination, item in sorted(multigroup_buckets.items()):
        combination_rows.append({"acute_group_combination": combination,
                                 "ecg_count": int(item["records"]),
                                 "unique_patient_count": len(item["patients"])})
    write_csv(output / "multigroup_combinations.csv", combination_rows,
              ["acute_group_combination", "ecg_count", "unique_patient_count"])

    # These three contain identifiers and therefore remain server-side only.
    detail_fields = ["record_id", "patient_id", "acs", "omi", "status", "culprit_group",
                     "acute_groups", "acute_fields", "cto_fields", "cto_flag", "exception"]
    def detail(records: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{field: json.dumps(row[field]) if isinstance(row.get(field), list) else row.get(field) for field in detail_fields}
                for row in records]
    write_csv(output / "unresolved_omi.csv", detail(unresolved), detail_fields)
    write_csv(output / "multigroup_omi.csv", detail(multigroup), detail_fields)
    write_csv(output / "cto_cases.csv", detail(with_cto), detail_fields)

    class_rows = []
    patient_rows = []
    total = len(audited)
    for joint_id, name in enumerate(JOINT10_CLASSES):
        selected = [row for row in audited if row["joint10_id"] == joint_id]
        class_rows.append({"joint10_id": joint_id, "joint10_class": name, "ecg_count": len(selected),
                           "unique_patient_count": len({str(row["patient_id"]) for row in selected}),
                           "prevalence_in_full_cohort": len(selected) / total})
        patient_rows.append({"joint10_id": joint_id, "joint10_class": name,
                             "unique_patient_count": len({str(row["patient_id"]) for row in selected})})
    write_csv(output / "joint10_class_counts.csv", class_rows)
    write_csv(output / "joint10_patient_counts.csv", patient_rows)

    audited_by_record = {str(row["record_id"]): row for row in audited}
    fold_count_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    split_errors: list[str] = []
    split_seeds: set[int] = set()
    outer_record_sets: list[set[str]] = []
    outer_patient_sets: list[set[str]] = []
    for fold in args.folds:
        fold_dir = manifest_root / f"fold{fold}"
        assignments = read_csv(fold_dir / "patient_assignments.csv")
        assignment_record_ids = {str(row["record_id"]).zfill(5) for row in assignments}
        if assignment_record_ids != set(record_ids) or len(assignments) != len(assignment_record_ids):
            raise RuntimeError(
                f"fold{fold} assignment coverage mismatch or duplicate IDs: "
                f"rows={len(assignments)}, unique={len(assignment_record_ids)}"
            )
        split_provenance = json.loads((fold_dir / "split_provenance.json").read_text(encoding="utf-8"))
        split_seeds.add(int(split_provenance["seed"]))
        if canonical_assignment_hash(assignments) != split_provenance["assignment_sha256"]:
            raise RuntimeError(f"fold{fold} canonical assignment hash mismatch")
        recorded_file_hash = split_provenance.get("files", {}).get("patient_assignments.csv")
        if recorded_file_hash and recorded_file_hash != sha256_file(fold_dir / "patient_assignments.csv"):
            raise RuntimeError(f"fold{fold} assignment file hash mismatch")
        subset_sets: dict[str, dict[str, set[str]]] = {}
        buckets: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"records": 0, "patients": set()})
        for assignment in assignments:
            record_id = str(assignment["record_id"]).zfill(5)
            row = audited_by_record.get(record_id)
            if row is None:
                split_errors.append(f"fold{fold}: missing source label for {record_id}")
                continue
            subset = assignment["subset"]
            if int(assignment["fold"]) != fold:
                split_errors.append(f"fold{fold}: assignment embeds fold={assignment['fold']}")
            if subset not in ("inner_train", "inner_validation", "outer_test"):
                split_errors.append(f"fold{fold}: invalid subset={subset}")
            if int(assignment["idx"]) != int(row["idx"]):
                split_errors.append(f"fold{fold}: idx mismatch for {record_id}")
            if assignment["patient_id"] != row["patient_id"]:
                split_errors.append(f"fold{fold}: patient mismatch for {record_id}")
            if assignment["acs"] != row["acs"] or int(assignment["omi"]) != int(row["omi"]):
                split_errors.append(f"fold{fold}: assignment label mismatch for {record_id}")
            subset_entry = subset_sets.setdefault(subset, {"records": set(), "patients": set()})
            subset_entry["records"].add(record_id)
            subset_entry["patients"].add(assignment["patient_id"])
            if row["joint10_class"] is not None:
                category = str(row["joint10_class"])
            else:
                category = f"UNRESOLVED_{row['status']}"
            bucket = buckets[(subset, category)]
            bucket["records"] = int(bucket["records"]) + 1
            patients = bucket["patients"]
            assert isinstance(patients, set)
            patients.add(assignment["patient_id"])
        for subset in ("inner_train", "inner_validation", "outer_test"):
            subset_total = len(subset_sets.get(subset, {}).get("records", set()))
            categories = list(JOINT10_CLASSES) + ["UNRESOLVED_NO_GROUP", "UNRESOLVED_MULTI_GROUP", "UNRESOLVED_UA_OMI_INVALID"]
            for category in categories:
                item = buckets.get((subset, category), {"records": 0, "patients": set()})
                fold_count_rows.append({"fold": fold, "subset": subset, "joint10_class_or_status": category,
                                        "ecg_count": int(item["records"]),
                                        "unique_patient_count": len(item["patients"]),
                                        "prevalence": int(item["records"]) / subset_total if subset_total else 0.0})
            record_set = subset_sets.get(subset, {}).get("records", set())
            patient_set = subset_sets.get(subset, {}).get("patients", set())
            cv_rows.append({"fold": fold, "subset": subset, "record_count": len(record_set),
                            "patient_count": len(patient_set), "record_ids_sha256": sha256_values(record_set),
                            "patient_ids_sha256": sha256_values(patient_set),
                            "assignment_file_sha256": sha256_file(fold_dir / "patient_assignments.csv"),
                            "canonical_assignment_sha256": split_provenance["assignment_sha256"],
                            "split_seed": split_provenance["seed"], "patient_overlap_count": 0})
        subsets = {key: value["patients"] for key, value in subset_sets.items()}
        outer_record_sets.append(set(subset_sets.get("outer_test", {}).get("records", set())))
        outer_patient_sets.append(set(subset_sets.get("outer_test", {}).get("patients", set())))
        for left, right in (("inner_train", "inner_validation"), ("inner_train", "outer_test"),
                            ("inner_validation", "outer_test")):
            overlap = subsets.get(left, set()) & subsets.get(right, set())
            if overlap:
                split_errors.append(f"fold{fold}: {left}/{right} patient overlap={len(overlap)}")
    if len(split_seeds) != 1:
        split_errors.append(f"fold split seeds differ: {sorted(split_seeds)}")
    outer_record_union = set().union(*outer_record_sets)
    outer_patient_union = set().union(*outer_patient_sets)
    expected_patients = {str(row["patient_id"]) for row in audited}
    if outer_record_union != set(record_ids) or sum(map(len, outer_record_sets)) != len(outer_record_union):
        split_errors.append("outer-test record folds are not disjoint complete coverage")
    if outer_patient_union != expected_patients or sum(map(len, outer_patient_sets)) != len(outer_patient_union):
        split_errors.append("outer-test patient folds are not disjoint complete coverage")
    if split_errors:
        raise RuntimeError("; ".join(split_errors[:20]))
    write_csv(output / "joint10_counts_by_fold.csv", fold_count_rows)
    write_csv(output / "cv_fingerprint.csv", cv_rows)

    rules = {
        "status": "audited_not_activated",
        "joint10_class_order": {str(index): name for index, name in enumerate(JOINT10_CLASSES)},
        "coronary_groups": CORONARY_GROUPS,
        "acute_evidence": "coronary segment field == 1 only",
        "cto_evidence": "coronary segment field == 2; never used as acute evidence",
        "within_group_multisegment": "allowed",
        "between_group_multilesion": "MULTI_GROUP_UNRESOLVED; never guessed",
        "no_acute_group": "NO_GROUP; never guessed",
        "ua_omi_guard": "UA+OMI > 0 invalidates the Joint-10 label space",
        "other_omi_policy": "preserve raw OMI; candidate maps to OTHER with explicit exception",
        "sources": {
            "dataset_dictionary": "https://www.nature.com/articles/s41597-026-07278-0",
            "nature_method": "https://www.nature.com/articles/s41467-026-73023-1",
            "official_code": "https://github.com/stefan-gustafsson-work/omi",
        },
    }
    (output / "joint10_mapping_rules.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")

    unresolved_fraction = len(unresolved) / len(omi_rows) if omi_rows else 0.0
    status = "PASS"
    reasons: list[str] = []
    if ua_omi:
        status = "JOINT10_LABEL_SPACE_INVALID"
        reasons.append(f"UA+OMI={len(ua_omi)}")
    if unresolved:
        status = "BLOCKED_BY_LABEL_AUDIT"
        reasons.append(f"{len(unresolved)} OMI records lack a unique large-vessel culprit label")
    summary = {
        "status": status,
        "training_allowed": status == "PASS",
        "stop_reason": "; ".join(reasons) or None,
        "dataset_rows": len(audited),
        "dataset_patients": len({str(row["patient_id"]) for row in audited}),
        "omi_total_ecg": len(omi_rows),
        "omi_total_patients": len({str(row["patient_id"]) for row in omi_rows}),
        "ua_omi_ecg": len(ua_omi),
        "ua_omi_patients": len({str(row["patient_id"]) for row in ua_omi}),
        "other_omi_ecg": len(other_omi),
        "other_omi_patients": len({str(row["patient_id"]) for row in other_omi}),
        "standard_omi_ecg": len(standard_omi),
        "unique_culprit_group_ecg": len(unique),
        "unique_culprit_group_patients": len({str(row["patient_id"]) for row in unique}),
        "no_culprit_group_ecg": len(no_group),
        "multiple_culprit_groups_ecg": len(multigroup),
        "unresolved_omi_ecg": len(unresolved),
        "unresolved_fraction_of_all_omi": unresolved_fraction,
        "omi_with_cto_flag_or_segment_value2": len(with_cto),
        "cto_flag_positive_all_ecg": len(all_with_cto_flag),
        "any_coronary_segment_value2_all_ecg": len(all_with_segment_cto),
        "omi_with_cto_flag_positive": len(omi_with_cto_flag),
        "omi_with_any_coronary_segment_value2": len(omi_with_segment_cto),
        "omi_with_both_acute_value1_and_cto_value2_segments": len(omi_with_acute_and_cto),
        "candidate_joint10_mapped_ecg": sum(int(row["ecg_count"]) for row in class_rows),
        "candidate_joint10_unmapped_ecg": len(audited) - sum(int(row["ecg_count"]) for row in class_rows),
        "fold_count": len(args.folds),
        "split_seed": next(iter(split_seeds)),
        "patient_overlap": 0,
        "outer_test_record_coverage": len(outer_record_union),
        "outer_test_patient_coverage": len(outer_patient_union),
        "ami_without_stemi_or_nstemi_ecg": len(ami_without_subtype),
        "stemi_or_nstemi_without_ami_ecg": len(subtype_without_ami),
        "source_hashes": {
            "label_csv_sha256": sha256_file(label_csv),
            "master_csv_sha256": sha256_file(master_csv),
            "source_labels_sha256": sha256_file(source_labels_path),
        },
        "provenance_mismatch_count": len(provenance_errors),
        "outputs_with_patient_identifiers_server_only": ["unresolved_omi.csv", "multigroup_omi.csv", "cto_cases.csv"],
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "training_gate.json").write_text(json.dumps({
        "status": "not_run" if status != "PASS" else "ready",
        "joint10_training_allowed": status == "PASS",
        "baseline_rerun": False,
        "reason": summary["stop_reason"] if status != "PASS" else "label audit passed",
        "outer_test_model_inference": False,
        "external_test_accessed": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    aggregate_outputs = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {
            "unresolved_omi.csv", "multigroup_omi.csv", "cto_cases.csv", "run_manifest.json"
        }
    )
    run_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "expected_exit_code": 0 if status == "PASS" else 2,
        "status": status,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_hashes": summary["source_hashes"],
        "aggregate_output_hashes": {path.name: sha256_file(path) for path in aggregate_outputs},
        "identifier_level_outputs_excluded_from_git": [
            "unresolved_omi.csv", "multigroup_omi.csv", "cto_cases.csv"
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
