"""Read-only deterministic integrity checks for ACS OMI/AMI project."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
meta = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
split = json.loads((ROOT / "data" / "splits" / "patient_split.json").read_text())
labels = np.asarray(meta["labels"], dtype=int)
pids = np.asarray(meta["patient_ids"]).astype(str)
record_ids = np.asarray(meta["record_ids"]).astype(str)
train = np.asarray(split["train"], dtype=int)
val = np.asarray(split["val"], dtype=int)


def dist(idx):
    y = labels[idx]
    return {
        "n_records": int(len(idx)),
        "n_patients": int(len(np.unique(pids[idx]))),
        "omi_pos": int(y[:, 0].sum()),
        "ami_pos": int(y[:, 1].sum()),
        "omi_rate": float(y[:, 0].mean()),
        "ami_rate": float(y[:, 1].mean()),
    }


patient_rows = defaultdict(list)
for i, pid in enumerate(pids):
    patient_rows[pid].append(i)
inconsistent = {"OMI": 0, "AMI": 0}
for idxs in patient_rows.values():
    for col, name in enumerate(("OMI", "AMI")):
        if len(np.unique(labels[idxs, col])) > 1:
            inconsistent[name] += 1

array_info = {}
for name in ("P0_train", "P1_train", "P2_train"):
    arr = np.load(ROOT / "data" / "arrays" / f"{name}.npy", mmap_mode="r")
    sample_idx = np.linspace(0, len(arr) - 1, min(256, len(arr)), dtype=int)
    sample = np.asarray(arr[sample_idx])
    array_info[name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sample_all_finite": bool(np.isfinite(sample).all()),
        "sample_abs_q999": float(np.quantile(np.abs(sample), 0.999)),
        "sample_abs_max": float(np.max(np.abs(sample))),
    }

test_meta = np.load(ROOT / "data" / "arrays" / "meta_test.npz", allow_pickle=True)
test_ids = np.asarray(test_meta["record_ids"]).astype(str)
with (ROOT / "data" / "CSV" / "test.csv").open(encoding="utf-8-sig", newline="") as f:
    test_csv_rows = list(csv.DictReader(f))

out = {
    "n_total": int(len(labels)),
    "labels_shape": list(labels.shape),
    "record_ids_unique": bool(len(np.unique(record_ids)) == len(record_ids)),
    "empty_patient_id_records": int(np.sum(np.char.strip(pids) == "")),
    "split": {
        "train": dist(train),
        "val": dist(val),
        "index_overlap": int(len(set(train.tolist()) & set(val.tolist()))),
        "patient_overlap": int(len(set(pids[train]) & set(pids[val]))),
        "indices_cover_all_once": bool(
            len(train) + len(val) == len(labels)
            and len(np.unique(np.concatenate([train, val]))) == len(labels)
        ),
    },
    "patients": {
        "n_unique": int(len(patient_rows)),
        "n_with_multiple_records": int(sum(len(v) > 1 for v in patient_rows.values())),
        "label_inconsistent_patient_counts": inconsistent,
    },
    "label_relationship": {
        "OMI1_AMI0": int(np.sum((labels[:, 0] == 1) & (labels[:, 1] == 0))),
        "OMI1_AMI1": int(np.sum((labels[:, 0] == 1) & (labels[:, 1] == 1))),
        "OMI0_AMI1": int(np.sum((labels[:, 0] == 0) & (labels[:, 1] == 1))),
        "OMI0_AMI0": int(np.sum((labels[:, 0] == 0) & (labels[:, 1] == 0))),
    },
    "arrays": array_info,
    "official_test": {
        "meta_n": int(len(test_ids)),
        "csv_n": int(len(test_csv_rows)),
        "meta_ids_unique": bool(len(np.unique(test_ids)) == len(test_ids)),
        "csv_has_label_columns": [
            name for name in ("OMI", "AMI") if name in (test_csv_rows[0] if test_csv_rows else {})
        ],
    },
}
print(json.dumps(out, indent=2, ensure_ascii=False))
