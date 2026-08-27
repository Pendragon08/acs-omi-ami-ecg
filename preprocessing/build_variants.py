"""One-time data preparation: consolidate per-record npy into single arrays and
build preprocessing variants P0/P1/P2 (实施方案 section 5-7).

P0 = existing per-record per-lead min-max (normalized/ npy).
P1 = unnormalized + Global Robust Scaling.
P2 = unnormalized + 0.5-40Hz bandpass + Global Robust Scaling.
P3 = P2 + online augmentation (no extra array; augmentation happens in Dataset).

Outputs under data/arrays/:
    meta_train.npz  {record_ids, patient_ids, labels}  (labels = [OMI, AMI])
    meta_test.npz   {record_ids, patient_ids}
    P0_train.npy / P0_test.npy
    P1_train.npy / P1_test.npy
    P2_train.npy / P2_test.npy
    scalers.json    {"P1": {...}, "P2": {...}}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.bandpass import bandpass_filter          # noqa: E402
from preprocessing.robust_scaler import (                   # noqa: E402
    apply_robust_scaler,
    fit_robust_scaler,
)

DATA = ROOT / "data"
CSV_DIR = DATA / "CSV"
NPY_DIR = DATA / "preprocessed_100hz"
OUT = DATA / "arrays"
TARGETS = ("OMI", "AMI")


def _read_csv_records(csv_path):
    """Return dict record_id -> row (dict)."""
    rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = Path((row.get("ecg_row_record") or "").strip()).stem
            if rid:
                rows[rid] = row
    return rows


def _stack_npy(npy_dir, record_ids):
    """Load npy files named <rid>.npy in the given order -> [N,1000,12]."""
    arrs = []
    for rid in record_ids:
        p = Path(npy_dir) / f"{rid}.npy"
        arrs.append(np.load(p, allow_pickle=False).astype(np.float32))
    return np.stack(arrs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scalers = {}

    # ---- order: sorted record ids from unnormalized/train -------------------
    train_rows = _read_csv_records(CSV_DIR / "train.csv")
    test_rows = _read_csv_records(CSV_DIR / "test.csv")

    train_npy_ids = sorted(p.stem for p in (NPY_DIR / "unnormalized" / "train").glob("*.npy"))
    test_npy_ids = sorted(p.stem for p in (NPY_DIR / "unnormalized" / "test").glob("*.npy"))

    train_ids = [r for r in train_npy_ids if r in train_rows]
    test_ids = [r for r in test_npy_ids if r in test_rows]
    print(f"train matched {len(train_ids)}/{len(train_npy_ids)}; "
          f"test matched {len(test_ids)}/{len(test_npy_ids)}")

    # ---- load raw (unnormalized) and normalized -----------------------------
    print("loading unnormalized train/test ...")
    X_tr_raw = _stack_npy(NPY_DIR / "unnormalized" / "train", train_ids)
    X_te_raw = _stack_npy(NPY_DIR / "unnormalized" / "test", test_ids)
    print(f"raw shapes: {X_tr_raw.shape} {X_te_raw.shape}")

    # P0: normalized (existing min-max)
    print("loading normalized train/test ...")
    P0_train = _stack_npy(NPY_DIR / "normalized" / "train", train_ids)
    P0_test = _stack_npy(NPY_DIR / "normalized" / "test", test_ids)

    # ---- P1: global robust on raw ------------------------------------------
    s1 = fit_robust_scaler(X_tr_raw)
    P1_train = apply_robust_scaler(X_tr_raw, s1)
    P1_test = apply_robust_scaler(X_te_raw, s1)
    scalers["P1"] = s1
    del X_te_raw  # free

    # ---- P2: bandpass + global robust --------------------------------------
    print("bandpass filtering train ...")
    X_tr_bp = bandpass_filter(X_tr_raw).astype(np.float32)
    print("bandpass filtering test ...")
    X_te_bp = bandpass_filter(
        _stack_npy(NPY_DIR / "unnormalized" / "test", test_ids)
    ).astype(np.float32)
    s2 = fit_robust_scaler(X_tr_bp)
    P2_train = apply_robust_scaler(X_tr_bp, s2)
    P2_test = apply_robust_scaler(X_te_bp, s2)
    scalers["P2"] = s2

    # ---- save arrays --------------------------------------------------------
    for name, arr in [("P0_train", P0_train), ("P0_test", P0_test),
                      ("P1_train", P1_train), ("P1_test", P1_test),
                      ("P2_train", P2_train), ("P2_test", P2_test)]:
        np.save(OUT / f"{name}.npy", arr)
        print(f"saved {name}.npy {arr.shape}")

    # ---- meta ---------------------------------------------------------------
    train_patient = np.array([str(train_rows[r].get("Patient_id", "")).strip() for r in train_ids])
    train_labels = np.stack([[int(train_rows[r][t]) for t in TARGETS] for r in train_ids]).astype(np.int64)
    np.savez(OUT / "meta_train.npz",
             record_ids=np.array(train_ids), patient_ids=train_patient,
             labels=train_labels)

    test_patient = np.array([str(test_rows[r].get("Patient_id", "")).strip() for r in test_ids])
    np.savez(OUT / "meta_test.npz",
             record_ids=np.array(test_ids), patient_ids=test_patient)

    (OUT / "scalers.json").write_text(json.dumps(scalers), encoding="utf-8")
    print("label dist train: OMI pos=%d, AMI pos=%d" % (train_labels[:, 0].sum(), train_labels[:, 1].sum()))
    print("DONE")


if __name__ == "__main__":
    main()
