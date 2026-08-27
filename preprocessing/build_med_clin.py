"""Build median-beat (med) and clinical-feature (clin) arrays aligned to the
existing record order, for the multimodal XceptionTimeFusion model.

Median beat (.med): int16 little-endian, 500 samples x 12 leads (interleaved)
at 500 Hz -> a 1s median-averaged beat.  Pipeline (mirroring the raw P2 path):
    read int16 -> reshape [500,12] -> resample 500->100 Hz -> [100,12]
    -> 0.5-40 Hz bandpass -> global robust scaling (fit on train only).

Clinical features: gender (0/1) + age (z-scored on train).  Time_Interval is
dropped (replaced by the median beat per user decision).

Outputs under data/arrays/:
    med_P2_train.npy / med_P2_test.npy   ([N,100,12])
    clin_train.npy   / clin_test.npy     ([N,2] = gender, age)
    scalers.json     += {"MED": {...}, "CLIN": {"age_mean", "age_std"}}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.bandpass import bandpass_filter          # noqa: E402
from preprocessing.robust_scaler import (                   # noqa: E402
    apply_robust_scaler,
    fit_robust_scaler,
)

DATA = ROOT / "data"
CSV_DIR = DATA / "CSV"
MED_DIR = DATA / "med_data"
OUT = DATA / "arrays"

ORIG_FS = 500
TARGET_FS = 100
MED_LEN = int(500 * TARGET_FS / ORIG_FS)      # 100


def _read_csv_records(csv_path):
    """dict record_id (stem of ecg_row_record) -> row."""
    rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = Path((row.get("ecg_row_record") or "").strip()).stem
            if rid:
                rows[rid] = row
    return rows


def _read_med(med_path):
    """int16 LE interleaved [500 samples x 12 leads] @500Hz -> [100, 12] @100Hz."""
    raw = np.fromfile(med_path, dtype="<i2")            # [6000]
    a = raw.reshape(500, 12).T                          # [12, 500] lead-major
    num = int(a.shape[1] * TARGET_FS / ORIG_FS)         # 100
    resampled = resample(a, num, axis=1)                # [12, 100]
    return np.ascontiguousarray(resampled.T)            # [100, 12]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train_rows = _read_csv_records(CSV_DIR / "train.csv")
    test_rows = _read_csv_records(CSV_DIR / "test.csv")

    # Reuse the exact record order of the existing raw arrays (build_variants.py),
    # so every row of med_*/clin_* aligns 1:1 with P2_*.npy and meta_*.npz labels.
    meta_tr = np.load(OUT / "meta_train.npz", allow_pickle=True)
    meta_te = np.load(OUT / "meta_test.npz", allow_pickle=True)
    train_ids = [str(r) for r in meta_tr["record_ids"]]
    test_ids = [str(r) for r in meta_te["record_ids"]]

    # ---- median beats --------------------------------------------------------
    def _stack_med(ids, rows):
        arrs, missing = [], 0
        for rid in ids:
            med_stem = Path((rows[rid].get("ecg_med_record") or "").strip()).stem or rid
            p = MED_DIR / f"{med_stem}.med"
            if not p.exists():
                missing += 1
                arrs.append(np.zeros((MED_LEN, 12), dtype=np.float32))
                continue
            arrs.append(_read_med(p).astype(np.float32))
        if missing:
            print(f"  WARNING: {missing} missing .med -> zero-filled")
        return np.stack(arrs)                            # [N,100,12]

    print(f"stacking median beats (train {len(train_ids)}) ...")
    med_train = _stack_med(train_ids, train_rows)
    print(f"stacking median beats (test {len(test_ids)}) ...")
    med_test = _stack_med(test_ids, test_rows)

    print("bandpass filtering median beats ...")
    med_train = bandpass_filter(med_train).astype(np.float32)
    med_test = bandpass_filter(med_test).astype(np.float32)

    print("robust scaling median beats (fit on train) ...")
    smed = fit_robust_scaler(med_train)
    med_train = apply_robust_scaler(med_train, smed)
    med_test = apply_robust_scaler(med_test, smed)

    np.save(OUT / "med_P2_train.npy", med_train)
    np.save(OUT / "med_P2_test.npy", med_test)
    print(f"saved med_P2_train.npy {med_train.shape} / med_P2_test.npy {med_test.shape}")

    # ---- clinical features (gender, age) ------------------------------------
    def _clin(ids, rows):
        g = np.array([float(rows[rid].get("gender") or 0.0) for rid in ids], dtype=np.float32)
        a = np.array([float(rows[rid].get("age") or 0.0) for rid in ids], dtype=np.float32)
        return np.stack([g, a], axis=1)                  # [N,2]

    clin_train = _clin(train_ids, train_rows)
    clin_test = _clin(test_ids, test_rows)

    age_mean = float(clin_train[:, 1].mean())
    age_std = float(clin_train[:, 1].std()) or 1.0
    clin_train[:, 1] = (clin_train[:, 1] - age_mean) / age_std
    clin_test[:, 1] = (clin_test[:, 1] - age_mean) / age_std

    np.save(OUT / "clin_train.npy", clin_train)
    np.save(OUT / "clin_test.npy", clin_test)
    print(f"saved clin_train.npy {clin_train.shape} / clin_test.npy {clin_test.shape}")
    print(f"clin age z-score: mean={age_mean:.2f}, std={age_std:.2f}")

    # ---- scaler stats --------------------------------------------------------
    scaler_path = OUT / "scalers.json"
    scalers = json.loads(scaler_path.read_text(encoding="utf-8")) if scaler_path.exists() else {}
    scalers["MED"] = smed
    scalers["CLIN"] = {"age_mean": age_mean, "age_std": age_std}
    scaler_path.write_text(json.dumps(scalers), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
