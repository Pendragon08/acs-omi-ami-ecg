"""Patient-level train/val split (no internal test).

Groups records by Patient_id and splits *patients* (never records) so no patient
appears in more than one split.  The official test.csv is the test set (labels
hidden, used only for final prediction/submission), so we split train.csv only:
~90% train / ~10% validation.  Written once to data/splits/patient_split.json
and reused by every model/seed for fairness.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent


def build_patient_split(patient_ids, seed=2026, train_frac=0.9, val_frac=0.1):
    """patient_ids: [N] (str/int). Returns dict of record-index lists."""
    patient_ids = np.asarray(patient_ids)
    n = len(patient_ids)
    indices = np.arange(n)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(gss.split(indices, groups=patient_ids))
    train_idx = indices[train_idx]
    val_idx = indices[val_idx]

    # Leakage audit: patient sets must be disjoint.
    p_train = set(patient_ids[train_idx])
    p_val = set(patient_ids[val_idx])
    assert p_train.isdisjoint(p_val), "train/val patient leakage!"

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
    }


def main():
    meta = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
    patient_ids = meta["patient_ids"]
    split = build_patient_split(patient_ids)

    out_path = ROOT / "data" / "splits" / "patient_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split), encoding="utf-8")

    # Report label distribution + patient counts per split for sanity.
    labels = meta["labels"]
    for k, idx in split.items():
        idx = np.asarray(idx)
        ys = labels[idx]
        n_pat = len(set(patient_ids[idx]))
        print(f"{k}: n_ecg={len(idx)}, n_patient={n_pat}, "
              f"OMI_pos={int(ys[:, 0].sum())}, AMI_pos={int(ys[:, 1].sum())}")
    print(f"total_patient={len(set(patient_ids))}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
