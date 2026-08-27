"""Build the leakage-controlled ACS protocol-v2 split and preprocessing spec.

Protocol v2 keeps the historical outer validation patient set as `evaluation`
and splits the historical training patients into `fit`, `selection`, and
`calibration`.
All learned preprocessing parameters are fitted on `fit` only.

The existing P2 array is an affine transform of band-passed ECG using a scaler
fitted on all official-training records.  Rather than rewriting ~1 GB arrays,
we fit median/MAD in that coordinate system and apply
    clean = (P2 - fit_median_in_P2_space) / fit_scale_in_P2_space
at load time.  Algebraically this is the same fold-fit affine robust scaling,
up to the float32 quantization already present in P2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "protocol_v2"
SEED = 2026
HOLDOUT_FRACTION = 0.20
CAL_WITHIN_HOLDOUT_FRACTION = 0.50
SEARCH_CANDIDATES = 2000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def subset_stats(indices: np.ndarray, patient_ids: np.ndarray, labels: np.ndarray) -> dict:
    y = labels[indices]
    return {
        "n_records": int(len(indices)),
        "n_patients": int(len(np.unique(patient_ids[indices]))),
        "OMI_positive": int(y[:, 0].sum()),
        "AMI_positive": int(y[:, 1].sum()),
        "OMI_rate": float(y[:, 0].mean()),
        "AMI_rate": float(y[:, 1].mean()),
    }


def choose_group_holdout_split(
    base_indices: np.ndarray,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    holdout_fraction: float,
    seed: int,
    holdout_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    groups = patient_ids[base_indices]
    base_y = labels[base_indices]
    target_rate = base_y.mean(axis=0)
    best = None
    gss = GroupShuffleSplit(
        n_splits=SEARCH_CANDIDATES,
        test_size=holdout_fraction,
        random_state=seed,
    )
    dummy = np.zeros(len(base_indices), dtype=np.int8)
    for candidate_id, (fit_rel, holdout_rel) in enumerate(gss.split(dummy, groups=groups)):
        fit_idx = base_indices[fit_rel]
        holdout_idx = base_indices[holdout_rel]
        holdout_rate = labels[holdout_idx].mean(axis=0)
        size_error = abs(len(holdout_idx) / len(base_indices) - holdout_fraction)
        prevalence_error = np.sum(
            np.abs(holdout_rate - target_rate) / np.maximum(target_rate, 1e-6)
        )
        # Size receives enough weight to avoid a prevalence-perfect but badly
        # sized calibration set caused by variable records per patient.
        score = float(10.0 * size_error + prevalence_error)
        item = (score, candidate_id, fit_idx, holdout_idx, holdout_rate)
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None
    score, candidate_id, fit_idx, holdout_idx, holdout_rate = best
    return fit_idx, holdout_idx, {
        "holdout_name": holdout_name,
        "candidate_id": int(candidate_id),
        "objective": float(score),
        "target_rates_in_parent": {
            "OMI": float(target_rate[0]), "AMI": float(target_rate[1])
        },
        "selected_holdout_rates": {
            "OMI": float(holdout_rate[0]), "AMI": float(holdout_rate[1])
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = DATA / "arrays" / "meta_train.npz"
    historical_split_path = DATA / "splits" / "patient_split.json"
    base_array_path = DATA / "arrays" / "P2_train.npy"

    meta = np.load(meta_path, allow_pickle=True)
    patient_ids = np.asarray(meta["patient_ids"]).astype(str)
    labels = np.asarray(meta["labels"], dtype=np.int64)
    historical = json.loads(historical_split_path.read_text(encoding="utf-8"))
    historical_train = np.asarray(historical["train"], dtype=np.int64)
    evaluation = np.asarray(historical["val"], dtype=np.int64)

    fit, selection_calibration, outer_search = choose_group_holdout_split(
        historical_train, patient_ids, labels, HOLDOUT_FRACTION, SEED,
        "selection_plus_calibration",
    )
    selection, calibration, inner_search = choose_group_holdout_split(
        selection_calibration, patient_ids, labels,
        CAL_WITHIN_HOLDOUT_FRACTION, SEED + 1, "calibration",
    )

    all_indices = np.concatenate([fit, selection, calibration, evaluation])
    assert len(np.unique(all_indices)) == len(labels) == len(all_indices)
    patient_sets = {
        "fit": set(patient_ids[fit]),
        "selection": set(patient_ids[selection]),
        "calibration": set(patient_ids[calibration]),
        "evaluation": set(patient_ids[evaluation]),
    }
    assert patient_sets["fit"].isdisjoint(patient_sets["calibration"])
    assert patient_sets["fit"].isdisjoint(patient_sets["selection"])
    assert patient_sets["fit"].isdisjoint(patient_sets["evaluation"])
    assert patient_sets["selection"].isdisjoint(patient_sets["calibration"])
    assert patient_sets["selection"].isdisjoint(patient_sets["evaluation"])
    assert patient_sets["calibration"].isdisjoint(patient_sets["evaluation"])

    split_payload = {
        "protocol": "ACS_PROTOCOL_V2",
        "seed": SEED,
        "source_split": "data/splits/patient_split.json",
        "source_split_sha256": sha256_file(historical_split_path),
        "construction": (
            "historical train split into patient-disjoint fit/selection/calibration "
            "via two deterministic 2000-candidate GroupShuffleSplit searches; "
            "historical val retained as evaluation"
        ),
        "split_search": {"outer": outer_search, "inner": inner_search},
        "fit": fit.tolist(),
        "selection": selection.tolist(),
        "calibration": calibration.tolist(),
        "evaluation": evaluation.tolist(),
        "stats": {
            "fit": subset_stats(fit, patient_ids, labels),
            "selection": subset_stats(selection, patient_ids, labels),
            "calibration": subset_stats(calibration, patient_ids, labels),
            "evaluation": subset_stats(evaluation, patient_ids, labels),
        },
        "integrity": {
            "all_records_covered_once": True,
            "patient_disjoint": True,
            "empty_patient_ids": int(np.sum(np.char.strip(patient_ids) == "")),
        },
    }
    split_out = OUT / "protocol_v2_split.json"
    split_out.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    # Exact fit-only robust scaler in the existing P2 coordinate system.
    base = np.load(base_array_path, mmap_mode="r")
    fit_values = np.asarray(base[fit], dtype=np.float32)
    fit_median = float(np.median(fit_values))
    fit_mad = float(np.median(np.abs(fit_values - fit_median)))
    fit_scale = float(1.4826 * fit_mad)
    if not np.isfinite(fit_scale) or fit_scale <= 1e-8:
        raise RuntimeError(f"invalid fit-only robust scale: {fit_scale}")

    # Fit-only, deterministic robust clipping: estimate the 99.9th percentile
    # of absolute cleaned values on a patient-fit record sample.
    rng = np.random.default_rng(SEED)
    sample_n = min(2048, len(fit))
    sample_idx = rng.choice(fit, size=sample_n, replace=False)
    sample = np.asarray(base[sample_idx], dtype=np.float32)
    sample_clean = (sample - fit_median) / fit_scale
    clip_abs = float(np.quantile(np.abs(sample_clean), 0.999))

    preprocessing = {
        "protocol": "ACS_PROTOCOL_V2",
        "variant": "P4",
        "definition": (
            "P2 band-pass signal + fit-only robust affine scaling + fit-only "
            "99.9% absolute-value clipping + deterministic train-only augmentation"
        ),
        "base_array": "data/arrays/P2_train.npy",
        "base_array_sha256": sha256_file(base_array_path),
        "meta_train_sha256": sha256_file(meta_path),
        "protocol_split_sha256": sha256_file(split_out),
        "fit_only_affine_in_P2_space": {
            "median": fit_median,
            "scale_1p4826_mad": fit_scale,
            "eps": 1e-8,
        },
        "fit_only_clip": {
            "method": "sampled_abs_quantile",
            "quantile": 0.999,
            "sample_records": sample_n,
            "seed": SEED,
            "clip_abs": clip_abs,
        },
        "leakage_contract": {
            "learned_from": "fit indices only",
            "applied_to": ["fit", "selection", "calibration", "evaluation", "official_test"],
            "evaluation_used_for_fit": False,
            "selection_used_for_fit": False,
            "calibration_used_for_fit": False,
        },
    }
    preprocessing_out = OUT / "protocol_v2_preprocessing.json"
    preprocessing_out.write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")

    del fit_values, sample, sample_clean
    print(json.dumps({
        "split": split_payload["stats"],
        "split_search": split_payload["split_search"],
        "preprocessing": preprocessing,
        "outputs": [str(split_out), str(preprocessing_out)],
    }, indent=2))


if __name__ == "__main__":
    main()
