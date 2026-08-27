"""Official Test inference for a trained single-label (OMI / AMI) or multilabel model.

Loads the best checkpoint + validation threshold(s) from a completed training
run, applies the SAME preprocessing used at train time (P3 -> P2 array: 0.5-40 Hz
bandpass + train-fitted global robust scaler; NO augmentation, NO scaler refit),
runs model.forward -> sigmoid -> validation threshold, and writes prediction
CSV(s).

Usage:
    python scripts/ecg/predict_official_test.py --model TimesNet --task omi \
        --exp_name comparison --seed 2026
    python scripts/ecg/predict_official_test.py --model TimesNet --task multilabel \
        --exp_name comparison_ml --seed 2026

Internal output columns (single-label): ecg_row_record, <OMI|AMI>_probability,
<OMI|AMI>_prediction.  (multilabel): both OMI and AMI probability/prediction.
Also writes platform-ready submission CSV(s) with columns `ecg_row_record,<label>`
(0/1) -- confirmed against the official platform (http://39.105.59.221:8080/):
id column must match test.csv exactly (e.g. "16210.dat"), label columns are
named exactly `OMI` / `AMI`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# This copy lives directly under ``scripts/`` (not ``scripts/ecg/``).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nets import build_model  # noqa: E402

TASK_LABELS = {
    "omi": ["OMI"],
    "ami": ["AMI"],
    "multilabel": ["OMI", "AMI"],
    "stemi_nstemi_ua": ["STEMI", "NSTEMI", "UA"],
}


def _tag(model, task, loss, variant, seed):
    # multilabel JSON has no label name (exp_acs.run_exp); single-label does.
    return f"{model}_{loss}_{variant}_seed{seed}"


def _read_csv_records(csv_path):
    """Return dict stem -> original ecg_row_record string."""
    rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("ecg_row_record") or "").strip()
            stem = Path(raw).stem
            if stem:
                rows[stem] = raw
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True,
                   choices=["omi", "ami", "multilabel", "stemi_nstemi_ua"])
    p.add_argument("--exp_name", default="comparison")
    p.add_argument("--loss", default="asl")
    p.add_argument("--variant", default="P3")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--use_fusion", action="store_true",
                   help="multimodal input: raw waveform + median beat + clinical features")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = _tag(a.model, a.task, a.loss, a.variant, a.seed)

    # ---- best checkpoint + validation threshold(s) -------------------------
    res_dir = ROOT / "results" / a.exp_name
    result = json.loads((res_dir / f"{tag}.json").read_text(encoding="utf-8"))
    thresholds = [float(t) for t in result["thresholds"]]
    state = torch.load(res_dir / "checkpoints" / f"{tag}.pt", map_location=device)

    label_names = TASK_LABELS[a.task]
    num_class = len(label_names)
    assert len(thresholds) == num_class, \
        f"expected {num_class} thresholds, got {len(thresholds)}"

    # ---- model -------------------------------------------------------------
    model = build_model(a.model, num_class=num_class, device=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ---- official test data ----------------------------------------------
    # P3 -> P2 array: same bandpass + train-fitted robust scaler, no augmentation,
    # no scaler refit.  The test arrays were already built with the train scaler
    # (see preprocessing/build_variants.py).
    array_key = "P2" if a.variant == "P3" else a.variant
    X = np.load(ROOT / "data" / "arrays" / f"{array_key}_test.npy", mmap_mode="r")
    meta = np.load(ROOT / "data" / "arrays" / "meta_test.npz", allow_pickle=True)
    record_ids = [str(r) for r in meta["record_ids"]]

    # Original ecg_row_record strings.  Reorder predictions to the exact
    # test.csv order before writing/submitting; this avoids relying on the
    # platform to join rows by identifier.
    csv_map = _read_csv_records(ROOT / "data" / "CSV" / "test.csv")
    ids_meta = [csv_map.get(rid, rid) for rid in record_ids]
    with (ROOT / "data" / "CSV" / "test.csv").open(newline="") as f:
        ids_out = [(row.get("ecg_row_record") or "").strip()
                   for row in csv.DictReader(f)]
    id_to_meta = {rid: i for i, rid in enumerate(ids_meta)}
    if len(ids_out) != len(record_ids) or set(ids_out) != set(ids_meta):
        raise ValueError("test.csv identifiers do not match meta_test identifiers")
    test_order = np.asarray([id_to_meta[rid] for rid in ids_out], dtype=np.int64)

    # optional multimodal inputs (median beat + clinical features)
    med_test = clin_test = None
    if a.use_fusion:
        med_test = np.load(ROOT / "data" / "arrays" / "med_P2_test.npy", mmap_mode="r")
        clin_test = np.load(ROOT / "data" / "arrays" / "clin_test.npy", mmap_mode="r")

    # ---- inference --------------------------------------------------------
    probs = []
    with torch.no_grad():
        for i in range(0, len(record_ids), a.batch_size):
            xb = torch.from_numpy(X[i:i + a.batch_size].astype(np.float32)).to(device)
            if a.use_fusion:
                mb = torch.from_numpy(med_test[i:i + a.batch_size].astype(np.float32)).to(device)
                cb = torch.from_numpy(clin_test[i:i + a.batch_size].astype(np.float32)).to(device)
                probs.append(torch.sigmoid(model(xb, med=mb, clin=cb)).cpu().numpy())
            else:
                probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs = np.concatenate(probs, 0)            # [N, num_class]
    if probs.ndim == 1:
        probs = probs[:, None]
    probs = probs[test_order]
    preds = (probs >= np.asarray(thresholds)).astype(int)  # [N, num_class]

    # ---- internal CSV: ecg_row_record + per-label probability/prediction ---
    out = res_dir / f"official_test_{a.task}_{a.model}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["ecg_row_record"]
        for lbl in label_names:
            header += [f"{lbl}_probability", f"{lbl}_prediction"]
        w.writerow(header)
        for rid, pv, pdl in zip(ids_out, probs, preds):
            row = [rid]
            for j in range(num_class):
                row += [f"{pv[j]:.6f}", int(pdl[j])]
            w.writerow(row)

    # ---- official submission CSV(s): id + label column (0/1), per platform ---
    for j, lbl in enumerate(label_names):
        sub = res_dir / f"submission_{lbl.lower()}_{a.model}.csv"
        with open(sub, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ecg_row_record", lbl])
            for rid, pdl in zip(ids_out, preds[:, j]):
                w.writerow([rid, int(pdl)])

    # The ACS evaluation platform accepts several label columns in one CSV.
    # Keep a combined file so multi-label tasks are scored on the same upload.
    combined = res_dir / f"submission_{a.task}_{a.model}.csv"
    with open(combined, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ecg_row_record", *label_names])
        for rid, row in zip(ids_out, preds):
            w.writerow([rid, *[int(v) for v in row]])

    n_pos = {lbl: int(preds[:, j].sum()) for j, lbl in enumerate(label_names)}
    print(f"wrote {out} + {num_class} per-label CSV(s) + {combined.name} "
          f"({len(probs)} rows, thresholds={thresholds}, n_pos_pred={n_pos})")


if __name__ == "__main__":
    main()
