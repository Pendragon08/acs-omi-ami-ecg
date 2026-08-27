"""Deterministic verifier for ACS protocol-v2 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "protocol_v2"
sys.path.insert(0, str(ROOT))
from nets import build_model  # noqa: E402


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_point(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    joint = 2 * f1 * sens / (f1 + sens)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sens": sens, "spec": spec, "f1": f1, "joint_f1_sens_hmean": joint}


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    checks = []
    split = json.loads((ROOT / "data/protocol_v2/protocol_v2_split.json").read_text())
    meta = np.load(ROOT / "data/arrays/meta_train.npz", allow_pickle=True)
    patients = np.asarray(meta["patient_ids"]).astype(str)
    parts = ["fit", "selection", "calibration", "evaluation"]
    all_idx = np.concatenate([np.asarray(split[k], dtype=int) for k in parts])
    checks.append({"check": "partition_complete_once", "pass": len(all_idx) == len(np.unique(all_idx)) == len(patients)})
    patient_sets = {k: set(patients[np.asarray(split[k], dtype=int)]) for k in parts}
    overlap = {f"{a}:{b}": len(patient_sets[a] & patient_sets[b])
               for i, a in enumerate(parts) for b in parts[i + 1:]}
    checks.append({"check": "patient_disjoint", "pass": max(overlap.values()) == 0, "overlap": overlap})

    for task in ("OMI", "AMI"):
        tag = f"XceptionTime_{task}_bce_P4_protocolv2_seed2026"
        result_path = OUT / f"{tag}.json"
        result = json.loads(result_path.read_text())
        artifact_ok = True
        artifact_details = {}
        for key, value in result["artifacts"].items():
            if key.endswith("_sha256"):
                continue
            path = ROOT / value
            expected = result["artifacts"][f"{key}_sha256"]
            actual = sha256(path)
            artifact_details[key] = {"exists": path.exists(), "sha256_match": actual == expected}
            artifact_ok &= path.exists() and actual == expected
        checks.append({"check": f"{task}_artifact_hashes", "pass": artifact_ok, "details": artifact_details})

        state = torch.load(ROOT / result["artifacts"]["checkpoint"], map_location="cpu", weights_only=True)
        model = build_model("XceptionTime", num_class=1, device="cpu")
        mismatch = model.load_state_dict(state, strict=True)
        checks.append({"check": f"{task}_checkpoint_strict_load", "pass": not mismatch.missing_keys and not mismatch.unexpected_keys})

        pred_path = ROOT / result["artifacts"]["evaluation_predictions"]
        with pred_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y = np.asarray([int(r[task]) for r in rows])
        p = np.asarray([float(r["probability_calibrated"]) for r in rows])
        t = result["metrics"]["evaluation"]["operating_point"]["threshold"]
        point = recompute_point(y, p, t)
        stored = result["metrics"]["evaluation"]["operating_point"]
        ranking = result["metrics"]["evaluation"]["ranking_calibrated"]
        metrics_ok = (
            len(rows) == len(split["evaluation"])
            and close(roc_auc_score(y, p), ranking["auroc"], 1e-9)
            and close(average_precision_score(y, p), ranking["auprc"], 1e-9)
            and all(close(point[k], stored[k], 1e-9) for k in point)
        )
        checks.append({"check": f"{task}_evaluation_recompute", "pass": metrics_ok,
                       "n": len(rows), "recomputed": {"auroc": roc_auc_score(y, p),
                       "auprc": average_precision_score(y, p), **point}})

        with (ROOT / result["artifacts"]["threshold_grid_csv"]).open(newline="", encoding="utf-8") as f:
            grid = list(csv.DictReader(f))
        best = max(grid, key=lambda r: (float(r["joint_f1_sens_hmean"]), -float(r["threshold"])))
        selected = result["threshold_selection"]["selected"]
        checks.append({"check": f"{task}_threshold_grid", "pass": len(grid) == 99 and close(best["threshold"], selected["threshold"]),
                       "n_grid": len(grid), "selected_threshold": selected["threshold"]})

    log = (ROOT / "logs/protocol_v2/protocol_v2_gpu1_master.log").read_text()
    checks.append({"check": "gpu1_run_complete", "pass": "physical_gpu=1" in log and "RUN_COMPLETE" in log})
    payload = {
        "verifier": "scripts/ecg/verify_protocol_v2.py",
        "evaluation_type": "real_gt_internal_evaluation_historically_seen",
        "checks": checks,
        "overall_pass": all(item["pass"] for item in checks),
    }
    out = OUT / "verification.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["overall_pass"] else 1)


if __name__ == "__main__":
    main()
