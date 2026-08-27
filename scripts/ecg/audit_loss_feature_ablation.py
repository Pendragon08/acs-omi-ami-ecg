"""Deterministic artifact and metric audit for the loss/feature ablation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/loss_feature_ablation"
EXPECTED_SPLIT = "5bb9ad6f34e8af19a5c7dff1e141c686eec9b6b662edfde997339a4f8cf5dd85"
SEEDS = [42, 2024, 2025, 2026, 3407]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, atol=1e-9):
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0))


def verify_run(path):
    item = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if item["split_sha256"] != EXPECTED_SPLIT:
        errors.append("split hash mismatch")
    if item["sampling_strategy"] != "shuffle_no_oversampling":
        errors.append("unexpected sampling")
    if item["scheduler"] is not None:
        errors.append("scheduler changed")
    if item["study"] == "loss" and item["model"] != "XceptionTime":
        errors.append("loss study model changed")
    if item["study"] == "feature" and (item["model"] != "XceptionTimeFusion" or item["loss"] != "bce"):
        errors.append("feature study model/loss changed")
    if item["loss"] == "cb_focal":
        cfg = item["loss_config"]
        if "pos_weight" in cfg or cfg.get("weight_source") != "effective_number_training_set_only":
            errors.append("CB-Focal weighting invalid")
    pred_path = ROOT / item["artifacts"]["validation_predictions_csv"]
    checkpoint_path = ROOT / item["artifacts"]["checkpoint"]
    if not pred_path.exists() or not checkpoint_path.exists():
        errors.append("missing prediction/checkpoint artifact")
        return item, errors
    if sha256(checkpoint_path) != item["checkpoint_sha256"]:
        errors.append("checkpoint hash mismatch")
    pred = pd.read_csv(pred_path)
    if len(pred) != 1786:
        errors.append(f"validation row count {len(pred)}")
    y = pred["y_true"].to_numpy(dtype=int)
    p = pred["probability"].to_numpy(dtype=float)
    if not close(roc_auc_score(y, p), item["val"]["auroc"], 2e-9):
        errors.append("AUROC recomputation mismatch")
    if not close(average_precision_score(y, p), item["val"]["auprc"], 2e-9):
        errors.append("AUPRC recomputation mismatch")
    for name, op in item["val"]["operating_points"].items():
        if op is None:
            continue
        binary = (p >= op["threshold"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
        if [int(tp), int(tn), int(fp), int(fn)] != [op["tp"], op["tn"], op["fp"], op["fn"]]:
            errors.append(f"{name} confusion mismatch")
        if not close(f1_score(y, binary, zero_division=0), op["f1"], 2e-9):
            errors.append(f"{name} F1 mismatch")
        if not close(matthews_corrcoef(y, binary), op["mcc"], 2e-9):
            errors.append(f"{name} MCC mismatch")
    return item, errors


def main():
    candidate_paths = sorted(RESULTS.glob("loss_*.json")) + sorted(RESULTS.glob("feature_*.json"))
    result_paths = []
    for path in candidate_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("study") in {"loss", "feature"}:
            result_paths.append(path)
    all_errors = {}
    items = []
    for path in result_paths:
        item, errors = verify_run(path)
        items.append(item)
        if errors:
            all_errors[path.name] = errors

    selection = json.loads((RESULTS / "stage1_selection.json").read_text(encoding="utf-8"))
    best_focal = selection["best_focal"]
    best_cb = selection["best_cb_focal"]
    def matching_seeds(loss, cfg_match):
        return sorted(item["seed"] for item in items if item.get("study") == "loss" and item["loss"] == loss and cfg_match(item["loss_config"]))
    seed_checks = {
        "baseline": matching_seeds("bce", lambda cfg: True),
        "weighted_bce": matching_seeds("wbce", lambda cfg: True),
        "focal": matching_seeds("focal", lambda cfg: close(cfg.get("alpha"), best_focal["alpha"]) and close(cfg.get("gamma"), best_focal["gamma"])),
        "cb_focal": matching_seeds("cb_focal", lambda cfg: close(cfg.get("beta"), best_cb["beta"]) and close(cfg.get("gamma"), best_cb["gamma"])),
    }
    for name, seeds in seed_checks.items():
        if seeds != sorted(SEEDS):
            all_errors[f"seed_check_{name}"] = [f"got {seeds}, expected {sorted(SEEDS)}"]

    new_ckpt = torch.load(RESULTS / "checkpoints/loss_bce_plain_omi_seed2026.pt", map_location="cpu", weights_only=True)
    old_ckpt = torch.load(ROOT / "results/final_fusion_focal/checkpoints/XceptionTime_OMI_bce_P3_seed2026.pt", map_location="cpu", weights_only=True)
    baseline_exact = new_ckpt.keys() == old_ckpt.keys() and all(torch.equal(new_ckpt[k], old_ckpt[k]) for k in new_ckpt)
    if not baseline_exact:
        all_errors["baseline_reproduction"] = ["checkpoint tensors differ"]

    source = (ROOT / "exp/exp_acs_ablation.py").read_text(encoding="utf-8")
    test_access_in_runner = any(
        token in source
        for token in ("meta_test.npz", "P2_test.npy", "data/CSV/test.csv", "predict_official_test")
    )
    if test_access_in_runner:
        all_errors["external_test_access"] = ["ablation runner references external test"]

    verdict = "WARN" if not all_errors else "FAIL"
    audit = {
        "audit_skill": "experiment-audit",
        "review_independence": "unavailable_multi_agent_delegation_not_authorized",
        "acceptance_status": "provisional_deterministic_checks_only",
        "overall_verdict": verdict,
        "integrity_status": verdict.lower(),
        "deterministic_checks_passed": not bool(all_errors),
        "reason_for_warn": "Historical preprocessing leakage remains and no fresh independent semantic reviewer was authorized.",
        "result_json_count": len(result_paths),
        "seed_checks": seed_checks,
        "baseline_checkpoint_exact": baseline_exact,
        "external_test_referenced_by_runner": test_access_in_runner,
        "errors": all_errors,
        "evaluation_type": "real_gt_internal_validation",
    }
    (RESULTS / "EXPERIMENT_AUDIT_ABLATION.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Loss/feature ablation integrity audit",
        "",
        f"Overall: **{verdict}** (provisional deterministic audit; no fresh independent semantic reviewer was authorized).",
        "",
        f"- Parsed and recomputed {len(result_paths)} run JSON files and their validation prediction CSVs.",
        f"- Baseline checkpoint exactly reproduces the historical checkpoint: {baseline_exact}.",
        f"- Ablation runner references the external test: {test_access_in_runner}.",
        f"- Final loss seed sets: {seed_checks}.",
        f"- Deterministic errors: {all_errors if all_errors else 'none'}.",
        "",
        "WARN is retained even when deterministic checks pass because the historical scalers include internal-validation data and the required fresh-agent semantic audit could not be performed under the active no-delegation constraint.",
    ]
    (RESULTS / "EXPERIMENT_AUDIT_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
