"""Staged Joint-6 optimization. Candidate screening is INNER-ONLY by construction."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.acs_joint_task import select_omi_threshold, sha256, save_curves, save_roc_pr_curves
from experiments.ecg_lfm.evaluate_joint import _read_csv
from experiments.ecg_lfm.evaluate_joint6_fold import (
    arrays, build_rows, infer_joint6, json_safe, metric_bundle, write_rows,
)
from experiments.ecg_lfm.joint6 import JOINT6_CLASSES
from experiments.ecg_lfm.training_reporting import read_fold_history, history_cost

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/joint6_optimization_20260907"
MANIFEST = ROOT / "manifests/ecg_lfm_nested/fold0"
REFERENCE = ROOT / "results/joint6_cv/ecg_lfm/lora/fold0"
BASE_EVAL = ROOT / "results/joint6_cv/evaluation/lora/fold0"
LEARNING_RATES = {"baseline_lr1e4": 1e-4, "lr5e5": 5e-5, "lr3e5": 3e-5}
POINTS = ("fixed_0.5", "max_f1", "youden_j", "sens90", "sens95", "spec90")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inner_assignments():
    assignments = _read_csv(MANIFEST / "patient_assignments.csv")
    patients = {subset: {x["patient_id"] for x in assignments if x["subset"] == subset}
                for subset in ("inner_train", "inner_validation", "outer_test")}
    if any(patients[a] & patients[b] for a, b in (("inner_train", "inner_validation"), ("inner_train", "outer_test"), ("inner_validation", "outer_test"))):
        raise ValueError("patient split overlap")
    return [x for x in assignments if x["subset"] == "inner_validation"]


def verify_reference():
    gate = json.loads((ROOT / "results/joint6_cv/summary/reporting_repair_complete.json").read_text())
    if gate.get("status") != "pass" or gate.get("repaired_folds") != 10:
        raise RuntimeError("reporting repair must pass first")
    completion = json.loads((ROOT / "results/joint6_cv/summary/completion_status.json").read_text())
    if not completion.get("complete_all_comparators") or completion.get("rejected_results"):
        raise RuntimeError("all historical results must retain valid provenance after reporting repair")
    reference = json.loads((REFERENCE / "launch_provenance.json").read_text())
    pairs = {
        "config_sha256": ROOT / "configs/ecg_lfm/corrected_joint6_lora.yaml",
        "criterion_source_sha256": ROOT / "third_party/ECG-LFM/fairseq_signals/criterions/acs_joint6.py",
        "model_source_sha256": ROOT / "third_party/ECG-LFM/fairseq_signals/models/classification/ecg_transformer_classifier.py",
        "pretrained_sha256": ROOT / "checkpoints/ecg_lfm/pretrain.pt",
        "joint6_label_sha256": MANIFEST.parent / "y_joint_6.npy",
        "split_provenance_sha256": MANIFEST / "split_provenance.json",
        "train_segments_sha256": MANIFEST / "train_segments.csv",
        "valid_segments_sha256": MANIFEST / "valid_segments.csv",
    }
    for key, path in pairs.items():
        if sha256(path) != reference[key]:
            raise RuntimeError(f"reference mismatch: {key}")
    if reference["seed"] != 2026 or reference["fold"] != 0:
        raise RuntimeError("wrong reference seed/fold")
    inner_assignments()
    return {key: reference[key] for key in pairs}


def evaluate_rows(rows):
    _, _, _, _, labels = arrays(rows)
    prob = np.asarray([row["p_omi"] for row in rows])
    thresholds, details = {}, {}
    for name in POINTS:
        thresholds[name], details[name] = select_omi_threshold(labels, prob, name)
    return {
        "evaluation_subset": "inner_validation",
        "outer_test_accessed": False, "external_test_accessed": False,
        "threshold_protocol": "historical grid_0.01_0.99; inner-only; unchanged in LR study",
        "thresholds": thresholds, "threshold_details": details,
        "inner_validation": metric_bundle(rows, thresholds),
    }


def scalars(payload):
    m = payload["inner_validation"]
    o = m["omi"]["youden_j"]
    return {
        "joint6_macro_auprc": m["joint6"]["macro_auprc_ovr"],
        "joint6_macro_auroc": m["joint6"]["macro_auroc_ovr"],
        "joint6_macro_f1": m["joint6"]["macro_f1"],
        "joint6_accuracy": m["joint6"]["accuracy"],
        "acs_macro_auprc": m["acs"]["macro_auprc_ovr"],
        "omi_auprc": o["auprc"], "omi_auroc": o["auroc"],
        "omi_f1": o["f1"], "omi_sensitivity": o["sensitivity"],
        "omi_specificity": o["specificity"], "omi_ppv": o["ppv"], "omi_fn": o["fn"],
        "nste_omi_auprc": m["omi_among_nste_acs"]["auprc"],
        "nste_omi_sensitivity": m["omi_among_nste_acs"]["sensitivity"],
        "nstemi_omi_auprc": m["joint6"]["per_class"]["NSTEMI+OMI"]["auprc"],
        "nstemi_vs_ua_auprc": m["nstemi_vs_ua"]["auprc"],
    }


def choose_learning_rate(results):
    """Predeclared INNER gate, not a clinical non-inferiority test."""
    if set(results) != set(LEARNING_RATES):
        return {"status": "pending", "missing": sorted(set(LEARNING_RATES) - set(results))}
    baseline = results["baseline_lr1e4"]
    eligible, reasons = [], {}
    for name, values in results.items():
        reasons[name] = []
        if any(v is None or not np.isfinite(v) for v in values.values()):
            reasons[name].append("undefined/nonfinite metric")
        else:
            for key, tolerance in (("omi_auprc", .005), ("nste_omi_auprc", .005), ("omi_sensitivity", .02)):
                if values[key] < baseline[key] - tolerance:
                    reasons[name].append(f"{key} below baseline minus {tolerance}")
        if not reasons[name]:
            eligible.append(name)
    if not eligible:
        return {"status": "blocked", "reasons": reasons}
    winner = max(eligible, key=lambda name: (results[name]["joint6_macro_auprc"], name == "baseline_lr1e4"))
    return {"status": "complete", "selected": winner, "learning_rate": LEARNING_RATES[winner],
            "eligible": eligible, "reasons": reasons, "selection_data": "inner_validation only"}


def summarize():
    rows, values = [], {}
    for name, lr in LEARNING_RATES.items():
        path = STUDY / name / "inner_metrics.json"
        if path.exists():
            payload = json.loads(path.read_text())
            receipt = json.loads((path.parent / "inner_evaluation_complete.json").read_text())
            if receipt["metrics_sha256"] != sha256(path):
                raise RuntimeError(f"invalid inner evaluation receipt: {name}")
            values[name] = scalars(payload)
            rows.append({"candidate": name, "learning_rate": lr, "status": "complete", **values[name]})
    STUDY.mkdir(parents=True, exist_ok=True)
    if rows:
        with (STUDY / "lr_inner_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    selection = choose_learning_rate(values)
    atomic_json(STUDY / "lr_selection.json", selection)
    table = ["| 候选 | LR | 状态 | Joint-6 AUPRC | ACS AUPRC | OMI AUPRC | OMI Sensitivity | NSTE OMI AUPRC |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, lr in LEARNING_RATES.items():
        v = values.get(name)
        status_path = STUDY / name / "run_state.json"
        status = json.loads(status_path.read_text()).get("status", "pending") if status_path.exists() else "pending"
        formatted = " | ".join(f"{v[k]:.6f}" for k in ("joint6_macro_auprc", "acs_macro_auprc", "omi_auprc", "omi_sensitivity", "nste_omi_auprc")) if v else "— | — | — | — | —"
        table.append(f"| {name} | {lr:g} | {'complete' if v else status} | {formatted} |")
    block = "\n".join(table) + "\n\n以上全部是 inner-validation 筛选指标，不是 outer/外部测试，不代表独立泛化性能。\n\n"
    block += "选择状态：`" + json.dumps(selection, ensure_ascii=False) + "`。\n"
    if "baseline_lr1e4" in values:
        baseline = values["baseline_lr1e4"]
        for name, v in values.items():
            if name != "baseline_lr1e4":
                delta = v["joint6_macro_auprc"] - baseline["joint6_macro_auprc"]
                block += f"\n- {name} vs baseline：Joint-6 AUPRC Δ={delta:+.6f}，相对变化={delta / baseline['joint6_macro_auprc'] * 100:+.2f}%。"
    report = ROOT / "docs/ECG_LFM_Joint6_Optimization_Report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in text:
            text = text.split(start)[0] + start + "\n" + block + "\n" + end + text.split(end, 1)[1]
        else:
            text += "\n## 5. 学习率实验实时汇总\n\n" + start + "\n" + block + "\n" + end + "\n"
        temporary = report.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8"); temporary.replace(report)
    print(json.dumps({"completed_lr_candidates": len(rows), "selection": selection}), flush=True)


def prepare_baseline():
    identity = verify_reference()
    out = STUDY / "baseline_lr1e4"
    out.mkdir(parents=True, exist_ok=True)
    path = BASE_EVAL / "inner_validation_predictions.csv"
    receipt = json.loads((BASE_EVAL / "evaluation_complete.json").read_text())
    if receipt["artifact_sha256"][path.name] != sha256(path):
        raise ValueError("baseline inner prediction hash mismatch")
    saved = _read_csv(path)
    assignments = inner_assignments()
    by_id = {x["record_id"].zfill(5): x for x in assignments}
    records = [{"record_id": x["record_id"], "patient_id": x["patient_id"],
                "master_idx": int(by_id[x["record_id"].zfill(5)]["idx"])} for x in saved]
    q = np.asarray([[float(row[f"p_{name.lower().replace('+', '_').replace('/', '_')}"]) for name in JOINT6_CLASSES] for row in saved])
    if not np.isfinite(q).all() or (q <= 0).any() or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError("baseline cached joint probabilities invalid; re-inference required")
    source = np.load(MANIFEST.parent / "y_all.npy", mmap_mode="r")
    rows = build_rows(records, np.log(q), assignments, source)
    for old, new in zip(saved, rows):
        for key in ("raw_acs_true", "raw_omi_true", "joint6_true"):
            if int(old[key]) != new[key]:
                raise ValueError("cached baseline label/master join mismatch")
    payload = evaluate_rows(rows)
    payload["provenance"] = {**identity, "source": str(path.relative_to(ROOT)), "source_sha256": sha256(path),
                             "reference_checkpoint_sha256": receipt["checkpoint_sha256"],
                             "cached_reference_not_retrained": True}
    atomic_json(out / "inner_metrics.json", payload)
    atomic_json(out / "inner_evaluation_complete.json", {"status": "pass", "metrics_sha256": sha256(out / "inner_metrics.json")})
    atomic_json(out / "run_state.json", {"status": "complete_cached_inner_reference"})
    summarize()


def run_candidate(name, dry_run=False):
    if name not in ("lr5e5", "lr3e5"):
        raise ValueError("only the two predeclared new LR candidates are allowed")
    identity = verify_reference()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ("0", "1"):
        raise RuntimeError("bind one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1")
    if shutil.disk_usage(ROOT).free < 25 * 2**30:
        raise RuntimeError("less than 25 GiB free; preserve all old artifacts and stop")
    out = STUDY / name
    out.mkdir(parents=True, exist_ok=True)
    run_root = out / "training"
    run = run_root / "lora/fold0"
    if (run / "launch_provenance.json").exists():
        raise RuntimeError("candidate already attempted; inspect state before any restart")
    config = yaml.safe_load((ROOT / "configs/ecg_lfm/corrected_joint6_lora.yaml").read_text())
    config["optimization"]["lr"] = [LEARNING_RATES[name]]
    atomic_json(out / "experiment_config.json", {"candidate": name, "config": config, "reference_hashes": identity,
                "changed_training_variable": "learning rate only", "fold": 0, "seed": 2026,
                "outer_test_accessed": False, "calibration_applied": False,
                "instrumentation": "profiled_hydra_train; no change to gradients or optimizer updates"})
    command = [sys.executable, "-m", "experiments.ecg_lfm.run_joint6_fold", "--method", "lora", "--fold", "0", "--seed", "2026",
               "--output_root", str(run_root.relative_to(ROOT)), "--reuse_label_audit", "--profile_training",
               f"optimization.lr=[{LEARNING_RATES[name]}]"]
    if dry_run:
        print(json.dumps({"status": "dry_run", "command": command, "reference_verified": True}), flush=True)
        return
    atomic_json(out / "run_state.json", {"status": "training", "started_at": datetime.now(timezone.utc).isoformat(),
                                       "cuda_visible_devices": visible, "command": command})
    subprocess.run(command, cwd=ROOT, check=True)
    atomic_json(out / "run_state.json", {"status": "inner_inference", "cuda_visible_devices": visible})
    checkpoint = run / "checkpoint_best.pt"
    records, logits, stats = infer_joint6(checkpoint, MANIFEST, "valid", MANIFEST / "valid_segments.csv", torch.device("cuda:0"), 8, 64)
    rows = build_rows(records, logits, inner_assignments(), np.load(MANIFEST.parent / "y_all.npy", mmap_mode="r"))
    predictions = out / "inner_validation_predictions.csv"
    write_rows(predictions, rows)
    np.save(out / "inner_logits.npy", logits)
    payload = evaluate_rows(rows)
    history = read_fold_history(run / "fairseq_train.jsonl", 0)
    save_curves(out, [history], f"Joint-6 LoRA {name} Fold0 inner screening")
    _, _, y_acs, p_acs, y_omi = arrays(rows)
    save_roc_pr_curves(out, y_acs, p_acs, y_omi, np.asarray([x["p_omi"] for x in rows]))
    payload["model_stats"] = {**stats, **history_cost(history)}
    payload["provenance"] = {**identity, "checkpoint_sha256": sha256(checkpoint),
                             "prediction_sha256": sha256(predictions), "record_id_join": "pass",
                             "evaluation_source_sha256": sha256(Path(__file__)), "candidate": name}
    atomic_json(out / "inner_metrics.json", payload)
    atomic_json(out / "inner_evaluation_complete.json", {"status": "pass", "metrics_sha256": sha256(out / "inner_metrics.json"),
                "prediction_sha256": sha256(predictions), "checkpoint_sha256": payload["provenance"]["checkpoint_sha256"]})
    atomic_json(out / "run_state.json", {"status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(), "cuda_visible_devices": visible})
    summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "run", "summarize"))
    parser.add_argument("--candidate", choices=("lr5e5", "lr3e5"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.action == "baseline":
        prepare_baseline()
    elif args.action == "summarize":
        summarize()
    else:
        try:
            run_candidate(args.candidate, args.dry_run)
        except Exception as exc:
            if args.candidate and not args.dry_run:
                atomic_json(STUDY / args.candidate / "run_state.json", {"status": "failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
