"""Create protocol-v2 comparison and a complete machine-readable result inventory."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "protocol_v2"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_status(rel: Path) -> str:
    top = rel.parts[0]
    if top in {"exp1", "smoke", "smoke_1label"}:
        return "legacy_stale_split"
    if top == "protocol_v2":
        return "protocol_v2_formal"
    if top == "protocol_v2_smoke":
        return "protocol_v2_smoke"
    return "historical_exploratory"


def make_comparison():
    posthoc = read_json(RESULTS / "final_fusion_focal" / "threshold_selection_summary.json")
    rows = []
    payload = {
        "comparison_scope": (
            "same historical evaluation records, but different training/selection protocols; "
            "old numbers are adaptively selected exploratory results and v2 evaluation is not a pristine holdout"
        ),
        "seed_scope": "one complete run per task, seed=2026; no across-seed uncertainty claim",
        "tasks": {},
    }
    for task in ("OMI", "AMI"):
        old = read_json(RESULTS / "final_fusion_focal" / f"XceptionTime_{task}_bce_P3_seed2026.json")
        new = read_json(OUT / f"XceptionTime_{task}_bce_P4_protocolv2_seed2026.json")
        old_val = old["val"]
        new_eval = new["metrics"]["evaluation"]
        new_op = new_eval["operating_point"]
        old_joint = 2 * old_val[f"{task}_f1"] * old_val[f"{task}_sens"] / (
            old_val[f"{task}_f1"] + old_val[f"{task}_sens"]
        )
        values = {
            "AUROC": (old_val[f"{task}_auroc"], new_eval["ranking_raw"]["auroc"]),
            "AUPRC": (old_val[f"{task}_auprc"], new_eval["ranking_raw"]["auprc"]),
            "threshold": (old_val[f"{task}_threshold"], new_op["threshold"]),
            "Sensitivity": (old_val[f"{task}_sens"], new_op["sens"]),
            "Specificity": (old_val[f"{task}_spec"], new_op["spec"]),
            "PPV": (old_val[f"{task}_ppv"], new_op["ppv"]),
            "NPV": (old_val[f"{task}_npv"], new_op["npv"]),
            "Accuracy": (old_val[f"{task}_acc"], new_op["acc"]),
            "F1": (old_val[f"{task}_f1"], new_op["f1"]),
            "HMean_F1_Sensitivity": (old_joint, new_op["joint_f1_sens_hmean"]),
        }
        for metric, (old_value, new_value) in values.items():
            rows.append({
                "task": task,
                "metric": metric,
                "old_exploratory": old_value,
                "protocol_v2_evaluation": new_value,
                "delta_v2_minus_old": new_value - old_value,
            })
        payload["tasks"][task] = {
            "old_stored_operating_point": {k: v[0] for k, v in values.items()},
            "old_posthoc_joint_operating_point": posthoc["tasks"][task]["recommended"],
            "protocol_v2": {
                "best_epoch": new["model_info"]["best_epoch"],
                "epochs_completed": new["config"]["epochs_completed"],
                "calibration_selected": new["threshold_selection"]["selected"],
                "calibration_max_f1_reference": new["threshold_selection"]["max_f1_reference"],
                "evaluation": new_eval,
                "platt": new["calibration_model"],
                "train_time_seconds": new["model_info"]["train_time_seconds"],
                "peak_gpu_memory_mb": new["model_info"]["peak_gpu_memory_mb"],
            },
            "delta_v2_minus_old": {k: v[1] - v[0] for k, v in values.items()},
        }

    csv_path = OUT / "protocol_v2_vs_old.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT / "protocol_v2_vs_old.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def make_inventory():
    excluded = {"EXPERIMENT_INVENTORY.csv", "EXPERIMENT_INVENTORY.json"}
    rows = []
    parse_errors = []
    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file() or path.name in excluded or path.suffix.lower() not in {".json", ".csv", ".md", ".pt"}:
            continue
        rel = path.relative_to(RESULTS)
        row = {
            "path": str(rel),
            "family": rel.parts[0],
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "status": legacy_status(rel),
            "protocol": "",
            "run_id": "",
            "task": "",
            "model": "",
        }
        if path.suffix.lower() == ".json":
            try:
                obj = read_json(path)
                if isinstance(obj, dict):
                    row["protocol"] = obj.get("protocol", "")
                    row["run_id"] = obj.get("run_id", "")
                    row["task"] = obj.get("task", "")
                    row["model"] = obj.get("model", "")
            except Exception as exc:
                parse_errors.append({"path": str(rel), "error": repr(exc)})
        rows.append(row)

    csv_path = RESULTS / "EXPERIMENT_INVENTORY.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "generated_by": "scripts/ecg/build_protocol_v2_summary.py",
        "n_artifacts": len(rows),
        "parse_errors": parse_errors,
        "counts_by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
        "counts_by_status": dict(sorted(Counter(r["status"] for r in rows).items())),
        "counts_by_extension": dict(sorted(Counter(r["extension"] for r in rows).items())),
        "inventory_csv": "results/EXPERIMENT_INVENTORY.csv",
        "scope": "All JSON, CSV, Markdown and checkpoint files below results/ at generation time.",
    }
    json_path = RESULTS / "EXPERIMENT_INVENTORY.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path, summary


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    comp = make_comparison()
    inv = make_inventory()
    print(json.dumps({
        "comparison": [str(x.relative_to(ROOT)) for x in comp],
        "inventory": [str(inv[0].relative_to(ROOT)), str(inv[1].relative_to(ROOT))],
        "inventory_summary": inv[2],
    }, ensure_ascii=False, indent=2))
