"""Summarize the complete pre-registered inner-validation RL smoke grid."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    result_root = ROOT / "results/ecg_lfm_rl"
    output = result_root / "inner_grid_v1"
    output.mkdir(parents=True, exist_ok=True)
    summaries = sorted(result_root.glob("R1_clinical_rl_smoke_*/fold0/training_summary.json"))
    if len(summaries) != 8:
        raise RuntimeError(f"expected all 8 registered grid candidates, found {len(summaries)}")
    rows = []
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        provenance = json.loads((path.parent / "launch_provenance.json").read_text(encoding="utf-8"))
        best = summary.get("best")
        if summary.get("status") != "smoke_pass" or best is None:
            raise RuntimeError(f"incomplete or invalid grid candidate: {path}")
        metrics = best["metrics"]
        config = provenance["config"]
        rows.append({
            "run_name": path.parents[1].name,
            "learning_rate": config["optimization"]["learning_rate"],
            "kl_coefficient": config["kl"]["coefficient"],
            "hard_reward_scale": provenance["hard_reward_scale"],
            "safety_pass": best["safety"]["pass"],
            "clinical_utility": best["utility"],
            "acs_macro_auprc": metrics["acs"]["macro_auprc_ovr"],
            "nstemi_auprc": metrics["acs"]["per_class"]["NSTEMI"]["auprc"],
            "nstemi_f1": metrics["acs"]["per_class"]["NSTEMI"]["f1"],
            "nstemi_sensitivity": metrics["acs"]["per_class"]["NSTEMI"]["sensitivity"],
            "omi_auprc": metrics["omi"]["auprc"],
            "omi_f1": metrics["omi"]["f1"],
            "omi_sensitivity": metrics["omi"]["sensitivity"],
            "omi_specificity": metrics["omi"]["specificity"],
            "omi_ppv": metrics["omi"]["ppv"],
            "nste_omi_auprc": metrics["omi_among_nste_acs"]["auprc"],
            "acs_brier": metrics["calibration"]["acs"]["multiclass_brier"],
            "omi_brier": metrics["calibration"]["omi"]["brier"],
            "kl_total": metrics["kl"]["total"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_gpu_memory_gib": summary["peak_gpu_memory_bytes"] / 2**30,
        })
    rows.sort(key=lambda row: (not row["safety_pass"], -row["clinical_utility"], row["kl_total"]))
    with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    passing = [row for row in rows if row["safety_pass"]]
    decision = {
        "protocol": "inner-validation-only pre-registered 2-update sensitivity grid",
        "outer_test_used": False, "n_candidates": len(rows),
        "n_safety_pass": len(passing), "selection_rule": "max clinical utility after safety gate",
        "selected": passing[0] if passing else None, "candidates": rows,
    }
    (output / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "selected": decision["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
