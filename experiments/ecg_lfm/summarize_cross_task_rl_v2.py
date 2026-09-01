"""Summarize the v2A Head-only candidates using inner-validation only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _row(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    provenance = json.loads((run_dir / "launch_provenance.json").read_text(encoding="utf-8"))
    best = summary.get("best")
    base = {
        "run_name": path.parents[1].name,
        "status": summary.get("status"),
        "safety_pass": bool(best and best.get("safety", {}).get("pass", False)),
        "updates": summary.get("updates", 0),
        "learning_rate": provenance.get("config", {}).get("optimization", {}).get("learning_rate"),
        "target_kl": provenance.get("target_kl"),
        "ranking_coefficient": provenance.get("ranking_coefficient"),
        "clinical_utility": None,
        "outer_test_loaded": bool(summary.get("outer_test_loaded", False)),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "peak_gpu_memory_gib": (summary.get("peak_gpu_memory_bytes") or 0) / 2**30,
    }
    if not best:
        return base
    metrics = best["metrics"]
    nstemi = metrics["acs"]["per_class"]["NSTEMI"]
    omi = metrics["omi"]
    nste = metrics["omi_among_nste_acs"]
    nstemi_omi = metrics["omi_among_nstemi"]
    ua_omi = metrics["omi_among_ua"]
    safety = best["safety"]
    hacking = safety.get("reward_hacking_guard", {})
    base.update(
        {
            "selected_update": best["update"],
            "clinical_utility": best["utility"],
            "acs_macro_auroc": metrics["acs"]["macro_auroc_ovr"],
            "acs_macro_auprc": metrics["acs"]["macro_auprc_ovr"],
            "acs_macro_f1": metrics["acs"]["macro_f1"],
            "nstemi_auprc": nstemi["auprc"],
            "nstemi_f1": nstemi["f1"],
            "nstemi_sensitivity": nstemi["sensitivity"],
            "nstemi_vs_ua_auprc": metrics["nstemi_vs_ua"]["auprc"],
            "omi_auroc": omi["auroc"],
            "omi_auprc": omi["auprc"],
            "omi_f1": omi["f1"],
            "omi_sensitivity": omi["sensitivity"],
            "omi_specificity": omi["specificity"],
            "omi_ppv": omi["ppv"],
            "omi_nste_auprc": nste["auprc"],
            "omi_nstemi_auprc": nstemi_omi["auprc"],
            "omi_ua_auprc": ua_omi["auprc"],
            "acs_brier": metrics["calibration"]["acs"]["multiclass_brier"],
            "acs_ece": metrics["calibration"]["acs"]["top_label_ece"],
            "omi_brier": metrics["calibration"]["omi"]["brier"],
            "omi_ece": metrics["calibration"]["omi"]["ece"],
            "mean_kl": metrics["kl"]["total"],
            "reward_hacking_pass": hacking.get("pass", False),
            "nste_pred_omi_rate": hacking.get("candidate_nste_pred_omi_rate"),
            "nste_pred_omi_rate_ratio": hacking.get("candidate_to_reference_rate_ratio"),
            "nste_ppv_drop": hacking.get("ppv_drop"),
        }
    )
    return base


def main() -> None:
    root = ROOT / "results/ecg_lfm_cross_task_rl_v2"
    output = root / "head_only_selection"
    output.mkdir(parents=True, exist_ok=True)
    summaries = sorted(
        path
        for path in root.glob("R2A_cross_task_head_*/fold0/training_summary.json")
        if "smoke" not in path.parts[-3].lower()
    )
    if len(summaries) != 4:
        raise RuntimeError(f"expected 4 formal v2A candidates, found {len(summaries)}")
    rows = [_row(path) for path in summaries]
    rows.sort(
        key=lambda row: (
            not bool(row["safety_pass"]),
            -(float(row["clinical_utility"]) if row["clinical_utility"] is not None else -1e9),
            row["run_name"],
        )
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passing = [row for row in rows if row["safety_pass"]]
    selected = passing[0] if passing else None
    if selected is None:
        decision_status = "v2A_no_safe_candidate"
    else:
        decision_status = "v2A_safe_candidate"
    decision = {
        "status": decision_status,
        "study": "Cross-task Clinical RL v2A Head-only",
        "outer_test_used": False,
        "n_candidates": len(rows),
        "n_safety_pass": len(passing),
        "selection_rule": "Safety PASS first, then maximum inner-validation clinical utility",
        "primary_endpoints": ["acs_macro_auprc", "omi_auprc"],
        "clinical_utility": "NSTEMI F1 + OMI F1 + OMI AUPRC among NSTE-ACS",
        "selected": selected,
        "v2a_all_failed": not bool(passing),
        "candidates": rows,
    }
    (output / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": decision_status, "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
