"""Build the R1-v2 Head-only candidate pool from inner-validation artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    root = ROOT / "results/ecg_lfm_rl_v2"
    output = root / "head_only_selection"
    output.mkdir(parents=True, exist_ok=True)
    summaries = sorted(root.glob("R1_v2_head_only_*/fold0/training_summary.json"))
    if len(summaries) != 4:
        raise RuntimeError(f"expected 4 Head-only candidates, found {len(summaries)}")
    rows = []
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        provenance = json.loads((path.parent / "launch_provenance.json").read_text(encoding="utf-8"))
        best = summary.get("best")
        if best is None:
            rows.append({"run_name": path.parents[1].name, "status": summary["status"], "safety_pass": False,
                         "updates": summary["updates"], "learning_rate": provenance["config"]["optimization"]["learning_rate"],
                         "target_kl": provenance["target_kl"], "clinical_utility": None})
            continue
        m = best["metrics"]
        rows.append({
            "run_name": path.parents[1].name, "status": summary["status"], "safety_pass": bool(best["safety"]["pass"]),
            "updates": summary["updates"], "selected_update": best["update"],
            "learning_rate": provenance["config"]["optimization"]["learning_rate"], "target_kl": provenance["target_kl"],
            "clinical_utility": best["utility"], "acs_macro_auprc": m["acs"]["macro_auprc_ovr"],
            "nstemi_auprc": m["acs"]["per_class"]["NSTEMI"]["auprc"], "nstemi_f1": m["acs"]["per_class"]["NSTEMI"]["f1"],
            "nstemi_sensitivity": m["acs"]["per_class"]["NSTEMI"]["sensitivity"], "omi_auprc": m["omi"]["auprc"],
            "omi_f1": m["omi"]["f1"], "omi_sensitivity": m["omi"]["sensitivity"],
            "omi_specificity": m["omi"]["specificity"], "omi_ppv": m["omi"]["ppv"],
            "nste_omi_auprc": m["omi_among_nste_acs"]["auprc"],
            "acs_brier": m["calibration"]["acs"]["multiclass_brier"], "omi_brier": m["calibration"]["omi"]["brier"],
            "mean_kl": m["kl"]["total"], "elapsed_seconds": summary["elapsed_seconds"],
            "peak_gpu_memory_gib": summary["peak_gpu_memory_bytes"] / 2**30,
        })
    rows.sort(key=lambda row: (not row["safety_pass"], -(row.get("clinical_utility") or -1e9)))
    with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    passing = [row for row in rows if row["safety_pass"]]
    decision = {"status": "pass", "study": "R1-v2 stabilization Head-only", "outer_test_used": False,
                "n_candidates": len(rows), "n_safety_pass": len(passing),
                "selection_rule": "Safety PASS then maximum clinical utility", "selected": passing[0] if passing else None,
                "head_only_stop": not bool(passing), "candidates": rows}
    (output / "selection.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "selected": decision["selected"], "head_only_stop": decision["head_only_stop"]}, ensure_ascii=False))


if __name__ == "__main__": main()
