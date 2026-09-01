"""Finalize an operator-stopped RL run after a documented critical safety failure."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


CRITICAL = {"acs_brier_guard", "omi_brier_guard", "kl_guard"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.run_dir / "checkpoint_best.pt").exists() or (args.run_dir / "frozen_config.json").exists():
        raise RuntimeError("NO-GO finalizer refuses a saved or frozen checkpoint")
    epoch_files = sorted(args.run_dir.glob("inner_validation_epoch*.json"))
    if not epoch_files:
        raise FileNotFoundError("no completed inner-validation epoch")
    selected = json.loads(epoch_files[0].read_text(encoding="utf-8"))
    failed = {name for name, passed in selected["safety"]["checks"].items() if not passed}
    if not CRITICAL.issubset(failed):
        raise RuntimeError(f"critical safety failure is not established: {sorted(failed)}")
    rows = [json.loads(line) for line in (args.run_dir / "train_log.jsonl").read_text().splitlines() if line]
    completed_updates = [row for row in rows if row.get("event") == "update"]
    baseline = json.loads((args.run_dir / "reference_safety_guard.json").read_text(encoding="utf-8"))
    decision = {
        "status": "RL_NO_GO_CRITICAL_SAFETY_FAILURE",
        "decision_source": "pre-registered Fold0 inner-validation safety gate",
        "completed_validation_epoch": int(selected["epoch"]),
        "failed_guards": sorted(failed),
        "operator_action": "stopped partial Epoch2; no checkpoint saved; outer-test not loaded",
        "updates_logged_total": len(completed_updates),
        "partial_updates_after_selected_epoch_excluded_from_model_selection": int(
            sum(row.get("epoch") == 2 for row in completed_updates)
        ),
        "outer_test_loaded": False,
        "full_five_fold_allowed": False,
        "ablations_allowed": False,
        "metrics": selected["metrics"],
        "safety": selected["safety"],
        "baseline_inner_validation": baseline["baseline_bundle"],
        "finalized_at_unix": time.time(),
    }
    (args.run_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    summary = {
        "status": "critical_safety_gate_stop", "best": None,
        "completed_validation_epochs": 1, "updates_logged_total": len(completed_updates),
        "outer_test_loaded": False, "checkpoint_saved": False,
        "gate_decision": "gate_decision.json",
    }
    (args.run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
