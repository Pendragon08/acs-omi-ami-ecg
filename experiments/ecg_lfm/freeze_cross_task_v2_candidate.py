"""Freeze exactly one inner-validation v2A candidate before outer evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.acs_joint_task import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=ROOT / "results/ecg_lfm_cross_task_rl_v2/head_only_selection/selection.json")
    args = parser.parse_args()
    selection_path = args.selection if args.selection.is_absolute() else ROOT / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("outer_test_used"):
        raise RuntimeError("selection already records outer-test access")
    selected = selection.get("selected")
    if not selected or not selected.get("safety_pass"):
        raise RuntimeError("a safety-passing inner candidate is required before freeze")
    root = selection_path.parent.parent
    run_name = selected["run_name"]
    run_dir = root / run_name / "fold0"
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    candidate = json.loads((run_dir / "frozen_candidate_config.json").read_text(encoding="utf-8"))
    frozen = {
        "status": "frozen_before_outer_test",
        "study": "Cross-task Clinical RL v2A Head-only",
        "run_name": run_name,
        "outer_test_evaluations": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "selected_update": selected.get("selected_update"),
        "selection_sha256": sha256(selection_path),
        "candidate_freeze": candidate,
    }
    (run_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    manifest = {
        "status": "frozen_before_outer_test",
        "run_name": run_name,
        "outer_test_access": "forbidden_until_single_outer_evaluation",
        "selection_file": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "checkpoint_sha256": sha256(checkpoint),
        "candidate_safety_pass": True,
        "candidate_utility": selected.get("clinical_utility"),
        "outer_evaluations": 0,
    }
    (run_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    selection["final_frozen"] = {"run_name": run_name, "outer_test_evaluations": 0, "freeze_manifest": str(run_dir / "freeze_manifest.json")}
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "frozen_before_outer_test", "run_name": run_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
