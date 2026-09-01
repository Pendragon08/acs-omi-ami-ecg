"""Evaluate the one frozen Cross-task RL v2 candidate on outer Fold0 once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.acs_joint_task import sha256  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--run_root", default="results/ecg_lfm_cross_task_rl_v2")
    parser.add_argument("--output_root", default="results/ecg_lfm_cross_task_rl_v2_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("outer evaluation requires exactly one CUDA_VISIBLE_DEVICES value")
    run_dir = ROOT / args.run_root / args.run_name / "fold0"
    frozen_path = run_dir / "frozen_config.json"
    freeze_manifest_path = run_dir / "freeze_manifest.json"
    checkpoint = run_dir / "checkpoint_best.pt"
    if not frozen_path.exists() or not freeze_manifest_path.exists() or not checkpoint.exists():
        raise FileNotFoundError("frozen v2 candidate checkpoint and manifest are required")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_outer_test":
        raise RuntimeError("candidate is not in frozen_before_outer_test state")
    if int(frozen.get("outer_test_evaluations", 0)) != 0:
        raise RuntimeError("outer-test has already been evaluated for this candidate")
    marker = run_dir / "outer_evaluation_started.json"
    resume_failed = False
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        # The first attempt completed inference but failed before calibration
        # because the calibration script was invoked by absolute path without
        # the repository root on sys.path. Resume only that failed attempt;
        # never run outer inference a second time.
        if previous.get("status") == "failed" and (ROOT / args.output_root / args.run_name / "fold0" / "metrics.json").exists():
            resume_failed = True
        else:
            raise RuntimeError("outer evaluation marker already exists; refusing repeated evaluation")
    started = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "fold": 0,
        "cuda_visible_devices": visible,
        "purpose": "single frozen outer-test evaluation; not a tuning operation",
        "checkpoint_sha256": sha256(checkpoint),
    }
    _write(marker, started)
    evaluation_dir = ROOT / args.output_root / args.run_name / "fold0"
    evaluate = [
        sys.executable,
        str(ROOT / "experiments/ecg_lfm/evaluate_nested_fold.py"),
        "--run_name", args.run_name,
        "--fold", "0",
        "--run_root", args.run_root,
        "--output_root", args.output_root,
        "--device", args.device,
        "--workers", str(args.workers),
        "--batch", str(args.batch),
    ]
    calibrate = [
        sys.executable,
        str(ROOT / "experiments/ecg_lfm/calibrate_nested_fold.py"),
        "--evaluation_dir", str(evaluation_dir),
    ]
    try:
        if not resume_failed:
            subprocess.run(evaluate, cwd=ROOT, check=True)
        subprocess.run(calibrate, cwd=ROOT, check=True)
    except Exception as exc:
        started.update({"status": "failed", "error": repr(exc), "outer_metrics_must_not_be_used": True})
        _write(marker, started)
        raise
    finished = datetime.now(timezone.utc).isoformat()
    frozen.update({
        "status": "outer_test_evaluated_once",
        "outer_test_evaluations": 1,
        "outer_evaluation_finished_at": finished,
        "evaluation_dir": str(evaluation_dir),
    })
    _write(frozen_path, frozen)
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    freeze_manifest.update({
        "status": "outer_test_evaluated_once",
        "outer_evaluations": 1,
        "evaluation_dir": str(evaluation_dir),
        "evaluation_metrics_sha256": sha256(evaluation_dir / "metrics.json"),
    })
    _write(freeze_manifest_path, freeze_manifest)
    started.update({"status": "pass", "finished_at": finished, "evaluation_dir": str(evaluation_dir)})
    _write(marker, started)
    print(json.dumps({"status": "pass", "evaluation_dir": str(evaluation_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
