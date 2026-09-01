"""Evaluate one frozen Fold0 RL checkpoint on outer-test exactly once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("outer evaluation requires exactly one CUDA_VISIBLE_DEVICES value")
    run_dir = ROOT / "results/ecg_lfm_rl" / args.run_name / "fold0"
    frozen_path = run_dir / "frozen_config.json"
    if not frozen_path.exists() or not (run_dir / "checkpoint_best.pt").exists():
        raise FileNotFoundError("frozen safety-passing RL checkpoint is required")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_outer_test":
        raise RuntimeError("RL configuration is not frozen")
    if int(frozen.get("outer_test_evaluations", 0)) != 0:
        raise RuntimeError("outer-test has already been evaluated for this frozen RL run")
    marker = run_dir / "outer_evaluation_started.json"
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("outer evaluation marker already exists; refusing a repeated evaluation") from exc
    started = {
        "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name, "fold": 0, "cuda_visible_devices": visible,
        "purpose": "single frozen outer-test evaluation; never parameter selection",
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(started, handle, indent=2, ensure_ascii=False)
    evaluation_dir = ROOT / "results/ecg_lfm_rl_evaluation" / args.run_name / "fold0"
    evaluate = [
        sys.executable, str(ROOT / "experiments/ecg_lfm/evaluate_nested_fold.py"),
        "--run_name", args.run_name, "--fold", "0", "--run_root", "results/ecg_lfm_rl",
        "--output_root", "results/ecg_lfm_rl_evaluation", "--device", args.device,
        "--workers", str(args.workers), "--batch", str(args.batch),
    ]
    calibrate = [
        sys.executable, str(ROOT / "experiments/ecg_lfm/calibrate_nested_fold.py"),
        "--evaluation_dir", str(evaluation_dir),
    ]
    try:
        subprocess.run(evaluate, cwd=ROOT, check=True)
        subprocess.run(calibrate, cwd=ROOT, check=True)
    except Exception as exc:
        started.update({"status": "failed", "error": repr(exc), "outer_metrics_must_not_be_used": True})
        write(marker, started)
        raise
    frozen.update({
        "status": "outer_test_evaluated_once", "outer_test_evaluations": 1,
        "outer_evaluation_finished_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_dir": str(evaluation_dir),
    })
    write(frozen_path, frozen)
    started.update({"status": "pass", "finished_at": frozen["outer_evaluation_finished_at"]})
    write(marker, started)
    print(json.dumps({"status": "pass", "evaluation_dir": str(evaluation_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
