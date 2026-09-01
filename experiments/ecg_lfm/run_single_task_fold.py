"""Train one nested ECG-LFM LoRA single-task control on exactly one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ecg_lfm.run_joint_fold import ensure_joint_label_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_task_label_file(joint_label_file: Path, task_label_file: Path, task_mode: str) -> Path:
    joint = np.load(joint_label_file)
    if joint.ndim != 2 or joint.shape[1] != 5:
        raise ValueError(f"expected corrected joint labels [N,5], got {joint.shape}")
    values = joint[:, :4] if task_mode == "acs" else joint[:, 4:5]
    if task_mode == "acs" and np.any(values.sum(axis=1) != 1):
        raise ValueError("ACS labels are not exactly one-hot")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("single-task labels must be binary")
    task_label_file.parent.mkdir(parents=True, exist_ok=True)
    if task_label_file.exists():
        existing = np.load(task_label_file, mmap_mode="r")
        if existing.shape != values.shape or not np.array_equal(existing, values):
            raise ValueError(f"existing single-task labels do not match source: {task_label_file}")
    else:
        np.save(task_label_file, values.astype(np.int8))
    return task_label_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_mode", choices=("acs", "omi"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm_nested"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt"))
    parser.add_argument("--output_root", type=Path, default=Path("results/single_task_vs_joint"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("single-task controls require exactly one CUDA_VISIBLE_DEVICES value")
    protected = (
        "task.data=", "task.label_file=", "model.model_path=", "checkpoint.save_dir=",
        "common.log_file=", "criterion._name=", "criterion.task_mode=",
        "model.num_labels=", "model.output_semantics=",
    )
    if any(str(item).startswith(protected) for item in args.overrides):
        raise ValueError("protected single-task configuration cannot be overridden")

    root = args.project_root.resolve()
    manifests = (root / args.manifest_root).resolve() if not args.manifest_root.is_absolute() else args.manifest_root
    checkpoint = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    manifest_dir = manifests / f"fold{args.fold}"
    config_name = f"corrected_{args.task_mode}_only_lora"
    config_dir = root / "configs" / "ecg_lfm"
    config = config_dir / f"{config_name}.yaml"
    output = output_root / args.run_name / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)

    source_label_file = manifests / "y_all.npy"
    joint_label_file = ensure_joint_label_file(source_label_file, manifests / "y_joint_5.npy")
    label_file = ensure_task_label_file(
        joint_label_file, manifests / f"y_{args.task_mode}_only.npy", args.task_mode
    )
    required = [
        checkpoint, config, label_file, manifest_dir / "train.tsv", manifest_dir / "valid.tsv",
        manifest_dir / "train_segments.csv", manifest_dir / "valid_segments.csv",
        manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))

    command = [
        "fairseq-hydra-train",
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={checkpoint}",
        f"checkpoint.save_dir={output}",
        f"common.log_file={output / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}",
        *args.overrides,
        "--config-dir", str(config_dir),
        "--config-name", config_name,
    ]
    official = root / "third_party" / "ECG-LFM"
    criterion_source = official / "fairseq_signals" / "criterions" / "acs_omi_single_task.py"
    model_source = official / "fairseq_signals" / "models" / "classification" / "ecg_transformer_classifier.py"
    split_provenance = manifest_dir / "split_provenance.json"
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA" if args.task_mode == "acs" else "OMI independent binary",
        "task_mode": args.task_mode,
        "comparison_role": "single-task control against nested joint W0",
        "fold": args.fold,
        "run_name": args.run_name,
        "seed": args.seed,
        "adaptation": "ECG-LFM LoRA r=8 alpha=16 dropout=0.05",
        "loss": "unweighted CrossEntropyLoss" if args.task_mode == "acs" else "unweighted BCEWithLogitsLoss",
        "checkpoint_selection_objective": f"inner-validation {args.task_mode.upper()} loss only",
        "command": command,
        "config": str(config),
        "config_sha256": sha256(config),
        "criterion_source_sha256": sha256(criterion_source),
        "model_source_sha256": sha256(model_source),
        "runner_source_sha256": sha256(Path(__file__)),
        "source_joint_label_sha256": sha256(joint_label_file),
        "task_label_sha256": sha256(label_file),
        "train_segments_sha256": sha256(manifest_dir / "train_segments.csv"),
        "valid_segments_sha256": sha256(manifest_dir / "valid_segments.csv"),
        "split_provenance_sha256": sha256(split_provenance),
        "pretrained_sha256": sha256(checkpoint),
        "cuda_visible_devices": visible,
        "dry_run": bool(args.dry_run),
    }
    provenance_path = output / "launch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command}, ensure_ascii=False))
        return
    process = subprocess.run(command, cwd=official)
    provenance.update({"finished_at": datetime.now(timezone.utc).isoformat(), "returncode": process.returncode})
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
