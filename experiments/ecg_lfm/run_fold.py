"""Legacy independent-sigmoid ECG-LFM launcher.

This entry point is retained for historical reproducibility only.  It trains
four independent STEMI/NSTEMI/UA/OMI outputs and is invalid for the corrected
task because ACS subtypes are mutually exclusive.  Use
``experiments.ecg_lfm.run_joint_fold`` for current runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def training_class_balance(segment_index: Path, label_file: Path) -> dict[str, object]:
    """Compute fold-specific weights from unique training records only."""
    with segment_index.open(encoding="utf-8", newline="") as handle:
        train_indices = sorted({int(row["idx"]) for row in csv.DictReader(handle)})
    labels = np.load(label_file, mmap_mode="r")[train_indices]
    if labels.ndim != 2 or labels.shape[1] != len(LABELS):
        raise ValueError(f"Expected [records,4] labels, got {labels.shape}")
    positive = labels.sum(axis=0, dtype=np.int64)
    negative = len(train_indices) - positive
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("Every training-fold label must contain positive and negative records")
    pos_weight = negative.astype(np.float64) / positive.astype(np.float64)
    return {
        "source": "unique record indices in this fold's train_segments.csv; validation/external excluded",
        "n_unique_training_records": len(train_indices),
        "positive": {label: int(positive[j]) for j, label in enumerate(LABELS)},
        "negative": {label: int(negative[j]) for j, label in enumerate(LABELS)},
        "positive_fraction": {
            label: float(positive[j] / len(train_indices)) for j, label in enumerate(LABELS)
        },
        "pos_weight_negative_over_positive": {
            label: float(pos_weight[j]) for j, label in enumerate(LABELS)
        },
        "pos_weight_vector": pos_weight.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--adaptation", required=True, help="Hydra config basename")
    parser.add_argument("--run_name", help="Independent result-directory name; defaults to adaptation")
    parser.add_argument(
        "--loss", choices=("unweighted_bce", "weighted_bce"), default="unweighted_bce"
    )
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt"))
    parser.add_argument("--output_root", type=Path, default=Path("results/ecg_lfm_4label"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_root = (root / args.manifest_root).resolve() if not args.manifest_root.is_absolute() else args.manifest_root
    pretrained = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    config_dir = root / "configs" / "ecg_lfm"
    manifest_dir = manifest_root / f"fold{args.fold}"
    label_file = manifest_root / "y_all.npy"
    config_path = config_dir / f"{args.adaptation}.yaml"
    run_name = args.run_name or args.adaptation
    output_dir = output_root / run_name / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (
        pretrained,
        label_file,
        config_path,
        manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv",
        manifest_dir / "train_segments.csv",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    balance = training_class_balance(manifest_dir / "train_segments.csv", label_file)
    dynamic_overrides: list[str] = []
    if args.loss == "weighted_bce":
        if any(value.startswith("criterion.pos_weight=") for value in args.overrides):
            raise ValueError("Do not combine --loss weighted_bce with a manual criterion.pos_weight override")
        vector = balance["pos_weight_vector"]
        dynamic_overrides.append(
            "criterion.pos_weight=[" + ",".join(f"{float(value):.12g}" for value in vector) + "]"
        )

    command = [
        "fairseq-hydra-train",
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={pretrained}",
        f"checkpoint.save_dir={output_dir}",
        f"common.log_file={output_dir / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}",
        *dynamic_overrides,
        *args.overrides,
        "--config-dir", str(config_dir),
        "--config-name", args.adaptation,
    ]
    official_repo = root / "third_party" / "ECG-LFM"
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    official_diff = subprocess.check_output(
        ["git", "-C", str(official_repo), "diff", "--binary"]
    )
    official_semantic_diff = subprocess.check_output(
        ["git", "-C", str(official_repo), "diff", "--ignore-space-at-eol", "--binary"]
    )
    official_status = subprocess.check_output(
        ["git", "-C", str(official_repo), "status", "--short", "--untracked-files=all"],
        text=True,
    )
    official_untracked_files = []
    for status_line in official_status.splitlines():
        # ``?? path`` is the only status we need to capture here.  Tracked
        # modifications remain covered by the binary diff hashes above.
        if status_line.startswith("?? "):
            official_untracked_files.append(status_line[3:])
    semantic_numstat = subprocess.check_output(
        ["git", "-C", str(official_repo), "diff", "--ignore-space-at-eol", "--numstat"],
        text=True,
    )
    semantic_changed_files = []
    for line in semantic_numstat.splitlines():
        added, deleted, path = line.split("\t", 2)
        if added != "0" or deleted != "0":
            semantic_changed_files.append(path)
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "adaptation": args.adaptation,
        "run_name": run_name,
        "loss": args.loss,
        "training_class_balance": balance,
        "seed": args.seed,
        "command": command,
        "cwd": str(official_repo),
        "official_repo": "https://github.com/biomed-AI/ECG-LFM",
        "official_commit": official_commit,
        "official_worktree_diff_sha256": sha256_bytes(official_diff),
        "official_worktree_semantic_diff_ignore_eol_sha256": sha256_bytes(official_semantic_diff),
        "official_semantic_changed_files": semantic_changed_files,
        "official_untracked_files": official_untracked_files,
        "official_worktree_has_runtime_patch": bool(official_semantic_diff or official_untracked_files),
        "line_ending_only_changes_present": official_diff != official_semantic_diff,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "pretrained_checkpoint": str(pretrained),
        "pretrained_md5": md5(pretrained),
        "historical_split_sha256": sha256(root / "results/hmoe_kan_stemi_nstemi_ua_omi_e4/split.json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_dir / "launch_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    completed = subprocess.run(command, cwd=official_repo)
    provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
    provenance["returncode"] = completed.returncode
    (output_dir / "launch_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
