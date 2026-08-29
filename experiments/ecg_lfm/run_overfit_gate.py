"""Run the diagnostic small-subset overfit gate through the official trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/ecg_lfm_4label/gates/overfit"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    checkpoint = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    manifest_dir = root / "manifests/ecg_lfm/gates/overfit"
    label_file = root / "manifests/ecg_lfm/y_all.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (checkpoint, label_file, manifest_dir / "train.tsv", manifest_dir / "valid.tsv"):
        if not required.exists():
            raise FileNotFoundError(required)

    command = [
        "fairseq-hydra-train",
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={checkpoint}",
        f"checkpoint.save_dir={output_dir}",
        f"common.log_file={output_dir / 'fairseq_train.jsonl'}",
        "common.seed=2026",
        "optimization.max_epoch=10",
        "optimization.max_update=160",
        "optimization.update_freq=[1]",
        "optimization.lr=[0.001]",
        "dataset.batch_size=8",
        "checkpoint.patience=-1",
        "checkpoint.no_save=true",
        "model.freeze_finetune_updates=999999999",
        "--config-dir", str(root / "configs/ecg_lfm"),
        "--config-name", "finetune_fixed",
    ]
    provenance = {
        "gate": "diagnostic_only_not_reported_as_scientific_result",
        "purpose": "prove the official data/model/loss/training path can reduce loss on 64 repeated train/valid records",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "pretrained_md5": md5(checkpoint),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance_path = output_dir / "launch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    completed = subprocess.run(command, cwd=root / "third_party/ECG-LFM")
    provenance.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
    })
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
