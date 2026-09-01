"""Launch a pre-registered supervised-loss screening fold."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.ecg_lfm.run_joint_fold import ensure_joint_label_file


SCHEMES = {
    "focal": {
        "acs": "focal_ce(gamma=2.0,alpha=None)",
        "omi": "focal_bce(gamma=2.0,alpha=None)",
    },
    "omi_asl": {
        "acs": "unweighted_ce",
        "omi": "asl(gamma_neg=4,gamma_pos=0,clip=0.05)",
    },
    "label_smoothing": {
        "acs": "label_smoothed_ce(epsilon=0.1)",
        "omi": "unweighted_bce",
    },
    "omi_hnm": {
        "acs": "unweighted_ce",
        "omi": "hard_negative_bce(top_negative_fraction=0.25,scale_preserved)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=sorted(SCHEMES), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm_nested"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt"))
    parser.add_argument("--output_root", type=Path, default=Path("results/rl_supervised_screening"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("screening runs require exactly one CUDA_VISIBLE_DEVICES value")
    root = args.project_root.resolve()
    manifests = (root / args.manifest_root).resolve()
    checkpoint = (root / args.checkpoint).resolve()
    output_root = (root / args.output_root).resolve()
    manifest_dir = manifests / f"fold{args.fold}"
    config_dir = root / "configs" / "ecg_lfm"
    config_name = "corrected_joint_lora_supervised"
    config = config_dir / f"{config_name}.yaml"
    output = output_root / args.scheme / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    source_labels = manifests / "y_all.npy"
    joint_labels = ensure_joint_label_file(source_labels, manifests / "y_joint_5.npy")
    required = [
        checkpoint, config, joint_labels, manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv", manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    command = [
        "fairseq-hydra-train", f"task.data={manifest_dir}",
        f"task.label_file={joint_labels}", f"model.model_path={checkpoint}",
        f"checkpoint.save_dir={output}", f"common.log_file={output / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}", f"criterion.scheme={args.scheme}",
        "--config-dir", str(config_dir), "--config-name", config_name,
    ]
    official = root / "third_party" / "ECG-LFM"
    criterion_source = official / "fairseq_signals" / "criterions" / "acs_joint_supervised.py"
    model_source = official / "fairseq_signals" / "models" / "classification" / "ecg_transformer_classifier.py"
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + independent OMI binary",
        "protocol": "patient-level nested Fold0 supervised-loss screening",
        "fold": args.fold, "seed": args.seed, "scheme": args.scheme,
        "loss": {
            **SCHEMES[args.scheme], "omi_task_weight": 1.0,
            "validation_checkpoint_objective": "unweighted CE + unweighted BCE",
        },
        "fairness_constants": {
            "pretrained_checkpoint": "same as Joint W0", "lora_rank": 8,
            "lora_alpha": 16.0, "lora_dropout": 0.05, "learning_rate": 1e-4,
            "batch_size": 64, "update_freq": 4, "optimizer": "adam",
            "scheduler": "step", "early_stopping_patience": 10,
        },
        "hyperparameter_source": "pre-registered standard defaults; outer-test never used for selection",
        "command": command, "cuda_visible_devices": visible,
        "config": str(config), "config_sha256": sha256(config),
        "criterion_source": str(criterion_source), "criterion_source_sha256": sha256(criterion_source),
        "model_source": str(model_source), "model_source_sha256": sha256(model_source),
        "runner_source_sha256": sha256(Path(__file__)),
        "split_provenance_sha256": sha256(manifest_dir / "split_provenance.json"),
        "pretrained_sha256": sha256(checkpoint), "dry_run": bool(args.dry_run),
    }
    provenance_path = output / "launch_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **provenance}, ensure_ascii=False))
        return
    process = subprocess.run(command, cwd=official)
    provenance.update({
        "finished_at": datetime.now(timezone.utc).isoformat(), "returncode": process.returncode,
    })
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
