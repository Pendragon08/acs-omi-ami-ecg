"""Launch one nested patient-level ECG-LFM MOO fold through Fairseq.

Stage A intentionally trains only nested Fold0.  ``valid.tsv`` in the nested
manifest is the inner-validation partition; ``outer_test.tsv`` is never
passed to Fairseq and is evaluated only after the method has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ecg_lfm.run_joint_fold import ensure_joint_label_file


METHODS = ("linear_sum", "pcgrad", "cagrad", "fairgrad", "aligned_mtl")
UPSTREAM = {
    "pcgrad": {"url": "https://github.com/tianheyu927/PCGrad", "commit": "c5fbd7c856526373828074f06875230f7f3ee79e"},
    "cagrad": {"url": "https://github.com/Cranial-XIX/CAGrad", "commit": "dc3d48152b6196945cfd56144879b9d42353b095e"},
    "fairgrad": {"url": "https://github.com/OptMN-Lab/FairGrad", "commit": "b58bce810e57288f7bf2c4c9effb7133b856739a"},
    "aligned_mtl": {"url": "https://github.com/taohong08/MTL", "commit": "3f577f1365101fdd16094313b4a3b0d3845ce10d"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm_nested"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt"))
    parser.add_argument("--output_root", type=Path, default=Path("results/multi_objective"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("MOO runs require exactly one CUDA_VISIBLE_DEVICES value")
    protected = (
        "task.data=", "task.label_file=", "model.model_path=", "checkpoint.save_dir=",
        "common.log_file=", "criterion._name=", "model.num_labels=",
    )
    if any(str(item).startswith(protected) for item in args.overrides):
        raise ValueError("task/checkpoint/model configuration is protected")

    root = args.project_root.resolve()
    manifests = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    manifest_dir = manifests / f"fold{args.fold}"
    config_dir = root / "configs" / "ecg_lfm"
    config = config_dir / "corrected_joint_lora_moo.yaml"
    output = output_root / args.method / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    source_label_file = manifests / "y_all.npy"
    label_file = ensure_joint_label_file(source_label_file, manifests / "y_joint_5.npy")
    required = [
        checkpoint, config, label_file, manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv", manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))

    fairseq_train = str(Path(os.environ.get("CONDA_PREFIX", Path(sys.executable).parent)).resolve() / "bin" / "fairseq-hydra-train")
    if not Path(fairseq_train).exists():
        fairseq_train = str(Path(sys.executable).with_name("fairseq-hydra-train"))
    command = [
        fairseq_train,
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={checkpoint}",
        f"checkpoint.save_dir={output}",
        f"common.log_file={output / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}",
        f"criterion.method={args.method}",
        f"criterion.seed={args.seed}",
        *args.overrides,
        "--config-dir", str(config_dir),
        "--config-name", "corrected_joint_lora_moo",
    ]
    official = root / "third_party" / "ECG-LFM"
    criterion_source = official / "fairseq_signals" / "criterions" / "acs_joint_moo.py"
    moo_source = official / "fairseq_signals" / "modules" / "multi_objective.py"
    task_source = official / "fairseq_signals" / "tasks" / "task.py"
    model_source = official / "fairseq_signals" / "models" / "classification" / "ecg_transformer_classifier.py"
    split_provenance = manifest_dir / "split_provenance.json"
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "patient-level nested validation; Stage A Fold0 screening",
        "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + independent OMI binary",
        "fold": args.fold,
        "method": args.method,
        "seed": args.seed,
        "outer_test_access": False,
        "loss": {
            "acs": "unweighted CrossEntropyLoss",
            "omi": "unweighted BCEWithLogitsLoss",
            "joint": "L_ACS + L_OMI",
            "omi_task_weight": 1.0,
            "checkpoint_selection": "unweighted inner-validation CE+BCE (unchanged W0)",
        },
        "model": {
            "base": "official ECG-LFM",
            "adaptation": "LoRA",
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
            "head": "Linear(1024,5); rows 0:4 ACS, row 4 OMI",
        },
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.98],
            "weight_decay": 0.0,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "update_freq": 4,
            "scheduler": "step(period=112, decay=0.99)",
            "max_epoch": 100,
            "early_stopping_patience": 10,
        },
        "multi_objective": {
            "method": args.method,
            "c": 0.5,
            "alpha": 1.0,
            "task_prior": [0.5, 0.5],
            "shared_parameters": "trainable LoRA adapters only; projection rows remain task-specific",
            "upstream": UPSTREAM.get(args.method),
        },
        "command": command,
        "cuda_visible_devices": visible,
        "config": str(config),
        "config_sha256": sha256(config),
        "criterion_source": str(criterion_source),
        "criterion_source_sha256": sha256(criterion_source),
        "moo_source": str(moo_source),
        "moo_source_sha256": sha256(moo_source),
        "task_source": str(task_source),
        "task_source_sha256": sha256(task_source),
        "model_source": str(model_source),
        "model_source_sha256": sha256(model_source),
        "runner_source_sha256": sha256(Path(__file__)),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_sha256": sha256(checkpoint),
        "label_file": str(label_file),
        "label_file_sha256": sha256(label_file),
        "split_provenance": str(split_provenance),
        "split_provenance_sha256": sha256(split_provenance),
        "inner_train_manifest_sha256": sha256(manifest_dir / "train.tsv"),
        "inner_validation_manifest_sha256": sha256(manifest_dir / "valid.tsv"),
        "dry_run": bool(args.dry_run),
    }
    provenance_path = output / "launch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command, "method": args.method}, ensure_ascii=False))
        return
    process = subprocess.run(command, cwd=official)
    provenance.update({"finished_at": datetime.now(timezone.utc).isoformat(), "returncode": process.returncode})
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
