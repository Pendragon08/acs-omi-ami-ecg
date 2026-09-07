"""Launch one controlled observed Joint-6 ECG-LFM nested fold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.ecg_lfm.joint6 import JOINT6_CLASSES


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_indices(path: Path) -> list[int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return sorted({int(row["idx"]) for row in csv.DictReader(handle)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--method", choices=("lora", "partial"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project_root", type=Path, default=ROOT)
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--checkpoint", default="checkpoints/ecg_lfm/pretrain.pt")
    parser.add_argument("--output_root", default="results/joint6_cv/ecg_lfm")
    parser.add_argument("--audit_output", default="results/joint6_cv/label_audit")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reuse_label_audit", action="store_true", help="Read an existing passed audit; never rewrite shared labels")
    parser.add_argument("--profile_training", action="store_true", help="Record train/validation timing and allocator peaks without changing updates")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("Joint-6 runs require exactly one CUDA_VISIBLE_DEVICES value")
    protected = (
        "task.data=", "task.label_file=", "task.formulation=", "model.model_path=",
        "model.num_labels=", "model.output_semantics=", "model.adapter_method=",
        "criterion._name=", "checkpoint.save_dir=", "common.log_file=", "common.seed=",
    )
    if any(str(item).startswith(protected) for item in args.overrides):
        raise ValueError("protected Joint-6 configuration cannot be overridden")

    root = args.project_root.resolve()
    manifests = root / args.manifest_root
    manifest_dir = manifests / f"fold{args.fold}"
    pretrained = root / args.checkpoint
    config_name = f"corrected_joint6_{args.method}"
    config = root / "configs/ecg_lfm" / f"{config_name}.yaml"
    label_file = manifests / "y_joint_6.npy"
    audit_dir = root / args.audit_output
    out = root / args.output_root / args.method / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    audit_command = [
        os.sys.executable, "-m", "experiments.ecg_lfm.audit_joint6_labels",
        "--manifest_root", args.manifest_root,
        "--output", args.audit_output,
    ]
    if not args.reuse_label_audit:
        process = subprocess.run(audit_command, cwd=root)
        if process.returncode:
            raise RuntimeError(f"Joint-6 label audit failed with code {process.returncode}")
    audit_path = audit_dir / "audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("other_omi_exception_count") != 1 or audit.get("ua_omi_count") != 0:
        raise RuntimeError(f"Joint-6 label gate failed: {audit}")

    required = [
        pretrained, config, label_file, manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv", manifest_dir / "outer_test.tsv",
        manifest_dir / "train_segments.csv", manifest_dir / "valid_segments.csv",
        manifest_dir / "outer_test_segments.csv", manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    train_idx = unique_indices(manifest_dir / "train_segments.csv")
    labels = np.load(label_file, mmap_mode="r")[train_idx]
    counts = labels.sum(axis=0).astype(int).tolist()

    entrypoint = ["fairseq-hydra-train"]
    if args.profile_training:
        entrypoint = [os.sys.executable, "-m", "experiments.ecg_lfm.profiled_hydra_train"]
    command = [
        *entrypoint,
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={pretrained}",
        f"checkpoint.save_dir={out}",
        f"common.log_file={out / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}",
        *args.overrides,
        "--config-dir", str(root / "configs/ecg_lfm"),
        "--config-name", config_name,
    ]
    official = root / "third_party/ECG-LFM"
    criterion_source = official / "fairseq_signals/criterions/acs_joint6.py"
    model_source = official / "fairseq_signals/models/classification/ecg_transformer_classifier.py"
    split_provenance = manifest_dir / "split_provenance.json"
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "research_question": "observed Joint-6 model comparison",
        "task_formulation": "joint6_observed_outcome",
        "method": args.method,
        "joint_class_mapping": {str(k): name for k, name in enumerate(JOINT6_CLASSES)},
        "loss": "unweighted CrossEntropyLoss over six logits",
        "checkpoint_criterion": "minimum inner-validation Joint-6 CE",
        "outer_test_accessed_during_training": False,
        "external_test_accessed": False,
        "fold": args.fold,
        "seed": args.seed,
        "cuda_visible_devices": visible,
        "inner_train_class_counts": dict(zip(JOINT6_CLASSES, counts)),
        "command": command,
        "config": str(config),
        "config_sha256": sha256(config),
        "criterion_source_sha256": sha256(criterion_source),
        "model_source_sha256": sha256(model_source),
        "runner_source_sha256": sha256(Path(__file__)),
        "pretrained_checkpoint": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "joint6_label_sha256": sha256(label_file),
        "label_audit_sha256": sha256(audit_path),
        "split_provenance_sha256": sha256(split_provenance),
        "train_segments_sha256": sha256(manifest_dir / "train_segments.csv"),
        "valid_segments_sha256": sha256(manifest_dir / "valid_segments.csv"),
        "initialization_protocol": (
            "consume historical W0 five-row head RNG sequence before adapter injection; "
            "then independently Xavier-initialize the six-row head"
        ),
        "dry_run": bool(args.dry_run),
        "reuse_label_audit": bool(args.reuse_label_audit),
        "profile_training": bool(args.profile_training),
    }
    provenance_path = out / "launch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command}, ensure_ascii=False))
        return
    environment = os.environ.copy()
    if args.profile_training:
        environment["JOINT6_TELEMETRY_PATH"] = str(out / "training_telemetry.json")
    process = subprocess.run(command, cwd=official, env=environment)
    provenance.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": process.returncode,
    })
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
