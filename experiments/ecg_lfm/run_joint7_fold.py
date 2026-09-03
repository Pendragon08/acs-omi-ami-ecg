"""Launch one strictly controlled ECG-LFM LoRA Joint-7 nested fold."""
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


ROOT = Path(__file__).resolve().parents[2]
JOINT_CLASSES = (
    "OTHER", "STEMI+nOMI", "STEMI+OMI", "NSTEMI+nOMI",
    "NSTEMI+OMI", "UA+nOMI", "UA+OMI",
)


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
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project_root", type=Path, default=ROOT)
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--checkpoint", default="checkpoints/ecg_lfm/pretrain.pt")
    parser.add_argument("--output_root", default="results/joint7_stage_a")
    parser.add_argument("--run_name", default="joint7")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("Joint-7 runs require exactly one CUDA_VISIBLE_DEVICES value")
    protected = (
        "task.data=", "task.label_file=", "task.formulation=", "model.model_path=",
        "model.num_labels=", "model.output_semantics=", "criterion._name=",
        "checkpoint.save_dir=", "common.log_file=", "common.seed=",
    )
    if any(str(item).startswith(protected) for item in args.overrides):
        raise ValueError("protected Joint-7 configuration cannot be overridden")

    root = args.project_root.resolve()
    manifests = root / args.manifest_root
    manifest_dir = manifests / f"fold{args.fold}"
    pretrained = root / args.checkpoint
    config = root / "configs/ecg_lfm/corrected_joint7_lora.yaml"
    label_file = manifests / "y_joint_7.npy"
    audit_dir = root / args.output_root / "joint_label_audit"
    out = root / args.output_root / args.run_name / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    audit_command = [
        os.sys.executable, "-m", "experiments.ecg_lfm.audit_joint7_labels",
        "--manifest_root", args.manifest_root, "--fold", str(args.fold),
        "--output", str(Path(args.output_root) / "joint_label_audit"),
    ]
    audit_process = subprocess.run(audit_command, cwd=root)
    audit = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("other_omi_count") != 1:
        print(json.dumps({
            "status": "stopped_before_training",
            "reason": audit.get("stop_reason"),
            "audit": str(audit_dir / "audit_summary.json"),
            "outer_test_accessed": False,
            "external_test_accessed": False,
        }, ensure_ascii=False))
        raise SystemExit(2)
    if audit_process.returncode:
        raise RuntimeError(f"label audit process failed with code {audit_process.returncode}")

    required = [
        pretrained, config, label_file, manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv", manifest_dir / "train_segments.csv",
        manifest_dir / "valid_segments.csv", manifest_dir / "split_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    train_idx = unique_indices(manifest_dir / "train_segments.csv")
    labels = np.load(label_file, mmap_mode="r")[train_idx]
    counts = labels.sum(axis=0).astype(int).tolist()

    command = [
        "fairseq-hydra-train",
        f"task.data={manifest_dir}",
        f"task.label_file={label_file}",
        f"model.model_path={pretrained}",
        f"checkpoint.save_dir={out}",
        f"common.log_file={out / 'fairseq_train.jsonl'}",
        f"common.seed={args.seed}",
        *args.overrides,
        "--config-dir", str(root / "configs/ecg_lfm"),
        "--config-name", "corrected_joint7_lora",
    ]
    official = root / "third_party/ECG-LFM"
    criterion_source = official / "fairseq_signals/criterions/acs_joint7.py"
    model_source = official / "fairseq_signals/models/classification/ecg_transformer_classifier.py"
    split_provenance = manifest_dir / "split_provenance.json"
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "research_question": "P(ACS,OMI|ECG) Joint-7 versus factorized W0",
        "task_formulation": "joint_acs_omi",
        "joint_class_mapping": {str(k): name for k, name in enumerate(JOINT_CLASSES)},
        "loss": "unweighted CrossEntropyLoss over seven logits",
        "checkpoint_criterion": "minimum inner-validation joint CE",
        "outer_test_accessed": False,
        "external_test_accessed": False,
        "fold": args.fold,
        "seed": args.seed,
        "cuda_visible_devices": visible,
        "inner_train_class_counts": dict(zip(JOINT_CLASSES, counts)),
        "command": command,
        "config": str(config),
        "config_sha256": sha256(config),
        "criterion_source_sha256": sha256(criterion_source),
        "model_source_sha256": sha256(model_source),
        "runner_source_sha256": sha256(Path(__file__)),
        "pretrained_checkpoint": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "source_label_sha256": audit["source_label_sha256"],
        "joint_label_sha256": sha256(label_file),
        "split_provenance_sha256": sha256(split_provenance),
        "train_segments_sha256": sha256(manifest_dir / "train_segments.csv"),
        "valid_segments_sha256": sha256(manifest_dir / "valid_segments.csv"),
        "label_audit_sha256": sha256(audit_dir / "audit_summary.json"),
        "lora_initialization_protocol": (
            "consume historical W0 five-row head RNG sequence before LoRA injection; "
            "then independently Xavier-initialize seven-row head"
        ),
        "dry_run": bool(args.dry_run),
    }
    provenance_path = out / "launch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command}, ensure_ascii=False))
        return
    process = subprocess.run(command, cwd=official)
    provenance.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": process.returncode,
    })
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
