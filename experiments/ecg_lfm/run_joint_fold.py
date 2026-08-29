"""Launch one corrected ECG-LFM joint-task fold through official fairseq."""
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


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def ensure_joint_label_file(source_label_file: Path, joint_label_file: Path) -> Path:
    """Create Fairseq-compatible [OTHER, STEMI, NSTEMI, UA, OMI] labels.

    Fairseq's validation loop reshapes ``sample['label']`` to
    ``model.num_labels``. The source four-column labels cannot be reshaped to
    the corrected five-logit head, so the derived file is materialised on the
    controlled execution server (and is not versioned).
    """
    source = np.load(source_label_file)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(f"expected source y_all [N,4], got {source.shape}")
    if not np.isfinite(source).all() or not np.isin(source, (0, 1)).all():
        raise ValueError("y_all must be finite binary values")
    subtype_sum = source[:, :3].sum(axis=1).astype(np.int64)
    if np.any(subtype_sum > 1):
        raise ValueError("STEMI/NSTEMI/UA overlap in y_all")
    subtype = np.where(subtype_sum == 0, 0, np.argmax(source[:, :3], axis=1) + 1).astype(np.int64)
    joint = np.zeros((len(source), 5), dtype=np.int8)
    joint[np.arange(len(source)), subtype] = 1
    joint[:, 4] = source[:, 3].astype(np.int8)
    if not joint_label_file.exists() or joint_label_file.stat().st_size == 0:
        joint_label_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(joint_label_file, joint)
    else:
        existing = np.load(joint_label_file, mmap_mode="r")
        if existing.shape != joint.shape or not np.array_equal(existing, joint):
            raise ValueError(f"existing corrected label file does not match source: {joint_label_file}")
    return joint_label_file


def balance(segment_csv: Path, label_file: Path) -> dict:
    with segment_csv.open(encoding="utf-8", newline="") as f:
        indices = sorted({int(row["idx"]) for row in csv.DictReader(f)})
    y = np.load(label_file, mmap_mode="r")[indices]
    if y.ndim != 2 or y.shape[1] != 5: raise ValueError(f"expected joint labels [N,5], got {y.shape}")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all(): raise ValueError("y_all must be finite binary values")
    pos = y.sum(axis=0).astype(int); neg = len(indices) - pos
    acs_state = y[:, :4].argmax(axis=1)
    positive = {"STEMI": int(pos[1]), "NSTEMI": int(pos[2]), "UA": int(pos[3]), "OMI": int(pos[4])}
    negative = {"STEMI": int(neg[1]), "NSTEMI": int(neg[2]), "UA": int(neg[3]), "OMI": int(neg[4])}
    return {
        "n_unique_training_records": len(indices),
        "positive": positive,
        "negative": negative,
        "acs_class_counts": {"OTHER": int((acs_state == 0).sum()), "STEMI": int((acs_state == 1).sum()), "NSTEMI": int((acs_state == 2).sum()), "UA": int((acs_state == 3).sum())},
        "omi_pos_weight_if_requested": float(neg[4] / max(1, pos[4])),
        "source": "training-fold unique record indices only",
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, choices=range(5), required=True); p.add_argument("--adaptation", required=True); p.add_argument("--run_name", required=True); p.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2]); p.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm")); p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt")); p.add_argument("--output_root", type=Path, default=Path("results/corrected_ecg_lfm_joint")); p.add_argument("--seed", type=int, default=2026); p.add_argument("--omi_pos_weight", action="store_true"); p.add_argument("overrides", nargs="*"); args = p.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("corrected ECG-LFM runs require exactly one CUDA_VISIBLE_DEVICES value")
    protected = ("task.data=", "task.label_file=", "model.model_path=", "checkpoint.save_dir=", "common.log_file=", "criterion._name=", "model.num_labels=")
    if any(str(item).startswith(protected) for item in args.overrides):
        raise ValueError("protected corrected-task configuration cannot be overridden")
    root = args.project_root.resolve(); manifests = (root / args.manifest_root).resolve() if not args.manifest_root.is_absolute() else args.manifest_root; checkpoint = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint; output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root; manifest_dir = manifests / f"fold{args.fold}"; config_dir = root / "configs" / "ecg_lfm"; config = config_dir / f"{args.adaptation}.yaml"; out = output_root / args.run_name / f"fold{args.fold}"; out.mkdir(parents=True, exist_ok=True); source_label_file = manifests / "y_all.npy"; label_file = ensure_joint_label_file(source_label_file, manifests / "y_joint_5.npy")
    required = [checkpoint, config, label_file, manifest_dir / "train.tsv", manifest_dir / "valid.tsv", manifest_dir / "train_segments.csv"]
    missing = [str(x) for x in required if not x.exists()]
    if missing: raise FileNotFoundError("; ".join(missing))
    bal = balance(manifest_dir / "train_segments.csv", label_file)
    command = ["fairseq-hydra-train", f"task.data={manifest_dir}", f"task.label_file={label_file}", f"model.model_path={checkpoint}", f"checkpoint.save_dir={out}", f"common.log_file={out / 'fairseq_train.jsonl'}", f"common.seed={args.seed}", *args.overrides]
    if args.omi_pos_weight: command.append(f"criterion.omi_pos_weight={bal['omi_pos_weight_if_requested']:.12g}")
    command += ["--config-dir", str(config_dir), "--config-name", args.adaptation]
    official = root / "third_party" / "ECG-LFM"; provenance = {"started_at": datetime.now(timezone.utc).isoformat(), "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + OMI binary", "fold": args.fold, "adaptation": args.adaptation, "run_name": args.run_name, "seed": args.seed, "command": command, "training_balance": bal, "config": str(config), "config_sha256": sha256(config), "source_label_file": str(source_label_file), "source_label_file_sha256": sha256(source_label_file), "label_file": str(label_file), "label_file_sha256": sha256(label_file), "train_segments_sha256": sha256(manifest_dir / "train_segments.csv"), "pretrained_checkpoint": str(checkpoint), "pretrained_sha256": sha256(checkpoint), "official_repository": "https://github.com/biomed-AI/ECG-LFM", "cuda_visible_devices": visible}
    (out / "launch_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(command, cwd=official)
    provenance.update({"finished_at": datetime.now(timezone.utc).isoformat(), "returncode": proc.returncode})
    (out / "launch_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(proc.returncode)


if __name__ == "__main__": main()
