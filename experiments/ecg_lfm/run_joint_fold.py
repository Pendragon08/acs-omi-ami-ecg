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
    class_counts = np.bincount(acs_state, minlength=4).astype(np.float64)
    acs_weight = len(indices) / (4.0 * class_counts)
    acs_weight /= acs_weight.mean()
    return {
        "n_unique_training_records": len(indices),
        "positive": positive,
        "negative": negative,
        "acs_class_counts": {"OTHER": int((acs_state == 0).sum()), "STEMI": int((acs_state == 1).sum()), "NSTEMI": int((acs_state == 2).sum()), "UA": int((acs_state == 3).sum())},
        "acs_static_weight_if_requested": [float(value) for value in acs_weight],
        "omi_pos_weight_if_requested": float(neg[4] / max(1, pos[4])),
        "source": "training-fold unique record indices only",
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, choices=range(5), required=True); p.add_argument("--adaptation", required=True); p.add_argument("--run_name", required=True); p.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2]); p.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm")); p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ecg_lfm/pretrain.pt")); p.add_argument("--output_root", type=Path, default=Path("results/corrected_ecg_lfm_joint")); p.add_argument("--seed", type=int, default=2026); p.add_argument("--weighting", choices=("w0", "w1", "w2", "w3", "w4", "w5", "w6"), default="w0", help="w0=none; w1/w2/w3=static ACS/OMI/both; w4/w5/w6=RDR ACS/OMI/both"); p.add_argument("--rdr_warmup_updates", type=int, default=0); p.add_argument("--omi_pos_weight", action="store_true", help="legacy alias; prefer --weighting"); p.add_argument("--dry_run", action="store_true"); p.add_argument("overrides", nargs="*"); args = p.parse_args()
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
    use_acs_static = args.weighting in ("w1", "w3")
    use_omi_static = args.weighting in ("w2", "w3") or args.omi_pos_weight
    use_acs_rdr = args.weighting in ("w4", "w6")
    use_omi_rdr = args.weighting in ("w5", "w6")
    if (use_acs_rdr or use_omi_rdr) != (args.adaptation == "corrected_joint_lora_rdr"):
        raise ValueError("W4-W6 require corrected_joint_lora_rdr; non-RDR methods require the standard LoRA config")
    if use_acs_static:
        values = ",".join(f"{value:.12g}" for value in bal["acs_static_weight_if_requested"])
        command.append(f"criterion.acs_weight=[{values}]")
    if use_omi_static:
        command.append(f"criterion.omi_pos_weight={bal['omi_pos_weight_if_requested']:.12g}")
    if use_acs_rdr or use_omi_rdr:
        acs_counts = [bal["acs_class_counts"][name] for name in ("OTHER", "STEMI", "NSTEMI", "UA")]
        omi_counts = [bal["negative"]["OMI"], bal["positive"]["OMI"]]
        command.extend(
            [
                f"criterion.rdr_acs={str(use_acs_rdr).lower()}",
                f"criterion.rdr_omi={str(use_omi_rdr).lower()}",
                f"criterion.rdr_acs_class_counts=[{','.join(map(str, acs_counts))}]",
                f"criterion.rdr_omi_class_counts=[{','.join(map(str, omi_counts))}]",
                f"criterion.rdr_warmup_updates={args.rdr_warmup_updates}",
            ]
        )
    command += ["--config-dir", str(config_dir), "--config-name", args.adaptation]
    split_provenance = manifest_dir / "split_provenance.json"
    acs_loss_name = "rdr_weighted_ce" if use_acs_rdr else ("static_weighted_ce" if use_acs_static else "unweighted_ce")
    omi_loss_name = "rdr_weighted_bce" if use_omi_rdr else ("static_pos_weight_bce" if use_omi_static else "unweighted_bce")
    official = root / "third_party" / "ECG-LFM"
    criterion_source = official / "fairseq_signals" / "criterions" / (
        "acs_joint_rdr.py" if use_acs_rdr or use_omi_rdr else "acs_joint.py"
    )
    model_source = official / "fairseq_signals" / "models" / "classification" / "ecg_transformer_classifier.py"
    rdr_source = official / "fairseq_signals" / "modules" / "rdr_official.py"
    provenance = {"started_at": datetime.now(timezone.utc).isoformat(), "task": "ACS categorical OTHER/STEMI/NSTEMI/UA + OMI binary", "fold": args.fold, "adaptation": args.adaptation, "run_name": args.run_name, "seed": args.seed, "weighting": args.weighting, "loss": {"acs": acs_loss_name, "omi": omi_loss_name, "omi_task_weight": 1.0, "acs_weight": bal["acs_static_weight_if_requested"] if use_acs_static else None, "omi_pos_weight": bal["omi_pos_weight_if_requested"] if use_omi_static else None, "weight_source": "inner-train unique records only", "checkpoint_selection_objective": "unweighted inner-validation CE + BCE for every W0-W6 method", "rdr": None if not (use_acs_rdr or use_omi_rdr) else {"upstream_repository": "https://github.com/GoodMorningPeter/RDR", "upstream_commit": "bb55ea9f906f00a2d8938a97543fc232b5d26645", "license": "MIT", "canonical_feature_dim": 1024, "momentum": 0.9, "temperature": 1.0, "warmup_updates": args.rdr_warmup_updates, "warmup_note": "official code defaults to 160/200 epochs; this nested early-stopping protocol freezes a convergence-horizon-matched warmup selected from inner-validation only", "acs_enabled": use_acs_rdr, "omi_enabled": use_omi_rdr}}, "command": command, "training_balance": bal, "config": str(config), "config_sha256": sha256(config), "criterion_source": str(criterion_source), "criterion_source_sha256": sha256(criterion_source), "model_source": str(model_source), "model_source_sha256": sha256(model_source), "rdr_source_sha256": sha256(rdr_source) if use_acs_rdr or use_omi_rdr else None, "runner_source_sha256": sha256(Path(__file__)), "source_label_file": str(source_label_file), "source_label_file_sha256": sha256(source_label_file), "label_file": str(label_file), "label_file_sha256": sha256(label_file), "train_segments_sha256": sha256(manifest_dir / "train_segments.csv"), "valid_segments_sha256": sha256(manifest_dir / "valid_segments.csv"), "split_provenance": str(split_provenance) if split_provenance.exists() else None, "split_provenance_sha256": sha256(split_provenance) if split_provenance.exists() else None, "pretrained_checkpoint": str(checkpoint), "pretrained_sha256": sha256(checkpoint), "official_repository": "https://github.com/biomed-AI/ECG-LFM", "cuda_visible_devices": visible}
    provenance["dry_run"] = bool(args.dry_run)
    (out / "launch_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command, "loss": provenance["loss"]}, ensure_ascii=False))
        return
    proc = subprocess.run(command, cwd=official)
    provenance.update({"finished_at": datetime.now(timezone.utc).isoformat(), "returncode": proc.returncode})
    (out / "launch_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(proc.returncode)


if __name__ == "__main__": main()
