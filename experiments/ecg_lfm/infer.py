"""Record-level inference for official ECG-LFM fine-tuned checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.ecg_lfm_adapter import (  # noqa: E402
    LABELS,
    aggregate_numpy_logits,
    load_finetuned_model,
    parameter_summary,
)
from fairseq_signals.utils import utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "external"), required=True)
    parser.add_argument("--segment_index_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--segment_npz", type=Path)
    parser.add_argument("--y_all", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, saved_cfg, task = load_finetuned_model(args.checkpoint, args.manifest_dir)
    model.eval().to(device)
    task.load_dataset(args.split, combine=False, epoch=1, task_cfg=saved_cfg.task, label=False, shuffle=False)
    dataset = task.dataset(args.split)
    iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_signals=args.batch_size,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=2026,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=False)

    logits = np.full((len(dataset), len(LABELS)), np.nan, dtype=np.float32)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for sample in iterator:
            if not sample:
                continue
            local_ids = sample["id"].cpu().numpy()
            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**sample["net_input"])
                batch_logits = model.get_logits(output).float()
            logits[local_ids] = batch_logits.cpu().numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError("Inference did not fill every manifest position with finite logits")

    with args.segment_index_csv.open(encoding="utf-8", newline="") as handle:
        segment_rows = list(csv.DictReader(handle))
    if len(segment_rows) != len(logits):
        raise ValueError(f"Segment index has {len(segment_rows)} rows but inference has {len(logits)}")
    positions = np.asarray([int(row["manifest_position"]) for row in segment_rows], dtype=np.int64)
    if not np.array_equal(positions, np.arange(len(segment_rows))):
        raise ValueError("segment_index_csv is not in exact manifest order")
    record_indices = np.asarray([int(row["idx"]) for row in segment_rows], dtype=np.int64)
    unique_indices, record_logits = aggregate_numpy_logits(logits, record_indices)
    record_probabilities = 1.0 / (1.0 + np.exp(-record_logits))
    meta_by_idx = {int(row["idx"]): row for row in segment_rows}
    y_all = np.load(args.y_all) if args.y_all else None

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "record_id", "patient_id", "cohort"]
    if y_all is not None:
        fieldnames.extend([f"y_{label}" for label in LABELS])
    fieldnames.extend([f"logit_{label}" for label in LABELS])
    fieldnames.extend([f"prob_{label}" for label in LABELS])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_position, idx in enumerate(unique_indices):
            meta = meta_by_idx[int(idx)]
            output_row: dict[str, object] = {
                "idx": int(idx),
                "record_id": meta["record_id"],
                "patient_id": meta["patient_id"],
                "cohort": meta["cohort"],
            }
            if y_all is not None:
                output_row.update({f"y_{label}": int(y_all[idx, j]) for j, label in enumerate(LABELS)})
            output_row.update({f"logit_{label}": float(record_logits[row_position, j]) for j, label in enumerate(LABELS)})
            output_row.update({f"prob_{label}": float(record_probabilities[row_position, j]) for j, label in enumerate(LABELS)})
            writer.writerow(output_row)

    segment_npz = args.segment_npz or args.output_csv.with_suffix(".segments.npz")
    np.savez_compressed(segment_npz, logits=logits, record_indices=record_indices)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest_dir": str(args.manifest_dir.resolve()),
        "split": args.split,
        "n_segments": int(len(logits)),
        "n_records": int(len(unique_indices)),
        "aggregation": "mean two 5-second segment logits, then sigmoid",
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        **parameter_summary(model),
    }
    args.output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()

