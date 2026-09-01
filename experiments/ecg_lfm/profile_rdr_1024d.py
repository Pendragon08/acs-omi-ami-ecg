"""Profile canonical 1024-D RDR on a real nested Fold0 feature stream."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))
from experiments.ecg_lfm.evaluate_joint import _read_csv  # noqa: E402
from fairseq_signals.modules.rdr_official import OfficialRDRWeightEstimator  # noqa: E402
from fairseq_signals.utils import utils  # noqa: E402
from models.ecg_lfm_adapter import load_finetuned_model  # noqa: E402


def summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0 if finite else None,
        "nonfinite_count": len(values) - len(finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest_dir", type=Path, default=Path("manifests/ecg_lfm_nested/fold0"))
    parser.add_argument("--label_file", type=Path, default=Path("manifests/ecg_lfm_nested/y_joint_5.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/rdr_profile/fold0_1024d.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--small_subset_batches", type=int, default=8)
    args = parser.parse_args()
    checkpoint = (ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    manifest_dir = (ROOT / args.manifest_dir).resolve() if not args.manifest_dir.is_absolute() else args.manifest_dir
    label_file = (ROOT / args.label_file).resolve() if not args.label_file.is_absolute() else args.label_file
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.eval().to(device)
    task.load_dataset("train", combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False)
    dataset = task.dataset("train")
    iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_signals=args.batch,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=2026,
        num_shards=1,
        shard_id=0,
        num_workers=args.workers,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=False)
    segment_rows = _read_csv(manifest_dir / "train_segments.csv")
    labels = np.load(label_file, mmap_mode="r")
    unique_indices = sorted({int(row["idx"]) for row in segment_rows})
    unique_targets = np.asarray(labels[unique_indices])
    acs_states = unique_targets[:, :4].argmax(1)
    acs_counts = np.bincount(acs_states, minlength=4).astype(int).tolist()
    omi_positive = int(unique_targets[:, 4].sum())
    omi_counts = [len(unique_targets) - omi_positive, omi_positive]
    acs = OfficialRDRWeightEstimator(1024, acs_counts, 0.9, 1.0, profile_condition=True)
    omi = OfficialRDRWeightEstimator(1024, omi_counts, 0.9, 1.0, profile_condition=True)
    elapsed_ms: list[float] = []
    conditions: list[float] = []
    acs_weights_all: list[float] = []
    omi_weights_all: list[float] = []
    fallbacks = {"acs": 0, "omi": 0}
    peak_overhead: list[int] = []
    single_batch: dict | None = None
    for batch_index, sample in enumerate(iterator):
        if batch_index > args.small_subset_batches:
            break
        ids = sample["id"].cpu().numpy().astype(int)
        joint = np.asarray([labels[int(segment_rows[index]["idx"])] for index in ids])
        acs_target = torch.tensor(joint[:, :4].argmax(1), device=device, dtype=torch.long)
        omi_target = torch.tensor(joint[:, 4], device=device, dtype=torch.long)
        sample = utils.move_to_cuda(sample)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            net_output = model(**sample["net_input"])
        features = net_output["pooled_features"]
        model_peak = torch.cuda.max_memory_allocated(device)
        if batch_index == 0:
            acs.update_prototypes(features, acs_target)
            omi.update_prototypes(features, omi_target)
            continue
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        acs_weights, acs_diag = acs.weights(features, acs_target)
        omi_weights, omi_diag = omi.weights(features, omi_target)
        torch.cuda.synchronize(device)
        duration = (time.perf_counter() - start) * 1000.0
        rdr_peak = torch.cuda.max_memory_allocated(device)
        acs.update_prototypes(features, acs_target)
        omi.update_prototypes(features, omi_target)
        elapsed_ms.append(duration)
        conditions.extend(acs_diag.condition_numbers + omi_diag.condition_numbers)
        fallbacks["acs"] += acs_diag.fallback_count
        fallbacks["omi"] += omi_diag.fallback_count
        acs_values = acs_weights.detach().cpu().tolist()
        omi_values = omi_weights.detach().cpu().tolist()
        acs_weights_all.extend(acs_values)
        omi_weights_all.extend(omi_values)
        peak_overhead.append(max(0, int(rdr_peak - model_peak)))
        if single_batch is None:
            single_batch = {
                "batch_size": len(ids),
                "rdr_computation_ms": duration,
                "condition_number": summary(acs_diag.condition_numbers + omi_diag.condition_numbers),
                "acs_weight": summary(acs_values),
                "omi_weight": summary(omi_values),
                "fallback_count": {"acs": acs_diag.fallback_count, "omi": omi_diag.fallback_count},
                "nan_or_inf": not bool(torch.isfinite(acs_weights).all() and torch.isfinite(omi_weights).all()),
                "incremental_peak_memory_bytes": max(0, int(rdr_peak - model_peak)),
            }
    numerical_failure = (
        not all(np.isfinite(acs_weights_all + omi_weights_all))
        or single_batch is None
    )
    result = {
        "status": "fail" if numerical_failure else "pass",
        "profile": "canonical RDR on ECG-LFM pooled 1024-D representation",
        "upstream_commit": "bb55ea9f906f00a2d8938a97543fc232b5d26645",
        "feature_dim": 1024,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_dir),
        "single_batch": single_batch,
        "small_subset": {
            "profiled_batches": len(elapsed_ms),
            "rdr_computation_ms": summary(elapsed_ms),
            "condition_number": summary(conditions),
            "acs_weight": summary(acs_weights_all),
            "omi_weight": summary(omi_weights_all),
            "fallback_count": fallbacks,
            "nan_or_inf": not all(np.isfinite(acs_weights_all + omi_weights_all)),
            "incremental_peak_memory_bytes": summary([float(value) for value in peak_overhead]),
        },
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    if numerical_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
