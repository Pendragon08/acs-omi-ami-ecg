"""Opt-in instrumentation around official Trainer calls; no optimizer/loss edits."""
from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from fairseq_signals.trainer import Trainer
from models.ecg_lfm_adapter import parameter_summary


def capture_allocator_peaks(values):
    """Capture before fairseq internally resets its CUDA peak counters."""
    values["peak_allocated_mib"] = max(values["peak_allocated_mib"], torch.cuda.max_memory_allocated() / 2**20)
    values["peak_reserved_mib"] = max(values["peak_reserved_mib"], torch.cuda.max_memory_reserved() / 2**20)


def reset_with_capture(original, capture):
    @functools.wraps(original)
    def reset(*args, **kwargs):
        capture()
        return original(*args, **kwargs)
    return reset


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or " " in visible:
        raise RuntimeError("exactly one GPU is required")
    output = Path(os.environ["JOINT6_TELEMETRY_PATH"])
    started = time.perf_counter()
    stats = {"status": "running", "cuda_visible_devices": visible,
             "instrumentation_version": 2,
             "captures_fairseq_internal_peak_resets": True,
             "instrumentation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope": "synchronized Trainer train_step / valid_step calls; excludes data loading outside calls, model construction and checkpoint I/O",
             "allocator_units": "MiB; per-process PyTorch allocated/reserved; not nvidia-smi board usage"}
    for kind in ("train", "validation"):
        stats[kind] = {"calls": 0, "seconds": 0.0, "peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    original_reset = torch.cuda.reset_peak_memory_stats
    active_kind = [None]

    def snapshot():
        if active_kind[0] is not None:
            capture_allocator_peaks(stats[active_kind[0]])

    # Trainer.train_step resets counters before it returns. Observe each such
    # reset, rather than incorrectly reporting only the small post-step peak.
    torch.cuda.reset_peak_memory_stats = reset_with_capture(original_reset, snapshot)

    def save():
        stats["wrapper_wall_seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        temporary.replace(output)

    def wrap(original, kind):
        @functools.wraps(original)
        def measured(trainer, *args, **kwargs):
            if "model_stats" not in stats:
                stats["model_stats"] = parameter_summary(trainer.model)
            torch.cuda.synchronize()
            original_reset()
            previous_kind = active_kind[0]
            active_kind[0] = kind
            tick = time.perf_counter()
            try:
                return original(trainer, *args, **kwargs)
            finally:
                torch.cuda.synchronize()
                values = stats[kind]
                values["seconds"] += time.perf_counter() - tick
                values["calls"] += 1
                snapshot()
                active_kind[0] = previous_kind
                stats["num_updates"] = int(trainer.get_num_updates())
                if values["calls"] % 25 == 0:
                    save()
        return measured

    Trainer.train_step = wrap(Trainer.train_step, "train")
    Trainer.valid_step = wrap(Trainer.valid_step, "validation")
    atexit.register(save)
    try:
        from fairseq_cli.hydra_train import cli_main
        cli_main()
        stats["status"] = "complete"
    except BaseException:
        stats["status"] = "failed_or_interrupted"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
