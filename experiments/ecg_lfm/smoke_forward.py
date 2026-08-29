"""Single-record and two-segment forward gates for the official ECG-LFM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from fairseq_signals import tasks  # noqa: E402
from fairseq_signals.utils import utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ecg_lfm/linear_probe.yaml")
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--label_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    cfg = OmegaConf.load(args.config)
    cfg.task.data = str(args.manifest_dir.resolve())
    cfg.task.label_file = str(args.label_file.resolve())
    cfg.model.model_path = str(args.checkpoint.resolve())
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    task.load_dataset(args.split, combine=False, epoch=1, task_cfg=cfg.task, label=True, shuffle=False)
    dataset = task.dataset(args.split)
    if len(dataset) != 2:
        raise ValueError(f"Smoke manifest must contain exactly two segments, got {len(dataset)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    segment_logits = []
    with torch.no_grad():
        for index in range(2):
            sample = dataset.collator([dataset[index]])
            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            output = model(**sample["net_input"])
            logits = model.get_logits(output)
            if tuple(logits.shape) != (1, 4):
                raise AssertionError(f"Expected [1,4] logits, got {tuple(logits.shape)}")
            segment_logits.append(logits.float())
    stacked = torch.stack(segment_logits, dim=1)
    record_logits = stacked.mean(dim=1)
    probability = torch.sigmoid(record_logits)
    result = {
        "single_segment_output_shape": [1, 4],
        "two_segment_tensor_shape": list(stacked.shape),
        "record_output_shape": list(probability.shape),
        "finite": bool(torch.isfinite(probability).all()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "probabilities": probability.cpu().tolist(),
    }
    if not result["finite"]:
        raise AssertionError("Smoke probabilities are not finite")
    print("ECG_LFM_FORWARD_WITNESS " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
