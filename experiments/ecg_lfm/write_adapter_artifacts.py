"""Persist adapter target names and trainable-parameter audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.ecg_lfm_adapter import load_finetuned_model, parameter_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model, saved_cfg, _ = load_finetuned_model(args.checkpoint, args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = getattr(model, "adapter_summary", None)
    summary = parameter_summary(model)
    if adapter is not None:
        summary.update(
            {
                "adapter_method": adapter.method,
                "adapter_rank": adapter.rank,
                "adapter_alpha": adapter.alpha,
                "adapter_dropout": adapter.dropout,
                "adapter_target_module_count": len(adapter.target_modules),
                "adapter_target_modules": list(adapter.target_modules),
            }
        )
    else:
        summary.update({"adapter_method": "none", "adapter_target_module_count": 0})
    summary["checkpoint"] = str(args.checkpoint.resolve())
    summary["config_adapter_method"] = str(getattr(saved_cfg.model, "adapter_method", "none"))
    (args.output_dir / "adapter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if adapter is not None:
        (args.output_dir / "adapter_targets.txt").write_text(
            "\n".join(adapter.target_modules) + "\n", encoding="utf-8"
        )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    (args.output_dir / "trainable_parameter_names.txt").write_text(
        "\n".join(trainable) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
