"""官方 ECG-LFM 模型的轻量加载与参数统计工具。

当前联合训练使用五个 logit：全队列状态头为
``OTHER/STEMI/NSTEMI/UA``，另有一个独立 OMI logit。旧四个独立 sigmoid
推理辅助函数已删除，避免被误用于当前任务。
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_finetuned_model(checkpoint: Path, manifest_dir: Path):
    """Load an official fairseq-signals fine-tuned checkpoint for inference."""
    from fairseq_signals.utils import checkpoint_utils

    # Keep the saved ``model_path`` and ``no_pretrained_weights`` values.  The
    # official fine-tuning model is built by first reconstructing the encoder
    # from that pre-training checkpoint (24 layers / 16 heads for ECG-LFM) and
    # then loading the fine-tuned state dict.  Clearing these values makes the
    # loader fall back to the classifier dataclass defaults (12 heads), which
    # do not describe the trained encoder and cannot load this checkpoint.
    overrides = {
        "task": {"data": str(manifest_dir)},
    }
    model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        str(checkpoint), arg_overrides=overrides
    )
    return model, saved_cfg, task


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    requires_grad = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    effective = requires_grad
    encoder = getattr(model, "encoder", None)
    freeze_until = int(getattr(model, "freeze_finetune_updates", 0))
    num_updates = int(getattr(model, "num_updates", 0))
    if encoder is not None and num_updates < freeze_until:
        effective = int(sum(parameter.numel() for parameter in model.proj.parameters() if parameter.requires_grad))
    elif encoder is not None and float(getattr(encoder, "feature_grad_mult", 1.0)) <= 0:
        feature_extractor = getattr(encoder, "feature_extractor", None)
        if feature_extractor is not None:
            effective -= int(sum(parameter.numel() for parameter in feature_extractor.parameters() if parameter.requires_grad))
    summary = {
        "total_parameters": total,
        "requires_grad_parameters": requires_grad,
        "effective_trainable_parameters_at_checkpoint": effective,
        "checkpoint_num_updates": num_updates,
        "encoder_frozen_until_update": freeze_until,
    }
    adapter = getattr(model, "adapter_summary", None)
    if adapter is not None:
        summary.update(
            {
                "adapter_method": str(adapter.method),
                "adapter_rank": int(adapter.rank),
                "adapter_alpha": float(adapter.alpha),
                "adapter_dropout": float(adapter.dropout),
                "adapter_target_module_count": int(len(adapter.target_modules)),
            }
        )
    else:
        summary["adapter_method"] = "none"
    return summary
