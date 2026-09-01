"""Single-task controls for the corrected ACS/OMI ECG-LFM experiment."""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass


@dataclass
class ACSSingleTaskCriterionConfig(Dataclass):
    task_mode: str = field(default="acs")
    threshold: float = field(default=0.5)


@register_criterion("acs_omi_single_task", dataclass=ACSSingleTaskCriterionConfig)
class ACSOMISingleTaskCriterion(BaseCriterion):
    """Optimize exactly one head while retaining the common ECG-LFM encoder."""

    def __init__(self, cfg: ACSSingleTaskCriterionConfig, task):
        super().__init__(task)
        self.task_mode = str(cfg.task_mode).lower()
        self.threshold = float(cfg.threshold)
        if self.task_mode not in {"acs", "omi"}:
            raise ValueError(f"task_mode must be acs or omi, got {self.task_mode}")

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, normalize=False).float()
        raw_target = model.get_targets(sample, net_output).float()
        if raw_target.ndim == 1:
            raw_target = raw_target.unsqueeze(-1)

        if self.task_mode == "acs":
            if logits.ndim != 2 or logits.shape[-1] != 4:
                raise ValueError(f"ACS-only model must emit [B,4], got {tuple(logits.shape)}")
            if raw_target.shape[-1] != 4 or bool((raw_target.sum(dim=1) != 1.0).any()):
                raise ValueError("ACS-only labels must be one-hot [OTHER,STEMI,NSTEMI,UA]")
            target = raw_target.argmax(dim=1).long().to(logits.device)
            loss = F.cross_entropy(logits, target, reduction="sum")
            correct = int((logits.argmax(dim=1) == target).sum().item())
        else:
            if logits.ndim != 2 or logits.shape[-1] != 1:
                raise ValueError(f"OMI-only model must emit [B,1], got {tuple(logits.shape)}")
            if raw_target.shape[-1] != 1:
                raise ValueError("OMI-only labels must have shape [B,1]")
            target = raw_target[:, 0].to(logits.device)
            if not bool(torch.isfinite(target).all()) or not bool(
                torch.isin(target, torch.tensor((0.0, 1.0), device=target.device)).all()
            ):
                raise ValueError("OMI-only labels must be finite binary values")
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="sum")
            correct = int(((torch.sigmoid(logits[:, 0]) >= self.threshold).long() == target.long()).sum().item())

        if save_outputs:
            self.store(logits.detach(), target.detach())
        sample_size = int(logits.shape[0])
        return loss, sample_size, {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "sample_size": sample_size,
            "nsignals": int(sample["id"].numel()),
            "correct": correct,
        }

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(item.get("sample_size", 0)) for item in logging_outputs) or 1.0
        loss = sum(float(item.get("loss", 0.0)) for item in logging_outputs) / total
        correct = sum(int(item.get("correct", 0)) for item in logging_outputs)
        metrics.log_scalar(f"{prefix}loss", loss, total, round=5)
        metrics.log_scalar(f"{prefix}accuracy", correct / total, total, round=5)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True
