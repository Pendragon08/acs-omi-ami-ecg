"""Single-objective seven-class ACS-subtype x OMI criterion."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass


@dataclass
class ACSJoint7CriterionConfig(Dataclass):
    pass


@register_criterion("acs_joint7", dataclass=ACSJoint7CriterionConfig)
class ACSJoint7Criterion(BaseCriterion):
    """Unweighted CE over the fixed seven joint outcomes.

    Labels are stored as seven-column one-hot vectors because the ECG-LFM
    dataset indexes a shared numpy label table.  The criterion validates that
    representation and converts it to categorical indices here.
    """

    @staticmethod
    def _targets(raw: torch.Tensor) -> torch.Tensor:
        raw = raw.float()
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.ndim > 2:
            raw = raw.reshape(-1, raw.shape[-1])
        if raw.ndim != 2 or raw.shape[-1] != 7:
            raise ValueError(f"Joint-7 labels must have shape [B,7], got {tuple(raw.shape)}")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Joint-7 labels contain NaN/Inf")
        if not bool(torch.isin(raw, torch.tensor((0.0, 1.0), device=raw.device)).all()):
            raise ValueError("Joint-7 labels must be binary one-hot values")
        if not bool((raw.sum(dim=1) == 1.0).all()):
            raise ValueError("Joint-7 labels must contain exactly one positive class")
        return raw.argmax(dim=1).long()

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, normalize=False).float()
        target = self._targets(model.get_targets(sample, net_output)).to(logits.device)
        if logits.ndim != 2 or logits.shape[-1] != 7:
            raise ValueError(f"Joint-7 model must emit [B,7] logits, got {tuple(logits.shape)}")
        loss = F.cross_entropy(logits, target, reduction="sum")
        if save_outputs:
            self.store(logits.detach(), target.detach())
        logging_output = {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "joint_loss": loss.detach().item(),
            "sample_size": int(logits.shape[0]),
            "nsignals": int(sample["id"].numel()),
            "joint_correct": int((logits.argmax(dim=1) == target).sum().item()),
        }
        return loss, int(logits.shape[0]), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(item.get("sample_size", 0)) for item in logging_outputs) or 1.0
        loss = sum(float(item.get("loss", 0.0)) for item in logging_outputs) / total
        correct = sum(int(item.get("joint_correct", 0)) for item in logging_outputs)
        metrics.log_scalar(f"{prefix}loss", loss, total, round=5)
        metrics.log_scalar(f"{prefix}joint_loss", loss, total, round=5)
        metrics.log_scalar(f"{prefix}joint_accuracy", correct / total, total, round=5)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True
