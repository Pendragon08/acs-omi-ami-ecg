"""Stable joint ACS categorical + OMI binary criterion for ECG-LFM."""
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn.functional as F
from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.logging.meters import safe_round
from fairseq_signals.utils import utils


@dataclass
class ACSJointCriterionConfig(Dataclass):
    acs_weight: Optional[List[float]] = field(default=None)
    omi_pos_weight: Optional[float] = field(default=None)
    omi_loss_weight: float = field(default=1.0)
    threshold: float = field(default=0.5)


@register_criterion("acs_joint", dataclass=ACSJointCriterionConfig)
class ACSJointCriterion(BaseCriterion):
    """Cross entropy for ACS states plus numerically stable OMI BCE.

    The source audit is four columns (STEMI/NSTEMI/UA/OMI). Fairseq's
    validation loop reshapes labels to ``model.num_labels``; therefore the
    ECG-LFM runner materialises a five-column training label file
    (OTHER/STEMI/NSTEMI/UA/OMI). Accept both representations so the criterion
    remains explicit about the corrected semantics and can still validate a
    source-label batch in unit tests.
    All optional weights are supplied from the training fold only.
    """

    def __init__(self, cfg: ACSJointCriterionConfig, task):
        super().__init__(task)
        self.acs_weight = torch.tensor(cfg.acs_weight, dtype=torch.float32) if cfg.acs_weight else None
        self.omi_pos_weight = cfg.omi_pos_weight
        self.omi_loss_weight = float(cfg.omi_loss_weight)
        self.threshold = float(cfg.threshold)

    @staticmethod
    def _targets(raw):
        raw = raw.float()
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.shape[-1] not in (4, 5):
            raise ValueError(f"expected source labels [B,4] or joint labels [B,5], got {tuple(raw.shape)}")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isin(raw, torch.tensor((0.0, 1.0), device=raw.device)).all()):
            raise ValueError("source labels must be finite binary values in {0,1}")
        if raw.shape[-1] == 4:
            subtype_sum = raw[:, :3].sum(dim=1)
            if bool((subtype_sum > 1.0).any()):
                raise ValueError("STEMI/NSTEMI/UA overlap found in a categorical ACS batch")
            acs = torch.where(subtype_sum == 0, torch.zeros_like(subtype_sum), raw[:, :3].argmax(dim=1) + 1)
            return acs.long(), raw[:, 3].float()
        # Five-column joint labels are one-hot ACS state + independent OMI.
        acs_one_hot = raw[:, :4]
        if bool((acs_one_hot.sum(dim=1) != 1.0).any()):
            raise ValueError("joint ACS labels must contain exactly one OTHER/STEMI/NSTEMI/UA state")
        return acs_one_hot.argmax(dim=1).long(), raw[:, 4].float()

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        acs_target, omi_target = self._targets(model.get_targets(sample, net_output))
        if logits.ndim != 2 or logits.shape[-1] != 5:
            raise ValueError(f"ACS-joint model must emit [B,5] logits, got {tuple(logits.shape)}")
        # Weighting is an optimization intervention. Checkpoint selection must
        # remain identical across W0--W6, so validation reports the common
        # unweighted CE+BCE objective. This also matches official RDR, where
        # sample reweighting occurs in train() only.
        training = bool(model.training)
        weight = (
            self.acs_weight.to(logits.device)
            if training and self.acs_weight is not None
            else None
        )
        # Fairseq divides accumulated gradients by ``sample_size`` after
        # ``criterion.forward``.  Return summed per-record losses here so a
        # batch-size change does not silently scale the gradients twice.
        acs_loss = F.cross_entropy(logits[:, :4], acs_target.to(logits.device), weight=weight, reduction="sum")
        pos_weight = (
            torch.tensor(float(self.omi_pos_weight), device=logits.device)
            if training and self.omi_pos_weight is not None
            else None
        )
        omi_loss = F.binary_cross_entropy_with_logits(logits[:, 4], omi_target.to(logits.device), pos_weight=pos_weight, reduction="sum")
        loss = acs_loss + self.omi_loss_weight * omi_loss
        if save_outputs:
            self.store(logits.detach(), torch.stack([acs_target.float(), omi_target], dim=1))
        pred_acs = logits[:, :4].argmax(dim=1)
        pred_omi = (torch.sigmoid(logits[:, 4]) >= self.threshold).long()
        logging_output = {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "acs_loss": acs_loss.detach().item(), "omi_loss": omi_loss.detach().item(),
            "sample_size": int(logits.shape[0]), "nsignals": int(sample["id"].numel()),
            "acs_correct": int((pred_acs == acs_target.to(logits.device)).sum().item()),
            "omi_correct": int((pred_omi == omi_target.to(logits.device).long()).sum().item()),
        }
        return loss, int(logits.shape[0]), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(x.get("sample_size", 0)) for x in logging_outputs)
        if not total:
            total = 1.0
        loss = sum(float(x.get("loss", 0.0)) for x in logging_outputs) / total
        acs_loss = sum(float(x.get("acs_loss", 0.0)) for x in logging_outputs) / total
        omi_loss = sum(float(x.get("omi_loss", 0.0)) for x in logging_outputs) / total
        acs_correct = sum(int(x.get("acs_correct", 0)) for x in logging_outputs)
        omi_correct = sum(int(x.get("omi_correct", 0)) for x in logging_outputs)
        metrics.log_scalar(f"{prefix}loss", loss, total, round=5)
        metrics.log_scalar(f"{prefix}acs_loss", acs_loss, total, round=5)
        metrics.log_scalar(f"{prefix}omi_loss", omi_loss, total, round=5)
        metrics.log_scalar(f"{prefix}acs_accuracy", acs_correct / total, total, round=5)
        metrics.log_scalar(f"{prefix}omi_accuracy", omi_correct / total, total, round=5)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True
