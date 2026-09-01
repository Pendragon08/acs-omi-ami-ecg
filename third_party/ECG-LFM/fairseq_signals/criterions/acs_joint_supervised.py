"""Pre-registered supervised-loss screening for the corrected ACS+OMI task."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.criterions.acs_joint import ACSJointCriterion
from fairseq_signals.dataclass import Dataclass


SCHEMES = {"focal", "omi_asl", "label_smoothing", "omi_hnm"}


@dataclass
class ACSJointSupervisedCriterionConfig(Dataclass):
    scheme: str = "focal"
    omi_loss_weight: float = 1.0
    threshold: float = 0.5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    hnm_negative_fraction: float = 0.25


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    return ((1.0 - torch.exp(-ce)).pow(gamma) * ce).sum()


def focal_binary_cross_entropy(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return ((1.0 - torch.exp(-bce)).pow(gamma) * bce).sum()


def asymmetric_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
) -> torch.Tensor:
    """Official-ASL-equivalent binary formulation using raw logits."""
    target = target.float()
    positive = torch.sigmoid(logits.float())
    negative = 1.0 - positive
    if clip > 0:
        negative = (negative + clip).clamp(max=1.0)
    log_likelihood = target * torch.log(positive.clamp_min(1e-8))
    log_likelihood += (1.0 - target) * torch.log(negative.clamp_min(1e-8))
    with torch.no_grad():
        pt = positive * target + negative * (1.0 - target)
        gamma = gamma_pos * target + gamma_neg * (1.0 - target)
        weight = (1.0 - pt).pow(gamma)
    return -(weight * log_likelihood).sum()


def hard_negative_binary_loss(
    logits: torch.Tensor, target: torch.Tensor, negative_fraction: float
) -> torch.Tensor:
    """Keep every positive and concentrate total negative mass on hard negatives."""
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    positive = target > 0.5
    negative_losses = losses[~positive]
    positive_sum = losses[positive].sum()
    if negative_losses.numel() == 0:
        return positive_sum
    keep = max(1, int(torch.ceil(
        torch.tensor(negative_losses.numel() * negative_fraction)
    ).item()))
    hard = torch.topk(negative_losses, k=min(keep, negative_losses.numel())).values
    # Preserve the baseline negative-loss scale while changing only which
    # negatives receive gradients.
    scaled_negative = hard.mean() * negative_losses.numel()
    return positive_sum + scaled_negative


@register_criterion("acs_joint_supervised", dataclass=ACSJointSupervisedCriterionConfig)
class ACSJointSupervisedCriterion(BaseCriterion):
    def __init__(self, cfg: ACSJointSupervisedCriterionConfig, task):
        super().__init__(task)
        if cfg.scheme not in SCHEMES:
            raise ValueError(f"unknown supervised scheme {cfg.scheme!r}; expected {sorted(SCHEMES)}")
        if cfg.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if not 0 <= cfg.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0,1)")
        if not 0 < cfg.hnm_negative_fraction <= 1:
            raise ValueError("hnm_negative_fraction must be in (0,1]")
        self.scheme = cfg.scheme
        self.omi_loss_weight = float(cfg.omi_loss_weight)
        self.threshold = float(cfg.threshold)
        self.focal_gamma = float(cfg.focal_gamma)
        self.label_smoothing = float(cfg.label_smoothing)
        self.asl_gamma_neg = float(cfg.asl_gamma_neg)
        self.asl_gamma_pos = float(cfg.asl_gamma_pos)
        self.asl_clip = float(cfg.asl_clip)
        self.hnm_negative_fraction = float(cfg.hnm_negative_fraction)

    def _training_losses(self, logits, acs_target, omi_target):
        if self.scheme == "focal":
            return (
                focal_cross_entropy(logits[:, :4], acs_target, self.focal_gamma),
                focal_binary_cross_entropy(logits[:, 4], omi_target, self.focal_gamma),
            )
        if self.scheme == "omi_asl":
            return (
                F.cross_entropy(logits[:, :4], acs_target, reduction="sum"),
                asymmetric_binary_loss(
                    logits[:, 4], omi_target, self.asl_gamma_neg,
                    self.asl_gamma_pos, self.asl_clip,
                ),
            )
        if self.scheme == "label_smoothing":
            return (
                F.cross_entropy(
                    logits[:, :4], acs_target, label_smoothing=self.label_smoothing,
                    reduction="sum",
                ),
                F.binary_cross_entropy_with_logits(logits[:, 4], omi_target, reduction="sum"),
            )
        return (
            F.cross_entropy(logits[:, :4], acs_target, reduction="sum"),
            hard_negative_binary_loss(
                logits[:, 4], omi_target, self.hnm_negative_fraction
            ),
        )

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        acs_target, omi_target = ACSJointCriterion._targets(
            model.get_targets(sample, net_output)
        )
        if logits.ndim != 2 or logits.shape[-1] != 5:
            raise ValueError(f"ACS-joint model must emit [B,5] logits, got {tuple(logits.shape)}")
        acs_target, omi_target = acs_target.to(logits.device), omi_target.to(logits.device)
        if model.training:
            acs_loss, omi_loss = self._training_losses(logits, acs_target, omi_target)
        else:
            # Keep early stopping and checkpoint selection identical to W0.
            acs_loss = F.cross_entropy(logits[:, :4], acs_target, reduction="sum")
            omi_loss = F.binary_cross_entropy_with_logits(
                logits[:, 4], omi_target, reduction="sum"
            )
        loss = acs_loss + self.omi_loss_weight * omi_loss
        if save_outputs:
            self.store(logits.detach(), torch.stack([acs_target.float(), omi_target], dim=1))
        pred_acs = logits[:, :4].argmax(dim=1)
        pred_omi = (torch.sigmoid(logits[:, 4]) >= self.threshold).long()
        logging_output = {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "acs_loss": acs_loss.detach().item(), "omi_loss": omi_loss.detach().item(),
            "sample_size": int(logits.shape[0]), "nsignals": int(sample["id"].numel()),
            "acs_correct": int((pred_acs == acs_target).sum().item()),
            "omi_correct": int((pred_omi == omi_target.long()).sum().item()),
        }
        return loss, int(logits.shape[0]), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(item.get("sample_size", 0)) for item in logging_outputs) or 1.0
        for key in ("loss", "acs_loss", "omi_loss"):
            value = sum(float(item.get(key, 0.0)) for item in logging_outputs) / total
            metrics.log_scalar(f"{prefix}{key}", value, total, round=5)
        for task in ("acs", "omi"):
            correct = sum(int(item.get(f"{task}_correct", 0)) for item in logging_outputs)
            metrics.log_scalar(f"{prefix}{task}_accuracy", correct / total, total, round=5)

    @staticmethod
    def logging_outputs_can_be_summed():
        return True
