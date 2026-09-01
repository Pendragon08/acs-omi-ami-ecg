"""Joint ACS/OMI criterion using the official RDR sample-weighting equation."""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass

from fairseq_signals.criterions.acs_joint import ACSJointCriterion
from fairseq_signals.modules.rdr_official import (
    OfficialRDRWeightEstimator,
    class_weight_stats,
)


ACS_NAMES = ("OTHER", "STEMI", "NSTEMI", "UA")
OMI_NAMES = ("NOMI", "OMI")


@dataclass
class ACSJointRDRCriterionConfig(Dataclass):
    rdr_acs: bool = field(default=False)
    rdr_omi: bool = field(default=False)
    rdr_feature_dim: int = field(default=1024)
    rdr_acs_class_counts: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    rdr_omi_class_counts: List[int] = field(default_factory=lambda: [1, 1])
    rdr_momentum: float = field(default=0.9)
    rdr_temperature: float = field(default=1.0)
    rdr_warmup_updates: int = field(default=0)
    rdr_profile_condition: bool = field(default=False)
    omi_loss_weight: float = field(default=1.0)
    threshold: float = field(default=0.5)


@register_criterion("acs_joint_rdr", dataclass=ACSJointRDRCriterionConfig)
class ACSJointRDRCriterion(BaseCriterion):
    def __init__(self, cfg: ACSJointRDRCriterionConfig, task):
        super().__init__(task)
        if not cfg.rdr_acs and not cfg.rdr_omi:
            raise ValueError("acs_joint_rdr requires at least one enabled RDR task")
        self.rdr_acs = bool(cfg.rdr_acs)
        self.rdr_omi = bool(cfg.rdr_omi)
        self.warmup_updates = int(cfg.rdr_warmup_updates)
        self.omi_loss_weight = float(cfg.omi_loss_weight)
        self.threshold = float(cfg.threshold)
        common = dict(
            feature_dim=int(cfg.rdr_feature_dim),
            momentum=float(cfg.rdr_momentum),
            temperature=float(cfg.rdr_temperature),
            profile_condition=bool(cfg.rdr_profile_condition),
        )
        self.acs_estimator = (
            OfficialRDRWeightEstimator(class_counts=list(cfg.rdr_acs_class_counts), **common)
            if self.rdr_acs
            else None
        )
        self.omi_estimator = (
            OfficialRDRWeightEstimator(class_counts=list(cfg.rdr_omi_class_counts), **common)
            if self.rdr_omi
            else None
        )

    def _task_weights(self, estimator, features, targets, use_rdr, warmup):
        if not use_rdr or warmup:
            return torch.ones(targets.shape[0], device=features.device, dtype=torch.float32), 0
        values, diagnostics = estimator.weights(features, targets)
        return values, diagnostics.fallback_count

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        if "pooled_features" not in net_output:
            raise ValueError("RDR requires model net_output['pooled_features']")
        features = net_output["pooled_features"]
        if features.ndim != 2 or features.shape[1] != 1024:
            raise ValueError(f"canonical ECG-LFM RDR requires [B,1024], got {tuple(features.shape)}")
        acs_target, omi_target = ACSJointCriterion._targets(model.get_targets(sample, net_output))
        acs_target = acs_target.to(logits.device)
        omi_target = omi_target.to(logits.device)
        # RDR is a training-time reweighting rule. Validation uses the common
        # unweighted CE+BCE checkpoint objective shared with W0--W3.
        training = bool(model.training)
        warmup = int(getattr(model, "num_updates", 0)) < self.warmup_updates
        acs_weights, acs_fallback = self._task_weights(
            self.acs_estimator, features, acs_target, self.rdr_acs and training, warmup
        )
        omi_classes = omi_target.long()
        omi_weights, omi_fallback = self._task_weights(
            self.omi_estimator, features, omi_classes, self.rdr_omi and training, warmup
        )
        acs_per_sample = F.cross_entropy(logits[:, :4], acs_target, reduction="none")
        omi_per_sample = F.binary_cross_entropy_with_logits(logits[:, 4], omi_target, reduction="none")
        acs_loss = (acs_per_sample * acs_weights).sum()
        omi_loss = (omi_per_sample * omi_weights).sum()
        loss = acs_loss + self.omi_loss_weight * omi_loss
        if training:
            if self.acs_estimator is not None:
                self.acs_estimator.update_prototypes(features, acs_target)
            if self.omi_estimator is not None:
                self.omi_estimator.update_prototypes(features, omi_classes)
        if save_outputs:
            self.store(logits.detach(), torch.stack([acs_target.float(), omi_target], dim=1))
        logging_output = {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "acs_loss": acs_loss.detach().item(),
            "omi_loss": omi_loss.detach().item(),
            "sample_size": int(logits.shape[0]),
            "nsignals": int(sample["id"].numel()),
            "acs_correct": int((logits[:, :4].argmax(1) == acs_target).sum().item()),
            "omi_correct": int(((torch.sigmoid(logits[:, 4]) >= self.threshold).long() == omi_classes).sum().item()),
            "rdr_active": int(training and not warmup),
            "rdr_acs_fallback": int(acs_fallback),
            "rdr_omi_fallback": int(omi_fallback),
        }
        if training and self.rdr_acs:
            logging_output.update(class_weight_stats(acs_weights, acs_target, ACS_NAMES))
        if training and self.rdr_omi:
            logging_output.update(class_weight_stats(omi_weights, omi_classes, OMI_NAMES))
        return loss, int(logits.shape[0]), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(item.get("sample_size", 0)) for item in logging_outputs) or 1.0
        for name in ("loss", "acs_loss", "omi_loss"):
            value = sum(float(item.get(name, 0.0)) for item in logging_outputs) / total
            metrics.log_scalar(f"{prefix}{name}", value, total, round=6)
        for name in ("acs_correct", "omi_correct"):
            value = sum(int(item.get(name, 0)) for item in logging_outputs) / total
            metrics.log_scalar(f"{prefix}{name.replace('_correct', '_accuracy')}", value, total, round=6)
        metrics.log_scalar(
            f"{prefix}rdr_active_fraction",
            sum(int(item.get("rdr_active", 0)) for item in logging_outputs) / max(1, len(logging_outputs)),
            1,
            round=6,
        )
        for task, names in (("acs", ACS_NAMES), ("omi", OMI_NAMES)):
            for class_name in names:
                key = f"rdr_weight_{class_name}"
                count = sum(int(item.get(f"{key}_count", 0)) for item in logging_outputs)
                if not count:
                    continue
                total_weight = sum(float(item.get(f"{key}_sum", 0.0)) for item in logging_outputs)
                total_square = sum(float(item.get(f"{key}_sumsq", 0.0)) for item in logging_outputs)
                mean = total_weight / count
                variance = max(0.0, total_square / count - mean * mean)
                minimum = min(float(item.get(f"{key}_min", float("inf"))) for item in logging_outputs)
                maximum = max(float(item.get(f"{key}_max", float("-inf"))) for item in logging_outputs)
                metrics.log_scalar(f"{prefix}{task}_{class_name}_weight_mean", mean, count, round=6)
                metrics.log_scalar(f"{prefix}{task}_{class_name}_weight_std", variance ** 0.5, count, round=6)
                metrics.log_scalar(f"{prefix}{task}_{class_name}_weight_min", minimum, 1, round=6)
                metrics.log_scalar(f"{prefix}{task}_{class_name}_weight_max", maximum, 1, round=6)
        for task in ("acs", "omi"):
            metrics.log_scalar(
                f"{prefix}rdr_{task}_fallback_count",
                sum(int(item.get(f"rdr_{task}_fallback", 0)) for item in logging_outputs),
                1,
                round=0,
            )

    @staticmethod
    def logging_outputs_can_be_summed():
        return False
