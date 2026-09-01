"""Corrected ACS+OMI criterion with a supervised MOO gradient aggregator.

The forward losses are exactly the W0 losses (summed CE+BCE).  Only the
backward step for trainable shared LoRA parameters is changed.  ACS and OMI
rows of the classification projection remain task-specific and never undergo
gradient surgery.
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.criterions.acs_joint import ACSJointCriterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.modules.multi_objective import (
    METHODS,
    MultiObjectiveStrategy,
    merge_named_parameter_gradients,
)


@dataclass
class ACSJointMOOCriterionConfig(Dataclass):
    method: str = "linear_sum"
    c: float = 0.5
    alpha: float = 1.0
    task_prior: List[float] = field(default_factory=lambda: [0.5, 0.5])
    seed: int = 2026
    omi_loss_weight: float = 1.0
    threshold: float = 0.5


@register_criterion("acs_joint_moo", dataclass=ACSJointMOOCriterionConfig)
class ACSJointMOOCriterion(BaseCriterion):
    """Unweighted W0 objective with PCGrad/CAGrad/FairGrad/Aligned-MTL."""

    def __init__(self, cfg: ACSJointMOOCriterionConfig, task):
        super().__init__(task)
        method = str(cfg.method).lower()
        if method not in METHODS:
            raise ValueError(f"unknown MOO method {method!r}; expected {METHODS}")
        if float(cfg.omi_loss_weight) != 1.0:
            raise ValueError("MOO study fixes omi_loss_weight=1.0 to preserve W0")
        self.method = method
        self.omi_loss_weight = 1.0
        self.threshold = float(cfg.threshold)
        self.strategy = MultiObjectiveStrategy(
            method,
            c=float(cfg.c),
            alpha=float(cfg.alpha),
            task_prior=list(cfg.task_prior),
            seed=int(cfg.seed),
        )
        self._last_task_losses: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_logging_output: dict | None = None

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        acs_target, omi_target = ACSJointCriterion._targets(
            model.get_targets(sample, net_output)
        )
        if logits.ndim != 2 or logits.shape[-1] != 5:
            raise ValueError(f"ACS-joint model must emit [B,5] logits, got {tuple(logits.shape)}")
        acs_target = acs_target.to(logits.device)
        omi_target = omi_target.to(logits.device)
        acs_loss = F.cross_entropy(logits[:, :4], acs_target, reduction="sum")
        omi_loss = F.binary_cross_entropy_with_logits(
            logits[:, 4], omi_target, reduction="sum"
        )
        loss = acs_loss + omi_loss
        if model.training:
            # The graph is consumed by backward_multi_objective immediately in
            # task.train_step; validation never stores it.
            self._last_task_losses = (acs_loss, omi_loss)
        else:
            self._last_task_losses = None
        if save_outputs:
            self.store(logits.detach(), torch.stack([acs_target.float(), omi_target], dim=1))
        pred_acs = logits[:, :4].argmax(dim=1)
        pred_omi = (torch.sigmoid(logits[:, 4]) >= self.threshold).long()
        logging_output = {
            "loss": loss.detach().item() if reduce else loss.detach(),
            "acs_loss": acs_loss.detach().item(),
            "omi_loss": omi_loss.detach().item(),
            "sample_size": int(logits.shape[0]),
            "nsignals": int(sample["id"].numel()),
            "acs_correct": int((pred_acs == acs_target).sum().item()),
            "omi_correct": int((pred_omi == omi_target.long()).sum().item()),
            "moo_method": self.method,
        }
        self._last_logging_output = logging_output
        return loss, int(logits.shape[0]), logging_output

    def backward_multi_objective(self, model, optimizer, ignore_grad=False) -> None:
        """Compute task gradients, aggregate the shared part, and write grads.

        Fairseq's trainer performs accumulation, all-reduce, denominator
        scaling, clipping, and optimizer stepping after this hook exactly as it
        does for W0.  We multiply by the active Fairseq loss scale here so the
        FP16 optimizer's later unscale path remains unchanged.
        """
        if ignore_grad:
            self._last_task_losses = None
            return
        if self._last_task_losses is None:
            raise RuntimeError("MOO backward called without a training forward")
        losses = self._last_task_losses
        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named_params]
        grads_acs = torch.autograd.grad(
            losses[0], params, retain_graph=True, allow_unused=True
        )
        grads_omi = torch.autograd.grad(
            losses[1], params, retain_graph=False, allow_unused=True
        )
        merged, diagnostics = merge_named_parameter_gradients(
            named_params, grads_acs, grads_omi, self.strategy
        )
        scaler = getattr(optimizer, "scaler", None)
        scale = float(scaler.loss_scale) if scaler is not None else 1.0
        for (_, parameter), gradient in zip(named_params, merged):
            # MOO operates in float32; restore the model parameter dtype before
            # handing the result to Fairseq's unchanged optimizer path.
            value = gradient.detach() * scale
            if parameter.grad is None:
                parameter.grad = value.to(device=parameter.device, dtype=parameter.dtype)
            else:
                parameter.grad.add_(value.to(device=parameter.device, dtype=parameter.dtype))
        if hasattr(optimizer, "_needs_sync"):
            optimizer._needs_sync = True
        if self._last_logging_output is not None:
            self._last_logging_output.update(diagnostics.as_dict())
        self._last_task_losses = None

    @staticmethod
    def reduce_metrics(logging_outputs, prefix=None):
        prefix = "" if prefix is None else str(prefix).rstrip("_") + "_"
        total = sum(float(item.get("sample_size", 0)) for item in logging_outputs) or 1.0
        for key in ("loss", "acs_loss", "omi_loss"):
            value = sum(float(item.get(key, 0.0)) for item in logging_outputs) / total
            metrics.log_scalar(f"{prefix}{key}", value, total, round=6)
        for task_name in ("acs", "omi"):
            correct = sum(int(item.get(f"{task_name}_correct", 0)) for item in logging_outputs)
            metrics.log_scalar(f"{prefix}{task_name}_accuracy", correct / total, total, round=6)
        # Keep diagnostics additive so the existing fairseq log remains
        # machine-readable while the runner can reconstruct epoch summaries.
        for key in (
            "grad_cosine_acs_omi", "grad_norm_acs_shared", "grad_norm_omi_shared",
            "grad_norm_joint", "post_grad_cosine", "effective_task_weight_acs",
            "effective_task_weight_omi",
        ):
            values = [float(item[key]) for item in logging_outputs if key in item]
            if values:
                metrics.log_scalar(f"{prefix}{key}", sum(values) / len(values), len(values), round=6)
                if key == "grad_cosine_acs_omi":
                    ordered = sorted(values)
                    middle = len(ordered) // 2
                    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
                    metrics.log_scalar(f"{prefix}grad_cosine_acs_omi_median", median, len(values), round=6)
        values = [int(item["gradient_conflict"]) for item in logging_outputs if "gradient_conflict" in item]
        if values:
            metrics.log_scalar(f"{prefix}gradient_conflict_rate", sum(values) / len(values), len(values), round=6)

    @staticmethod
    def logging_outputs_can_be_summed():
        return False
