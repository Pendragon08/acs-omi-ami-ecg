"""Externally configured exploratory clinical reward and soft-F1 shaping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardWeights:
    acs_correct: float
    omi_correct: float
    nstemi_correct_bonus: float
    nstemi_to_ua_penalty: float
    stemi_to_other_penalty: float
    omi_tp_bonus: float
    omi_fn_penalty: float
    nste_omi_tp_bonus: float
    acs_soft_f1: float
    omi_soft_f1: float

    @classmethod
    def from_mapping(cls, payload: Mapping) -> "RewardWeights":
        return cls(
            acs_correct=float(payload["acs"]["correct"]),
            omi_correct=float(payload["omi"]["correct"]),
            nstemi_correct_bonus=float(payload["acs"]["nstemi_correct_bonus"]),
            nstemi_to_ua_penalty=float(payload["acs"]["nstemi_to_ua_penalty"]),
            stemi_to_other_penalty=float(payload["acs"]["stemi_to_other_penalty"]),
            omi_tp_bonus=float(payload["omi"]["tp_bonus"]),
            omi_fn_penalty=float(payload["omi"]["fn_penalty"]),
            nste_omi_tp_bonus=float(payload["nste_omi"]["correct_positive_bonus"]),
            acs_soft_f1=float(payload["soft_f1"]["acs_weight"]),
            omi_soft_f1=float(payload["soft_f1"]["omi_weight"]),
        )


def hard_reward(
    acs_action: Tensor,
    omi_action: Tensor,
    acs_target: Tensor,
    omi_target: Tensor,
    weights: RewardWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    acs_action, omi_action = acs_action.long(), omi_action.long()
    acs_target, omi_target = acs_target.long(), omi_target.long()
    tensors = (acs_action, omi_action, acs_target, omi_target)
    if any(item.ndim != 1 for item in tensors) or len({item.shape for item in tensors}) != 1:
        raise ValueError("actions and targets must share shape [B]")
    acs_base = (acs_action == acs_target).float() * weights.acs_correct
    omi_base = (omi_action == omi_target).float() * weights.omi_correct
    nstemi = (
        (acs_target == 2) & (acs_action == 2)
    ).float() * weights.nstemi_correct_bonus
    nstemi_to_ua = (
        (acs_target == 2) & (acs_action == 3)
    ).float() * weights.nstemi_to_ua_penalty
    stemi_to_other = (
        (acs_target == 1) & (acs_action == 0)
    ).float() * weights.stemi_to_other_penalty
    omi_tp = ((omi_target == 1) & (omi_action == 1)).float() * weights.omi_tp_bonus
    omi_fn = ((omi_target == 1) & (omi_action == 0)).float() * weights.omi_fn_penalty
    nste_omi = (
        ((acs_target == 2) | (acs_target == 3))
        & (omi_target == 1)
        & (omi_action == 1)
    ).float() * weights.nste_omi_tp_bonus
    components = {
        "acs_base": acs_base,
        "omi_base": omi_base,
        "nstemi": nstemi,
        "nstemi_to_ua": nstemi_to_ua,
        "stemi_to_other": stemi_to_other,
        "omi_tp": omi_tp,
        "omi_fn": omi_fn,
        "nste_omi": nste_omi,
    }
    total = sum(components.values())
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("clinical reward contains NaN/Inf")
    return total, components


def soft_f1_shaping(
    acs_probabilities: Tensor,
    omi_probability: Tensor,
    acs_target: Tensor,
    omi_target: Tensor,
    weights: RewardWeights,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    one_hot = torch.nn.functional.one_hot(acs_target.long(), num_classes=4).float()
    tp = (acs_probabilities * one_hot).sum(dim=0)
    fp = (acs_probabilities * (1.0 - one_hot)).sum(dim=0)
    fn = ((1.0 - acs_probabilities) * one_hot).sum(dim=0)
    acs_f1 = (2.0 * tp / (2.0 * tp + fp + fn + eps)).mean()
    omi_target = omi_target.float()
    omi_tp = (omi_probability * omi_target).sum()
    omi_fp = (omi_probability * (1.0 - omi_target)).sum()
    omi_fn = ((1.0 - omi_probability) * omi_target).sum()
    omi_f1 = 2.0 * omi_tp / (2.0 * omi_tp + omi_fp + omi_fn + eps)
    total = weights.acs_soft_f1 * acs_f1 + weights.omi_soft_f1 * omi_f1
    return total, {"acs_soft_f1": acs_f1, "omi_soft_f1": omi_f1}

