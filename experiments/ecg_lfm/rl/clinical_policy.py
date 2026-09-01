"""Factorized categorical-ACS × Bernoulli-OMI clinical policy."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical


@dataclass
class PolicyOutput:
    acs_probabilities: Tensor
    omi_probability: Tensor
    acs_action: Tensor
    omi_action: Tensor
    joint_action: Tensor
    joint_log_probability: Tensor
    acs_entropy: Tensor
    omi_entropy: Tensor


def validate_logits(logits: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError(f"joint policy expects [B,5] logits, got {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("joint policy logits contain NaN/Inf")


def probabilities(logits: Tensor) -> tuple[Tensor, Tensor]:
    validate_logits(logits)
    return torch.softmax(logits[:, :4].float(), dim=-1), torch.sigmoid(logits[:, 4].float())


def joint_log_probability(logits: Tensor, acs_action: Tensor, omi_action: Tensor) -> Tensor:
    acs_probabilities, omi_probability = probabilities(logits)
    if acs_action.shape != omi_action.shape or acs_action.ndim != 1:
        raise ValueError("ACS and OMI actions must both be [B]")
    acs_log = Categorical(probs=acs_probabilities).log_prob(acs_action.long())
    omi_log = Bernoulli(probs=omi_probability).log_prob(omi_action.float())
    return acs_log + omi_log


def sample(logits: Tensor) -> PolicyOutput:
    acs_probabilities, omi_probability = probabilities(logits)
    acs_distribution = Categorical(probs=acs_probabilities)
    omi_distribution = Bernoulli(probs=omi_probability)
    acs_action = acs_distribution.sample()
    omi_action = omi_distribution.sample().long()
    joint_action = acs_action * 2 + omi_action
    return PolicyOutput(
        acs_probabilities=acs_probabilities,
        omi_probability=omi_probability,
        acs_action=acs_action,
        omi_action=omi_action,
        joint_action=joint_action,
        joint_log_probability=(
            acs_distribution.log_prob(acs_action)
            + omi_distribution.log_prob(omi_action.float())
        ),
        acs_entropy=acs_distribution.entropy(),
        omi_entropy=omi_distribution.entropy(),
    )


def greedy_actions(logits: Tensor, omi_threshold: float = 0.5) -> tuple[Tensor, Tensor]:
    acs_probabilities, omi_probability = probabilities(logits)
    return acs_probabilities.argmax(dim=-1), (omi_probability >= omi_threshold).long()

