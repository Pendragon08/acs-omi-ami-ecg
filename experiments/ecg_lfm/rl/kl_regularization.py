"""Exact KL for the factorized ACS categorical and OMI Bernoulli policy."""
from __future__ import annotations

import torch
from torch import Tensor

from .clinical_policy import probabilities


def factorized_kl(policy_logits: Tensor, reference_logits: Tensor) -> dict[str, Tensor]:
    policy_acs, policy_omi = probabilities(policy_logits)
    reference_acs, reference_omi = probabilities(reference_logits)
    eps = torch.finfo(policy_acs.dtype).eps
    acs = (
        policy_acs
        * (policy_acs.clamp_min(eps).log() - reference_acs.clamp_min(eps).log())
    ).sum(dim=-1)
    policy_omi = policy_omi.clamp(eps, 1.0 - eps)
    reference_omi = reference_omi.clamp(eps, 1.0 - eps)
    omi = (
        policy_omi * (policy_omi.log() - reference_omi.log())
        + (1.0 - policy_omi)
        * ((1.0 - policy_omi).log() - (1.0 - reference_omi).log())
    )
    total = acs + omi
    return {"acs": acs, "omi": omi, "total": total}

