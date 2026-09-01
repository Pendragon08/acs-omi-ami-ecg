"""Explicit clinical reward for the NSTE/NSTEMI--OMI intersection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class CrossTaskRewardConfig:
    """Numerical cross-task utility, kept separate from the v1 base reward."""

    enabled: bool
    scale: float
    nstemi_exact: float
    nstemi_omi_only: float
    nstemi_phenotype_only: float
    nstemi_both_wrong: float
    ua_exact: float
    ua_nstemi_omi: float
    ua_phenotype_only: float
    ua_both_wrong: float

    @classmethod
    def from_mapping(cls, payload: Mapping) -> "CrossTaskRewardConfig":
        nstemi = payload["nstemi_omi"]
        ua = payload["ua_omi"]
        return cls(
            enabled=bool(payload.get("enabled", True)),
            scale=float(payload.get("scale", 1.0)),
            nstemi_exact=float(nstemi["exact"]),
            nstemi_omi_only=float(nstemi["omi_only"]),
            nstemi_phenotype_only=float(nstemi["phenotype_only"]),
            nstemi_both_wrong=float(nstemi["both_wrong"]),
            ua_exact=float(ua["exact"]),
            ua_nstemi_omi=float(ua["nstemi_omi"]),
            ua_phenotype_only=float(ua["phenotype_only"]),
            ua_both_wrong=float(ua["both_wrong"]),
        )


def cross_task_reward(
    acs_action: Tensor,
    omi_action: Tensor,
    acs_target: Tensor,
    omi_target: Tensor,
    config: CrossTaskRewardConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return reward only for true OMI in NSTEMI or UA.

    For a true ``NSTEMI+OMI`` record the configured ordering is
    ``NSTEMI+OMI > UA+OMI > NSTEMI+NOMI > UA+NOMI``.  UA+OMI has its own
    smaller configured utility and is logged separately, so the experiment
    cannot hide which NSTE subtype supplied the reward.
    """
    tensors = (acs_action, omi_action, acs_target, omi_target)
    if any(item.ndim != 1 for item in tensors) or len({item.shape for item in tensors}) != 1:
        raise ValueError("actions and targets must share shape [B]")
    acs_action, omi_action = acs_action.long(), omi_action.long()
    acs_target, omi_target = acs_target.long(), omi_target.long()
    zero = acs_action.float() * 0.0
    if not config.enabled:
        return zero, {"cross_nstemi_omi": zero, "cross_ua_omi": zero}

    true_nstemi_omi = (acs_target == 2) & (omi_target == 1)
    true_ua_omi = (acs_target == 3) & (omi_target == 1)

    nstemi_exact = true_nstemi_omi & (acs_action == 2) & (omi_action == 1)
    nstemi_omi_only = true_nstemi_omi & (acs_action != 2) & (omi_action == 1)
    nstemi_phenotype_only = true_nstemi_omi & (acs_action == 2) & (omi_action == 0)
    nstemi_both_wrong = true_nstemi_omi & (acs_action != 2) & (omi_action == 0)
    ua_exact = true_ua_omi & (acs_action == 3) & (omi_action == 1)
    ua_nstemi_omi = true_ua_omi & (acs_action == 2) & (omi_action == 1)
    ua_phenotype_only = true_ua_omi & (acs_action == 3) & (omi_action == 0)
    ua_both_wrong = true_ua_omi & (acs_action != 3) & (omi_action == 0)

    components = {
        "cross_nstemi_omi_exact": nstemi_exact.float() * config.nstemi_exact * config.scale,
        "cross_nstemi_omi_only": nstemi_omi_only.float() * config.nstemi_omi_only * config.scale,
        "cross_nstemi_phenotype_only": nstemi_phenotype_only.float() * config.nstemi_phenotype_only * config.scale,
        "cross_nstemi_both_wrong": nstemi_both_wrong.float() * config.nstemi_both_wrong * config.scale,
        "cross_ua_omi_exact": ua_exact.float() * config.ua_exact * config.scale,
        "cross_ua_nstemi_omi": ua_nstemi_omi.float() * config.ua_nstemi_omi * config.scale,
        "cross_ua_phenotype_only": ua_phenotype_only.float() * config.ua_phenotype_only * config.scale,
        "cross_ua_both_wrong": ua_both_wrong.float() * config.ua_both_wrong * config.scale,
    }
    components["cross_nstemi_omi"] = sum(
        components[name] for name in components if name.startswith("cross_nstemi_")
    )
    components["cross_ua_omi"] = sum(
        components[name] for name in components if name.startswith("cross_ua_")
    )
    total = components["cross_nstemi_omi"] + components["cross_ua_omi"]
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("cross-task reward contains NaN/Inf")
    return total, components
