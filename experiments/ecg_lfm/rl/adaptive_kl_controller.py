"""Adaptive target-KL controller used by the cross-task RL v2 study."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def update_kl_controller(
    coefficient: float,
    current_kl: float,
    target_kl: float,
    config: Mapping[str, float],
) -> float:
    """Adjust the KL penalty while keeping its scale bounded.

    ``config`` contains ``high_ratio``, ``low_ratio``, ``increase_factor``,
    ``decrease_factor``, ``min_coefficient`` and ``max_coefficient``.  The
    controller is deliberately stateless so every candidate can be restarted
    from the same W0 initialization.
    """
    if not np.isfinite(current_kl) or current_kl < 0:
        raise ValueError(f"current_kl must be finite and non-negative, got {current_kl}")
    if not np.isfinite(target_kl) or target_kl <= 0:
        raise ValueError(f"target_kl must be finite and positive, got {target_kl}")
    value = float(coefficient)
    if current_kl > target_kl * float(config["high_ratio"]):
        value *= float(config["increase_factor"])
    elif current_kl < target_kl * float(config["low_ratio"]):
        value *= float(config["decrease_factor"])
    return float(np.clip(value, float(config["min_coefficient"]), float(config["max_coefficient"])))
