"""Inner-validation-only safety guards for exploratory RL checkpoint selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SafetyGuard:
    lower_bounds: dict[str, float]
    upper_bounds: dict[str, float]
    max_kl: float

    @classmethod
    def from_reference(
        cls,
        reference: Mapping[str, float],
        margins: Mapping[str, float],
        max_kl: float,
    ) -> "SafetyGuard":
        lower_names = (
            "acs_macro_auprc", "omi_auprc", "omi_ppv", "omi_specificity", "stemi_f1"
        )
        upper_names = ("acs_brier", "omi_brier")
        return cls(
            lower_bounds={name: float(reference[name] - margins[name]) for name in lower_names},
            upper_bounds={name: float(reference[name] + margins[name]) for name in upper_names},
            max_kl=float(max_kl),
        )

    def evaluate(self, metrics: Mapping[str, float], kl_total: float) -> dict:
        checks = {
            f"{name}_guard": float(metrics[name]) >= bound
            for name, bound in self.lower_bounds.items()
        }
        checks.update(
            {
                f"{name}_guard": float(metrics[name]) <= bound
                for name, bound in self.upper_bounds.items()
            }
        )
        checks["kl_guard"] = float(kl_total) <= self.max_kl
        return {
            "pass": bool(all(checks.values())),
            "checks": checks,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "max_kl": self.max_kl,
        }
