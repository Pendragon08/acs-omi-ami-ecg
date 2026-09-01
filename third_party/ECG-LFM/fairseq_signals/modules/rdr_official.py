"""Auditable port of the official NeurIPS 2024 RDR sample weighting.

Upstream
--------
Repository: https://github.com/GoodMorningPeter/RDR
Commit: bb55ea9f906f00a2d8938a97543fc232b5d26645
License: MIT
Reference implementation: ``main.py::rdr_weights`` and ``update_Phi_nu``

The density-ratio equation, inverse-frequency multiplier, per-batch mean-one
normalization, prototype momentum, absent-class behavior, and lower clamp are
ported directly.  The only numerical adaptations are computation in float32,
an explicit non-finite check, and pseudoinverse fallback.  No upper clipping is
applied: weight explosion is a required experiment gate, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RDRDiagnostics:
    fallback_count: int
    condition_numbers: list[float]


class OfficialRDRWeightEstimator:
    """Stateful official RDR estimator for one categorical task."""

    def __init__(
        self,
        feature_dim: int,
        class_counts: list[int],
        momentum: float = 0.9,
        temperature: float = 1.0,
        profile_condition: bool = False,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not class_counts or any(int(value) <= 0 for value in class_counts):
            raise ValueError("RDR class counts must be positive inner-train counts")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("RDR momentum must be in [0,1)")
        if temperature <= 0.0:
            raise ValueError("RDR temperature must be positive")
        self.feature_dim = int(feature_dim)
        self.class_counts = torch.tensor(class_counts, dtype=torch.float32)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.profile_condition = bool(profile_condition)
        self.prototypes: torch.Tensor | None = None
        self.updated: torch.Tensor | None = None

    @property
    def num_classes(self) -> int:
        return int(self.class_counts.numel())

    def _ensure_state(self, device: torch.device) -> None:
        if self.prototypes is None:
            self.prototypes = torch.zeros(
                self.feature_dim, self.num_classes, device=device, dtype=torch.float32
            )
            self.updated = torch.zeros(self.num_classes, device=device, dtype=torch.bool)
        elif self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)
            self.updated = self.updated.to(device)

    @torch.no_grad()
    def weights(self, features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, RDRDiagnostics]:
        features = features.detach().float()
        targets = targets.detach().long()
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected RDR features [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("RDR target shape does not match features")
        if bool(((targets < 0) | (targets >= self.num_classes)).any()):
            raise ValueError("RDR targets outside configured classes")
        self._ensure_state(features.device)
        sample_weights = torch.zeros(features.shape[0], device=features.device, dtype=torch.float32)
        fallback_count = 0
        condition_numbers: list[float] = []
        for class_index in range(self.num_classes):
            mask = targets == class_index
            class_features = features[mask]
            if class_features.shape[0] == 0:
                # Official behavior: an absent minibatch class has no samples
                # to weight and its prototype is not updated.
                continue
            phi_de = class_features.T
            gram = phi_de.T @ phi_de
            if self.profile_condition:
                condition_numbers.append(float(torch.linalg.cond(gram).cpu()))
            try:
                inverse = torch.linalg.inv(gram)
                if not bool(torch.isfinite(inverse).all()):
                    raise RuntimeError("non-finite inverse")
            except RuntimeError:
                inverse = torch.linalg.pinv(gram)
                fallback_count += 1
            ratio = (inverse @ phi_de.T @ self.prototypes[:, class_index : class_index + 1])
            ratio = ratio * class_features.shape[0]
            ratio = torch.clamp(ratio, min=1e-9)
            sample_weights[mask] = ratio.squeeze(1)

        counts = self.class_counts.to(features.device)
        total = counts.sum()
        for class_index in range(self.num_classes):
            mask = targets == class_index
            sample_weights[mask] *= total / counts[class_index]
        sample_weights = sample_weights.pow(self.temperature)
        denominator = sample_weights.sum()
        if not bool(torch.isfinite(sample_weights).all()) or not bool(torch.isfinite(denominator)) or denominator <= 0:
            raise FloatingPointError("RDR produced non-finite or zero-sum weights")
        sample_weights = sample_weights / denominator * sample_weights.numel()
        return sample_weights, RDRDiagnostics(fallback_count, condition_numbers)

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        features = features.detach().float()
        targets = targets.detach().long()
        self._ensure_state(features.device)
        for class_index in range(self.num_classes):
            mask = targets == class_index
            if not bool(mask.any()):
                continue
            mean = features[mask].mean(dim=0)
            if bool(self.updated[class_index]):
                self.prototypes[:, class_index].mul_(self.momentum).add_(
                    mean, alpha=1.0 - self.momentum
                )
            else:
                self.prototypes[:, class_index].copy_(mean)
                self.updated[class_index] = True


def class_weight_stats(weights: torch.Tensor, targets: torch.Tensor, names: tuple[str, ...]) -> dict[str, float]:
    """Return mergeable per-class statistics for Fairseq epoch logging."""
    result: dict[str, float] = {}
    values = weights.detach().float()
    for index, name in enumerate(names):
        selected = values[targets == index]
        prefix = f"rdr_weight_{name}"
        result[f"{prefix}_count"] = int(selected.numel())
        result[f"{prefix}_sum"] = float(selected.sum().cpu()) if selected.numel() else 0.0
        result[f"{prefix}_sumsq"] = float((selected.square()).sum().cpu()) if selected.numel() else 0.0
        result[f"{prefix}_min"] = float(selected.min().cpu()) if selected.numel() else float("inf")
        result[f"{prefix}_max"] = float(selected.max().cpu()) if selected.numel() else float("-inf")
    return result
