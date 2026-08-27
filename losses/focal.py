"""Numerically stable focal losses for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Down-weights easy examples (most negatives under class imbalance) via the
``(1 - p_t)^gamma`` modulation.  With ``alpha=None`` this is the pure focal
modulation on top of plain BCEWithLogits, so it is directly comparable to
``bce`` (both unweighted) -- the only difference is the focal factor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        """Args:
            gamma: focusing parameter (paper default 2.0).
            alpha: positive-class weight in [0,1], or None for no balancing.
        """
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, x, y):
        x = x.float()
        y = y.float()

        # BCE-with-logits is the stable -log(p_t); exp(-BCE) recovers p_t
        # without separately evaluating log(sigmoid(x)).
        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        pt = torch.exp(-bce)

        if self.alpha is not None:
            alpha_t = self.alpha * y + (1.0 - self.alpha) * (1.0 - y)
        else:
            alpha_t = 1.0

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """Effective-number class-balanced focal loss for binary logits.

    The two weights are normalized so their empirical expectation on the
    training set equals one.  No ``pos_weight`` or sampling weight is stacked
    on top of these class-balanced weights.
    """

    def __init__(self, n_negative, n_positive, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.tensor([float(n_negative), float(n_positive)])
        if torch.any(counts <= 0):
            raise ValueError(f"class counts must be positive, got {counts.tolist()}")
        beta = float(beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        raw = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
        expected = (raw * counts).sum() / counts.sum()
        weights = raw / expected
        self.register_buffer("class_weights", weights.float())
        self.beta = beta
        self.gamma = float(gamma)

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        pt = torch.exp(-bce)
        weight_y = self.class_weights[1] * y + self.class_weights[0] * (1.0 - y)
        return (weight_y * (1.0 - pt).pow(self.gamma) * bce).mean()
