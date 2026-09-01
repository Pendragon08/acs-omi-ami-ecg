"""Sampled pairwise ranking preservation for ACS one-vs-rest and OMI."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def _sample_pairs(
    positive: Tensor,
    negative: Tensor,
    max_negatives_per_positive: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    if positive.numel() == 0 or negative.numel() == 0:
        return positive.new_empty((0,), dtype=torch.long), negative.new_empty((0,), dtype=torch.long)
    if max_negatives_per_positive < 1:
        raise ValueError("max_negatives_per_positive must be positive")
    pos_out, neg_out = [], []
    for pos_idx in positive.tolist():
        order = torch.randperm(negative.numel(), generator=generator)
        selected = negative[order[:max_negatives_per_positive]]
        pos_out.append(torch.full((selected.numel(),), int(pos_idx), dtype=torch.long))
        neg_out.append(selected.cpu())
    return torch.cat(pos_out).to(positive.device), torch.cat(neg_out).to(negative.device)


def _pair_loss(scores: Tensor, positive: Tensor, negative: Tensor, max_negatives: int, generator) -> tuple[Tensor, float, int]:
    pos_idx, neg_idx = _sample_pairs(positive, negative, max_negatives, generator)
    if pos_idx.numel() == 0:
        return scores.sum() * 0.0, 0.0, 0
    margins = scores[pos_idx] - scores[neg_idx]
    loss = F.softplus(-margins).mean()
    accuracy = float((margins > 0).float().mean().detach())
    return loss, accuracy, int(margins.numel())


def pairwise_ranking_losses(
    logits: Tensor,
    acs_target: Tensor,
    omi_target: Tensor,
    config: Mapping,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | float | int]:
    """Compute bounded sampled pair losses without constructing O(B²) pairs."""
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError(f"expected logits [B,5], got {tuple(logits.shape)}")
    max_negatives = int(config.get("max_negatives_per_positive", 2))
    losses: dict[str, Tensor] = {}
    accuracies: list[float] = []
    pair_count = 0
    acs_losses = []
    for index, name in enumerate(("OTHER", "STEMI", "NSTEMI", "UA")):
        positive = torch.nonzero(acs_target == index, as_tuple=False).flatten().cpu()
        negative = torch.nonzero(acs_target != index, as_tuple=False).flatten().cpu()
        loss, accuracy, count = _pair_loss(logits[:, index], positive, negative, max_negatives, generator)
        losses[f"{name.lower()}_rank_loss"] = loss
        acs_losses.append(loss)
        accuracies.append(accuracy); pair_count += count
    positive = torch.nonzero(omi_target == 1, as_tuple=False).flatten().cpu()
    negative = torch.nonzero(omi_target == 0, as_tuple=False).flatten().cpu()
    omi_loss, omi_accuracy, omi_pairs = _pair_loss(logits[:, 4], positive, negative, max_negatives, generator)
    losses["omi_rank_loss"] = omi_loss
    accuracies.append(omi_accuracy); pair_count += omi_pairs
    acs_total = torch.stack(acs_losses).mean()
    total = acs_total + omi_loss
    return {
        **losses,
        "acs_rank_loss": acs_total,
        "omi_rank_loss": omi_loss,
        "rank_loss": total,
        "pairwise_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "pair_count": int(pair_count),
    }
