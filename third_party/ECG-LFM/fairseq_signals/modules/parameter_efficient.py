"""Parameter-efficient linear adapters used by the ECG-LFM ablation.

The wrappers in this module deliberately operate on ordinary ``nn.Linear``
modules.  They are injected *after* the official ECG-LFM checkpoint has been
loaded, so the unmodified checkpoint remains a valid input and the baseline
path is bit-for-bit unchanged when no adapter is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer with a low-rank residual update.

    The update is initialized to zero (``lora_B``), preserving the pretrained
    function at step zero.  Bias and the original weight stay frozen; only the
    low-rank matrices are optimized.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + update * self.scaling


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) for ``nn.Linear``.

    DoRA decomposes each output row into a learnable magnitude and a
    direction.  The direction receives the LoRA update while the magnitude is
    initialized from the pretrained row norm.  This implementation follows
    the paper's inference-time equation and keeps the original bias frozen.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("DoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        # One magnitude per output row, initialized to ||W_i||_2.
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def effective_weight(self) -> Tensor:
        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        direction = self.base.weight + delta
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return direction * self.magnitude.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.effective_weight()
        return F.linear(self.dropout(x), weight, self.base.bias)


@dataclass(frozen=True)
class AdapterSummary:
    method: str
    rank: int
    alpha: float
    dropout: float
    target_modules: Tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def inject_linear_adapters(
    model: nn.Module,
    method: str,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    prefix: str = "encoder.encoder.layers.",
) -> AdapterSummary:
    """Replace every context-encoder linear projection with LoRA or DoRA.

    Only linear modules below ``encoder.encoder.layers`` are targeted.  This
    includes the Conformer attention projections (q/k/v/out and relative
    positional projection) and both feed-forward sublayers, while excluding
    the convolutional feature extractor, post-extract projection and the
    four-label classification head.
    """

    method = str(method).lower()
    if method not in {"lora", "dora"}:
        raise ValueError(f"Unsupported adapter method: {method}")
    names: List[str] = []
    for name, module in list(model.named_modules()):
        if not name.startswith(prefix) or not isinstance(module, nn.Linear):
            continue
        names.append(name)
    if not names:
        raise RuntimeError(f"No linear context modules found under {prefix!r}")
    wrapper = LoRALinear if method == "lora" else DoRALinear
    for name in names:
        module = dict(model.named_modules())[name]
        _replace_module(model, name, wrapper(module, rank, alpha, dropout))

    # Freeze the complete base model.  The wrappers' low-rank parameters (and
    # DoRA magnitudes) are re-enabled below; callers can then re-enable a task
    # head explicitly.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, (LoRALinear, DoRALinear)):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if isinstance(module, DoRALinear):
                module.magnitude.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return AdapterSummary(
        method=method,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        target_modules=tuple(names),
        trainable_parameters=trainable,
        total_parameters=total,
    )


def enable_head(model: nn.Module, head_name: str = "proj") -> None:
    """Enable the classification head after adapter injection."""

    head = getattr(model, head_name, None)
    if head is None:
        raise AttributeError(f"Model has no classification head {head_name!r}")
    for parameter in head.parameters():
        parameter.requires_grad = True


def parameter_counts(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters()),
    )
