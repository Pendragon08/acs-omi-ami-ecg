"""Multi-objective gradient aggregation for the corrected ECG-LFM task.

This module is intentionally small and contains only the gradient aggregation
step.  The task losses remain the original unweighted CE (ACS) and BCE
(OMI); no loss, sampler, model, optimizer, or scheduler is changed.  The
implementations follow the public reference repositories recorded in
``docs/logs/ECG_LFM_multi_objective_sources.md``.

The functions operate on detached, flat ``float32`` gradients.  Keeping the
solver separate from autograd makes the behaviour unit-testable and avoids
accidentally backpropagating through task weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


METHODS = ("linear_sum", "pcgrad", "cagrad", "fairgrad", "aligned_mtl")


def _safe_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-12:
        return torch.zeros((), device=first.device, dtype=torch.float32)
    return torch.dot(first, second) / denominator


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex (official MTL helper)."""
    values = np.asarray(values, dtype=np.float64)
    sorted_values = np.flip(np.sort(values), axis=0)
    cumulative = 0.0
    theta = (float(values.sum()) - 1.0) / len(values)
    for index in range(len(values) - 1):
        cumulative += float(sorted_values[index])
        candidate = (cumulative - 1.0) / (index + 1.0)
        if candidate > sorted_values[index + 1]:
            theta = candidate
            break
    return np.maximum(values - theta, 0.0)


def _scipy_minimize(objective, x0: np.ndarray, bounds, constraints):
    """Import scipy lazily so the unit-test module works without scipy."""
    from scipy.optimize import minimize

    return minimize(objective, x0, bounds=bounds, constraints=constraints)


def _scipy_least_squares(objective, x0: np.ndarray, bounds):
    from scipy.optimize import least_squares

    return least_squares(objective, x0, bounds=bounds)


@dataclass
class AggregationDiagnostics:
    method: str
    grad_cosine_acs_omi: float
    conflict: bool
    grad_norm_acs_shared: float
    grad_norm_omi_shared: float
    grad_norm_joint: float
    post_grad_cosine: float
    effective_task_weight_acs: float
    effective_task_weight_omi: float
    solver_fallback: bool = False

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "moo_method": self.method,
            "grad_cosine_acs_omi": self.grad_cosine_acs_omi,
            "gradient_conflict": int(self.conflict),
            "grad_norm_acs_shared": self.grad_norm_acs_shared,
            "grad_norm_omi_shared": self.grad_norm_omi_shared,
            "grad_norm_joint": self.grad_norm_joint,
            "post_grad_cosine": self.post_grad_cosine,
            "effective_task_weight_acs": self.effective_task_weight_acs,
            "effective_task_weight_omi": self.effective_task_weight_omi,
            "moo_solver_fallback": int(self.solver_fallback),
        }


class MultiObjectiveStrategy:
    """Aggregate exactly two task gradients on the shared parameter space."""

    def __init__(
        self,
        method: str = "linear_sum",
        *,
        c: float = 0.5,
        alpha: float = 1.0,
        task_prior: Sequence[float] = (0.5, 0.5),
        seed: int = 2026,
    ) -> None:
        method = str(method).lower()
        if method not in METHODS:
            raise ValueError(f"unsupported multi-objective method: {method}")
        prior = np.asarray(task_prior, dtype=np.float64)
        if prior.shape != (2,) or not np.isfinite(prior).all() or prior.sum() <= 0:
            raise ValueError("task_prior must contain two finite positive values")
        prior = prior / prior.sum()
        if c < 0 or alpha <= 0:
            raise ValueError("c must be non-negative and alpha must be positive")
        self.method = method
        self.c = float(c)
        self.alpha = float(alpha)
        self.task_prior = prior
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

    @staticmethod
    def _cagrad(grads: torch.Tensor, c: float, prior: np.ndarray) -> tuple[torch.Tensor, np.ndarray, bool]:
        """CAGrad solver copied from the public reference implementation."""
        gram = torch.matmul(grads, grads.T).detach().cpu().numpy().astype(np.float64)
        x0 = prior.astype(np.float64, copy=True)
        g0_norm = float(np.sqrt(max(float(gram.mean()), 0.0) + 1e-8))
        regularizer = (c * g0_norm) + 1e-8

        def objective(x: np.ndarray) -> float:
            left = x.reshape(1, 2).dot(gram).dot(x0.reshape(2, 1))
            right = np.sqrt(x.reshape(1, 2).dot(gram).dot(x.reshape(2, 1)) + 1e-8)
            return float((left + regularizer * right).sum())

        try:
            result = _scipy_minimize(
                objective,
                x0,
                bounds=tuple((0.0, 1.0) for _ in x0),
                constraints={"type": "eq", "fun": lambda x: 1.0 - float(x.sum())},
            )
            weights = np.asarray(result.x, dtype=np.float64)
            if not result.success or weights.shape != (2,) or not np.isfinite(weights).all():
                raise ValueError("CAGrad scipy solver did not converge")
            weights = _simplex_projection(weights)
        except Exception:
            # This is only an auditable dependency fallback.  It is not used
            # silently: ``solver_fallback`` is persisted in every train log.
            weights = x0
            fallback = True
        else:
            fallback = False

        weight_tensor = torch.as_tensor(weights, device=grads.device, dtype=grads.dtype)
        weighted = (grads * weight_tensor[:, None]).sum(dim=0)
        lam = regularizer / float(weighted.norm().detach().cpu() + 1e-8)
        merged = grads.mean(dim=0) + lam * weighted
        return merged, weights, fallback

    def _fairgrad(self, grads: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, bool]:
        """FairGrad proportional-fairness solver from the public repository."""
        gram = torch.matmul(grads, grads.T).detach().cpu().numpy().astype(np.float64)
        x0 = np.ones(2, dtype=np.float64) / 2.0

        def equation(x: np.ndarray) -> np.ndarray:
            # The published implementation uses this exact equation.
            return np.dot(gram, x) - np.power(1.0 / np.maximum(x, 1e-12), 1.0 / self.alpha)

        try:
            result = _scipy_least_squares(equation, x0, bounds=(1e-8, np.inf))
            weights = np.asarray(result.x, dtype=np.float64)
            if not result.success or weights.shape != (2,) or not np.isfinite(weights).all():
                raise ValueError("FairGrad scipy solver did not converge")
        except Exception:
            weights = np.ones(2, dtype=np.float64)
            fallback = True
        else:
            fallback = False
        weight_tensor = torch.as_tensor(weights, device=grads.device, dtype=grads.dtype)
        # Official FairGrad multiplies by n_tasks to match a summed loss.
        merged = (grads * weight_tensor[:, None]).sum(dim=0) * 2.0
        return merged, weights, fallback

    @staticmethod
    def _aligned_mtl(grads: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, bool]:
        """Aligned-MTL Procrustes solver (``scale_mode='min'``)."""
        gram = torch.matmul(grads, grads.T)
        eigenvalues, basis = torch.linalg.eigh(gram)
        maximum = float(eigenvalues.max().detach().cpu()) if eigenvalues.numel() else 0.0
        tolerance = maximum * max(gram.shape) * torch.finfo(gram.dtype).eps
        keep = eigenvalues > tolerance
        if not bool(keep.any()):
            return grads.sum(dim=0), np.ones(2, dtype=np.float64), True
        singulars = eigenvalues[keep]
        basis = basis[:, keep]
        order = torch.argsort(singulars, descending=True)
        singulars = singulars[order]
        basis = basis[:, order]
        scale = torch.sqrt(singulars[-1]).reshape(1, -1)
        weights = basis * scale
        weights = weights / torch.sqrt(singulars).reshape(1, -1)
        weights = weights @ basis.T
        # Official code stores task gradients as columns, transforms them,
        # then sums columns.  The row form below is algebraically identical.
        transformed = (grads.T @ weights.T).T
        merged = transformed.sum(dim=0)
        # Effective coefficients of the original task gradients in the
        # summed transformed gradient (W @ 1), used only for diagnostics.
        effective = weights @ torch.ones(2, device=weights.device, dtype=weights.dtype)
        return merged, effective.detach().cpu().numpy(), False

    def aggregate(self, grad_acs: torch.Tensor, grad_omi: torch.Tensor) -> tuple[torch.Tensor, AggregationDiagnostics]:
        if grad_acs.ndim != 1 or grad_omi.ndim != 1 or grad_acs.shape != grad_omi.shape:
            raise ValueError("task gradients must be flat vectors with identical shapes")
        g_acs = grad_acs.detach().float()
        g_omi = grad_omi.detach().float()
        cosine = _safe_cosine(g_acs, g_omi)
        conflict = bool(float(cosine) < 0.0)
        norms = (float(g_acs.norm()), float(g_omi.norm()))
        fallback = False
        effective = np.ones(2, dtype=np.float64)
        if self.method == "linear_sum":
            merged = g_acs + g_omi
        elif self.method == "pcgrad":
            tasks = torch.stack((g_acs, g_omi), dim=0)
            order = torch.randperm(2, generator=self._generator).tolist()
            projected = tasks.clone()
            for i in order:
                for j in order:
                    if i == j:
                        continue
                    dot = torch.dot(projected[i], tasks[j])
                    denom = torch.dot(tasks[j], tasks[j]) + 1e-5
                    if float(dot) < 0.0:
                        projected[i] = projected[i] - dot / denom * tasks[j]
            merged = projected.sum(dim=0)
        elif self.method == "cagrad":
            merged, effective, fallback = self._cagrad(torch.stack((g_acs, g_omi)), self.c, self.task_prior)
        elif self.method == "fairgrad":
            merged, effective, fallback = self._fairgrad(torch.stack((g_acs, g_omi)))
        else:
            merged, effective, fallback = self._aligned_mtl(torch.stack((g_acs, g_omi)))
        post_cosine = _safe_cosine(merged, g_acs + g_omi)
        diagnostics = AggregationDiagnostics(
            method=self.method,
            grad_cosine_acs_omi=float(cosine),
            conflict=conflict,
            grad_norm_acs_shared=norms[0],
            grad_norm_omi_shared=norms[1],
            grad_norm_joint=float(merged.norm()),
            post_grad_cosine=float(post_cosine),
            effective_task_weight_acs=float(effective[0]),
            effective_task_weight_omi=float(effective[1]),
            solver_fallback=fallback,
        )
        return merged, diagnostics


def flat_gradients(grads: Iterable[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Materialise an autograd result as a finite float32 flat vector."""
    chunks = []
    for grad, param in zip(grads, params):
        chunks.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1) if grad is None else grad.detach().float().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float32)


def merge_named_parameter_gradients(
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    grads_acs: Sequence[torch.Tensor | None],
    grads_omi: Sequence[torch.Tensor | None],
    strategy: MultiObjectiveStrategy,
) -> tuple[list[torch.Tensor], AggregationDiagnostics]:
    """Merge shared gradients while keeping ACS/OMI head rows task-specific.

    ``proj.weight`` and ``proj.bias`` are one PyTorch Parameter each, but rows
    0--3 belong exclusively to ACS and row 4 belongs exclusively to OMI.  The
    backbone/LoRA parameters are the shared space on which MOO operates.
    """
    shared_indices = [
        i for i, (name, parameter) in enumerate(named_params)
        if parameter.requires_grad and not name.endswith("proj.weight") and not name.endswith("proj.bias")
    ]
    if not shared_indices:
        diagnostics = AggregationDiagnostics(
            strategy.method, 0.0, False, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
        )
        return [torch.zeros_like(parameter) for _, parameter in named_params], diagnostics
    shared_params = [named_params[index][1] for index in shared_indices]
    g_acs = flat_gradients([grads_acs[index] for index in shared_indices], shared_params)
    g_omi = flat_gradients([grads_omi[index] for index in shared_indices], shared_params)
    merged, diagnostics = strategy.aggregate(g_acs, g_omi)
    result: list[torch.Tensor] = []
    offset = 0
    shared_lookup = set(shared_indices)
    for index, (name, parameter) in enumerate(named_params):
        ga = grads_acs[index]
        go = grads_omi[index]
        if index in shared_lookup:
            size = parameter.numel()
            result.append(merged[offset : offset + size].view_as(parameter).to(parameter.dtype))
            offset += size
        elif name.endswith("proj.weight") or name.endswith("proj.bias"):
            value = torch.zeros_like(parameter)
            if ga is not None:
                value[:4].copy_(ga[:4].detach().to(value.dtype))
            if go is not None:
                value[4:].copy_(go[4:].detach().to(value.dtype))
            result.append(value)
        else:
            # Frozen parameters are represented for convenience; the caller
            # skips them when writing grads.
            result.append(torch.zeros_like(parameter))
    return result, diagnostics
