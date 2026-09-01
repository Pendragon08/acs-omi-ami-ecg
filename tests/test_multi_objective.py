"""Small deterministic tests for the ECG-LFM MOO gradient interface."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from fairseq_signals.modules.multi_objective import (  # noqa: E402
    METHODS,
    MultiObjectiveStrategy,
    merge_named_parameter_gradients,
)


def test_01_no_conflict_pcgrad():
    strategy = MultiObjectiveStrategy("pcgrad", seed=42)
    merged, diag = strategy.aggregate(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert diag.conflict is False
    assert torch.allclose(merged, torch.tensor([1.0, 1.0]))


def test_02_conflict_pcgrad():
    strategy = MultiObjectiveStrategy("pcgrad", seed=42)
    merged, diag = strategy.aggregate(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))
    assert diag.conflict is True
    assert torch.isfinite(merged).all()
    assert float(torch.dot(merged, torch.tensor([1.0, 0.0]))) >= -1e-5


def test_03_shared_vs_head():
    acs_head = torch.nn.Parameter(torch.zeros(5, 2))
    shared = torch.nn.Parameter(torch.zeros(2, 2))
    named = [("proj.weight", acs_head), ("encoder.adapter", shared)]
    ga = [torch.ones_like(acs_head), torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    go = [torch.full_like(acs_head, 2.0), torch.tensor([[-1.0, 0.0], [0.0, 1.0]])]
    merged, _ = merge_named_parameter_gradients(
        named, ga, go, MultiObjectiveStrategy("linear_sum")
    )
    assert torch.allclose(merged[0][:4], torch.ones(4, 2))
    assert torch.allclose(merged[0][4:], torch.full((1, 2), 2.0))
    assert torch.allclose(merged[1], torch.tensor([[0.0, 0.0], [0.0, 2.0]]))


def test_04_linear_sum_equivalence():
    strategy = MultiObjectiveStrategy("linear_sum")
    first = torch.randn(17)
    second = torch.randn(17)
    merged, _ = strategy.aggregate(first, second)
    assert torch.allclose(merged, first + second, atol=1e-7, rtol=1e-7)


def test_05_no_nan():
    first = torch.tensor([1.0, 0.0, -2.0, 0.5])
    second = torch.tensor([-1.0, 2.0, 0.0, 0.5])
    for method in METHODS:
        merged, diag = MultiObjectiveStrategy(method, seed=2026).aggregate(first, second)
        assert torch.isfinite(merged).all(), method
        assert diag.solver_fallback in (True, False)


def test_06_frozen_encoder():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    assert all(parameter.grad is None for parameter in model[0].parameters())
    assert any(parameter.requires_grad for parameter in model[1].parameters())


def test_07_checkpoint_threshold_protocol():
    config = (ROOT / "configs" / "ecg_lfm" / "corrected_joint_lora_moo.yaml").read_text(encoding="utf-8")
    assert "best_checkpoint_metric: loss" in config
    assert "threshold: 0.5" in config
    assert "acs_joint_moo" in config
