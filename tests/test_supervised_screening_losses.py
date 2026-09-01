from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from fairseq_signals.criterions.acs_joint_supervised import (  # noqa: E402
    ACSJointSupervisedCriterion,
    ACSJointSupervisedCriterionConfig,
    asymmetric_binary_loss,
    focal_binary_cross_entropy,
    focal_cross_entropy,
    hard_negative_binary_loss,
)


class MockModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, **unused):
        return {"logits": self.logits}

    @staticmethod
    def get_logits(output):
        return output["logits"]

    @staticmethod
    def get_targets(sample, unused):
        return sample["target"]


class SupervisedScreeningLossTests(unittest.TestCase):
    def test_zero_gamma_focal_equals_unweighted_losses(self):
        acs_logits = torch.tensor([[2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 2.0, -1.0]])
        acs_target = torch.tensor([0, 2])
        omi_logits = torch.tensor([-1.0, 1.5])
        omi_target = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(
            focal_cross_entropy(acs_logits, acs_target, 0.0),
            F.cross_entropy(acs_logits, acs_target, reduction="sum"),
        )
        torch.testing.assert_close(
            focal_binary_cross_entropy(omi_logits, omi_target, 0.0),
            F.binary_cross_entropy_with_logits(omi_logits, omi_target, reduction="sum"),
        )

    def test_asl_is_finite_for_extreme_logits(self):
        logits = torch.tensor([-100.0, 100.0, 0.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0, 0.0])
        loss = asymmetric_binary_loss(logits, target, 4.0, 0.0, 0.05)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_hnm_keeps_positive_and_only_hardest_negative_gradients(self):
        logits = torch.tensor([0.0, -4.0, 2.0, -2.0, 4.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        loss = hard_negative_binary_loss(logits, target, 0.25)
        loss.backward()
        self.assertNotEqual(float(logits.grad[0]), 0.0)
        self.assertNotEqual(float(logits.grad[4]), 0.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad[1:4])), 0)

    def test_validation_objective_is_w0_for_every_scheme(self):
        logits = torch.tensor([
            [1.0, -1.0, 0.5, 0.0, -0.5],
            [0.1, 0.2, -0.2, 0.3, 1.0],
        ])
        labels = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32)
        expected = F.cross_entropy(logits[:, :4], torch.tensor([0, 3]), reduction="sum")
        expected += F.binary_cross_entropy_with_logits(
            logits[:, 4], torch.tensor([0.0, 1.0]), reduction="sum"
        )
        for scheme in ("focal", "omi_asl", "label_smoothing", "omi_hnm"):
            model = MockModel(logits)
            model.eval()
            criterion = ACSJointSupervisedCriterion(
                ACSJointSupervisedCriterionConfig(scheme=scheme), task=None
            )
            loss, sample_size, _ = criterion(
                model, {"net_input": {}, "target": labels, "id": torch.arange(2)}
            )
            self.assertEqual(sample_size, 2)
            torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
