from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.ecg_lfm.run_single_task_fold import ensure_task_label_file
from fairseq_signals.criterions.acs_omi_single_task import (
    ACSOMISingleTaskCriterion,
    ACSSingleTaskCriterionConfig,
)
from fairseq_signals.models.classification.ecg_transformer_classifier import (
    ECGTransformerClassificationModel,
)


class MockSingleTaskModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, **unused):
        return {"out": self.logits}

    @staticmethod
    def get_logits(output, normalize=False):
        return output["out"]

    @staticmethod
    def get_targets(sample, output):
        return sample["target"]


class SingleTaskControlTests(unittest.TestCase):
    def test_task_label_derivation(self):
        joint = np.asarray([[1, 0, 0, 0, 0], [0, 0, 1, 0, 1]], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint_path = root / "joint.npy"
            np.save(joint_path, joint)
            acs = np.load(ensure_task_label_file(joint_path, root / "acs.npy", "acs"))
            omi = np.load(ensure_task_label_file(joint_path, root / "omi.npy", "omi"))
        np.testing.assert_array_equal(acs, joint[:, :4])
        np.testing.assert_array_equal(omi, joint[:, 4:5])

    def test_acs_only_criterion_is_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.0, -1.0, 0.5], [0.0, 0.1, 1.2, -0.3]])
        labels = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
        criterion = ACSOMISingleTaskCriterion(ACSSingleTaskCriterionConfig(task_mode="acs"), None)
        loss, _, _ = criterion(MockSingleTaskModel(logits), {"net_input": {}, "target": labels, "id": torch.arange(2)})
        torch.testing.assert_close(loss, F.cross_entropy(logits, torch.tensor([0, 2]), reduction="sum"))

    def test_omi_only_criterion_is_stable_bce(self):
        logits = torch.tensor([[2.0], [-1.0]])
        labels = torch.tensor([[1.0], [0.0]])
        criterion = ACSOMISingleTaskCriterion(ACSSingleTaskCriterionConfig(task_mode="omi"), None)
        loss, _, _ = criterion(MockSingleTaskModel(logits), {"net_input": {}, "target": labels, "id": torch.arange(2)})
        torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0], reduction="sum"))

    def test_output_semantics_normalization(self):
        model = SimpleNamespace()
        model.output_semantics = "acs"
        acs = ECGTransformerClassificationModel.get_logits(model, {"out": torch.randn(3, 4)}, normalize=True)
        torch.testing.assert_close(acs.sum(dim=1), torch.ones(3))
        model.output_semantics = "omi"
        omi = ECGTransformerClassificationModel.get_logits(model, {"out": torch.tensor([[0.0]])}, normalize=True)
        torch.testing.assert_close(omi, torch.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
