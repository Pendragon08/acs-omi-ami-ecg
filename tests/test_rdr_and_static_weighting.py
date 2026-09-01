from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from experiments.ecg_lfm.run_joint_fold import balance  # noqa: E402
from fairseq_signals.criterions.acs_joint import (  # noqa: E402
    ACSJointCriterion,
    ACSJointCriterionConfig,
)
from fairseq_signals.modules.rdr_official import OfficialRDRWeightEstimator  # noqa: E402


class StaticWeightingTests(unittest.TestCase):
    def test_static_weights_are_inner_train_derived_and_mean_one(self):
        counts = [4, 2, 1, 1]
        labels = []
        for class_index, count in enumerate(counts):
            for item in range(count):
                target = [0, 0, 0, 0, int((class_index + item) % 2 == 0)]
                target[class_index] = 1
                labels.append(target)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_path = root / "labels.npy"
            np.save(label_path, np.asarray(labels, dtype=np.int8))
            segment_path = root / "train_segments.csv"
            with segment_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["idx"])
                writer.writeheader()
                for index in range(len(labels)):
                    writer.writerow({"idx": index})
                    writer.writerow({"idx": index})
            result = balance(segment_path, label_path)
        weights = np.asarray(result["acs_static_weight_if_requested"])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=12)
        expected = len(labels) / (4.0 * np.asarray(counts, dtype=float))
        expected /= expected.mean()
        np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-12)
        omi_positive = int(np.asarray(labels)[:, 4].sum())
        self.assertAlmostEqual(
            result["omi_pos_weight_if_requested"],
            (len(labels) - omi_positive) / omi_positive,
        )

    def test_static_weights_do_not_change_validation_checkpoint_objective(self):
        class MockModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits

            def forward(self, **unused):
                return {"logits": self.logits}

            @staticmethod
            def get_logits(net_output):
                return net_output["logits"]

            @staticmethod
            def get_targets(sample, net_output):
                return sample["target"]

        logits = torch.tensor([[1.0, -1.0, 0.5, 0.0, -0.5], [0.1, 0.2, -0.2, 0.3, 1.0]])
        target = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32)
        model = MockModel(logits)
        model.eval()
        criterion = ACSJointCriterion(
            ACSJointCriterionConfig(acs_weight=[0.1, 3.0, 4.0, 2.0], omi_pos_weight=10.0),
            task=None,
        )
        loss, _, _ = criterion(model, {"net_input": {}, "target": target, "id": torch.arange(2)})
        expected = F.cross_entropy(logits[:, :4], torch.tensor([0, 3]), reduction="sum")
        expected += F.binary_cross_entropy_with_logits(logits[:, 4], torch.tensor([0.0, 1.0]), reduction="sum")
        torch.testing.assert_close(loss, expected)


class OfficialRDRPortTests(unittest.TestCase):
    def test_weights_are_finite_positive_and_mean_one(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=torch.float32
        )
        targets = torch.tensor([0, 0, 1, 1])
        estimator = OfficialRDRWeightEstimator(2, [8, 2], momentum=0.9, temperature=1.0)
        estimator.update_prototypes(features, targets)
        weights, diagnostics = estimator.weights(features, targets)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertTrue(bool((weights > 0).all()))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreaterEqual(diagnostics.fallback_count, 0)

    def test_absent_class_does_not_update_prototype(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([0, 0])
        estimator = OfficialRDRWeightEstimator(2, [10, 2])
        estimator.update_prototypes(features, targets)
        self.assertTrue(bool(estimator.updated[0]))
        self.assertFalse(bool(estimator.updated[1]))
        torch.testing.assert_close(estimator.prototypes[:, 1], torch.zeros(2))

    def test_port_matches_official_equation(self):
        features = torch.tensor([[1.0, 0.2], [0.3, 1.0], [0.8, 0.4], [0.2, 0.9]])
        targets = torch.tensor([0, 0, 1, 1])
        estimator = OfficialRDRWeightEstimator(2, [6, 2], momentum=0.9, temperature=1.0)
        estimator.update_prototypes(features, targets)
        actual, _ = estimator.weights(features, targets)
        expected = torch.zeros(4)
        for class_index, class_count in enumerate((6, 2)):
            mask = targets == class_index
            class_features = features[mask]
            phi = class_features.T
            gram = phi.T @ phi
            try:
                inverse = torch.inverse(gram)
            except RuntimeError:
                inverse = torch.pinverse(gram)
            ratio = inverse @ phi.T @ estimator.prototypes[:, class_index : class_index + 1]
            ratio = torch.clamp(ratio * class_features.shape[0], min=1e-9)
            expected[mask] = ratio.squeeze() * (8.0 / class_count)
        expected = expected / expected.sum() * expected.numel()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
