from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.acs_joint_task import acs_metrics, binary_calibration_metrics
from experiments.ecg_lfm.evaluate_joint import load_record_ground_truth
from fairseq_signals.models.classification.ecg_transformer_classifier import (
    ECGTransformerClassificationModel,
)


def _normalized(logits: torch.Tensor) -> torch.Tensor:
    return ECGTransformerClassificationModel.get_logits(
        None, {"out": logits}, normalize=True
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CorrectedProbabilityTests(unittest.TestCase):
    def test_acs_softmax_sums_to_one(self):
        probabilities = _normalized(torch.tensor([[1.0, -1.0, 0.5, 2.0, -3.0]]))
        self.assertTrue(
            torch.allclose(probabilities[:, :4].sum(dim=1), torch.ones(1), atol=1e-7)
        )

    def test_omi_is_independent_sigmoid(self):
        first = _normalized(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.7]]))
        second = _normalized(torch.tensor([[100.0, -100.0, 50.0, -50.0, 0.7]]))
        self.assertTrue(
            torch.allclose(first[:, 4], torch.sigmoid(torch.tensor([0.7])), atol=1e-7)
        )
        self.assertTrue(torch.allclose(first[:, 4], second[:, 4], atol=1e-7))

    def test_no_5way_softmax(self):
        low_omi = _normalized(torch.tensor([[1.0, 2.0, 3.0, 4.0, -100.0]]))
        high_omi = _normalized(torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]]))
        self.assertTrue(torch.allclose(low_omi[:, :4], high_omi[:, :4], atol=1e-7))
        self.assertFalse(torch.allclose(high_omi.sum(dim=1), torch.ones(1), atol=1e-4))
        self.assertLess(float(low_omi[0, 4]), 1e-6)
        self.assertGreater(float(high_omi[0, 4]), 1 - 1e-6)

    def test_empty_calibration_bins_are_strict_json_serializable(self):
        metrics = binary_calibration_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0.10, 0.11, 0.89, 0.90]), n_bins=15
        )
        json.dumps(metrics, allow_nan=False)

    def test_per_class_balanced_accuracy_uses_argmax_predictions(self):
        y = np.asarray([0, 1, 2, 3])
        probabilities = np.eye(4, dtype=np.float64)
        metrics = acs_metrics(y, probabilities)
        for item in metrics["per_class"].values():
            self.assertEqual(item["balanced_accuracy"], 1.0)


class RecordAlignmentTests(unittest.TestCase):
    def test_record_id_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.csv"
            labels = root / "labels.csv"
            _write_csv(
                master,
                [
                    "idx",
                    "cohort",
                    "record_id",
                    "patient_id",
                    "STEMI",
                    "NSTEMI",
                    "UA",
                    "OMI",
                    "validation_fold",
                ],
                [
                    {"idx": 0, "cohort": "train", "record_id": "00001", "patient_id": "P1", "STEMI": 0, "NSTEMI": 0, "UA": 0, "OMI": 0, "validation_fold": 0},
                    {"idx": 1, "cohort": "train", "record_id": "00002", "patient_id": "P2", "STEMI": 0, "NSTEMI": 1, "UA": 0, "OMI": 1, "validation_fold": 1},
                ],
            )
            # Deliberately reverse label order; an index-only join would fail this test.
            _write_csv(
                labels,
                ["Patient_id", "ecg_row_record", "STEMI", "NSTEMI", "UA", "OMI"],
                [
                    {"Patient_id": "P2", "ecg_row_record": "00002.dat", "STEMI": 0, "NSTEMI": 1, "UA": 0, "OMI": 1},
                    {"Patient_id": "P1", "ecg_row_record": "00001.dat", "STEMI": 0, "NSTEMI": 0, "UA": 0, "OMI": 0},
                ],
            )
            joined, audit = load_record_ground_truth(master, labels)
            self.assertEqual([row["record_id"] for row in joined], ["00001", "00002"])
            self.assertEqual([row["y_acs"] for row in joined], [0, 2])
            self.assertEqual([row["y_omi"] for row in joined], [0, 1])
            self.assertEqual(audit["join_key"], "record_id")
            self.assertEqual(audit["missing_labels"], 0)


if __name__ == "__main__":
    unittest.main()
