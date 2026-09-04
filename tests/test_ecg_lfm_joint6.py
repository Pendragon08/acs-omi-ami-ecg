from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.ecg_lfm.joint6 import encode_source_labels, marginalize_joint6_probabilities
from experiments.ecg_lfm.evaluate_joint6_fold import build_rows, joint6_metrics
from fairseq_signals.criterions.acs_joint6 import ACSJoint6Criterion
from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
from fairseq_signals.modules.parameter_efficient import LoRALinear


class TinyJoint6(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter = LoRALinear(nn.Linear(4, 4), rank=2, alpha=4, dropout=0)
        self.proj = nn.Linear(4, 6)

    def forward(self, source):
        return {"out": self.proj(self.adapter(self.backbone(source)))}

    def get_logits(self, net_output, normalize=False, **kwargs):
        return torch.softmax(net_output["out"], -1) if normalize else net_output["out"]

    def get_targets(self, sample, net_output, **kwargs):
        return sample["label"]


class Joint6Tests(unittest.TestCase):
    def test_joint6_model_softmax(self):
        owner = SimpleNamespace(output_semantics="joint6_observed_outcome")
        probabilities = ECGTransformerClassificationModel.get_logits(
            owner, {"out": torch.randn(4, 6)}, normalize=True
        )
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(4), atol=1e-7))

    def test_joint_mapping(self):
        source = np.asarray([
            [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0],
        ])
        _, _, joint, exception = encode_source_labels(source)
        np.testing.assert_array_equal(joint, np.arange(6))
        self.assertFalse(exception.any())

    def test_other_omi_exception(self):
        acs, omi, joint, exception = encode_source_labels(np.asarray([[0, 0, 0, 1]]))
        self.assertEqual((acs[0], omi[0], joint[0]), (0, 1, 0))
        self.assertTrue(exception[0])

    def test_ua_omi_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "UA\\+OMI"):
            encode_source_labels(np.asarray([[0, 0, 1, 1]]))

    def test_joint_prob_sum(self):
        q = torch.softmax(torch.randn(5, 6), dim=1)
        self.assertTrue(torch.allclose(q.sum(1), torch.ones(5), atol=1e-7))

    def test_acs_marginal_sum(self):
        q = torch.softmax(torch.randn(5, 6), dim=1)
        self.assertTrue(torch.allclose(marginalize_joint6_probabilities(q)["acs"].sum(1), torch.ones(5), atol=1e-7))

    def test_omi_marginal(self):
        q = np.asarray([[.1, .1, .2, .1, .15, .35]])
        self.assertAlmostEqual(float(marginalize_joint6_probabilities(q)["omi"][0]), .35)

    def test_nste_probability(self):
        q = np.asarray([[.1, .1, .2, .1, .15, .35]])
        self.assertAlmostEqual(float(marginalize_joint6_probabilities(q)["nste"][0]), .60)

    def test_nstemi_vs_ua_probability(self):
        q = np.asarray([[.1, .1, .2, .1, .15, .35]])
        self.assertAlmostEqual(float(marginalize_joint6_probabilities(q)["nstemi_given_nste"][0]), .25 / .60)

    def test_joint_gradient(self):
        torch.manual_seed(7)
        model = TinyJoint6()
        criterion = ACSJoint6Criterion(task=None)
        sample = {
            "id": torch.arange(3),
            "net_input": {"source": torch.randn(3, 4)},
            "label": torch.eye(6)[torch.tensor([1, 4, 5])],
        }
        criterion(model, sample)[0].backward()
        self.assertGreater(float(model.proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.adapter.lora_B.grad.abs().sum()), 0.0)

    def test_frozen_encoder(self):
        model = TinyJoint6()
        criterion = ACSJoint6Criterion(task=None)
        sample = {
            "id": torch.arange(2),
            "net_input": {"source": torch.randn(2, 4)},
            "label": torch.eye(6)[torch.tensor([1, 2])],
        }
        criterion(model, sample)[0].backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))

    def test_old_w0_untouched(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.7]])
        probabilities = ECGTransformerClassificationModel.get_logits(None, {"out": logits}, normalize=True)
        self.assertTrue(torch.allclose(probabilities[:, :4].sum(1), torch.ones(1)))
        self.assertTrue(torch.allclose(probabilities[:, 4], torch.sigmoid(torch.tensor([.7]))))

    def test_checkpoint_save_load(self):
        torch.manual_seed(11)
        model = TinyJoint6()
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint6.pt"
            torch.save(model.state_dict(), path)
            restored = TinyJoint6()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)

    def test_joint6_complete_per_class_metrics(self):
        y = np.arange(6, dtype=np.int64)
        q = np.eye(6, dtype=np.float64) * .9 + np.ones((6, 6), dtype=np.float64) * .02
        q /= q.sum(axis=1, keepdims=True)
        result = joint6_metrics(y, q)
        self.assertEqual(set(result["per_class"]), {
            "OTHER", "STEMI+nOMI", "STEMI+OMI", "NSTEMI+nOMI", "NSTEMI+OMI", "UA",
        })
        for values in result["per_class"].values():
            self.assertIn("precision", values)
            self.assertIn("recall", values)
            self.assertEqual(values["support"], 1)

    def test_absent_joint6_class_is_na_not_zero(self):
        y = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        q = np.full((5, 6), .01, dtype=np.float64)
        q[np.arange(5), y] = .95
        q /= q.sum(axis=1, keepdims=True)
        result = joint6_metrics(y, q)
        ua = result["per_class"]["UA"]
        self.assertEqual(ua["support"], 0)
        for name in ("auroc", "auprc", "f1", "precision", "ppv", "sensitivity", "recall", "specificity"):
            self.assertIsNone(ua[name], name)
        self.assertIsNone(result["macro_auprc_ovr"])
        self.assertIsNotNone(result["supported_macro_auprc_ovr"])

    def test_prediction_export_has_required_aliases(self):
        records = [{"master_idx": 0, "record_id": "00001", "patient_id": "p1"}]
        assignments = [{
            "idx": "0", "record_id": "00001", "patient_id": "p1", "fold": "0",
            "subset": "outer_test", "acs": "OTHER", "omi": "0",
        }]
        rows = build_rows(records, np.zeros((1, 6)), assignments, np.asarray([[0, 0, 0, 0]]))
        self.assertEqual(rows[0]["joint_true"], rows[0]["joint6_true"])
        self.assertEqual(rows[0]["joint_pred"], rows[0]["joint6_pred"])
        for field in (
            "patient_id", "record_id", "fold", "p_other", "p_stemi_nomi", "p_stemi_omi",
            "p_nstemi_nomi", "p_nstemi_omi", "p_ua", "p_stemi", "p_nstemi", "p_omi",
            "raw_acs_true", "raw_omi_true",
        ):
            self.assertIn(field, rows[0])

    def test_prediction_join_rejects_master_index_mismatch(self):
        records = [{"master_idx": 1, "record_id": "00001", "patient_id": "p1"}]
        assignments = [{
            "idx": "0", "record_id": "00001", "patient_id": "p1", "fold": "0",
            "subset": "outer_test", "acs": "OTHER", "omi": "0",
        }]
        with self.assertRaisesRegex(ValueError, "master index mismatch"):
            build_rows(records, np.zeros((1, 6)), assignments, np.asarray([[0, 0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
