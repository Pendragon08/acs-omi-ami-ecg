from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.ecg_lfm.evaluate_joint import read_history
from experiments.ecg_lfm.joint6_optimization import choose_learning_rate, run_candidate
from models.ecg_lfm_adapter import parameter_summary
from experiments.ecg_lfm.summarize_joint6_cv import compatible_reporting_evaluator
from experiments.ecg_lfm import summarize_joint6_cv
from experiments.ecg_lfm.profiled_hydra_train import capture_allocator_peaks, reset_with_capture


def score(ap=.4, omi=.43, nste=.27, sens=.83):
    return {"joint6_macro_auprc": ap, "omi_auprc": omi, "nste_omi_auprc": nste, "omi_sensitivity": sens}


class OptimizationTests(unittest.TestCase):
    def test_allocator_peak_is_captured_before_trainer_reset(self):
        state = {"allocated": 30000.0, "reserved": 31000.0}
        values = {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
        def reset():
            state["allocated"] = 2800.0
        with patch("torch.cuda.max_memory_allocated", side_effect=lambda: state["allocated"] * 2**20), \
             patch("torch.cuda.max_memory_reserved", side_effect=lambda: state["reserved"] * 2**20):
            reset_with_capture(reset, lambda: capture_allocator_peaks(values))()
            capture_allocator_peaks(values)
        self.assertEqual(values["peak_allocated_mib"], 30000.0)
        self.assertEqual(state["allocated"], 2800.0)

    def test_unrecognized_legacy_evaluator_is_rejected(self):
        self.assertFalse(compatible_reporting_evaluator("unknown", "different"))

    def test_reviewed_reporting_version_is_accepted(self):
        entry = json.loads((summarize_joint6_cv.ROOT / "configs/ecg_lfm/joint6_reporting_compatibility.json").read_text())
        self.assertTrue(compatible_reporting_evaluator(entry["historical_evaluator_sha256"], entry["current_evaluator_sha256"]))

    def test_reporting_registry_cannot_ignore_changed_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "configs/ecg_lfm/joint6_reporting_compatibility.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"historical_evaluator_sha256": "old", "current_evaluator_sha256": "new",
                                          "required_current_sources": {"missing.py": "expected"}}))
            with patch.object(summarize_joint6_cv, "ROOT", root):
                self.assertFalse(compatible_reporting_evaluator("old", "new"))

    def test_selection_waits_for_every_predeclared_candidate(self):
        self.assertEqual(choose_learning_rate({"baseline_lr1e4": score()})["status"], "pending")

    def test_primary_improvement_cannot_override_safety_failure(self):
        results = {"baseline_lr1e4": score(), "lr5e5": score(.5, sens=.7), "lr3e5": score(.405)}
        self.assertEqual(choose_learning_rate(results)["selected"], "lr3e5")

    def test_negative_results_retain_baseline(self):
        results = {"baseline_lr1e4": score(), "lr5e5": score(.39), "lr3e5": score(.38)}
        self.assertEqual(choose_learning_rate(results)["selected"], "baseline_lr1e4")

    def test_ties_prefer_reference(self):
        results = {name: score() for name in ("baseline_lr1e4", "lr5e5", "lr3e5")}
        self.assertEqual(choose_learning_rate(results)["selected"], "baseline_lr1e4")

    def test_no_outer_inference_entrypoint(self):
        source = inspect.getsource(run_candidate)
        self.assertIn('MANIFEST, "valid", MANIFEST / "valid_segments.csv"', source)
        self.assertNotIn('"outer_test", MANIFEST', source)

    def test_single_fold_history_does_not_mix_other_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in range(5):
                directory = root / "lora" / f"fold{fold}"
                directory.mkdir(parents=True)
                (directory / "fairseq_train.jsonl").write_text('{"epoch":1,"train_loss":1,"train_lr":0.0001}\n')
            histories = read_history(root, "lora", fold=3)
        self.assertEqual([len(rows) for rows in histories], [0, 0, 0, 1, 0])
        self.assertEqual(histories[3][0]["fold"], 3)

    def test_effective_params_exclude_forward_frozen_convolutions(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.feature_extractor = torch.nn.Linear(3, 4)
        model.encoder.feature_grad_mult = 0
        model.proj = torch.nn.Linear(4, 6)
        stats = parameter_summary(model)
        self.assertEqual(stats["requires_grad_parameters"], 46)
        self.assertEqual(stats["effective_trainable_parameters_at_checkpoint"], 30)


if __name__ == "__main__":
    unittest.main()
