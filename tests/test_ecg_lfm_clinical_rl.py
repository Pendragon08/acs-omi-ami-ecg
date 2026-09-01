from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ECG-LFM"))

from experiments.ecg_lfm.rl.clinical_constraints import SafetyGuard  # noqa: E402
from experiments.ecg_lfm.rl.clinical_policy import (  # noqa: E402
    joint_log_probability,
    probabilities,
    sample,
)
from experiments.ecg_lfm.rl.clinical_reward import (  # noqa: E402
    RewardWeights,
    hard_reward,
    soft_f1_shaping,
)
from experiments.ecg_lfm.rl.adaptive_kl_controller import (  # noqa: E402
    update_kl_controller as adaptive_update_kl_controller,
)
from experiments.ecg_lfm.rl.cross_task_reward import (  # noqa: E402
    CrossTaskRewardConfig,
    cross_task_reward,
)
from experiments.ecg_lfm.rl.kl_regularization import factorized_kl  # noqa: E402
from experiments.ecg_lfm.rl.ranking_preservation import pairwise_ranking_losses  # noqa: E402
from experiments.ecg_lfm.rl.reference_guard import (  # noqa: E402
    guard_metrics,
    patient_bootstrap_margins,
)
from experiments.ecg_lfm.train_rl_v2_head_only import update_kl_controller  # noqa: E402
from fairseq_signals.modules.parameter_efficient import (  # noqa: E402
    LoRALinear,
    enable_head,
    inject_linear_adapters,
)


def weights() -> RewardWeights:
    return RewardWeights(
        acs_correct=1.0, omi_correct=1.0, nstemi_correct_bonus=0.5,
        nstemi_to_ua_penalty=-0.25, stemi_to_other_penalty=-0.25,
        omi_tp_bonus=0.5, omi_fn_penalty=-0.5, nste_omi_tp_bonus=0.5,
        acs_soft_f1=0.1, omi_soft_f1=0.1,
    )


class ClinicalPolicyTests(unittest.TestCase):
    def test_policy_factorization(self):
        logits = torch.tensor([[0.2, -0.1, 0.4, 0.0, 0.7]])
        acs, omi = probabilities(logits)
        action_acs, action_omi = torch.tensor([2]), torch.tensor([1])
        actual = joint_log_probability(logits, action_acs, action_omi).exp()
        expected = acs[0, 2] * omi[0]
        torch.testing.assert_close(actual[0], expected)

    def test_acs_softmax(self):
        acs, _ = probabilities(torch.randn(7, 5))
        torch.testing.assert_close(acs.sum(dim=1), torch.ones(7))

    def test_omi_sigmoid_is_independent(self):
        first = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.7]])
        second = torch.tensor([[100.0, -100.0, 50.0, -50.0, 0.7]])
        self.assertEqual(float(probabilities(first)[1]), float(probabilities(second)[1]))

    def test_joint_action_space_size_8(self):
        actions = {int(acs * 2 + omi) for acs in range(4) for omi in range(2)}
        self.assertEqual(actions, set(range(8)))

    def test_action_distribution_and_entropy_logging_fields(self):
        output = sample(torch.zeros(32, 5))
        self.assertEqual(tuple(output.joint_action.shape), (32,))
        self.assertTrue(bool(((output.joint_action >= 0) & (output.joint_action < 8)).all()))
        self.assertTrue(bool(torch.isfinite(output.acs_entropy).all()))
        self.assertTrue(bool(torch.isfinite(output.omi_entropy).all()))

    def test_reference_and_policy_equal_at_init_and_kl_zero(self):
        reference = torch.randn(16, 5)
        policy = reference.clone()
        ref_prob = probabilities(reference)
        policy_prob = probabilities(policy)
        torch.testing.assert_close(ref_prob[0], policy_prob[0])
        torch.testing.assert_close(ref_prob[1], policy_prob[1])
        kl = factorized_kl(policy, reference)
        torch.testing.assert_close(kl["total"], torch.zeros(16), atol=1e-7, rtol=0)

    def test_kl_acs_and_omi_are_nonnegative(self):
        policy, reference = torch.randn(64, 5), torch.randn(64, 5)
        kl = factorized_kl(policy, reference)
        self.assertGreaterEqual(float(kl["acs"].min()), -1e-6)
        self.assertGreaterEqual(float(kl["omi"].min()), -1e-6)
        torch.testing.assert_close(kl["total"], kl["acs"] + kl["omi"])


class ClinicalRewardTests(unittest.TestCase):
    def test_reward_no_nan(self):
        total, components = hard_reward(
            torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 1, 1]), weights(),
        )
        self.assertTrue(bool(torch.isfinite(total).all()))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in components.values()))

    def test_nstemi_reward_condition(self):
        _, components = hard_reward(
            torch.tensor([2, 3, 2]), torch.zeros(3, dtype=torch.long),
            torch.tensor([2, 2, 3]), torch.zeros(3, dtype=torch.long), weights(),
        )
        self.assertEqual(components["nstemi"].tolist(), [0.5, 0.0, 0.0])
        self.assertEqual(components["nstemi_to_ua"].tolist(), [0.0, -0.25, 0.0])

    def test_nste_omi_reward_condition(self):
        _, components = hard_reward(
            torch.tensor([2, 3, 1]), torch.tensor([1, 1, 1]),
            torch.tensor([2, 3, 1]), torch.tensor([1, 1, 1]), weights(),
        )
        self.assertEqual(components["nste_omi"].tolist(), [0.5, 0.5, 0.0])

    def test_soft_f1_reward_is_finite_and_differentiable(self):
        logits = torch.randn(8, 5, requires_grad=True)
        acs, omi = probabilities(logits)
        score, parts = soft_f1_shaping(
            acs, omi, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), weights(),
        )
        score.backward()
        self.assertTrue(bool(torch.isfinite(score)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertEqual(set(parts), {"acs_soft_f1", "omi_soft_f1"})

    def test_cross_task_reward_is_restricted_to_true_nstemi_or_ua_omi(self):
        config = CrossTaskRewardConfig(
            enabled=True, scale=1.0,
            nstemi_exact=1.0, nstemi_omi_only=0.5, nstemi_phenotype_only=0.0, nstemi_both_wrong=-0.5,
            ua_exact=0.5, ua_nstemi_omi=0.25, ua_phenotype_only=0.0, ua_both_wrong=-0.25,
        )
        acs_action = torch.tensor([2, 0, 2, 3, 1])
        omi_action = torch.tensor([1, 1, 1, 1, 1])
        acs_target = torch.tensor([2, 0, 3, 3, 1])
        omi_target = torch.tensor([1, 1, 1, 1, 0])
        total, parts = cross_task_reward(acs_action, omi_action, acs_target, omi_target, config)
        self.assertEqual(parts["cross_nstemi_omi"].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(parts["cross_ua_omi"].tolist(), [0.0, 0.0, 0.25, 0.5, 0.0])
        self.assertEqual(total.tolist(), [1.0, 0.0, 0.25, 0.5, 0.0])

    def test_cross_task_reward_ordering_for_nstemi_omi(self):
        config = CrossTaskRewardConfig(
            enabled=True, scale=1.0,
            nstemi_exact=1.0, nstemi_omi_only=0.5, nstemi_phenotype_only=0.0, nstemi_both_wrong=-0.5,
            ua_exact=0.5, ua_nstemi_omi=0.25, ua_phenotype_only=0.0, ua_both_wrong=-0.25,
        )
        target_acs = torch.tensor([2, 2, 2, 2])
        target_omi = torch.ones(4, dtype=torch.long)
        action_acs = torch.tensor([2, 0, 2, 0])
        action_omi = torch.tensor([1, 1, 0, 0])
        total, _ = cross_task_reward(action_acs, action_omi, target_acs, target_omi, config)
        self.assertEqual(total.tolist(), [1.0, 0.5, 0.0, -0.5])

    def test_pairwise_ranking_is_sampled_finite_and_differentiable(self):
        logits = torch.randn(16, 5, requires_grad=True)
        acs_target = torch.tensor([0, 1, 2, 3] * 4)
        omi_target = torch.tensor([0, 1] * 8)
        generator = torch.Generator().manual_seed(2026)
        result = pairwise_ranking_losses(
            logits, acs_target, omi_target,
            {"max_negatives_per_positive": 2}, generator=generator,
        )
        result["rank_loss"].backward()
        self.assertTrue(bool(torch.isfinite(result["rank_loss"])))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(int(result["pair_count"]), 0)
        self.assertLessEqual(int(result["pair_count"]), 16 * 4 * 2 + 16 * 2)
        self.assertIn("nstemi_rank_loss", result)


class ClinicalSafetyAndFreezingTests(unittest.TestCase):
    def test_v2_update_budget_head_only_config_and_controller(self):
        config = yaml.safe_load((ROOT / "configs/ecg_lfm/rl/clinical_rl_v2_head_only.yaml").read_text())
        self.assertEqual(config["policy"]["trainable"], "head_only")
        self.assertEqual(config["optimization"]["max_updates"], 100)
        self.assertEqual(config["optimization"]["eval_interval_updates"], 25)
        self.assertEqual(update_kl_controller(0.05, 0.02, 0.005, config), 0.1)
        self.assertEqual(update_kl_controller(0.05, 0.001, 0.005, config), 0.025)
        source = (ROOT / "experiments/ecg_lfm/train_rl_v2_head_only.py").read_text()
        self.assertNotIn('load_dataset("outer_test"', source)
        self.assertNotIn("outer_test_segments.csv", source)

    def test_v2a_cross_task_config_and_adaptive_controller(self):
        config = yaml.safe_load((ROOT / "configs/ecg_lfm/rl/v2a_head_cross_task_targetkl.yaml").read_text())
        self.assertEqual(config["policy"]["trainable"], "head_only")
        self.assertTrue(config["reward"]["cross_task"]["enabled"])
        self.assertTrue(config["ranking"]["omi_pairwise"])
        self.assertEqual(config["optimization"]["max_updates"], 100)
        self.assertEqual(config["optimization"]["eval_interval_updates"], 25)
        self.assertEqual(adaptive_update_kl_controller(0.05, 0.02, 0.005, config["kl"]), 0.1)
        self.assertEqual(adaptive_update_kl_controller(0.05, 0.001, 0.005, config["kl"]), 0.025)
        source = (ROOT / "experiments/ecg_lfm/train_rl_v2_cross_task.py").read_text()
        self.assertNotIn('load_dataset("outer_test"', source)
        self.assertNotIn("outer_test_segments.csv", source)

    @staticmethod
    def synthetic_rows():
        rows = []
        for index in range(80):
            y_acs = index % 4
            y_omi = int(index % 5 == 0)
            p_acs = torch.full((4,), 0.05)
            p_acs[y_acs] = 0.85
            p_acs /= p_acs.sum()
            rows.append({
                "patient_id": f"p{index // 2}", "y_acs": y_acs, "y_omi": y_omi,
                **{f"p_{name}": float(p_acs[k]) for k, name in enumerate(("OTHER", "STEMI", "NSTEMI", "UA"))},
                "p_omi": 0.8 if y_omi else 0.1,
            })
        return rows

    def test_reference_guard_is_inner_patient_bootstrap(self):
        rows = self.synthetic_rows()
        reference = guard_metrics(rows)
        margins, audit = patient_bootstrap_margins(rows, 20, 2026, 0.005)
        self.assertEqual(audit["audit"]["n_patients"], 40)
        self.assertEqual(audit["audit"]["replicates_requested"], 20)
        self.assertTrue(all(value >= 0.005 for value in margins.values()))
        self.assertAlmostEqual(reference["omi_threshold"], audit["reference"]["omi_threshold"])

    def test_safety_gate(self):
        reference = {
            "acs_macro_auprc": 0.55, "omi_auprc": 0.46, "omi_ppv": 0.26,
            "omi_specificity": 0.82, "stemi_f1": 0.67,
            "acs_brier": 0.51, "omi_brier": 0.05,
        }
        margins = {name: 0.01 for name in reference}
        guard = SafetyGuard.from_reference(reference, margins, max_kl=0.05)
        self.assertTrue(guard.evaluate(reference, kl_total=0.0)["pass"])
        failed = dict(reference, omi_ppv=0.1)
        self.assertFalse(guard.evaluate(failed, kl_total=0.0)["pass"])

    def test_lora_and_head_trainable_base_frozen(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.encoder = nn.Module()
                self.encoder.encoder.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])
                self.proj = nn.Linear(4, 5)

        model = Dummy()
        inject_linear_adapters(model, "lora", rank=2, alpha=4, dropout=0.0)
        enable_head(model)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any("lora_A" in name for name in trainable))
        self.assertTrue(any("lora_B" in name for name in trainable))
        self.assertIn("proj.weight", trainable)
        self.assertIn("proj.bias", trainable)
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                self.assertFalse(module.base.weight.requires_grad, name)

    def test_outer_test_not_used_for_tuning(self):
        for filename in (
            "clinical_rl_joint_w0.yaml", "clinical_rl_no_kl.yaml",
            "clinical_rl_no_hard_reward.yaml",
        ):
            payload = yaml.safe_load((ROOT / "configs" / "ecg_lfm" / "rl" / filename).read_text())
            self.assertFalse(payload["outer_test"]["allowed_during_tuning"])
            self.assertTrue(payload["outer_test"]["evaluate_once_after_freeze"])
            self.assertEqual(payload["safety"]["baseline_source"], "fold0_inner_validation_w0_only")

        training_source = (ROOT / "experiments" / "ecg_lfm" / "train_rl_fold.py").read_text()
        self.assertNotIn('load_dataset("outer_test"', training_source)
        self.assertNotIn("outer_test_segments.csv", training_source)


if __name__ == "__main__":
    unittest.main()
