import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.ecg_lfm.summarize_joint_vs_single_nested5fold import (
    acs_fold,
    aligned,
    format_ms,
    hard_binary,
    omi_fold,
    write_report,
)


class JointVsSingleSummaryTests(unittest.TestCase):
    def test_fold_metric_parser_uses_nested_nste_metrics(self):
        metrics = {
            "outer_test": {
                "acs": {
                    "macro_auroc_ovr": 0.8, "macro_auprc_ovr": 0.5,
                    "macro_f1": 0.4, "balanced_accuracy": 0.45, "accuracy": 0.6,
                    "per_class": {
                        "NSTEMI": {"auprc": 0.2, "f1": 0.1, "sensitivity": 0.15}
                    },
                },
                "nstemi_vs_ua": {"auprc": 0.3},
                "omi": {"youden_j": {
                    "auroc": 0.9, "auprc": 0.4, "f1": 0.3, "sensitivity": 0.8,
                    "specificity": 0.7, "ppv": 0.2, "npv": 0.98, "mcc": 0.3,
                    "accuracy": 0.71, "threshold": 0.04,
                }},
                "omi_among_nste_acs": {"n": 100, "metrics": {"auprc": 0.25}},
            }
        }
        self.assertEqual(acs_fold(metrics)["nstemi_vs_ua_auprc"], 0.3)
        self.assertEqual(omi_fold(metrics)["nste_auprc"], 0.25)

    def test_hard_binary_confusion_and_metrics(self):
        result = hard_binary(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1]))
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertEqual(result["sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["f1"], 0.5)

    def test_alignment_rejects_record_mismatch(self):
        left = [{"record_id": "r1", "patient_id": "p1", "y_acs": "2"}]
        right = [{"record_id": "r2", "patient_id": "p1", "y_acs": "2"}]
        with self.assertRaisesRegex(ValueError, "record/patient mismatch"):
            aligned(left, right, "acs")

    def test_chinese_report_is_utf8_and_readable(self):
        metric = {"mean": 0.5, "std": 0.1}
        acs = {
            key: metric for key in (
                "macro_auroc", "macro_auprc", "macro_f1", "balanced_accuracy",
                "nstemi_auprc", "nstemi_sensitivity", "nstemi_vs_ua_auprc",
            )
        }
        omi = {
            key: metric for key in (
                "auroc", "auprc", "f1", "sensitivity", "specificity", "ppv", "npv", "nste_auprc",
            )
        }
        comparison = {
            "endpoint": {"delta": 0.1, "ci95": [0.01, 0.2], "ci_crosses_zero": False}
        }
        result = {
            "fold_mean_std": {
                "joint_acs": acs, "acs_only": acs, "joint_omi": omi, "omi_only": omi,
            },
            "paired_patient_bootstrap": {
                "joint_vs_acs_only": comparison, "joint_vs_omi_only": comparison,
            },
            "positive_fold_counts": {"acs_macro_auprc": 5},
            "phase_s0_decision": "PASS", "n_records": 10, "n_patients": 8,
            "bootstrap_replicates": 1000, "bootstrap_seed": 2026,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, result)
            text = path.read_text(encoding="utf-8")
        self.assertIn("五折 mean ± std", text)
        self.assertIn("全部 pass", text)
        self.assertNotIn("�", text)
        self.assertEqual(format_ms(metric), "0.500000 ± 0.100000")


if __name__ == "__main__":
    unittest.main()
