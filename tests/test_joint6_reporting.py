from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.ecg_lfm.training_reporting import read_fold_history, history_cost


class HistoryTests(unittest.TestCase):
    def test_epoch_merge_duplicate_filter_and_train_lr(self):
        train = {"epoch": 1, "train_loss": "1.2", "train_lr": "0.0001", "train_num_updates": "91", "train_train_wall": "300"}
        val = {"epoch": 1, "valid_loss": "1.3", "valid_num_updates": "91"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fairseq_train.jsonl"
            path.write_text("configuration text\n" + "\n".join(json.dumps(x) for x in (
                train, val, train, val, {"epoch": 1, "update": 0.5, "train_loss": 88},
            )))
            rows = read_fold_history(path, 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fold"], 3)
        self.assertEqual(rows[0]["learning_rate"], .0001)
        self.assertEqual(rows[0]["val_total_loss"], 1.3)
        self.assertEqual(history_cost(rows)["train_compute_seconds"], 300)

    def test_conflicting_restart_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log"
            path.write_text('\n'.join(json.dumps(x) for x in (
                {"epoch": 1, "train_loss": 1}, {"epoch": 1, "train_loss": 2},
            )))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                read_fold_history(path, 0)

    def test_validation_best_not_last(self):
        rows = [{"epoch": 1, "val_total_loss": 1}, {"epoch": 2, "val_total_loss": .8}, {"epoch": 3, "val_total_loss": 2}]
        self.assertEqual(history_cost(rows)["best_epoch"], 2)

    def test_explicit_completed_retry_is_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log"
            path.write_text("{'_name': None,\n" + json.dumps({"epoch": 1, "train_loss": 9}) +
                            "\n{'_name': None,\n" + json.dumps({"epoch": 1, "train_loss": 1}) +
                            "\ndone training in 10 seconds\n")
            rows = read_fold_history(path, 4)
        self.assertEqual(rows[0]["train_total_loss"], 1)
        self.assertEqual(rows[0]["discarded_prior_log_runs"], 1)


if __name__ == "__main__":
    unittest.main()
