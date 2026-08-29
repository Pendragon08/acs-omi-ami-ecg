"""Fast contract tests required before ECG-LFM model training."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.ecg_lfm_adapter import LABELS, aggregate_two_segment_logits  # noqa: E402
from experiments.ecg_lfm.run_fold import training_class_balance  # noqa: E402
from fairseq_signals.criterions.binary_cross_entropy_with_logits import (  # noqa: E402
    BinaryCrossEntropyWithLogitsCriterion,
)
from fairseq_signals import tasks  # noqa: E402
from fairseq_signals.utils import checkpoint_utils  # noqa: E402


def checkpoint_selection_contract() -> dict[str, object]:
    """Prove that the runtime I/O patch saves only strict validation improvements."""
    with tempfile.TemporaryDirectory(prefix="ecg_lfm_checkpoint_contract_") as directory:
        cfg = SimpleNamespace(
            save_dir=directory,
            maximize_best_checkpoint_metric=False,
            no_save=False,
            no_epoch_checkpoints=True,
            save_interval=1,
            save_interval_updates=0,
            keep_best_checkpoints=-1,
            no_last_checkpoints=True,
            best_checkpoint_metric="loss",
            keep_interval_updates=-1,
            keep_interval_updates_pattern=-1,
            keep_last_epochs=0,
        )

        class FakeEpoch:
            epoch = 1

            @staticmethod
            def end_of_epoch() -> bool:
                return True

            @staticmethod
            def state_dict() -> dict:
                return {}

        class FakeTrainer:
            data_parallel_rank = 0
            should_save_checkpoint_on_current_rank = True
            checkpoint_suffix = ""

            def __init__(self) -> None:
                self.saved: list[str] = []

            @staticmethod
            def get_num_updates() -> int:
                return 1

            def save_checkpoint(self, path: str, extra_state: dict) -> None:
                del extra_state
                self.saved.append(Path(path).name)
                Path(path).write_bytes(b"checkpoint-contract")

        if hasattr(checkpoint_utils.save_checkpoint, "best"):
            delattr(checkpoint_utils.save_checkpoint, "best")
        trainer = FakeTrainer()
        checkpoint_utils.save_checkpoint(cfg, trainer, FakeEpoch(), 0.50)
        checkpoint_utils.save_checkpoint(cfg, trainer, FakeEpoch(), 0.60)
        checkpoint_utils.save_checkpoint(cfg, trainer, FakeEpoch(), 0.40)
        best = float(checkpoint_utils.save_checkpoint.best)
        if trainer.saved != ["checkpoint_best.pt", "checkpoint_best.pt"] or best != 0.40:
            raise AssertionError(
                f"Strict-best checkpoint contract failed: saved={trainer.saved}, best={best}"
            )
        delattr(checkpoint_utils.save_checkpoint, "best")
        return {"loss_sequence": [0.50, 0.60, 0.40], "save_events": 2, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=Path, default=Path("manifests/ecg_lfm/master_records.csv"))
    parser.add_argument("--manifest_root", type=Path, default=Path("manifests/ecg_lfm"))
    parser.add_argument("--config", type=Path, default=Path("configs/ecg_lfm/linear_probe.yaml"))
    parser.add_argument("--label_file", type=Path, default=Path("manifests/ecg_lfm/y_all.npy"))
    args = parser.parse_args()

    torch.manual_seed(2026)
    logits = torch.randn(7, 4, dtype=torch.float64)
    target = torch.randint(0, 2, (7, 4), dtype=torch.int64).to(torch.float64)
    criterion = BinaryCrossEntropyWithLogitsCriterion.__new__(BinaryCrossEntropyWithLogitsCriterion)
    criterion.pos_weight = None
    criterion.weight = None
    official, _ = criterion.compute_loss(logits, target, reduce=True)
    reference = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    if not torch.allclose(official, reference, rtol=0.0, atol=1e-12):
        raise AssertionError("Official criterion is not equivalent to unweighted BCEWithLogitsLoss")

    pos_weight = torch.tensor([2.0, 3.0, 0.5, 4.0], dtype=torch.float64)
    criterion.pos_weight = pos_weight
    weighted_official, _ = criterion.compute_loss(logits, target, reduce=True)
    weighted_reference = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="sum"
    )
    if not torch.allclose(weighted_official, weighted_reference, rtol=0.0, atol=1e-12):
        raise AssertionError("Official criterion does not implement PyTorch pos_weight BCE exactly")

    segment_logits = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    aggregated = aggregate_two_segment_logits(segment_logits)
    if tuple(aggregated.shape) != (3, 4) or not torch.equal(aggregated, segment_logits.mean(dim=1)):
        raise AssertionError("Two-segment logit aggregation contract failed")
    if tuple(LABELS) != ("STEMI", "NSTEMI", "UA", "OMI"):
        raise AssertionError("Label order contract failed")

    cfg = OmegaConf.load(args.config)
    cfg.task.data = str(args.manifest_root.resolve())
    cfg.task.label_file = str((args.manifest_root / "y_all.npy").resolve())
    task = tasks.setup_task(cfg.task)
    task.load_dataset("smoke", combine=False, epoch=1, task_cfg=cfg.task, label=True, shuffle=False)
    smoke_dataset = task.dataset("smoke")
    smoke_batch = smoke_dataset.collator([smoke_dataset[0], smoke_dataset[1]])
    if tuple(smoke_batch["net_input"]["source"].shape) != (2, 12, 2500):
        raise AssertionError(f"Official loader input shape mismatch: {smoke_batch['net_input']['source'].shape}")
    if tuple(smoke_batch["label"].shape) != (2, 4):
        raise AssertionError(f"Official loader label shape mismatch: {smoke_batch['label'].shape}")
    if not torch.equal(smoke_batch["label"][0], smoke_batch["label"][1]):
        raise AssertionError("Two segments from one record do not share labels")

    train_patients: dict[int, set[str]] = {fold: set() for fold in range(5)}
    valid_patients: dict[int, set[str]] = {fold: set() for fold in range(5)}
    train_ids: set[str] = set()
    external_ids: set[str] = set()
    all_train_patients: set[str] = set()
    external_patients: set[str] = set()
    with args.master_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation_fold_by_idx = {
        int(row["idx"]): int(row["validation_fold"])
        for row in rows
        if row["cohort"] == "train"
    }
    for row in rows:
        if row["cohort"] == "external":
            external_ids.add(row["record_id"])
            external_patients.add(row["patient_id"])
            continue
        train_ids.add(row["record_id"])
        all_train_patients.add(row["patient_id"])
        held_out_fold = int(row["validation_fold"])
        for fold in range(5):
            destination = valid_patients if fold == held_out_fold else train_patients
            destination[fold].add(row["patient_id"])
    leakage = {fold: sorted(train_patients[fold] & valid_patients[fold]) for fold in range(5)}
    if any(leakage.values()) or train_ids & external_ids or all_train_patients & external_patients:
        raise AssertionError("Patient-level or external split leakage contract failed")

    checkpoint_contract = checkpoint_selection_contract()
    fold_balance = {}
    for fold in range(5):
        segment_index = args.manifest_root / f"fold{fold}" / "train_segments.csv"
        with segment_index.open(encoding="utf-8", newline="") as handle:
            indices = {int(row["idx"]) for row in csv.DictReader(handle)}
        if any(validation_fold_by_idx[idx] == fold for idx in indices):
            raise AssertionError(f"Fold {fold} class-weight calculation includes validation records")
        expected_indices = {idx for idx, held_out in validation_fold_by_idx.items() if held_out != fold}
        if indices != expected_indices:
            raise AssertionError(f"Fold {fold} training indices do not exactly match historical split")
        fold_balance[str(fold)] = training_class_balance(segment_index, args.label_file)

    result = {
        "label_order": list(LABELS),
        "bce_equivalent": True,
        "bce_absolute_difference": float(abs(official.item() - reference.item())),
        "weighted_bce_equivalent": True,
        "weighted_bce_absolute_difference": float(
            abs(weighted_official.item() - weighted_reference.item())
        ),
        "fold_specific_training_only_class_balance": fold_balance,
        "segment_aggregation_shape": list(aggregated.shape),
        "official_loader_source_shape": list(smoke_batch["net_input"]["source"].shape),
        "official_loader_label_shape": list(smoke_batch["label"].shape),
        "train_records": len(train_ids),
        "external_records": len(external_ids),
        "patient_overlap_by_fold": {str(fold): len(leakage[fold]) for fold in range(5)},
        "external_record_overlap": len(train_ids & external_ids),
        "external_patient_overlap": len(all_train_patients & external_patients),
        "checkpoint_strict_improvement_contract": checkpoint_contract,
    }
    print("ECG_LFM_CONTRACT_WITNESS " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
