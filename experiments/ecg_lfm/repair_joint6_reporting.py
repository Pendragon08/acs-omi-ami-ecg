"""Repair historical reporting only. No inference, training, or prediction edits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import torch

from experiments.acs_joint_task import save_curves, sha256
from experiments.ecg_lfm.training_reporting import read_fold_history, history_cost

ROOT = Path(__file__).resolve().parents[2]
VERSION = "reporting_v2_20260907"


def repair(root: Path) -> list[dict]:
    costs = []
    for method in ("lora", "partial"):
        for fold in range(5):
            run = root / "ecg_lfm" / method / f"fold{fold}"
            out = root / "evaluation" / method / f"fold{fold}"
            original = out / "metrics.json"
            metrics = json.loads(original.read_text(encoding="utf-8"))
            # Never edit metrics.json: its hash is bound to the evaluation receipt.
            protected = [p for p in out.iterdir() if p.is_file() and (
                p.name.endswith("predictions.csv") or p.name in ("metrics.json", "evaluation_complete.json")
            )]
            hashes = {p.name: sha256(p) for p in protected}
            log = run / "fairseq_train.jsonl"
            history = read_fold_history(log, fold)
            if not history or not any(row.get("learning_rate") for row in history):
                raise RuntimeError(f"missing trustworthy epoch/LR history: {run}")
            archive = out / "legacy_reporting_v1"
            archive.mkdir(exist_ok=True)
            for name in ("training_curves.csv", "training_validation_loss_lr.png"):
                if (out / name).exists() and not (archive / name).exists():
                    shutil.copy2(out / name, archive / name)
            histories = [[] for _ in range(5)]
            histories[fold] = history
            save_curves(out, histories, f"Joint-6 {method} fold {fold} (epoch summaries)")
            if (out / "curve_plot_error.txt").exists():
                raise RuntimeError((out / "curve_plot_error.txt").read_text())
            stats = dict(metrics["model_stats"])
            requires_grad = int(stats.get("requires_grad_parameters", stats["trainable_parameters"]))
            frozen_conv = 0
            checkpoint = run / "checkpoint_best.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            best_epoch = history_cost(history)["best_epoch"]
            best_row = next(row for row in history if row["epoch"] == best_epoch)
            # fairseq iterator state points to NEXT epoch after an epoch ends.
            # Bind the actual saved update and validation loss, not that cursor.
            checkpoint_update = payload["optimizer_history"][-1]["num_updates"]
            checkpoint_loss = payload["extra_state"]["val_loss"]
            if int(checkpoint_update) != int(best_row["val_num_updates"]) or not math.isclose(
                float(checkpoint_loss), best_row["val_total_loss"], abs_tol=1e-6
            ):
                raise ValueError("checkpoint/log best update or validation loss mismatch")
            cfg = payload["cfg"]
            model_cfg = cfg["model"]
            if method == "partial" and float(model_cfg.get("feature_grad_mult", 1.0)) <= 0:
                candidates = {k: v for k, v in payload["model"].items() if k.startswith("encoder.feature_extractor.")}
                # The convolution stack has parameters only (weight/bias or weight_norm).
                if not candidates or any(k.rsplit(".", 1)[-1] not in ("weight", "bias", "weight_g", "weight_v") for k in candidates):
                    raise ValueError("unexpected convolution state schema; cannot infer effective parameters")
                frozen_conv = sum(v.numel() for v in candidates.values())
            freeze_until = int(model_cfg.get("freeze_finetune_updates", 0))
            if freeze_until != 0:
                raise ValueError("historical correction expects freeze_finetune_updates=0")
            effective = requires_grad - frozen_conv
            del payload
            stats.update({
                "requires_grad_parameters": requires_grad,
                "forward_no_grad_parameters": frozen_conv,
                "effective_trainable_parameters_at_checkpoint": effective,
                "trainable_parameters": effective,
                "trainable_percent": 100.0 * effective / stats["total_parameters"],
                "trainable_parameter_definition": "effective after feature_grad_mult=0 no_grad guard",
                **history_cost(history),
            })
            correction = {
                "reporting_version": VERSION,
                "model_stats": stats,
                "source_metrics_sha256": hashes["metrics.json"],
                "source_log_sha256": sha256(log),
                "source_checkpoint_sha256": sha256(checkpoint),
                "protected_artifacts_sha256": hashes,
                "corrected_curve_sha256": sha256(out / "training_curves.csv"),
                "selected_log_run_index": history[0]["log_run_index"],
                "discarded_prior_log_runs": history[0]["discarded_prior_log_runs"],
                "notes": "Reporting-only repair. Original metrics, predictions and receipts unchanged. Legacy curves archived.",
            }
            if any(sha256(out / name) != value for name, value in hashes.items()):
                raise RuntimeError("protected evaluation artifact changed")
            (out / "reporting_corrections.json").write_text(json.dumps(correction, indent=2), encoding="utf-8")
            costs.append({"method": method, "fold": fold, **{k: stats[k] for k in (
                "total_parameters", "requires_grad_parameters", "effective_trainable_parameters_at_checkpoint",
                "trainable_percent", "best_epoch", "best_inner_validation_loss", "completed_epochs", "train_compute_seconds",
            )}})
            print(json.dumps({"method": method, "fold": fold, "epochs": len(history), "status": "repaired", "protected_unchanged": True}), flush=True)
    summary = root / "summary"
    with (summary / "corrected_training_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    (summary / "reporting_repair_complete.json").write_text(json.dumps({
        "status": "pass", "version": VERSION, "repaired_folds": len(costs),
        "prediction_and_metrics_unchanged": True,
        "repair_source_sha256": sha256(Path(__file__)),
        "historical_memory_limit": "ECG-LFM = sampled whole-board nvidia-smi memory; traditional = torch max_memory_allocated. Not directly comparable.",
    }, indent=2), encoding="utf-8")
    return costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/joint6_cv")
    repair(parser.parse_args().root)
