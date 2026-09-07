"""Fold-local, end-of-epoch reporting; never reads predictions or changes training."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_fold_history(path: Path, fold: int) -> list[dict]:
    """Merge final train/valid summaries, excluding interval logs and duplicates.

    fairseq's JSON interval events have ``update``. End-of-epoch events do not.
    Identical JSON events can be repeated by multiple logging handlers. If a
    log contains conflicting epoch summaries (e.g. appended restarts), fail
    instead of silently combining different training attempts.
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("{'_name':")]
    run_index = max(0, len(starts) - 1)
    if len(starts) > 1:
        # An appended retry is a different run, not an epoch duplicate. Select
        # the final run only when its explicit completion footer is present.
        final_run = lines[starts[-1]:]
        if not any(line.startswith("done training in ") for line in final_run):
            raise ValueError(f"appended final run has not completed: {path}")
        lines = final_run
    summaries = {}
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict) or "update" in event:
            continue
        epoch = _number(event.get("epoch"))
        if epoch is None or not epoch.is_integer():
            continue
        epoch = int(epoch)
        kind = "train" if "train_loss" in event else "valid" if "valid_loss" in event else None
        if kind is None:
            continue
        prefix = "train" if kind == "train" else "valid"
        row = {
            f"{kind}_total_loss": _number(event.get(f"{prefix}_loss")),
            f"{kind}_joint_accuracy": _number(event.get(f"{prefix}_joint_accuracy")),
            f"{kind}_num_updates": _number(event.get(f"{prefix}_num_updates")),
        }
        if kind == "train":
            row.update({
                "learning_rate": _number(event.get("train_lr", event.get("lr"))),
                "train_compute_seconds": _number(event.get("train_train_wall")),
                "elapsed_wall_seconds": _number(event.get("train_wall")),
                "gradient_norm": _number(event.get("train_gnorm")),
            })
        key = (epoch, kind)
        if key in summaries and summaries[key] != row:
            raise ValueError(f"conflicting {kind} epoch {epoch} summaries in {path}")
        summaries[key] = row
    result = []
    for epoch in sorted({key[0] for key in summaries}):
        row = {"fold": int(fold), "epoch": epoch, "log_run_index": run_index,
               "discarded_prior_log_runs": run_index}
        row.update(summaries.get((epoch, "train"), {}))
        valid = summaries.get((epoch, "valid"), {})
        row.update({key.replace("valid_", "val_", 1): value for key, value in valid.items()})
        result.append(row)
    return result


def history_cost(rows: list[dict]) -> dict:
    validation = [row for row in rows if row.get("val_total_loss") is not None]
    best = min(validation, key=lambda row: (row["val_total_loss"], row["epoch"])) if validation else None
    durations = [row["train_compute_seconds"] for row in rows if row.get("train_compute_seconds") is not None]
    return {
        "best_epoch": None if best is None else best["epoch"],
        "best_inner_validation_loss": None if best is None else best["val_total_loss"],
        "completed_epochs": max((row["epoch"] for row in rows), default=0),
        "train_compute_seconds": sum(durations) if durations else None,
        "train_compute_seconds_source": "sum of deduplicated fairseq train_train_wall; excludes validation; rounded epoch timers",
    }
