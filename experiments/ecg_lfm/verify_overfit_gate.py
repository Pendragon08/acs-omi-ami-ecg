"""Verify that the diagnostic ECG-LFM subset training actually reduced loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=Path,
        default=Path("results/ecg_lfm_4label/gates/overfit/fairseq_train.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/ecg_lfm_4label/gates/overfit/overfit_gate.json"),
    )
    args = parser.parse_args()

    valid_losses: list[float] = []
    train_losses: list[float] = []
    with args.log.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, destination in (("valid_loss", valid_losses), ("train_loss", train_losses)):
                if key in row:
                    try:
                        destination.append(float(row[key]))
                    except (TypeError, ValueError):
                        pass
    if len(valid_losses) < 2:
        raise RuntimeError(f"Expected at least two validation losses, got {valid_losses}")
    best = min(valid_losses)
    relative_drop = (valid_losses[0] - best) / max(abs(valid_losses[0]), 1e-12)
    passed = bool(best < valid_losses[0] and relative_drop >= 0.10)
    result = {
        "gate": "64-record repeated train/valid diagnostic only",
        "first_valid_loss": valid_losses[0],
        "best_valid_loss": best,
        "last_valid_loss": valid_losses[-1],
        "relative_drop": relative_drop,
        "n_validation_points": len(valid_losses),
        "first_train_loss": train_losses[0] if train_losses else None,
        "best_train_loss": min(train_losses) if train_losses else None,
        "passed_minimum_10_percent_loss_drop": passed,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ECG_LFM_OVERFIT_WITNESS " + json.dumps(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
