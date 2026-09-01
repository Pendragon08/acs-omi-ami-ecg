"""Freeze an ECG-LFM RDR warmup from W0 inner-validation convergence only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("results/adaptive_weighting/w0/fold0/fairseq_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_weighting/rdr_warmup_selection.json"))
    parser.add_argument("--official_warmup_epochs", type=int, default=160)
    parser.add_argument("--official_total_epochs", type=int, default=200)
    args = parser.parse_args()
    validation: dict[int, dict] = {}
    for line in args.log.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" not in item or "valid_loss" not in item:
            continue
        epoch = int(float(item["epoch"]))
        validation[epoch] = {
            "epoch": epoch,
            "valid_loss": float(item["valid_loss"]),
            "num_updates": int(float(item["valid_num_updates"])),
        }
    if not validation:
        raise RuntimeError(f"no inner-validation epoch summaries found in {args.log}")
    best = min(validation.values(), key=lambda item: (item["valid_loss"], item["epoch"]))
    ratio = args.official_warmup_epochs / args.official_total_epochs
    warmup_updates = max(1, int(math.floor(best["num_updates"] * ratio)))
    result = {
        "status": "frozen",
        "selection_source": "W0 Fold0 inner-validation loss only",
        "outer_test_used": False,
        "official_repository": "https://github.com/GoodMorningPeter/RDR",
        "official_commit": "bb55ea9f906f00a2d8938a97543fc232b5d26645",
        "official_schedule": {
            "warmup_epochs": args.official_warmup_epochs,
            "total_epochs": args.official_total_epochs,
            "ratio": ratio,
        },
        "ecg_lfm_convergence_horizon": best,
        "frozen_rdr_warmup_updates": warmup_updates,
        "all_inner_validation_epochs": [validation[index] for index in sorted(validation)],
        "source_log": str(args.log),
        "source_log_sha256": sha256(args.log),
        "note": "The official RDR equation and momentum are unchanged. The 80% official warmup ratio is mapped to the W0 nested inner-validation convergence horizon because the ECG-LFM protocol early-stops long before 160 epochs.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "warmup_updates": warmup_updates, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
