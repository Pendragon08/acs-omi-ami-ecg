#!/usr/bin/env python3
"""导出可公开归档的逐 epoch 训练日志。

只保留 loss、验证集指标、学习率和 fold/epoch 等实验字段；不会复制
患者编号、波形、checkpoint、服务器绝对路径或逐例预测。该脚本同时支持
PyTorch history_fold*.json 和 ECG-LFM Fairseq 的 JSONL 日志。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_total_loss",
    "train_acs_ce",
    "train_omi_bce",
    "val_total_loss",
    "val_acs_ce",
    "val_omi_bce",
    "val_acs_accuracy",
    "val_acs_macro_auroc",
    "val_omi_auroc",
    "val_omi_auprc",
    "val_joint_score",
]

FAIRSEQ_FIELDS = [
    "epoch",
    "train_loss",
    "train_acs_loss",
    "train_omi_loss",
    "train_acs_accuracy",
    "train_omi_accuracy",
    "train_lr",
    "valid_loss",
    "valid_acs_loss",
    "valid_omi_loss",
    "valid_acs_accuracy",
    "valid_omi_accuracy",
    "valid_best_loss",
]


def _number(value: Any) -> Any:
    """Normalize numeric strings while leaving missing values empty."""

    if value in (None, ""):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def history_rows(root: Path) -> Iterable[dict[str, Any]]:
    """Yield whitelisted rows from every fold history below *root*."""

    for history_path in sorted(root.glob("history_fold*.json")):
        stem = history_path.stem
        try:
            fold = int(stem.replace("history_fold", ""))
        except ValueError:
            continue
        records = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            row = {"fold": fold}
            row.update({field: _number(record.get(field)) for field in HISTORY_FIELDS})
            yield row


def fairseq_rows(path: Path, run_name: str) -> Iterable[dict[str, Any]]:
    """Yield only epoch-level metric records from an ECG-LFM JSONL log."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            # Fairseq repeats records at different logging hooks.  Keep only
            # records that contain train_* or valid_* epoch aggregates.
            if not any(key.startswith(("train_", "valid_")) for key in record):
                continue
            row = {"run": run_name}
            row.update({field: _number(record.get(field)) for field in FAIRSEQ_FIELDS})
            yield row


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="含 history_fold*.json 的结果目录")
    parser.add_argument("--jsonl", type=Path, help="ECG-LFM Fairseq JSONL 日志")
    parser.add_argument("--run-name", default="ecg_lfm", help="JSONL 对应的运行名称")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.root) == bool(args.jsonl):
        parser.error("必须且只能指定 --root 或 --jsonl")
    if args.root:
        fields = ["fold", *HISTORY_FIELDS]
        count = write_csv(args.output, history_rows(args.root), fields)
    else:
        fields = ["run", *FAIRSEQ_FIELDS]
        count = write_csv(args.output, fairseq_rows(args.jsonl, args.run_name), fields)
    print(f"wrote {count} sanitized epoch rows to {args.output}")


if __name__ == "__main__":
    main()
