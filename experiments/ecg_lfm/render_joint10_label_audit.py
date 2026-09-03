"""Render aggregate-only figures from the Joint-10 pre-training audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bar(labels: list[str], values: list[int], title: str, ylabel: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.8))
    bars = axis.bar(labels, values, color="#4472C4")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    for item, value in zip(bars, values):
        axis.text(item.get_x() + item.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", type=Path, default=Path("results/joint10_label_audit"))
    args = parser.parse_args()
    root = args.audit_dir.resolve()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    status = audit["status"]
    classes = read_rows(root / "joint10_class_counts.csv")
    bar([row["joint10_class"] for row in classes], [int(row["ecg_count"]) for row in classes],
        f"[{status}] Candidate Joint-10 distribution (unresolved OMI excluded)", "ECG count",
        figures / "joint10_class_distribution.png")
    locations = read_rows(root / "omi_location_counts.csv")
    bar([row["location_or_status"] for row in locations], [int(row["ecg_count"]) for row in locations],
        f"[{status}] OMI culprit-region resolution audit", "OMI ECG count",
        figures / "omi_culprit_vessel_distribution.png")


if __name__ == "__main__":
    main()
