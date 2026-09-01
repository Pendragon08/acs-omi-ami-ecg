"""Extract per-epoch official-RDR weights from Fairseq JSON logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ACS = ("OTHER", "STEMI", "NSTEMI", "UA")
OMI = ("NOMI", "OMI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_epoch: dict[int, dict] = {}
    for line in (args.run_dir / "fairseq_train.jsonl").read_text(errors="ignore").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" not in item or "train_loss" not in item:
            continue
        # Fairseq installs more than one JSON logger in this project, so an
        # epoch summary can be written repeatedly.  Keep one canonical row per
        # epoch; otherwise the weight trajectory and downstream statistics are
        # silently over-counted.
        epoch = int(float(item["epoch"]))
        row = {"method": args.method, "fold": args.fold, "epoch": epoch}
        for task, names in (("acs", ACS), ("omi", OMI)):
            for name in names:
                for statistic in ("mean", "std", "min", "max"):
                    key = f"train_{task}_{name}_weight_{statistic}"
                    row[key] = item.get(key)
        row["train_rdr_active_fraction"] = item.get("train_rdr_active_fraction")
        rows_by_epoch[epoch] = row
    rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]
    if not rows:
        raise RuntimeError(f"no RDR epoch rows found: {args.run_dir}")
    fields = list(rows[0])
    csv_path = args.output_dir / "rdr_weight_dynamics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for task, names in (("acs", ACS), ("omi", OMI)):
        available = [name for name in names if any(row.get(f"train_{task}_{name}_weight_mean") is not None for row in rows)]
        if not available:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        for name in available:
            axis.plot(
                [row["epoch"] for row in rows],
                [float(row[f"train_{task}_{name}_weight_mean"]) for row in rows],
                marker="o",
                label=name,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Epoch", ylabel="Mean RDR sample weight", title=f"{args.method} Fold{args.fold} {task.upper()} RDR dynamics")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"rdr_{task}_weight_dynamics.png", dpi=180)
        plt.close(figure)
    print(json.dumps({"status": "pass", "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
