"""Export deduplicated ECG-LFM training curves and an overfitting audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FIELDS = (
    "train_loss",
    "valid_loss",
    "valid_auroc",
    "valid_auprc",
    "valid_f1",
    "valid_accuracy",
    "train_lr",
    "train_gnorm",
    "train_gb_free",
    "train_wall",
)


def number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_epochs(path: Path) -> list[dict[str, float | int | None]]:
    epochs: dict[int, dict[str, float | int | None]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch_value = number(payload.get("epoch"))
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            row = epochs.setdefault(epoch, {"epoch": epoch})
            for field in FIELDS:
                value = number(payload.get(field))
                if value is not None:
                    row[field] = value
    return [epochs[epoch] for epoch in sorted(epochs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    rows = load_epochs(args.log)
    completed = [row for row in rows if row.get("valid_loss") is not None]
    if not completed:
        raise RuntimeError(f"No validation epochs found in {args.log}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", *FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best = min(completed, key=lambda row: float(row["valid_loss"]))
    last = completed[-1]
    train_at_best = number(best.get("train_loss"))
    last_train = number(last.get("train_loss"))
    validation_deterioration = float(last["valid_loss"]) - float(best["valid_loss"])
    train_change_after_best = (
        None if train_at_best is None or last_train is None else last_train - train_at_best
    )
    epochs_after_best = int(last["epoch"]) - int(best["epoch"])
    possible_overfit = bool(
        epochs_after_best > 0
        and validation_deterioration >= 0.005
        and train_change_after_best is not None
        and train_change_after_best < 0
    )
    audit = {
        "source_log": str(args.log),
        "n_validation_epochs": len(completed),
        "best_validation_loss_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["valid_loss"]),
        "best_epoch_valid_auroc": number(best.get("valid_auroc")),
        "best_epoch_valid_auprc": number(best.get("valid_auprc")),
        "best_epoch_valid_f1_at_0_5": number(best.get("valid_f1")),
        "last_epoch": int(last["epoch"]),
        "last_validation_loss": float(last["valid_loss"]),
        "epochs_after_best": epochs_after_best,
        "validation_loss_deterioration_after_best": validation_deterioration,
        "train_loss_change_after_best": train_change_after_best,
        "possible_overfit": possible_overfit,
        "patience": args.patience,
        "note": "Health audit only. Final model is the strict minimum-validation-loss checkpoint; final thresholds are selected from record-level OOF probabilities.",
    }
    (args.output_dir / "overfitting_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    epochs = [int(row["epoch"]) for row in completed]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [number(row.get("train_loss")) for row in completed], label="train loss")
    axes[0, 0].plot(epochs, [number(row.get("valid_loss")) for row in completed], label="valid loss")
    axes[0, 0].axvline(int(best["epoch"]), color="black", linestyle="--", linewidth=1, label="best val loss")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0, 0].legend()

    for key, label in (("valid_auroc", "AUROC"), ("valid_auprc", "AUPRC"), ("valid_f1", "F1@0.5")):
        axes[0, 1].plot(epochs, [number(row.get(key)) for row in completed], label=label)
    axes[0, 1].set(title="Validation ranking and threshold metrics", xlabel="Epoch", ylabel="Metric")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [number(row.get("train_lr")) for row in completed])
    axes[1, 0].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[1, 1].plot(epochs, [number(row.get("train_gnorm")) for row in completed], label="gradient norm")
    axes[1, 1].set(title="Optimization stability", xlabel="Epoch", ylabel="Gradient norm")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure_path = args.output_dir / "training_validation_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps({**audit, "curves_csv": str(csv_path), "figure": str(figure_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
