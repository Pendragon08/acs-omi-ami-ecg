"""Create auditable RL optimization and inner-validation curves from JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "train_log.jsonl").read_text().splitlines() if line]
    updates = [row for row in rows if row.get("event") == "update"]
    # R1-v2 uses a fixed optimizer-update budget and emits
    # ``inner_validation`` records instead of epoch records.  Keep the
    # original epoch handling for R1 while treating both event types as
    # validation checkpoints here.
    validations = [row for row in rows if row.get("event") in {"epoch", "inner_validation"}]
    if not updates:
        raise RuntimeError("no RL updates found")
    import matplotlib.pyplot as plt

    x = [row["update"] for row in updates]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(x, [row["loss"] for row in updates], label="RL loss")
    axes[0, 0].plot(x, [row["policy_objective"] for row in updates], label="policy objective")
    axes[0, 0].plot(x, [row["soft_f1"] for row in updates], label="soft-F1 shaping")
    axes[0, 0].legend(); axes[0, 0].set_title("Optimization objectives")
    axes[0, 1].plot(x, [row["reward_mean"] for row in updates], label="reward mean")
    axes[0, 1].plot(x, [row["reward_std"] for row in updates], label="reward std")
    axes[0, 1].legend(); axes[0, 1].set_title("Sampled reward")
    axes[1, 0].plot(x, [row["kl_acs"] for row in updates], label="KL ACS")
    axes[1, 0].plot(x, [row["kl_omi"] for row in updates], label="KL OMI")
    axes[1, 0].plot(x, [row["kl_total"] for row in updates], label="KL total")
    axes[1, 0].legend(); axes[1, 0].set_title("Drift from frozen W0")
    axes[1, 1].plot(x, [row["acs_entropy"] for row in updates], label="ACS entropy")
    axes[1, 1].plot(x, [row["omi_entropy"] for row in updates], label="OMI entropy")
    axes[1, 1].plot(x, [row["clip_fraction"] for row in updates], label="clip fraction")
    axes[1, 1].legend(); axes[1, 1].set_title("Policy diagnostics")
    for axis in axes.flat: axis.set_xlabel("optimizer update"); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.run_dir / "rl_training_curves.png", dpi=180); plt.close(fig)

    if validations:
        e = [row.get("update", row.get("epoch")) for row in validations]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].plot(e, [row["metrics"]["acs"]["macro_auprc_ovr"] for row in validations], marker="o", label="ACS Macro AUPRC")
        axes[0].plot(e, [row["metrics"]["acs"]["per_class"]["NSTEMI"]["auprc"] for row in validations], marker="o", label="NSTEMI AUPRC")
        axes[1].plot(e, [row["metrics"]["omi"]["auprc"] for row in validations], marker="o", label="OMI AUPRC")
        axes[1].plot(e, [row["metrics"]["omi"]["f1"] for row in validations], marker="o", label="OMI F1")
        axes[2].plot(e, [row["metrics"]["kl"]["total"] for row in validations], marker="o", label="Validation mean KL")
        axes[2].plot(e, [row["clinical_utility"] for row in validations], marker="o", label="Clinical utility")
        for axis in axes: axis.set_xlabel("optimizer update"); axis.grid(alpha=0.25); axis.legend()
        fig.tight_layout(); fig.savefig(args.run_dir / "rl_inner_validation_curves.png", dpi=180); plt.close(fig)
    print(json.dumps({"status": "pass", "updates": len(updates), "validations": len(validations)}))


if __name__ == "__main__":
    main()
