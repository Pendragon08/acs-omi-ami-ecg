"""Finalize the project experiment log after all summaries and audit exist."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/loss_feature_ablation"


def main():
    path = RESULTS / "EXPERIMENT_LOG.md"
    text = path.read_text(encoding="utf-8").replace("status: running", "status: completed", 1)
    loss = (RESULTS / "loss_ablation_summary.md").read_text(encoding="utf-8")
    feature = (RESULTS / "feature_ablation_summary.md").read_text(encoding="utf-8")
    audit = (RESULTS / "EXPERIMENT_AUDIT_ABLATION.md").read_text(encoding="utf-8")
    marker = "## Results\n\nPending completion. Negative results will be retained."
    replacement = "## Results\n\n" + loss + "\n\n" + feature + "\n\n## Integrity audit\n\n" + audit
    if marker not in text:
        raise SystemExit("experiment log pending marker missing")
    path.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    print(f"finalized {path}")


if __name__ == "__main__":
    main()
