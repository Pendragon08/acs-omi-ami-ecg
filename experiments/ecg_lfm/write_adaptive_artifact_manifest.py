"""Hash publishable adaptive-weighting artifacts and reject protected files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_NAMES = (
    "inner_validation_predictions.csv",
    "outer_test_predictions.csv",
    "patient_assignments.csv",
    "checkpoint_best.pt",
)
FORBIDDEN_PARTS = ("patient_id", "record_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_weighting/artifact_manifest.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    result_roots = (
        root / "results" / "adaptive_weighting",
        root / "results" / "adaptive_weighting_evaluation",
        root / "results" / "single_task_vs_joint",
        root / "results" / "single_task_vs_joint_evaluation",
    )
    artifacts = []
    excluded = []
    for result_root in result_roots:
        for path in sorted(result_root.rglob("*")):
            if not path.is_file() or path.resolve() == (root / args.output).resolve():
                continue
            relative = path.relative_to(root).as_posix()
            lower = relative.lower()
            protected = (
                path.name in FORBIDDEN_NAMES
                or path.suffix.lower() in {".pt", ".pth", ".jsonl", ".log"}
                or "predictions" in path.name.lower()
                or "dryrun" in lower
                or any(part in lower for part in FORBIDDEN_PARTS)
            )
            if protected:
                excluded.append(relative)
                continue
            artifacts.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    leaked = [item for item in artifacts if any(token in item["path"].lower() for token in FORBIDDEN_PARTS)]
    if leaked:
        raise RuntimeError(f"protected artifact entered publishable manifest: {leaked}")
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "scope": "publishable aggregate adaptive-weighting and single-task-control artifacts",
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
        "excluded_protected_artifacts": excluded,
        "policy": "no record/patient predictions, patient assignments, checkpoints, raw logs, or event streams",
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "artifacts": len(artifacts), "excluded": len(excluded), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
