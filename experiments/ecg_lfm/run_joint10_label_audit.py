"""Run the Joint-10 audit and persist the observed process exit code."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/joint10_label_audit"))
    args, passthrough = parser.parse_known_args()
    output = (ROOT / args.output).resolve()
    command = [
        sys.executable,
        str(ROOT / "experiments/ecg_lfm/audit_joint10_labels.py"),
        "--output",
        str(output),
        *passthrough,
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    summary_path = output / "audit_summary.json"
    status = None
    if summary_path.exists():
        status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
    receipt = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "observed_exit_code": process.returncode,
        "audit_status": status,
        "gate_consistent": (status == "PASS" and process.returncode == 0)
        or (status != "PASS" and process.returncode == 2),
        "stdout_sha256": digest(process.stdout),
        "stderr_sha256": digest(process.stderr),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_execution_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
