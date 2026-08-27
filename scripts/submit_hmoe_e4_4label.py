"""Submit the frozen four-label E4 CSV to the ACS external evaluation platform.

This script performs exactly one POST of the pre-existing submission CSV.  It
does not train, select thresholds, or read hidden test labels.  The platform
accepts a CSV containing any subset of its known binary label columns, so this
run uploads STEMI, NSTEMI, UA and OMI together and records every returned row.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLATFORM = "http://39.105.59.221:8080/"
LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
HEADERS = ("label", "samples", "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def parse_all_rows(page: str) -> list[dict]:
    """Parse all metric rows instead of only the first label row."""
    table = re.search(r"<th>Label</th>.*?</table>", page, re.S | re.I)
    if not table:
        return []
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S | re.I):
        cells = [strip_tags(cell) for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
        if len(cells) < len(HEADERS) or cells[0].lower() == "label":
            continue
        row = {}
        for key, value in zip(HEADERS, cells):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def flash_message(page: str) -> str | None:
    match = re.search(r'<li class="flash-message[^"]*">(.*?)</li>', page, re.S | re.I)
    return strip_tags(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="results/hmoe_kan_stemi_nstemi_ua_omi_e4")
    parser.add_argument("--submission", default="submission_stemi_nstemi_ua_omi_e4_youden.csv")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    submission = run_dir / args.submission
    if not submission.exists():
        raise FileNotFoundError(submission)

    result = subprocess.run(
        ["curl", "-sS", "-m", str(args.timeout), "-L", "-c", "/tmp/acs_cj.txt",
         "-F", f"file=@{submission}", PLATFORM + "submit"],
        capture_output=True, text=True,
    )
    page = result.stdout
    (run_dir / "official_platform_4label_response.html").write_text(page, encoding="utf-8")
    rows = parse_all_rows(page)
    by_label = {str(row.get("label")): row for row in rows}
    payload = {
        "platform": PLATFORM,
        "labels_submitted": list(LABELS),
        "submission": str(submission),
        "submission_sha256": sha256(submission),
        "curl_returncode": int(result.returncode),
        "curl_stderr": result.stderr.strip(),
        "flash_message": flash_message(page),
        "metrics": rows,
        "missing_metric_labels": [label for label in LABELS if label not in by_label],
        "test_labels_used_locally": False,
    }
    (run_dir / "official_platform_4label_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if result.returncode != 0 or payload["missing_metric_labels"]:
        raise RuntimeError("Platform submission did not return all requested label metrics")


if __name__ == "__main__":
    main()
