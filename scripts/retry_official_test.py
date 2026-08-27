"""Retry failed platform submissions in an official_test_summary.json.

The platform occasionally rate-limits ("temporarily unable to handle your
request").  This re-POSTs any ok=False entries (their submission CSVs already
exist from predict_official_test.py) with backoff, updating the summary in place.

Usage:
    python scripts/ecg/retry_official_test.py --exp_name comparison
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_official_test_all import _submit, _parse_result, _flash, ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="comparison")
    ap.add_argument("--summary", default="official_test_summary.json",
                    help="summary filename inside results/<exp_name>/")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--sleep_base", type=int, default=5,
                    help="backoff base seconds (linear: sleep_base * attempt)")
    a = ap.parse_args()

    summary_path = ROOT / "results" / a.exp_name / a.summary
    rows = json.loads(summary_path.read_text(encoding="utf-8"))

    n_retry = 0
    for rec in rows:
        if rec.get("ok"):
            continue
        model, task, lbl = rec["model"], rec["task"], rec["label"]
        src_dir = rec.get("src_dir", a.exp_name)
        csv_path = ROOT / "results" / src_dir / f"submission_{lbl.lower()}_{model}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {model}/{task}/{lbl}: csv missing {csv_path}", flush=True)
            continue
        ok = False
        for attempt in range(1, a.attempts + 1):
            html = _submit(str(csv_path))
            m = _parse_result(html)
            if m:
                rec["ok"] = True
                rec.update(m)
                rec["label"] = lbl
                rec.pop("error", None)
                print(f"[RETRY-OK] {model}/{task}/{lbl} (attempt {attempt}): "
                      f"Sens={m['sensitivity']:.4f} F1={m['f1']:.4f}", flush=True)
                ok = True
                break
            err = _flash(html)
            print(f"[RETRY-FAIL] {model}/{task}/{lbl} (attempt {attempt}): "
                  f"{err or 'parse failed'}", flush=True)
            time.sleep(a.sleep_base * attempt)  # linear backoff
        if not ok:
            rec["error"] = "retry_failed"
        n_retry += 1
        time.sleep(2)

    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"wrote {summary_path} ({n_retry} retried)", flush=True)


if __name__ == "__main__":
    main()
