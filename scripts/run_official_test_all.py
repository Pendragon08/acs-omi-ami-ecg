"""Run Official Test inference + platform submission for all runs.

For each (model, task), generates the submission CSV(s) (via
predict_official_test.py), POSTs them to the platform, parses the returned
metric table, and writes results/<exp_name>/official_test_summary.json.

Single-label usage (after run_comparison.py):
    python scripts/ecg/run_official_test_all.py --exp_name comparison --seed 2026

Multi-label usage (after run_comparison_multilabel.py) -- submits both OMI & AMI:
    python scripts/ecg/run_official_test_all.py --exp_name comparison_ml --seed 2026 --tasks multilabel
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["TimesNet", "Informer", "Nonstationary_Transformer",
          "LSTM", "ResNet1D_Wang", "InceptionTime", "ViT"]
TASK_LABEL = {"omi": "OMI", "ami": "AMI"}
PLATFORM = "http://39.105.59.221:8080/"

RESULT_HEADERS = ["label", "samples", "sensitivity", "specificity",
                  "ppv", "npv", "accuracy", "f1"]


def _submission_entries(task, model):
    """Return [(label_name, submission_csv_filename), ...] for a task."""
    if task == "multilabel":
        return [("OMI", f"submission_omi_{model}.csv"),
                ("AMI", f"submission_ami_{model}.csv")]
    return [(TASK_LABEL[task], f"submission_{task}_{model}.csv")]


def _submit(csv_path):
    r = subprocess.run(
        ["curl", "-s", "-m", "30", "-L", "-c", "/tmp/acs_cj.txt",
         "-F", f"file=@{csv_path}", PLATFORM + "submit"],
        capture_output=True, text=True)
    return r.stdout


def _parse_result(html):
    """Extract the metric table row (Label, Samples, Sens, Spec, PPV, NPV, Acc, F1)."""
    m = re.search(r"<th>Label</th>.*?</tr>(.*?)</table>", html, re.S)
    if not m:
        return None
    cells = re.findall(r"<td>([^<]*)</td>", m.group(1))
    if len(cells) < len(RESULT_HEADERS):
        return None
    row = {}
    for name, c in zip(RESULT_HEADERS, cells):
        s = c.strip()
        try:
            row[name] = float(s)
        except ValueError:
            row[name] = s
    return row


def _flash(html):
    m = re.search(r'<li class="flash-message[^"]*">(.*?)</li>', html, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="comparison")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", default="", help="comma-separated model subset")
    ap.add_argument("--tasks", default="omi,ami",
                    help="comma-separated tasks: omi,ami (default) or multilabel")
    ap.add_argument("--skip_submit", action="store_true",
                    help="only generate CSVs, do not POST to platform")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()] if a.models else MODELS
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    rows = []
    for model in models:
        for task in tasks:
            tag = f"{model}/{task}"
            # 1) generate submission CSV(s)
            pred = subprocess.run(
                [sys.executable, "scripts/ecg/predict_official_test.py",
                 "--model", model, "--task", task,
                 "--exp_name", a.exp_name, "--seed", str(a.seed)],
                cwd=str(ROOT), capture_output=True, text=True)

            for lbl, fname in _submission_entries(task, model):
                sub_csv = ROOT / "results" / a.exp_name / fname
                if not sub_csv.exists():
                    print(f"[FAIL] {tag}/{lbl}: predict failed: "
                          f"{pred.stderr.strip()[-200:]}", flush=True)
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": False, "error": "predict_failed"})
                    continue

                if a.skip_submit:
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": True, "csv": str(sub_csv),
                                 "submitted": False})
                    print(f"[GEN] {tag}/{lbl} -> {sub_csv.name}", flush=True)
                    continue

                # 2) submit to platform
                html = _submit(sub_csv)
                metrics = _parse_result(html)
                err = _flash(html)
                if metrics:
                    rec = {"model": model, "task": task, "label": lbl, "ok": True}
                    rec.update(metrics)      # sens/spec/ppv/npv/acc/f1 (+ parsed label)
                    rec["label"] = lbl       # authoritative label
                    rows.append(rec)
                    print(f"[OK] {tag}/{lbl}: Sens={metrics['sensitivity']:.4f} "
                          f"Spec={metrics['specificity']:.4f} "
                          f"PPV={metrics['ppv']:.4f} NPV={metrics['npv']:.4f} "
                          f"Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}",
                          flush=True)
                else:
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": False, "error": err or "parse_failed"})
                    print(f"[FAIL] {tag}/{lbl}: {err or 'parse failed'}", flush=True)
                time.sleep(3)  # gentler pace to avoid platform rate-limiting

    out = ROOT / "results" / a.exp_name / "official_test_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
