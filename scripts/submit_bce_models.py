"""Submit the 7 COMPLETED bce x P3 single-label models to the official platform.

The completed comparison runs are split across two result dirs (because the
7-model comparison reused the bce runs from the loss study):

  - loss_comparison/:  LSTM, Informer, Nonstationary_Transformer
  - model_comparison/: ResNet1D_Wang, InceptionTime, ViT, TimesNet

For each (model, task in {omi, ami}) this:
  1) generates the submission CSV(s) via predict_official_test.py
     (--loss bce --variant P3, correct per-model source dir),
  2) POSTs them to the platform,
  3) parses the returned metric table (Sens/Spec/PPV/NPV/Acc/F1),
  4) writes a single consolidated summary to results/model_comparison/.

Usage:
    python scripts/ecg/submit_bce_models.py --seed 2026
    python scripts/ecg/submit_bce_models.py --models LSTM --tasks omi --skip_submit
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_official_test_all import (  # noqa: E402
    _submit, _parse_result, _flash, _submission_entries, ROOT,
)

MODELS = ["TimesNet", "Informer", "Nonstationary_Transformer",
          "LSTM", "ResNet1D_Wang", "InceptionTime", "ViT"]

# Which result dir holds each model's bce/P3 checkpoint + JSON.  The 3 models
# whose bce run came from the loss study live in loss_comparison/; the rest
# (and any future model_comparison run) in model_comparison/.
SRC_DIR = {
    "LSTM": "loss_comparison",
    "Informer": "loss_comparison",
    "Nonstationary_Transformer": "loss_comparison",
}
DEFAULT_SRC = "model_comparison"
LOSS = "bce"
VARIANT = "P3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", default="", help="comma-separated model subset")
    ap.add_argument("--tasks", default="omi,ami", help="comma-separated tasks")
    ap.add_argument("--skip_submit", action="store_true",
                    help="only generate CSVs, do not POST to platform")
    ap.add_argument("--out_name", default="official_test_summary_bce.json")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()] if a.models else MODELS
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    rows = []
    for model in models:
        src = SRC_DIR.get(model, DEFAULT_SRC)
        for task in tasks:
            tag = f"{model}/{task}"
            pred = subprocess.run(
                [sys.executable, "scripts/ecg/predict_official_test.py",
                 "--model", model, "--task", task,
                 "--exp_name", src, "--loss", LOSS, "--variant", VARIANT,
                 "--seed", str(a.seed)],
                cwd=str(ROOT), capture_output=True, text=True)

            for lbl, fname in _submission_entries(task, model):
                sub_csv = ROOT / "results" / src / fname
                if not sub_csv.exists():
                    print(f"[FAIL] {tag}/{lbl}: predict failed: "
                          f"{pred.stderr.strip()[-200:]}", flush=True)
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": False, "error": "predict_failed",
                                 "src_dir": src})
                    continue

                if a.skip_submit:
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": True, "csv": str(sub_csv),
                                 "submitted": False, "src_dir": src})
                    print(f"[GEN] {tag}/{lbl} -> {sub_csv.name}", flush=True)
                    continue

                html = _submit(str(sub_csv))
                metrics = _parse_result(html)
                err = _flash(html)
                if metrics:
                    rec = {"model": model, "task": task, "label": lbl,
                           "ok": True, "src_dir": src}
                    rec.update(metrics)
                    rec["label"] = lbl
                    rows.append(rec)
                    print(f"[OK] {tag}/{lbl}: Sens={metrics['sensitivity']:.4f} "
                          f"Spec={metrics['specificity']:.4f} "
                          f"PPV={metrics['ppv']:.4f} NPV={metrics['npv']:.4f} "
                          f"Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}",
                          flush=True)
                else:
                    rows.append({"model": model, "task": task, "label": lbl,
                                 "ok": False, "error": err or "parse_failed",
                                 "src_dir": src})
                    print(f"[FAIL] {tag}/{lbl}: {err or 'parse failed'}", flush=True)
                time.sleep(3)  # gentle pace to avoid platform rate-limiting

    out = ROOT / "results" / DEFAULT_SRC / a.out_name
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
