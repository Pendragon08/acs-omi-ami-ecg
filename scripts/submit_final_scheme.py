"""Submit the final-scheme (XceptionTime + bce + P3) OMI/AMI submission CSVs to
the official platform and print the returned metric table (Sens/Spec/PPV/NPV/
Acc/F1).  Reuses the _submit/_parse_result/_flash helpers from
run_official_test_all.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_official_test_all import _submit, _parse_result, _flash  # noqa: E402

RES = ROOT / "results" / "final_fusion_focal"


def main():
    out_rows = []
    for task in ("omi", "ami"):
        csv_path = RES / f"submission_{task}_XceptionTime.csv"
        if not csv_path.exists():
            print(f"[FAIL] {task}: missing {csv_path}")
            out_rows.append({"task": task, "ok": False, "error": "missing_csv"})
            continue
        html = _submit(str(csv_path))
        metrics = _parse_result(html)
        err = _flash(html)
        if metrics:
            rec = {"task": task, "label": metrics.get("label", task.upper()),
                   "ok": True}
            rec.update({k: metrics[k] for k in
                        ("samples", "sensitivity", "specificity", "ppv",
                         "npv", "accuracy", "f1") if k in metrics})
            out_rows.append(rec)
            print(f"[OK] {task.upper()}: Sens={metrics['sensitivity']:.4f} "
                  f"Spec={metrics['specificity']:.4f} PPV={metrics['ppv']:.4f} "
                  f"NPV={metrics['npv']:.4f} Acc={metrics['accuracy']:.4f} "
                  f"F1={metrics['f1']:.4f}")
        else:
            out_rows.append({"task": task, "ok": False, "error": err or "parse_failed"})
            print(f"[FAIL] {task.upper()}: {err or 'parse_failed'}")
            print("RAW_HTML:", html[:800])

    (RES / "official_test_summary_final.json").write_text(
        json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
