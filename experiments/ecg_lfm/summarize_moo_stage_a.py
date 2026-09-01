"""Summarise MOO Stage-A inner-validation results and diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METHODS = ("linear_sum", "pcgrad", "cagrad", "fairgrad", "aligned_mtl")


def read_json_lines(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"(\{.*\})\s*$", line)
        if not match:
            continue
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def training_summary(path: Path) -> dict:
    rows = read_json_lines(path)
    train = [r for r in rows if "train_loss" in r]
    valid = [r for r in rows if "valid_loss" in r]
    summary = {"epochs_logged": len(train), "validation_epochs_logged": len(valid)}
    for prefix, source in (("train", train), ("valid", valid)):
        for key in ("loss", "acs_loss", "omi_loss", "gradient_conflict_rate", "grad_cosine_acs_omi", "grad_norm_acs_shared", "grad_norm_omi_shared", "grad_norm_joint", "effective_task_weight_acs", "effective_task_weight_omi"):
            values = [float(item[f"{prefix}_{key}"]) for item in source if f"{prefix}_{key}" in item]
            if values:
                summary[f"{prefix}_{key}_mean"] = sum(values) / len(values)
                summary[f"{prefix}_{key}_last"] = values[-1]
        free = [float(item["train_gb_free"]) for item in source if "train_gb_free" in item]
        if free:
            summary["min_reported_free_gb"] = min(free)
            summary["peak_reported_used_gb"] = 47.411 - min(free)
    full = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    matches = re.findall(r"done training in ([0-9.]+) seconds", full)
    if matches:
        summary["training_seconds"] = float(matches[-1])
    return summary


def get_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_method(method: str, metrics: dict, train: dict, w0: dict | None) -> dict:
    inner = metrics["inner_validation"]
    acs = inner["acs"]
    omi_youden = inner["omi"].get("youden_j") or inner["omi"].get("fixed_0.5")
    nstemi = inner["conditional_acs"]["per_class"]["NSTEMI"]
    values = {
        "method": method,
        "fold": metrics.get("fold", 0),
        "threshold_youden": metrics.get("thresholds", {}).get("youden_j"),
        "acs_macro_auroc": acs.get("macro_auroc_ovr"),
        "acs_macro_auprc": acs.get("macro_auprc_ovr"),
        "acs_macro_f1": acs.get("macro_f1"),
        "acs_balanced_accuracy": acs.get("balanced_accuracy"),
        "acs_accuracy": acs.get("accuracy"),
        "nstemi_auprc": nstemi.get("auprc"),
        "nstemi_f1": nstemi.get("f1"),
        "nstemi_sensitivity": nstemi.get("sensitivity"),
        "nstemi_vs_ua_auprc": inner["nstemi_vs_ua"].get("auprc"),
        "omi_auroc": omi_youden.get("auroc") if omi_youden else None,
        "omi_auprc": omi_youden.get("auprc") if omi_youden else None,
        "omi_f1": omi_youden.get("f1") if omi_youden else None,
        "omi_sensitivity": omi_youden.get("sensitivity") if omi_youden else None,
        "omi_specificity": omi_youden.get("specificity") if omi_youden else None,
        "omi_ppv": omi_youden.get("ppv") if omi_youden else None,
        "omi_npv": omi_youden.get("npv") if omi_youden else None,
        "omi_mcc": omi_youden.get("mcc") if omi_youden else None,
        "nste_omi_auprc": inner["omi_among_nste_acs"]["metrics"].get("auprc") if inner["omi_among_nste_acs"].get("metrics") else None,
        "conflict_rate": train.get("train_gradient_conflict_rate_mean"),
        "grad_cosine_mean": train.get("train_grad_cosine_acs_omi_mean"),
        "grad_norm_acs_mean": train.get("train_grad_norm_acs_shared_mean"),
        "grad_norm_omi_mean": train.get("train_grad_norm_omi_shared_mean"),
        "grad_norm_joint_mean": train.get("train_grad_norm_joint_mean"),
        "effective_weight_acs_mean": train.get("train_effective_task_weight_acs_mean"),
        "effective_weight_omi_mean": train.get("train_effective_task_weight_omi_mean"),
        "training_seconds": train.get("training_seconds"),
        "peak_reported_used_gb": train.get("peak_reported_used_gb"),
    }
    if w0 is not None:
        ref = w0["inner_validation"]
        ref_omi = ref["omi"].get("youden_j") or ref["omi"].get("fixed_0.5")
        values["delta_acs_macro_auprc"] = values["acs_macro_auprc"] - ref["acs"]["macro_auprc_ovr"]
        values["delta_omi_auprc"] = values["omi_auprc"] - ref_omi["auprc"]
        values["relative_acs_gain"] = values["delta_acs_macro_auprc"] / ref["acs"]["macro_auprc_ovr"]
        values["relative_omi_gain"] = values["delta_omi_auprc"] / ref_omi["auprc"]
        values["balanced_gain"] = min(values["relative_acs_gain"], values["relative_omi_gain"])
        values["joint_geometric_score"] = (values["acs_macro_auprc"] * values["omi_auprc"]) ** 0.5
        values["screening_gate"] = bool(
            (min(values["delta_acs_macro_auprc"], values["delta_omi_auprc"]) >= -0.003 and max(values["delta_acs_macro_auprc"], values["delta_omi_auprc"]) >= 0.005)
            or (values["delta_acs_macro_auprc"] >= 0.003 and values["delta_omi_auprc"] >= 0.003)
        )
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        methods = [r["method"] for r in rows]
        def bar(name, ylabel, filename):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(methods, [r.get(name) or 0 for r in rows])
            ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
        bar("conflict_rate", "mean conflict rate", "moo_conflict_rate.png")
        bar("grad_cosine_mean", "mean ACS–OMI gradient cosine", "moo_gradient_cosine.png")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(methods, [r.get("grad_norm_acs_mean") or 0 for r in rows], marker="o", label="ACS")
        ax.plot(methods, [r.get("grad_norm_omi_mean") or 0 for r in rows], marker="o", label="OMI")
        ax.plot(methods, [r.get("grad_norm_joint_mean") or 0 for r in rows], marker="o", label="joint")
        ax.set_ylabel("gradient norm"); ax.legend(); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out / "moo_gradient_norms.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(methods, [r.get("acs_macro_auprc") or 0 for r in rows], marker="o", label="ACS macro AUPRC")
        ax.plot(methods, [r.get("omi_auprc") or 0 for r in rows], marker="o", label="OMI AUPRC")
        ax.set_ylabel("inner-validation AUPRC"); ax.legend(); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out / "moo_validation_auprc.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(methods, [r.get("effective_weight_acs_mean") or 0 for r in rows], marker="o", label="ACS")
        ax.plot(methods, [r.get("effective_weight_omi_mean") or 0 for r in rows], marker="o", label="OMI")
        ax.set_ylabel("effective task coefficient"); ax.legend(); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out / "moo_effective_task_weights.png", dpi=160); plt.close(fig)
    except Exception as exc:
        (out / "plot_error.txt").write_text(repr(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="results/multi_objective")
    parser.add_argument("--eval_root", default="results/multi_objective_evaluation")
    parser.add_argument("--output_root", default="results/multi_objective_stage_a")
    parser.add_argument("--w0_metrics", default="results/multi_objective_evaluation/w0/inner_metrics.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    w0 = get_metrics(root / args.w0_metrics)
    rows = []
    for method in METHODS:
        metrics = get_metrics(root / args.eval_root / method / "fold0" / "inner_metrics.json")
        if metrics is None:
            continue
        train = training_summary(root / args.run_root / method / "fold0" / "fairseq_train.jsonl")
        rows.append(flatten_method(method, metrics, train, w0))
    write_csv(out / "stage_a_metrics.csv", rows)
    save_plots(out, rows)
    (out / "stage_a_summary.json").write_text(json.dumps({"w0_reference": w0, "methods": rows}, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# ECG-LFM 多目标优化 Stage A（Fold0 内部验证）", "", "> 本表只用于 inner-validation 工程筛选；没有读取 outer-test。", "", "| 方法 | ACS Macro AUPRC | OMI AUPRC | NSTEMI vs UA AUPRC | NSTEMI AUPRC | OMI MCC | 冲突率 | gate |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['acs_macro_auprc']:.6f} | {row['omi_auprc']:.6f} | {row['nstemi_vs_ua_auprc']:.6f} | {row['nstemi_auprc']:.6f} | {row['omi_mcc']:.6f} | {row.get('conflict_rate') if row.get('conflict_rate') is not None else 'NA'} | {'PASS' if row.get('screening_gate') else 'FAIL'} |")
    lines += ["", "详细数值见 `stage_a_metrics.csv` 和 `stage_a_summary.json`；诊断图见同目录 PNG。"]
    (out / "stage_a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "methods": [row["method"] for row in rows], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

