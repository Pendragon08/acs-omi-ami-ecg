"""Summarize complete Joint-6 nested comparisons into aggregate artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.acs_joint_task import acs_metrics, binary_metrics, nstemi_vs_ua_metrics
from experiments.ecg_lfm.evaluate_joint6_fold import joint6_metrics
from experiments.ecg_lfm.joint6 import JOINT6_CLASSES


ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "ECG-LFM LoRA": ("ecg_lfm", "lora"),
    "ECG-LFM Partial FT": ("ecg_lfm", "partial"),
    "InceptionTime": ("traditional", "InceptionTime"),
    "ResNet1D_Wang": ("traditional", "ResNet1D_Wang"),
    "LSTM": ("traditional", "LSTM"),
    "Transformer1D": ("traditional", "Transformer1D"),
    "ViT": ("traditional", "ViT"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_dir(root: Path, kind: str, key: str, fold: int) -> Path:
    if kind == "ecg_lfm":
        return root / "evaluation" / key / f"fold{fold}"
    return root / "traditional" / key / f"fold{fold}"


def fold_scalars(metrics: dict) -> dict[str, float]:
    outer = metrics["outer_test"]
    omi = outer["omi"]["youden_j"]
    nstemi = outer["acs"]["per_class"]["NSTEMI"]
    nste = outer["omi_among_nste_acs"]
    return {
        "joint6_macro_auroc": outer["joint6"]["macro_auroc_ovr"],
        "joint6_macro_auprc": outer["joint6"]["macro_auprc_ovr"],
        "joint6_macro_f1": outer["joint6"]["macro_f1"],
        "joint6_balanced_accuracy": outer["joint6"]["balanced_accuracy"],
        "joint6_accuracy": outer["joint6"]["accuracy"],
        "acs_macro_auroc": outer["acs"]["macro_auroc_ovr"],
        "acs_macro_auprc": outer["acs"]["macro_auprc_ovr"],
        "acs_macro_f1": outer["acs"]["macro_f1"],
        "acs_balanced_accuracy": outer["acs"]["balanced_accuracy"],
        "acs_accuracy": outer["acs"]["accuracy"],
        "nstemi_auprc": nstemi["auprc"],
        "nstemi_f1": nstemi["f1"],
        "nstemi_sensitivity": nstemi["sensitivity"],
        "nstemi_vs_ua_auprc": outer["nstemi_vs_ua"]["auprc"],
        "omi_auroc": omi["auroc"],
        "omi_auprc": omi["auprc"],
        "omi_f1": omi["f1"],
        "omi_sensitivity": omi["sensitivity"],
        "omi_specificity": omi["specificity"],
        "omi_ppv": omi["ppv"],
        "omi_npv": omi["npv"],
        "omi_mcc": omi["mcc"],
        "omi_accuracy": omi["accuracy"],
        "omi_threshold": omi["threshold"],
        "nste_omi_auprc": nste["auprc"],
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def duration_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("finished_at"):
        return None
    return (datetime.fromisoformat(payload["finished_at"]) - datetime.fromisoformat(payload["started_at"])).total_seconds()


def profiled_peak_memory(path: Path) -> float | None:
    """Read the maximum device-used MiB from the launcher's 30-second profile."""
    if not path.exists():
        return None
    values = []
    for row in read_csv(path):
        for key, raw in row.items():
            if key.strip() == "memory.used" and raw not in (None, ""):
                values.append(float(raw.strip()))
    return max(values) if values else None


def pooled_metrics(rows: list[dict], fold_thresholds: dict[int, float]) -> dict:
    y_joint = np.asarray([int(row["joint6_true"]) for row in rows])
    q = np.asarray([
        [float(row[f"p_{name.lower().replace('+', '_').replace('/', '_')}"]) for name in JOINT6_CLASSES]
        for row in rows
    ])
    y_acs = np.asarray([int(row["raw_acs_true"]) for row in rows])
    p_acs = np.asarray([[float(row[f"p_{name}"]) for name in ("other", "stemi", "nstemi", "ua")] for row in rows])
    y_omi = np.asarray([int(row["raw_omi_true"]) for row in rows])
    p_omi = np.asarray([float(row["p_omi"]) for row in rows])
    pred_omi = np.asarray([p_omi[index] >= fold_thresholds[int(row["fold"])] for index, row in enumerate(rows)], dtype=int)
    hard = binary_metrics(y_omi, pred_omi.astype(float), 0.5)
    hard["auroc"] = float(roc_auc_score(y_omi, p_omi))
    hard["auprc"] = float(average_precision_score(y_omi, p_omi))
    nste = np.isin(y_acs, (2, 3))
    return {
        "joint6": joint6_metrics(y_joint, q),
        "acs": acs_metrics(y_acs, p_acs),
        "omi_fold_specific_youden": hard,
        "nstemi_vs_ua": nstemi_vs_ua_metrics(y_acs, p_acs),
        "omi_among_nste_acs": {
            "n": int(nste.sum()),
            "auroc": float(roc_auc_score(y_omi[nste], p_omi[nste])),
            "auprc": float(average_precision_score(y_omi[nste], p_omi[nste])),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_ms(row: dict, metric: str) -> str:
    return f"{row[metric + '_mean']:.4f} ± {row[metric + '_std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/joint6_cv")
    parser.add_argument("--report", default="docs/ECG_LFM_Joint6_CV_Comparison_Report.md")
    args = parser.parse_args()
    root = ROOT / args.root
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    fold_rows, summary_rows, per_class_rows, cost_rows = [], [], [], []
    pooled = {}
    missing = []
    for method, (kind, key) in METHODS.items():
        metrics_by_fold = []
        prediction_rows = []
        thresholds = {}
        for fold in range(5):
            directory = result_dir(root, kind, key, fold)
            metrics_path = directory / "metrics.json"
            predictions_path = directory / "outer_test_predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                missing.append(f"{method}/fold{fold}")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            values = fold_scalars(metrics)
            fold_rows.append({"method": method, "fold": fold, **values})
            metrics_by_fold.append((fold, metrics, values))
            prediction_rows.extend(read_csv(predictions_path))
            thresholds[fold] = float(metrics["thresholds"]["youden_j"])
            for family in ("joint6", "acs"):
                for class_name, class_metrics in metrics["outer_test"][family]["per_class"].items():
                    per_class_rows.append({
                        "method": method, "fold": fold, "family": family,
                        "class": class_name,
                        **{name: class_metrics.get(name) for name in ("auroc", "auprc", "f1", "sensitivity", "specificity", "ppv", "support")},
                    })
            stats = metrics.get("model_stats", {})
            training_seconds = stats.get("training_seconds")
            peak_memory = stats.get("peak_gpu_memory_mib")
            if kind == "ecg_lfm":
                training_seconds = duration_seconds(root / "ecg_lfm" / key / f"fold{fold}" / "launch_provenance.json")
                peak_memory = profiled_peak_memory(root / "resource" / f"ecg_lfm_{key}_fold{fold}.csv")
            cost_rows.append({
                "method": method,
                "fold": fold,
                "total_parameters": stats.get("total_parameters"),
                "trainable_parameters": stats.get("trainable_parameters"),
                "trainable_percent": stats.get("trainable_percent"),
                "peak_gpu_memory_mib": peak_memory,
                "training_seconds": training_seconds,
                "best_epoch": stats.get("best_epoch"),
            })
        if len(metrics_by_fold) != 5:
            continue
        scalar_names = list(metrics_by_fold[0][2])
        summary = {"method": method, "completed_folds": 5}
        for metric in scalar_names:
            mean, std = mean_std([item[2][metric] for item in metrics_by_fold])
            summary[metric + "_mean"] = mean
            summary[metric + "_std"] = std
        summary_rows.append(summary)
        pooled[method] = pooled_metrics(prediction_rows, thresholds)

    if fold_rows:
        write_csv(output / "fold_metrics.csv", fold_rows)
    if summary_rows:
        write_csv(output / "summary_metrics.csv", summary_rows)
    if per_class_rows:
        write_csv(output / "per_class_metrics.csv", per_class_rows)
    if cost_rows:
        write_csv(output / "training_cost.csv", cost_rows)
    (output / "pooled_oof_metrics.json").write_text(json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8")
    status = {"complete": not missing, "missing": missing, "completed_methods": [row["method"] for row in summary_rows]}
    (output / "completion_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = ROOT / args.report
    if missing:
        report = "# ECG-LFM Joint-6 五折对比实验报告\n\n当前实验尚未完成。\n\n缺失运行：\n\n" + "\n".join(f"- {item}" for item in missing) + "\n"
        report_path.write_text(report, encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False))
        return

    primary_sorted = sorted(summary_rows, key=lambda row: row["joint6_macro_auprc_mean"], reverse=True)
    acs_sorted = sorted(summary_rows, key=lambda row: row["acs_macro_auprc_mean"], reverse=True)
    omi_sorted = sorted(summary_rows, key=lambda row: row["omi_auprc_mean"], reverse=True)
    table = "\n".join(
        f"| {row['method']} | {fmt_ms(row, 'joint6_macro_auprc')} | {fmt_ms(row, 'joint6_macro_f1')} | {fmt_ms(row, 'acs_macro_auprc')} | {fmt_ms(row, 'acs_macro_f1')} | {fmt_ms(row, 'acs_accuracy')} | {fmt_ms(row, 'omi_auprc')} | {fmt_ms(row, 'omi_f1')} | {fmt_ms(row, 'omi_sensitivity')} | {fmt_ms(row, 'omi_specificity')} |"
        for row in summary_rows
    )
    acs_table = "\n".join(
        f"| {row['method']} | {fmt_ms(row, 'acs_macro_auroc')} | {fmt_ms(row, 'acs_macro_auprc')} | {fmt_ms(row, 'acs_macro_f1')} | {fmt_ms(row, 'acs_balanced_accuracy')} | {fmt_ms(row, 'acs_accuracy')} |"
        for row in summary_rows
    )
    omi_table = "\n".join(
        f"| {row['method']} | {fmt_ms(row, 'omi_auroc')} | {fmt_ms(row, 'omi_auprc')} | {fmt_ms(row, 'omi_f1')} | {fmt_ms(row, 'omi_sensitivity')} | {fmt_ms(row, 'omi_specificity')} | {fmt_ms(row, 'omi_ppv')} | {fmt_ms(row, 'omi_npv')} | {fmt_ms(row, 'omi_mcc')} | {fmt_ms(row, 'omi_accuracy')} | {fmt_ms(row, 'omi_threshold')} |"
        for row in summary_rows
    )
    subgroup_table = "\n".join(
        f"| {row['method']} | {fmt_ms(row, 'nstemi_auprc')} | {fmt_ms(row, 'nstemi_f1')} | {fmt_ms(row, 'nstemi_sensitivity')} | {fmt_ms(row, 'nstemi_vs_ua_auprc')} | {fmt_ms(row, 'nste_omi_auprc')} |"
        for row in summary_rows
    )
    fold_table = "\n".join(
        f"| {row['method']} | {row['fold']} | {fmt(row['joint6_macro_auprc'])} | {fmt(row['acs_macro_auprc'])} | {fmt(row['acs_macro_f1'])} | {fmt(row['acs_accuracy'])} | {fmt(row['omi_auprc'])} | {fmt(row['omi_f1'])} | {fmt(row['omi_sensitivity'])} | {fmt(row['omi_specificity'])} |"
        for row in fold_rows
    )
    class_groups = {}
    for row in per_class_rows:
        class_groups.setdefault((row["method"], row["family"], row["class"]), []).append(row)
    class_table_rows = []
    for (method, family, class_name), rows in class_groups.items():
        cells = []
        for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity", "ppv"):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            cells.append("N/A" if not values else f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
        support = int(sum(int(row["support"]) for row in rows))
        class_table_rows.append(
            f"| {method} | {family} | {class_name} | " + " | ".join(cells) + f" | {support} |"
        )
    class_table = "\n".join(class_table_rows)
    cost_table_rows = []
    for method in METHODS:
        rows = [row for row in cost_rows if row["method"] == method]
        if not rows:
            continue
        def cost_cell(name, digits=1):
            values = [float(row[name]) for row in rows if row[name] is not None]
            if not values:
                return "N/A"
            return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"
        total = next((row["total_parameters"] for row in rows if row["total_parameters"] is not None), None)
        trainable = next((row["trainable_parameters"] for row in rows if row["trainable_parameters"] is not None), None)
        percent = next((row["trainable_percent"] for row in rows if row["trainable_percent"] is not None), None)
        percent_text = "N/A" if percent is None else f"{float(percent):.4f}"
        cost_table_rows.append(
            f"| {method} | {total if total is not None else 'N/A'} | {trainable if trainable is not None else 'N/A'} | "
            f"{percent_text} | {cost_cell('peak_gpu_memory_mib')} | {cost_cell('training_seconds')} |"
        )
    cost_table = "\n".join(cost_table_rows)
    report = f"""# ECG-LFM Joint-6 五折对比实验报告

## 1. 任务定义

Joint-6 为六个互斥输出：OTHER、STEMI+nOMI、STEMI+OMI、NSTEMI+nOMI、NSTEMI+OMI、UA。模型输出六类 Softmax，训练损失为未加权 CrossEntropyLoss。OTHER+OMI singleton 的原始 OMI 标签保留用于 OMI 主评价；UA+OMI 为 0。

## 2. 实验协议

- 数据：17,960 条有标签 ECG，17,018 名患者。
- 完全复用冻结的 patient-level nested 5-fold。
- inner-train 用于训练，inner-validation 用于早停、checkpoint 与 OMI threshold；outer-test 每折只评价一次。
- ECG-LFM 比较 LoRA 与部分参数微调；传统模型为 InceptionTime、ResNet1D_Wang、LSTM、Transformer1D、ViT。
- 所有方法均按 inner-validation Joint-6 CE 选择 checkpoint。

## 3. 五折主要结果（mean ± std）

| 方法 | Joint-6 Macro AUPRC | Joint-6 Macro F1 | ACS Macro AUPRC | ACS Macro F1 | ACS Accuracy | OMI AUPRC | OMI F1 | OMI Sensitivity | OMI Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 4. ACS 四分类完整指标（mean ± std）

| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
{acs_table}

## 5. OMI 二分类完整指标（各折 inner-validation Youden 阈值，mean ± std）

| 方法 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV | MCC | Accuracy | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{omi_table}

## 6. 临床重点子组指标（mean ± std）

| 方法 | NSTEMI AUPRC | NSTEMI F1 | NSTEMI Sensitivity | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUPRC |
|---|---:|---:|---:|---:|---:|
{subgroup_table}

## 7. 五折逐折关键指标

| 方法 | Fold | Joint-6 Macro AUPRC | ACS Macro AUPRC | ACS Macro F1 | ACS Accuracy | OMI AUPRC | OMI F1 | OMI Sensitivity | OMI Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_table}

## 8. Joint-6 与 ACS 逐类别指标（五折 mean ± std）

`support` 为五个 outer folds 的总样本数；由于五折 outer-test 恰好覆盖训练池一次，它等于完整训练池中的类别支持数。

| 方法 | 指标族 | 类别 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | support |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{class_table}

## 9. 训练成本

| 方法 | Total params | Trainable params | Trainable % | Peak GPU memory MiB | Training seconds/fold |
|---|---:|---:|---:|---:|---:|
{cost_table}

ECG-LFM 显存为 `nvidia-smi` 30 秒采样的整卡峰值；传统模型为 PyTorch allocator 峰值，两者统计口径在表中保留说明，不作过度精确的横向推断。

## 10. 简单分析

- Joint-6 Macro AUPRC 最佳：**{primary_sorted[0]['method']}**（{fmt_ms(primary_sorted[0], 'joint6_macro_auprc')}）。
- ACS Macro AUPRC 最佳：**{acs_sorted[0]['method']}**（{fmt_ms(acs_sorted[0], 'acs_macro_auprc')}）。
- OMI AUPRC 最佳：**{omi_sorted[0]['method']}**（{fmt_ms(omi_sorted[0], 'omi_auprc')}）。
- OMI 的 F1/Sensitivity/Specificity 使用各折 inner-validation 选择并冻结的 Youden threshold，未在 outer-test 调参。
- Joint-6 把 ACS 与 OMI 的依赖直接放入标签空间，但 OTHER+OMI singleton 无独立输出分支；其影响另以 sensitivity analysis 记录。

## 11. 可追溯产物

- `results/joint6_cv/summary/fold_metrics.csv`
- `results/joint6_cv/summary/summary_metrics.csv`
- `results/joint6_cv/summary/per_class_metrics.csv`
- `results/joint6_cv/summary/pooled_oof_metrics.json`
- `results/joint6_cv/summary/training_cost.csv`
- 各方法 Fold0–4 的 config、checkpoint、训练日志、预测、指标和图保存在服务器对应结果目录。

## 12. 限制

本报告比较的是当前数据支持的 observed Joint-6，不是包含责任血管定位的 Joint-10。传统模型使用项目既有 P2 输入表示，而 ECG-LFM 使用原生预处理表示；因此比较反映完整工程方案差异，不是只改变网络架构的严格单变量消融。未进行外部测试，不能把 nested outer-test 写成 external validation。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
