"""Update the traceable Markdown report from frozen experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")
START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def mean_std(values: dict) -> str:
    return f"{values['mean']:.4f} ± {values['std']:.4f}"


def run_config(run_name: str) -> str:
    configs = {
        "finetune_paper": "LR=1e-6；physical batch=32；accumulation=8；effective batch=256；unweighted BCE",
        "lr3e7_bce_b256": "LR=3e-7；physical batch=32；accumulation=8；effective batch=256；unweighted BCE",
        "lr3e6_bce_b256": "LR=3e-6；physical batch=32；accumulation=8；effective batch=256；unweighted BCE",
        "lr1e6_bce_b128": "LR=1e-6；physical batch=32；accumulation=4；effective batch=128；unweighted BCE",
        "lr1e6_bce_b512": "LR=1e-6；physical batch=32；accumulation=16；effective batch=512；unweighted BCE",
        "lr3e7_wbce_b256": "LR=3e-7；physical batch=32；accumulation=8；effective batch=256；training-fold Weighted BCE",
        "lr1e6_wbce_b256": "LR=1e-6；physical batch=32；accumulation=8；effective batch=256；training-fold Weighted BCE",
        "lr3e6_wbce_b256": "LR=3e-6；physical batch=32；accumulation=8；effective batch=256；training-fold Weighted BCE",
    }
    return configs.get(run_name, run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--final_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, default=Path("results/ecg_lfm_4label/finetune_paper/summary"))
    parser.add_argument("--fold0_table", type=Path, default=Path("results/ecg_lfm_4label_tuning/summary/fold0_comparison.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/ecg_lfm_hyperparameter_loss_ablation_20260827.md"))
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    final_eval = json.loads((args.final_dir / "oof_evaluation.json").read_text(encoding="utf-8"))
    baseline_eval = None
    if (args.baseline_dir / "oof_evaluation.json").exists():
        baseline_eval = json.loads((args.baseline_dir / "oof_evaluation.json").read_text(encoding="utf-8"))
    winner = selection["selected_final"]
    run_root = args.final_dir.parent
    provenance = json.loads((run_root / "fold0" / "launch_provenance.json").read_text(encoding="utf-8"))

    lines = [START, "", "## 11. 自动更新的最终实验结果", ""]
    lines += [f"最终内部方案：**`{winner}`**。", "", f"固定配置：{run_config(winner)}。", ""]
    lines += [
        "选择依据为内部 patient-level validation：Fold 0 用于筛选，入选的普通 BCE 与 Weighted BCE 在 Fold 1–2 复核；最终按三折 mean macro-AUPRC 优先，依次参考 NSTEMI/OMI AUPRC、macro-AUROC、F1、Specificity 和 PPV。外部测试指标未参与选择。",
        "",
        "### 11.1 Fold 0 全候选筛选",
        "",
        "| Rank | Run | Loss | Macro AUROC | Macro AUPRC | ΔAUPRC vs baseline | Sensitivity | Specificity | PPV | F1 | Best epoch | Overfit |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if args.fold0_table.exists():
        with args.fold0_table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                lines.append(
                    f"| {row['rank']} | {row['run_name']} | {row['loss']} | {fmt(row['macro_auroc'])} | "
                    f"{fmt(row['macro_auprc'])} | {fmt(row['delta_macro_auprc_vs_baseline'])} | "
                    f"{fmt(row['macro_sensitivity'])} | {fmt(row['macro_specificity'])} | {fmt(row['macro_ppv'])} | "
                    f"{fmt(row['macro_f1'])} | {row['best_validation_loss_epoch']} | {row['possible_overfit']} |"
                )
    lines += ["", "### 11.2 三折确认与最终锁定", ""]
    lines += [
        "| Rank | Run | Loss | Macro AUROC | Macro AUPRC | Minority AUPRC | Sensitivity | Specificity | PPV | F1 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(selection["ranking"], 1):
        def ms(metric: str) -> str:
            return f"{row[f'macro_{metric}_mean']:.4f} ± {row[f'macro_{metric}_std']:.4f}"
        lines.append(
            f"| {rank} | {row['run_name']} | {row['loss']} | {ms('auroc')} | {ms('auprc')} | "
            f"{row['minority_auprc_mean']:.4f} | {ms('sensitivity')} | {ms('specificity')} | "
            f"{ms('ppv')} | {ms('f1')} |"
        )

    lines += ["", "### 11.3 最终五折内部 OOF 指标", ""]
    cv = final_eval["fivefold_mean_std_at_global_oof_thresholds"]
    lines += [
        "| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (*LABELS, "macro"):
        lines.append("| " + label + " | " + " | ".join(mean_std(cv[label][metric]) for metric in METRICS) + " |")
    lines += ["", "最终 OOF Youden 阈值：" + "；".join(f"{label}={final_eval['thresholds'][label]:.2f}" for label in LABELS) + "。", ""]
    pooled = final_eval["oof_youden"]
    lines += [
        "Pooled OOF confusion matrix：",
        "",
        "| Label | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        item = pooled[label]
        lines.append(f"| {label} | {item['tp']} | {item['tn']} | {item['fp']} | {item['fn']} |")

    lines += ["", "### 11.4 训练曲线与过拟合审计", ""]
    lines += ["| Fold | Best validation-loss epoch | Best validation loss | Last epoch | Overfit flag |", "|---:|---:|---:|---:|---|"]
    audits = []
    for fold in range(5):
        path = run_root / f"fold{fold}" / "curves" / "overfitting_audit.json"
        if not path.exists():
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        audits.append(audit)
        lines.append(
            f"| {fold} | {audit.get('best_validation_loss_epoch')} | {audit.get('best_validation_loss')} | "
            f"{audit.get('last_epoch')} | {audit.get('possible_overfit')} |"
        )
    lines += [
        "",
        "Checkpoint 始终按最小 validation loss 保存，early stopping patience=10；AUROC/AUPRC/F1 曲线只用于诊断，不反向挑选 checkpoint。每折 PNG、CSV 和 JSON 审计文件均保存在对应 `fold*/curves/`。",
        "",
    ]

    lines += ["### 11.5 与原 baseline 的内部差值", ""]
    if baseline_eval is None:
        lines += ["Baseline 完整五折尚未生成，差值待补。", ""]
    else:
        base_cv = baseline_eval["fivefold_mean_std_at_global_oof_thresholds"]["macro"]
        final_cv = cv["macro"]
        lines += ["| Metric | Baseline | Final | Δ Final−Baseline |", "|---|---:|---:|---:|"]
        for metric in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "f1"):
            delta = final_cv[metric]["mean"] - base_cv[metric]["mean"]
            lines.append(f"| {metric.upper()} | {mean_std(base_cv[metric])} | {mean_std(final_cv[metric])} | {delta:+.4f} |")
        lines.append("")

    lines += ["### 11.6 外部测试平台", ""]
    external_path = args.final_dir / "official_platform_4label_metrics.json"
    if external_path.exists():
        external = json.loads(external_path.read_text(encoding="utf-8"))
        lines += [
            f"提交文件 SHA-256：`{external['submission_sha256']}`。平台原始响应保存在 `official_platform_4label_response.html`；测试标签未在本地读取。",
            "",
            "| Label | Samples | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in external["metrics"]:
            lines.append(
                f"| {row['label']} | {row['samples']} | {row['sensitivity']} | {row['specificity']} | "
                f"{row['ppv']} | {row['npv']} | {row['accuracy']} | {row['f1']} |"
            )
        lines += ["", "平台只返回阈值化 hard-label 指标，未返回连续概率，因此不报告或臆造外部 AUROC/AUPRC。", ""]
    else:
        lines += ["最终方案已锁定，外部平台提交尚未完成。", ""]

    loss = provenance.get("loss")
    lines += ["### 11.7 结果分析", ""]
    if loss == "weighted_bce":
        lines.append("最终入选 Weighted BCE，说明按各训练折 `N_negative/N_positive` 计算正类权重的方案在多折验证中优于普通 BCE；结论基于多折平均而非单折最高值。")
    else:
        lines.append("最终未入选 Weighted BCE；类别加权没有在预注册的多折规则下稳定超过普通 BCE，因此不把它描述为有效创新点，也不继续根据外部结果强行调权。")
    if audits:
        flagged = sum(bool(item.get("possible_overfit")) for item in audits)
        lines.append(f"五折中自动曲线审计标记 {flagged}/{len(audits)} 折存在训练损失继续下降而验证损失恶化的信号；strict-best checkpoint 与 early stopping 已限制该风险。")
    lines += [
        "Sensitivity、Specificity、PPV 和 F1 均在 OOF Youden 阈值下解释；模型优先选择指标是 macro-AUPRC，因为四标签尤其 NSTEMI/OMI 类别不平衡，Accuracy 仅作为辅助指标。",
        "",
        "### 11.8 可溯源产物",
        "",
        f"- 最终运行目录：`{run_root.as_posix()}`",
        f"- 三折选择记录：`{args.selection.as_posix()}`",
        f"- 五折 OOF：`{(args.final_dir / 'oof_evaluation.json').as_posix()}`",
        f"- 外部提交目录：`{args.final_dir.as_posix()}`",
        f"- Fold 0 启动 provenance：`{(run_root / 'fold0' / 'launch_provenance.json').as_posix()}`",
        "",
        END,
    ]
    generated = "\n".join(lines)
    text = args.report.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("Report auto-results markers are missing")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    args.report.write_text(before + generated + after, encoding="utf-8")
    print(json.dumps({"winner": winner, "report": str(args.report), "external_present": external_path.exists()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
