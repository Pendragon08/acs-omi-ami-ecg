"""Create the aggregate-only Phase-1 corrected evaluation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "lora", "dora")


def _f(value) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/revalidated")
    parser.add_argument("--report", default="docs/ECG_LFM_corrected_evaluation_report.md")
    args = parser.parse_args()
    result_root = ROOT / args.result_root
    metrics = {
        method: json.loads((result_root / method / "metrics.json").read_text(encoding="utf-8"))
        for method in METHODS
    }
    alignment = {
        method: json.loads(
            (result_root / method / "provenance_alignment_report.json").read_text(encoding="utf-8")
        )
        for method in METHODS
    }
    if any(item.get("status") != "pass" for item in alignment.values()):
        raise RuntimeError("Phase-1 alignment has not passed for every method")

    lines = [
        "# ECG-LFM 修正评估报告",
        "",
        "> 本报告由现有 Baseline/LoRA/DoRA 共 15 个 Fold0–4 checkpoint 重新推理生成；没有重新训练。",
        "> 当前仍使用 outer validation 选择 checkpoint，并在同一 pooled OOF 选择阈值，故 hard metrics 是描述性 OOF；nested validation 从下一阶段开始。",
        "",
        "## 1. Provenance 与对齐门禁",
        "",
        "| 方法 | record-ID join | 重复预测 | 缺失预测 | 缺失标签 | patient overlap | fold coverage | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        item = alignment[method]
        lines.append(
            f"| {method} | {item['join_key']} | {item['duplicate_prediction_record_ids']} | "
            f"{item['missing_predictions']} | {item['missing_labels']} | {item['patient_overlap']} | "
            f"{item['fold_coverage_fraction']:.1%} | {item['status']} |"
        )
    lines += [
        "",
        "标签由 prediction `record_id` 与 `master_records.csv`、原始 `train.csv` 显式连接；numeric master index 只作交叉核对，不再作为唯一真值索引。逐例 CSV 含 patient_id，仅保存在受控执行服务器并被 Git 忽略。",
        "",
        "## 2. 全队列 ACS 四状态 pooled OOF",
        "",
        "| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]["acs"]
        lines.append(
            f"| {method} | {_f(item['macro_auroc_ovr'])} | {_f(item['macro_auprc_ovr'])} | "
            f"{_f(item['macro_f1'])} | {_f(item['balanced_accuracy'])} | {_f(item['accuracy'])} |"
        )
    lines += [
        "",
        "### 2.1 ACS 逐类",
        "",
        "| 方法 | 类别 | AUROC | AUPRC | F1 | Sensitivity | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for name, item in metrics[method]["acs"]["per_class"].items():
            lines.append(
                f"| {method} | {name} | {_f(item['auroc'])} | {_f(item['auprc'])} | "
                f"{_f(item['f1'])} | {_f(item['sensitivity'])} | {_f(item['specificity'])} |"
            )
    lines += [
        "",
        "## 3. Conditional ACS（真实 ACS 病例）",
        "",
        "连续概率严格按 `P(k|ACS)=P(k)/(P(STEMI)+P(NSTEMI)+P(UA))` 计算。",
        "",
        "| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]["conditional_acs"]
        lines.append(
            f"| {method} | {_f(item['macro_auroc_ovr'])} | {_f(item['macro_auprc_ovr'])} | "
            f"{_f(item['macro_f1'])} | {_f(item['balanced_accuracy'])} | {_f(item['accuracy'])} |"
        )
    lines += [
        "",
        "## 4. NSTEMI vs UA",
        "",
        "NSTEMI 为 positive class，使用连续 `p_NSTEMI/(p_NSTEMI+p_UA)`。",
        "",
        "| 方法 | AUROC | AUPRC | F1 | Balanced Accuracy | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]["nstemi_vs_ua"]
        lines.append(
            f"| {method} | {_f(item['auroc'])} | {_f(item['auprc'])} | {_f(item['f1'])} | "
            f"{_f(item['balanced_accuracy'])} | {_f(item['sensitivity'])} | {_f(item['specificity'])} |"
        )
    lines += [
        "",
        "## 5. OMI pooled OOF（全队列 Youden 工作点）",
        "",
        "| 方法 | AUROC | AUPRC | Threshold | F1 | Sensitivity | Specificity | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]["omi"]["youden_j"]
        lines.append(
            f"| {method} | {_f(item['auroc'])} | {_f(item['auprc'])} | {_f(item['threshold'])} | "
            f"{_f(item['f1'])} | {_f(item['sensitivity'])} | {_f(item['specificity'])} | "
            f"{_f(item['ppv'])} | {_f(item['npv'])} |"
        )
    lines += [
        "",
        "## 6. OMI among NSTE-ACS（NSTEMI + UA）",
        "",
        "使用全队列 pooled OOF Youden threshold，不在子集上再次调阈值。",
        "",
        "| 方法 | N | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        group = metrics[method]["omi_among_nste_acs"]
        item = group["metrics"]
        lines.append(
            f"| {method} | {group['n']} | {_f(item['auroc'])} | {_f(item['auprc'])} | "
            f"{_f(item['f1'])} | {_f(item['sensitivity'])} | {_f(item['specificity'])} | "
            f"{_f(item['ppv'])} | {_f(item['npv'])} |"
        )
    lines += [
        "",
        "## 7. 未校准概率的 calibration audit",
        "",
        "当前只测量 calibration，没有拟合 temperature/Platt/isotonic/beta 模型。",
        "",
        "| 方法 | ACS multiclass Brier | ACS top-label ECE | OMI Brier | OMI ECE | OMI slope | OMI intercept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        acs = metrics[method]["calibration"]["acs"]
        omi = metrics[method]["calibration"]["omi"]
        lines.append(
            f"| {method} | {_f(acs['multiclass_brier'])} | {_f(acs['top_label_ece'])} | "
            f"{_f(omi['brier'])} | {_f(omi['ece'])} | {_f(omi['calibration_slope'])} | "
            f"{_f(omi['calibration_intercept'])} |"
        )
    lines += [
        "",
        "## 8. Phase-1 结论与边界",
        "",
        "- 只有三个方法的 alignment status 都为 `pass`，才允许进入 nested split 阶段。",
        "- 本阶段修复 evaluator/provenance，但没有消除旧 checkpoint 由 outer validation 选择造成的乐观偏差；新训练必须使用 inner validation。",
        "- 现有平台集已经参与工程选型，本报告不把它当作 untouched confirmatory test，也不重新提交平台。",
        "- 本阶段没有修改 Loss，没有实现 RDR，更没有实现 RL。",
        "",
        "## 9. Source hashes",
        "",
    ]
    for method in METHODS:
        item = alignment[method]
        lines.append(
            f"- `{method}`: evaluator `{item['evaluator_sha256']}`, task metrics "
            f"`{item['task_metrics_sha256']}`, model source `{item['model_source_sha256']}`."
        )
    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase": "evaluation_repair",
        "status": "pass",
        "methods": list(METHODS),
        "alignment": {method: alignment[method]["status"] for method in METHODS},
        "report": str(report),
    }
    (result_root / "revalidation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
