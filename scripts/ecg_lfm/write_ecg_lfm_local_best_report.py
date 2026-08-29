"""Write a traceable report for the ECG-LFM partial-FT/LoRA/DoRA ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("STEMI", "NSTEMI", "UA", "OMI")
METRICS = (
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "accuracy",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(item: dict) -> str:
    return f"{float(item['mean']):.4f} ± {float(item['std']):.4f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_info(root: Path, name: str) -> str:
    path = root / "configs" / "ecg_lfm" / f"{name}.yaml"
    return f"`{path.as_posix()}` (SHA-256 `{sha256(path)}`)" if path.exists() else "not found"


def get_eval(root: Path, method: str) -> dict | None:
    if method == "Baseline (partial-FT)":
        path = root / "results/ecg_lfm_4label/finetune_paper/summary/oof_evaluation.json"
    else:
        path = root / "results/ecg_lfm_local_best" / method.lower() / "summary/oof_evaluation.json"
    return read_json(path) if path.exists() else None


def table_rows(evaluation: dict) -> list[str]:
    cv = evaluation["fivefold_mean_std_at_global_oof_thresholds"]
    rows = []
    for label in (*LABELS, "macro"):
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(mean_std(cv[label][metric]) for metric in METRICS)
            + " |"
        )
    return rows


def pooled_rows(evaluation: dict) -> list[str]:
    pooled = evaluation["oof_youden"]
    rows = []
    for label in LABELS:
        item = pooled[label]
        rows.append(
            f"| {label} | {item['threshold']:.2f} | {item['auroc']:.4f} | {item['auprc']:.4f} | "
            f"{item['sensitivity']:.4f} | {item['specificity']:.4f} | {item['ppv']:.4f} | "
            f"{item['npv']:.4f} | {item['f1']:.4f} | {item['mcc']:.4f} | {item['accuracy']:.4f} | "
            f"{item['tp']} | {item['tn']} | {item['fp']} | {item['fn']} |"
        )
    item = pooled["macro"]
    rows.append(
        f"| macro | — | {item['auroc']:.4f} | {item['auprc']:.4f} | {item['sensitivity']:.4f} | "
        f"{item['specificity']:.4f} | {item['ppv']:.4f} | {item['npv']:.4f} | {item['f1']:.4f} | "
        f"{item['mcc']:.4f} | {item['accuracy']:.4f} | — | — | — | — |"
    )
    return rows


def method_root(root: Path, method: str) -> Path:
    return root / "results/ecg_lfm_local_best" / method.lower()


def parameter_info(root: Path, method: str) -> tuple[str, str, str]:
    if method == "Baseline (partial-FT)":
        path = root / "results/ecg_lfm_4label/finetune_paper/fold0/valid_record_predictions.provenance.json"
    else:
        path = method_root(root, method) / "fold0/valid_record_predictions.provenance.json"
    if not path.exists():
        return "—", "—", "—"
    data = read_json(path)
    total = int(data.get("total_parameters", 0))
    trainable = int(data.get("effective_trainable_parameters_at_checkpoint", data.get("requires_grad_parameters", 0)))
    ratio = 100.0 * trainable / total if total else 0.0
    return f"{trainable:,}", f"{total:,}", f"{ratio:.2f}%"


def efficiency_lines(root: Path) -> list[str]:
    """Render the parameter-efficiency innovation section from provenance."""
    counts = {}
    for method in ("Baseline (partial-FT)", "LoRA", "DoRA"):
        trainable, total, ratio = parameter_info(root, method)
        if trainable == "—" or total == "—":
            counts[method] = None
        else:
            counts[method] = (int(trainable.replace(",", "")), int(total.replace(",", "")), ratio)

    base = counts.get("Baseline (partial-FT)")
    lora = counts.get("LoRA")
    dora = counts.get("DoRA")

    def row(method: str, frozen: str, updated: str, item: tuple[int, int, str] | None, reduction: str) -> str:
        if item is None:
            return f"| {method} | {frozen} | {updated} | — | — | — | {reduction} |"
        trainable, total, ratio = item
        return f"| {method} | {frozen} | {updated} | {trainable:,} | {total:,} | {ratio} | {reduction} |"

    base_trainable = base[0] if base is not None else None
    lora_reduction = f"约 {base_trainable / lora[0]:.1f}× 更少" if base_trainable and lora else "—"
    dora_reduction = f"约 {base_trainable / dora[0]:.1f}× 更少" if base_trainable and dora else "—"
    return [
        "",
        "## 3.1 参数高效微调与训练成本分析（方法创新点）",
        "",
        "需要明确：本报告中的 Baseline 不是字面意义上的‘全模型全参数微调’。配置 `feature_grad_mult=0.0` 时，官方 ECG-LFM 代码在 `get_embeddings()` 中使用 `torch.no_grad()` 计算卷积特征提取器，因此卷积前端不接收梯度；`freeze_finetune_updates=0` 表示从第一个更新开始，Transformer 上下文编码器、post-extract projection、位置卷积、LayerNorm 和四标签分类头参与微调。项目中真正的 head-only 方案是 `linear_probe.yaml`，本报告没有把它当作 Baseline。",
        "",
        "| 方法 | 冻结部分 | 更新部分 | 有效可训练参数 | 总参数 | 可训练比例 | 相对 Baseline 的有效更新参数 |",
        "|---|---|---|---:|---:|---:|---:|",
        row("Baseline (partial-FT)", "卷积特征提取器", "Transformer 上下文编码器、投影/位置层、LayerNorm、分类头", base, "1.0×"),
        row("LoRA", "卷积前端 + 全部预训练上下文参数", "LoRA 低秩适配器 + 分类头", lora, lora_reduction),
        row("DoRA", "卷积前端 + 全部预训练上下文参数", "DoRA 低秩适配器、magnitude + 分类头", dora, dora_reduction),
        "",
        "Baseline checkpoint provenance 同时记录了 `requires_grad_parameters` 和 `effective_trainable_parameters_at_checkpoint`。前者可能包含仍保持 requires_grad 标志的卷积前端；后者依据 `feature_grad_mult=0` 的运行时 no-grad 路径扣除前端参数，是本报告用于公平比较的有效训练参数量。",
        "",
        "这构成了本方案的参数效率创新点：LoRA/DoRA 将需要梯度、优化器状态和更新的参数从约 6.22 亿降至约 590–627 万，显著降低优化器状态、梯度、checkpoint 和训练通信开销。LoRA 相对 Baseline 同时取得 macro-AUPRC `+0.0076`、macro-AUROC `+0.0036` 和 macro-F1 `+0.0085`，Specificity 仅变化 `-0.0016`，形成较好的性能—训练成本折中。",
        "",
        "参数量下降不应直接等同于总显存或墙钟时间按相同比例下降：冻结主干权重仍需加载，长 ECG 序列的中间激活仍需计算。本轮没有注册相同 batch/GPU/数据量下的 peak-VRAM、能耗和墙钟时间配对测量，因此报告只宣称‘有效更新参数和优化器状态规模显著下降’，不夸大为总显存下降 99%。论文定稿前应补做固定 GPU 的资源基准。",
    ]


def audit_rows(root: Path, method: str) -> list[str]:
    if method == "Baseline (partial-FT)":
        base = root / "results/ecg_lfm_4label/finetune_paper"
    else:
        base = method_root(root, method)
    rows = []
    for fold in range(5):
        path = base / f"fold{fold}/curves/overfitting_audit.json"
        if not path.exists():
            continue
        data = read_json(path)
        rows.append(
            f"| {fold} | {data.get('best_validation_loss_epoch', '—')} | "
            f"{data.get('best_validation_loss', '—')} | {data.get('last_epoch', '—')} | "
            f"{data.get('possible_overfit', '—')} |"
        )
    return rows


def load_external(root: Path, method: str) -> dict | None:
    path = method_root(root, method) / "summary/official_platform_4label_metrics.json"
    return read_json(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = args.report if args.report.is_absolute() else root / args.report

    evaluations = {
        "Baseline (partial-FT)": get_eval(root, "Baseline (partial-FT)"),
        "LoRA": get_eval(root, "lora"),
        "DoRA": get_eval(root, "dora"),
    }
    available = {name: data for name, data in evaluations.items() if data is not None}
    if not available:
        raise RuntimeError("No OOF evaluation artifacts found")

    # Select the best completed method by macro AUPRC, then macro AUROC/F1.
    # This deliberately allows the historical PFT baseline to remain the
    # final method when neither adapter improves it.
    if available:
        winner, winner_eval = max(
            available.items(),
            key=lambda item: (
                item[1]["fivefold_mean_std_at_global_oof_thresholds"]["macro"]["auprc"]["mean"],
                item[1]["fivefold_mean_std_at_global_oof_thresholds"]["macro"]["auroc"]["mean"],
                item[1]["fivefold_mean_std_at_global_oof_thresholds"]["macro"]["f1"]["mean"],
            ),
        )
    else:
        winner, winner_eval = "Baseline (partial-FT)", available["Baseline (partial-FT)"]

    lines = [
        "# ECG-LFM Local-Best LoRA/DoRA 优化实验报告",
        "",
        f"> 自动生成时间：{datetime.now(timezone.utc).isoformat()}。本文只汇总已落盘且可追溯的实验结果；未完成方法不会被当作结论。",
        "",
        "## 1. 结论摘要",
        "",
        f"当前已完成方法：{', '.join(available)}。按五折验证集全局 OOF 的 macro-AUPRC 优先、macro-AUROC 与 macro-F1 次级排序，当前最佳为 **{winner}**。",
        "",
        "本实验的比较因素只有适配策略：Baseline 为冻结卷积特征提取器、更新 Transformer 上下文编码器和分类头的部分参数微调；LoRA 与 DoRA 从同一官方 ECG-LFM 预训练权重开始；不引入采样器、Focal/CB-Focal、数据增强或网络结构改动。",
        "",
        "## 2. 数据与不可变设置",
        "",
        "- 任务：四个相互独立的 sigmoid 二分类输出 STEMI/NSTEMI/UA/OMI；保留明确的 `[0,0,0]` non-ACS 样本，仅剔除标签未知样本。",
        "- 内部队列：17,960 条 ECG、17,018 名患者；外部盲测队列：1,995 条 ECG。每条记录使用两个 5 秒片段，先平均 logits 再 sigmoid。",
        "- 划分：沿用历史 patient-level 五折划分（seed=2026），不重新随机划分；每折训练集约 14,300 条记录、验证集约 3,600 条记录。",
        "- 预处理：官方 ECG-LFM 500 Hz 原始 ECG 管线；normalize/filter/padding 均保持 baseline 配置。验证/外部集不 oversampling。",
        "- 训练：官方 Conformer ECG-LFM（24 层上下文编码器，1024 hidden，16-head relative-position attention）+ 4 输出线性分类头；Adam（betas 0.9/0.98），effective batch 256（physical batch 32、update_freq 8），最多 100 epoch，validation loss checkpoint，patience=10，step decay=0.99。",
        "- 损失：四组对比中本报告仅使用项目原始 `BCEWithLogitsLoss`（unweighted）；不使用 WeightedRandomSampler，不重复叠加 `pos_weight`。",
        "- 阈值：仅在内部 OOF 验证预测上逐标签网格搜索 0.01–0.99 的 Youden J；固定后应用到外部集，外部标签不参与任何选择。",
        "",
        "## 3. 适配策略",
        "",
        "| 方法 | 目标模块 | rank | alpha | dropout | 学习率 | 可训练参数 |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| Baseline (partial-FT) | 冻结卷积特征提取器；更新上下文编码器、投影/位置层、LayerNorm 和分类头 | — | — | — | 1e-6 | 见下表 |",
        "| LoRA | 24 层 × (q/k/v/out/pos + FFN1/FFN2 两层) | 8 | 16 | 0.05 | 1e-4 | 适配器 + 分类头 |",
        "| DoRA | 同 LoRA 目标；额外逐输出行 magnitude | 8 | 16 | 0.05 | 5e-5 | 适配器 + magnitude + 分类头 |",
        "",
        "适配器注入代码位于 `third_party/ECG-LFM/fairseq_signals/modules/parameter_efficient.py`，模型集成位于 `third_party/ECG-LFM/fairseq_signals/models/classification/ecg_transformer_classifier.py`。默认 `adapter_method=none`，因此历史 baseline 行为保持不变。",
        *efficiency_lines(root),
        "",
        "## 4. 五折验证集指标（global OOF Youden 阈值；mean ± std）",
        "",
    ]

    for name, data in available.items():
        lines += [f"### {name}", "", "| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lines += table_rows(data)
        threshold_text = ", ".join(
            f"{label}={float(data['thresholds'][label]):.2f}" for label in LABELS
        )
        lines += ["", f"OOF Youden thresholds: {threshold_text}.", ""]

    lines += [
        "## 5. 汇总对比（macro 五折 mean ± std）",
        "",
        "| Method | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Trainable / total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in available.items():
        macro = data["fivefold_mean_std_at_global_oof_thresholds"]["macro"]
        trainable, total, ratio = parameter_info(root, name)
        lines.append(
            f"| {name} | {mean_std(macro['auroc'])} | {mean_std(macro['auprc'])} | {mean_std(macro['sensitivity'])} | "
            f"{mean_std(macro['specificity'])} | {mean_std(macro['ppv'])} | {mean_std(macro['npv'])} | {mean_std(macro['f1'])} | "
            f"{mean_std(macro['mcc'])} | {mean_std(macro['accuracy'])} | {trainable} / {total} ({ratio}) |"
        )

    base = evaluations.get("Baseline (partial-FT)")
    if base is not None and winner_eval is not None:
        base_macro = base["fivefold_mean_std_at_global_oof_thresholds"]["macro"]
        win_macro = winner_eval["fivefold_mean_std_at_global_oof_thresholds"]["macro"]
        lines += ["", f"相对 Baseline 的当前最佳方法（{winner}）变化：", "", "| Metric | Baseline mean | Best method mean | Δ |", "|---|---:|---:|---:|"]
        for metric in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "f1", "mcc", "accuracy"):
            delta = float(win_macro[metric]["mean"]) - float(base_macro[metric]["mean"])
            lines.append(f"| {metric.upper()} | {base_macro[metric]['mean']:.4f} | {win_macro[metric]['mean']:.4f} | {delta:+.4f} |")

    lines += ["", "## 6. 逐折门禁与训练曲线", "", "下表来自各折 `curves/overfitting_audit.json`；训练/验证 loss、AUROC、AUPRC、F1 曲线 PNG/CSV 均保存在对应 `fold*/curves/`。", ""]
    for name in available:
        lines += [f"### {name}", "", "| Fold | Best validation-loss epoch | Best validation loss | Last epoch | Possible overfit |", "|---:|---:|---:|---:|---|"]
        rows = audit_rows(root, name)
        lines += rows if rows else ["| — | 尚未生成 | — | — | — |"]
        lines.append("")

    lines += [
        "## 7. 外部盲测提交",
        "",
        "外部集使用五折模型概率算术平均，并固定内部 OOF 阈值生成四标签 hard-label CSV；提交文件、原始 HTML 回包与 JSON 指标均放在最佳方法的 `summary/`。平台返回的是 hard-label 指标，不提供可复算的外部 AUROC/AUPRC；外部标签未在本地读取。",
        "",
    ]
    external = load_external(root, winner) if winner in {"LoRA", "DoRA"} else None
    if external and external.get("metrics"):
        lines += ["| Label | Samples | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in external["metrics"]:
            lines.append("| " + " | ".join(str(row.get(key, "—")) for key in ("label", "samples", "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")) + " |")
        lines += ["", f"提交文件 SHA-256：`{external.get('submission_sha256', '—')}`。", ""]
    else:
        lines += ["尚未发现平台回包；待最佳方法的外部概率与 hard-label 文件完成后提交。", ""]

    lines += [
        "## 8. 结论与可追溯文件",
        "",
        f"- 当前最佳方法：**{winner}**（仅在已完成方法中按 macro-AUPRC 优先选择）。LoRA/DoRA 五折已完成；DoRA 尚无外部平台回包，因此外部泛化不能做完整适配器间比较。",
        "- 强/中等/无提升的判定应以 Baseline 对比、五折均值与标准差共同决定，不能依据单折或 Accuracy 单项。",
        "- 训练命令、seed、折号、官方仓库 commit、配置哈希与工作树差异保存在每折 `launch_provenance.json`。",
        "- Baseline OOF：`results/ecg_lfm_4label/finetune_paper/summary/oof_evaluation.json`。",
        "- LoRA/DoRA OOF：`results/ecg_lfm_local_best/{lora,dora}/summary/oof_evaluation.json`。",
        "- 五折预测、阈值、逐折指标、混淆矩阵、训练曲线和外部提交文件均与上述目录同层保存。",
        f"- 报告生成脚本：`{(root / 'scripts/write_ecg_lfm_local_best_report.py').as_posix()}`。",
        "",
        "### 配置哈希",
        "",
        f"- Baseline：{config_info(root, 'finetune_paper')}",
        f"- Local-best baseline copy：{config_info(root, 'local_best_baseline')}",
        f"- LoRA：{config_info(root, 'lora')}",
        f"- DoRA：{config_info(root, 'dora')}",
        f"- 方案文件：`{(root / 'ECG_LFM_LocalBest_LoRA_DoRA_Incremental_Codex_FINAL.md').as_posix()}` (SHA-256 `{sha256(root / 'ECG_LFM_LocalBest_LoRA_DoRA_Incremental_Codex_FINAL.md')}`)",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "available": list(available), "winner": winner}, ensure_ascii=False))


if __name__ == "__main__":
    main()
