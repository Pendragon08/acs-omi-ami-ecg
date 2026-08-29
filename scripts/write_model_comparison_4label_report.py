"""Create the auditable markdown report for the four-label comparison.

The report is intentionally generated from JSON/CSV artifacts rather than
hand-entered numbers.  It can be run after either GPU worker completes and is
safe to re-run when additional model summaries (for example the ECG-LFM
fine-tune) arrive.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
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


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(value: object) -> str:
    if isinstance(value, dict) and "mean" in value:
        mean = float(value["mean"])
        std = float(value.get("std", float("nan")))
        if math.isnan(std):
            return "%.4f" % mean
        return "%.4f ± %.4f" % (mean, std)
    if isinstance(value, (float, int)):
        return "%.4f" % float(value)
    return "NA"


def _summary_paths(out_dir: Path) -> Iterable[Path]:
    for path in sorted(out_dir.glob("*/summary.json")):
        yield path


def _find_historical_summary() -> Optional[Path]:
    candidates = [
        ROOT / "results" / "hmoe_kan_stemi_nstemi_ua_omi_e4" / "e4_4label_summary.json",
        ROOT / "results" / "hmoe_kan_stemi_nstemi_ua_omi_e4" / "summary.json",
    ]
    candidates.extend(ROOT.glob("results/**/e4_4label_summary.json"))
    for path in candidates:
        if path.exists() and _read_json(path):
            return path
    return None


def _find_ecg_lfm_artifacts() -> list[Path]:
    paths = []
    root = ROOT / "results" / "ecg_lfm_4label"
    if root.exists():
        paths = sorted(root.rglob("*.json"))
    return paths


def _extract_mean_std(summary: dict) -> Optional[dict]:
    value = summary.get("mean_std_youden")
    if isinstance(value, dict) and "macro" in value:
        return value
    # Historical E4 used this spelling in some revisions.
    value = summary.get("mean_std")
    if isinstance(value, dict) and "macro" in value:
        return value
    return None


def _find_ecg_lfm_summary() -> Optional[tuple[Path, dict]]:
    """Return a completed ECG-LFM OOF summary when all five folds exist.

    ECG-LFM's official evaluator names the aggregate keys differently from
    the E4/architecture runner.  Normalize only the aggregate metrics here;
    single-fold ``fold_evaluation.json`` files are deliberately ignored.
    """
    root = ROOT / "results" / "ecg_lfm_4label"
    candidates = sorted(root.glob("**/oof_evaluation.json")) if root.exists() else []
    preferred = [path for path in candidates if "finetune_paper" in str(path)] or candidates
    for path in preferred:
        raw = _read_json(path)
        if not raw or not raw.get("fivefold_mean_std_at_global_oof_thresholds"):
            continue
        normalized = {
            "model": "ECG-LFM Fine-tune",
            "mean_std_youden": raw["fivefold_mean_std_at_global_oof_thresholds"],
            "recommended_threshold_method": "pooled OOF Youden J (official ECG-LFM evaluator)",
            "thresholds": {
                "youden_j": [raw.get("thresholds", {}).get(label, "NA") for label in LABELS]
            },
            "scope": raw.get("task", "official ECG-LFM record-level OOF"),
            "source_path": str(path),
        }
        return path, normalized
    return None


def _model_table_rows(summaries: Dict[str, dict]) -> list[str]:
    rows = []
    for model, summary in summaries.items():
        mean_std = _extract_mean_std(summary)
        if not mean_std:
            continue
        macro = mean_std.get("macro", {})
        rows.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                model,
                _fmt(macro.get("auroc")),
                _fmt(macro.get("auprc")),
                _fmt(macro.get("sensitivity")),
                _fmt(macro.get("specificity")),
                _fmt(macro.get("ppv")),
                _fmt(macro.get("npv")),
                _fmt(macro.get("f1")),
                _fmt(macro.get("mcc")),
                _fmt(macro.get("accuracy")),
                summary.get("recommended_threshold_method", "youden_j"),
            )
        )
    return rows


def _label_table_rows(summaries: Dict[str, dict]) -> list[str]:
    rows = []
    for model, summary in summaries.items():
        mean_std = _extract_mean_std(summary)
        if not mean_std:
            continue
        thresholds = summary.get("thresholds", {}).get("youden_j", ["NA"] * 4)
        if isinstance(thresholds, dict):
            thresholds = [thresholds.get(label, "NA") for label in LABELS]
        for index, label in enumerate(LABELS):
            metrics = mean_std.get(label, {})
            threshold = thresholds[index] if index < len(thresholds) else "NA"
            threshold_text = "%.2f" % float(threshold) if isinstance(threshold, (float, int)) else str(threshold)
            rows.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    model,
                    label,
                    threshold_text,
                    _fmt(metrics.get("auroc")),
                    _fmt(metrics.get("auprc")),
                    _fmt(metrics.get("sensitivity")),
                    _fmt(metrics.get("specificity")),
                    _fmt(metrics.get("ppv")),
                    _fmt(metrics.get("npv")),
                    _fmt(metrics.get("f1")),
                    _fmt(metrics.get("mcc")),
                    _fmt(metrics.get("accuracy")),
                )
            )
    return rows


def main() -> None:
    out_dir = ROOT / "results" / "model_comparison_4label"
    report_path = ROOT / "reports" / "model_comparison_4label_20260828.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(out_dir / "protocol.json") or {}
    label_audit = _read_json(out_dir / "label_audit.json") or {}
    split_audit = _read_json(out_dir / "split.json") or {}
    summaries: Dict[str, dict] = {}
    for path in _summary_paths(out_dir):
        summary = _read_json(path)
        if summary:
            summaries[str(summary.get("model", path.parent.name))] = summary

    historical_path = _find_historical_summary()
    if historical_path:
        historical = _read_json(historical_path)
        if historical and _extract_mean_std(historical):
            summaries.setdefault("Historical E4", historical)

    ecg_lfm_summary = _find_ecg_lfm_summary()
    if ecg_lfm_summary:
        _, normalized = ecg_lfm_summary
        summaries["ECG-LFM Fine-tune"] = normalized

    lfm_paths = _find_ecg_lfm_artifacts()
    lfm_status = "未发现本地 ECG-LFM 结果 JSON；其独立实验仍由 `reports/ecg_lfm_hyperparameter_loss_ablation_20260827.md` 记录。"
    if lfm_paths:
        lfm_status = "发现 %d 个 ECG-LFM 结果 JSON；本报告不把单折文件冒充五折均值，待完整聚合摘要后加入主表。" % len(lfm_paths)
    if ecg_lfm_summary:
        lfm_status = "已发现完整 ECG-LFM 五折 OOF 摘要并加入主表：`%s`。其官方 500 Hz 管线与 P2 传统模型不同，主表用于端到端方案比较。" % ecg_lfm_summary[0]

    counts = label_audit.get("record_label_counts", {})
    count_text = ", ".join(
        "%s=%s/%s" % (
            label,
            counts.get(label, {}).get("positive", "NA"),
            counts.get(label, {}).get("negative", "NA"),
        )
        for label in LABELS
    )
    lines = [
        "# 四标签模型对比实验结果报告（LEGACY—任务定义错误）",
        "",
        "> **INVALID FOR CURRENT TASK (2026-08-29)**：本生成器仅能汇总旧的四独立 sigmoid 结果；当前 ACS 亚型必须使用分类头。",
        "",
        "> 自动生成时间：%s（Asia/Shanghai）；所有数值均来自 `results/model_comparison_4label/` JSON/CSV 产物。" % datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "",
        "## 1. 任务与数据",
        "",
        "本任务是四个相互独立的 sigmoid 二分类输出：STEMI、NSTEMI、UA、OMI；不是互斥 softmax 四分类。明确的 subtype `[0,0,0]` non-ACS 记录保留为负样本，仅排除标签未知/非二值的记录。",
        "",
        "- 训练队列标签计数（正/负）：%s。" % count_text,
        "- 划分：历史方案的 patient-level iterative multilabel stratified five-fold，fold seed=%s；每一折患者集合不重叠，OOF 覆盖全部有效训练记录。" % protocol.get("fold_seed", split_audit.get("fold_seed", "2026")),
        "- 数据读取和 fold 内 robust scaling 复用 `experiments/hmoe_kan_e4_4label_multilabel.py`，验证集不增强、不重采样。",
        "- 本次传统模型只改变网络结构，loss 固定为当前 E4 的未加权 `BCEWithLogitsLoss`；不叠加 sampler、class weight 或 CB-Focal。",
        "",
        "## 2. 公平比较配置",
        "",
        "| 配置 | 值 |",
        "|---|---|",
        "| Optimizer | Adam |",
        "| Learning rate | %s |" % protocol.get("learning_rate", "0.001"),
        "| Batch size | %s |" % protocol.get("batch_size", "128"),
        "| Epoch / early stopping | %s / %s |" % (protocol.get("epochs_max", "100"), protocol.get("early_stopping_patience", "15")),
        "| Scheduler | none |",
        "| Train augmentation | existing train-only ECGDataset augment |",
        "| Checkpoint selection | validation macro AUROC（可由 protocol.json 溯源） |",
        "| Threshold | validation-only pooled OOF Youden J；同时保存 fixed 0.5/max-F1/Sensitivity≥0.90/0.95 |",
        "",
        "## 3. 模型来源",
        "",
        "| 模型 | 类型 | 实现/来源 |",
        "|---|---|---|",
        "| ECG-LFM Fine-tune | foundation encoder + linear head | 独立官方 ECG-LFM 适配器；结果沿用其专用报告，不与 P2 传统模型混写 |",
        "| InceptionTime | multi-scale CNN | `nets/traditional.py`，ACS-Baseline-main port |",
        "| ResNet1D_Wang | residual CNN | `nets/traditional.py`，ACS-Baseline-main port |",
        "| LSTM | two-layer recurrent model | `nets/traditional.py`，ACS-Baseline-main port |",
        "| ViT-1D | patch Transformer | `nets/traditional.py`，ACS-Baseline-main port |",
        "| Transformer1D | compact patch TransformerEncoder | 本项目脚本中的 PyTorch `nn.TransformerEncoder` |",
        "",
        "## 4. Macro 结果（五折 mean ± std）",
        "",
        "主表使用每个模型 pooled OOF 验证集上独立搜索的 Youden 阈值，并将同一阈值向量用于各折描述性 hard metrics。AUROC/AUPRC 是阈值无关指标；均值并非把患者当作独立样本重新切分。",
        "",
        "| Model | Macro AUROC | Macro AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Threshold rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = _model_table_rows(summaries)
    lines.extend(rows or ["| （传统模型尚未完成） | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | — |"])
    lines.extend([
        "",
        "## 5. 每标签结果（Youden 阈值；五折 mean ± std）",
        "",
        "| Model | Label | Threshold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(_label_table_rows(summaries) or ["| Pending | Pending | — | — | — | — | — | — | — | — | — | — |"])
    lines.extend([
        "",
        "## 6. 训练健康度与过拟合审计",
        "",
        "每个模型/折均保存 `foldN/history.json`、`history.csv`、`best_model.pt`，并生成 `training_validation_curves.png`。曲线同时记录 train/validation loss、验证 macro AUROC 和 AUPRC；early stopping 只使用注册的排序指标，不能根据 test 集挑选模型或阈值。",
        "",
        "## 7. 当前 ECG-LFM 状态",
        "",
        lfm_status,
        "ECG-LFM 采用官方 500 Hz 预处理和独立 fine-tune 训练流程，与本次传统模型使用的 P2 100 Hz 输入不同。因此，报告中若出现 ECG-LFM 行，应解释为端到端方案比较，不应声称是严格的单变量 architecture-only ablation。",
        "",
        "## 8. 可追溯产物",
        "",
        "- 实验协议：`results/model_comparison_4label/protocol.json`",
        "- 标签审计：`results/model_comparison_4label/label_audit.json`",
        "- 患者级切分：`results/model_comparison_4label/split.json`",
        "- 每模型摘要：`results/model_comparison_4label/<model>/summary.json`",
        "- OOF 概率/预测：`results/model_comparison_4label/<model>/oof_probabilities.npz`、`oof_predictions.csv`",
        "- 每折指标与曲线：`fold_metrics_youden.csv`、`foldN/history.csv`、`training_validation_curves.png`",
        "- 运行脚本：`experiments/model_comparison_4label.py`",
        "",
        "## 9. 结论规则",
        "",
        "最终优胜模型不能只按 Accuracy 判断。优先比较 macro AUPRC、OMI/NSTEMI 的 Sensitivity、F1 和 PPV，同时检查 Specificity 是否明显下降，并要求五折大多数折方向一致。若 ECG-LFM 只有单折或不同数据管线结果，不将其写成已证实的五折提升。",
        "",
        "**状态：** 本报告会随着 GPU worker 产生新的 `summary.json` 自动重写；在全部模型和 ECG-LFM 五折摘要到齐前，Pending 行不是实验失败，也不是人为填入的估计值。",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(report_path))


if __name__ == "__main__":
    main()
