# 聚合实验产物

本目录只保存不含逐例标识符的汇总结果，例如每折训练历史、指标汇总、标签审计、阈值分析和外部平台聚合指标。

逐例 OOF 预测、测试预测、提交 CSV、患者级 split、checkpoint 和原始波形均不在默认 GitHub 发布包中。它们的路径与 SHA-256 可在 docs/reports 的 provenance 文件中审计。

## 当前结果族

- `ecg_lfm_4label/finetune_paper/summary/`：ECG-LFM Baseline（冻结卷积前端、更新上下文编码器与分类头）的四标签五折 OOF、阈值、外部平台和比较汇总。
- `ecg_lfm_local_best/lora/summary/` 与 `dora/summary/`：同一历史患者级划分上的 LoRA/DoRA 聚合结果；LoRA 的外部平台指标也在其目录中。
- `loss_feature_ablation/`：OMI 的 Baseline/WBCE/Focal/CB-Focal 多种子损失实验、原始 ECG/年龄/性别/中位数 ECG 特征实验、阈值网格和统计比较。
- `model_comparison_4label/`：InceptionTime、ResNet1D-Wang、LSTM、ViT-1D、Transformer1D 与 ECG-LFM 的逐标签聚合指标及外部平台返回指标。

每个结果族都只保留聚合 CSV/JSON/Markdown 和必要的图；文件名带有 `prediction`、`submission`、`split` 或模型权重的逐例/大文件已按仓库发布边界排除。原始运行日志的映射见 [`docs/logs/EXPERIMENT_LOG_INDEX.md`](../docs/logs/EXPERIMENT_LOG_INDEX.md)。
