# ECG-LFM Baseline

五折 pooled OOF、曲线和脱敏日志位于 `summary/` 与 `fold*/`。配置：学习率
`1e-6`、物理 batch 64、`update_freq=4`、Adam、step 调度和 patience=10。
外部平台聚合指标见 `official_platform_metrics.json`。

该方法用于与 LoRA/DoRA 公平比较；原始 ECG、checkpoint、逐例预测和提交 CSV
不进入 GitHub。
