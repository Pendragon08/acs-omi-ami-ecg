# ECG-LFM LoRA

当前推荐方案。配置：rank=8、alpha=16、dropout=0.05、学习率 `1e-4`、物理
batch 64、`update_freq=4`。患者级五折、预处理、优化器和训练预算与 Baseline、
DoRA 一致。

五折 pooled OOF、阈值、曲线和混淆矩阵位于 `summary/`；脱敏逐 epoch 日志位于
`fold*/training_log_sanitized.csv`；外部平台聚合指标见
`official_platform_metrics.json`。
