# ECG-LFM DoRA

配置：rank=8、alpha=16、dropout=0.05、学习率 `5e-5`。五折 pooled OOF、曲线、
混淆矩阵和阈值位于 `summary/`，脱敏日志位于 `fold*/`，外部平台聚合指标见
`official_platform_metrics.json`。

fold3 曾中断后恢复，fold4 在 GPU1 并行完成；归档 checkpoint 哈希已核对。
DoRA 在部分工作点略优，但总体排序指标和训练成本未稳定超过 LoRA，因此作为
适配器消融保留。
