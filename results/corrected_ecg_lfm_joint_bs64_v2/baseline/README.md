# ECG-LFM Baseline 校正联合任务

这是校正标签定义下的 ECG-LFM 全参数微调五折结果：ACS
`OTHER/STEMI/NSTEMI/UA` 互斥四分类 + 独立 OMI/NOMI 二分类。`summary/` 保存
pooled OOF 指标、阈值分析、ROC/PR、混淆矩阵和训练/验证损失-LR 曲线；每个
`fold*/training_log_sanitized.csv` 保存脱敏 epoch 训练日志。

本次配置学习率 `1e-6`、batch 64、`update_freq=4`、Adam、step 衰减和 patience=10。
外部平台 hard-label 指标见 `official_platform_4label_metrics.json`（1,995 条）。
原始 ECG、患者表格、checkpoint、逐例预测和提交 CSV 不在 GitHub。
