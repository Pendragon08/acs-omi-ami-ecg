# ECG-LFM LoRA 校正联合任务

这是当前校正标签定义下的 LoRA 五折结果：ACS `OTHER/STEMI/NSTEMI/UA` 互斥四
分类 + 独立 OMI/NOMI 二分类。`summary/summary.json` 和
`summary/threshold_analysis.json` 是 pooled OOF 指标；`summary/` 同时保存
ROC/PR、混淆矩阵和训练/验证损失-LR 曲线。

每个 `fold*/training_log_sanitized.csv` 只包含 epoch、训练/验证 loss、验证准确率
和学习率。外部平台结果见 `official_platform_4label_metrics.json`（1,995 条，
平台返回的 hard-label Sensitivity/Specificity/PPV/NPV/F1/Accuracy）。

本次配置为 rank=8、alpha=16、dropout=0.05、学习率 `1e-4`、batch 64、
`update_freq=4`；历史患者级五折和预处理与 Baseline/DoRA 相同。原始 ECG、患者
表格、checkpoint、逐例预测和提交 CSV 不在 GitHub。
