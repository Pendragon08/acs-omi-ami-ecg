# 传统架构联合任务对比结果

本目录保存 InceptionTime、ResNet1D-Wang、LSTM、ViT-1D 和 Transformer1D
在同一患者级五折、预处理、优化器和训练预算下的结果。每个模型目录包含汇总
JSON、脱敏训练日志、损失/LR、ROC/PR、混淆矩阵和
`official_platform_metrics.json`。

ACS 阳性亚型按单标签三分类解释；全队列内部保留 `OTHER` 状态；OMI 独立二分类。
总对比见 `docs/reports/实验结果总报告_最终.md`。
