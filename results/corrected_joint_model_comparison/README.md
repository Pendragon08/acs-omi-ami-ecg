# 校正联合任务：传统架构对比产物

任务定义为 ACS `OTHER/STEMI/NSTEMI/UA` 互斥四分类 + OMI/NOMI 独立二分类。
每个模型使用相同的历史患者级五折划分、训练预处理、优化器、学习率、batch、
调度器和早停设置；OMI 阈值在验证 OOF 上按 Youden 指数确定，ACS 使用 argmax。

模型目录：`InceptionTime/`、`ResNet1D_Wang/`、`LSTM/`、`ViT/`、
`Transformer1D/`。每个目录包含：

- `history_fold*.json`、`fold_metrics.json`、`summary.json`；
- `training_curves.csv` 和 `training_log_sanitized.csv`（逐 epoch loss/LR/验证指标）；
- `training_validation_loss_lr.png`、`roc_pr_curves.png`、`confusion_matrices.png`；
- 外部平台聚合指标 JSON 与预测 provenance/hash（不含逐例预测）。

原始 ECG、患者级表格、checkpoint、逐例 OOF/外部预测、提交 CSV 和原始服务器
日志按安全边界保留在受控执行环境，不提交到 GitHub。总报告见
`docs/reports/实验结果总报告_20260829.md`。
