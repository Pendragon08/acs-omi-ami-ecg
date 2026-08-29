# 聚合实验产物

本目录保存不含逐例标识符的可复核结果：每折训练历史、汇总指标、标签审计、
阈值分析、外部平台聚合指标、脱敏逐 epoch 训练日志和可视化图。当前任务是
ACS `OTHER/STEMI/NSTEMI/UA` 互斥四分类 + OMI/NOMI 独立二分类。

## 当前有效结果

- `corrected_joint_acs_omi_e4/`：E4 HMoE/WavelNet/KAN 五折结果；含
  `training_log_sanitized.csv`、损失/LR、ROC/PR、混淆矩阵、阈值审计和外部平台指标。
- `corrected_joint_model_comparison/<模型>/`：InceptionTime、ResNet1D-Wang、
  LSTM、ViT-1D、Transformer1D；每个目录含同样的历史、汇总、脱敏日志和图表。
- `corrected_ecg_lfm_joint_bs64_v2/`：校正 ECG-LFM Baseline/LoRA/DoRA 队列，
  当前在 GPU1 运行；完成后追加同格式文件。
- `loss_feature_ablation/`：OMI 的 Baseline/WBCE/Focal/CB-Focal 多种子损失实验，
  以及原始 ECG/年龄/性别/中位数 ECG 特征对比。

## 安全边界

原始 ECG、患者级表格、checkpoint、逐例 OOF/外部预测、提交 CSV 和原始服务器
日志不在 GitHub。`training_log_sanitized.csv` 仅包含 fold、epoch、loss、验证
指标和学习率；其导出脚本为 `scripts/export_sanitized_training_logs.py`。
报告和运行日志索引见 [`docs/reports/README.md`](../docs/reports/README.md) 与
[`docs/logs/EXPERIMENT_LOG_INDEX.md`](../docs/logs/EXPERIMENT_LOG_INDEX.md)。
