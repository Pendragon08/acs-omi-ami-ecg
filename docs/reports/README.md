# ACS 实验报告索引

本目录的当前主线是“ACS 四分类 + OMI 二分类”联合任务：
`OTHER / STEMI / NSTEMI / UA` 是一个互斥 ACS 分类头，OMI/NOMI 是独立二分类
头。任何把 STEMI、NSTEMI、UA 当作三个独立 sigmoid 的结果都标记为
`LEGACY/INVALID`，仅用于审计，不用于当前结论。

## 当前有效报告

| 报告 | 内容 | 状态 |
|---|---|---|
| [实验结果总报告_20260829.md](实验结果总报告_20260829.md) | 数据、划分、校正实验、外部平台和 GitHub 归档总览 | 当前总报告 |
| [corrected_task_formulation_20260829.md](corrected_task_formulation_20260829.md) | 标签审计、代码修正、重跑登记 | 规范 |
| [corrected_joint_e4_20260829.md](corrected_joint_e4_20260829.md) | E4 HMoE/WavelNet/KAN 五折联合任务 | 已完成；GPU1 + 外部平台 |
| [corrected_joint_model_comparison_20260829.md](corrected_joint_model_comparison_20260829.md) | InceptionTime、ResNet1D-Wang、LSTM、ViT、Transformer1D 对比 | 已完成；GPU1 + 外部平台 |
| [corrected_ecg_lfm_joint_20260829.md](corrected_ecg_lfm_joint_20260829.md) | ECG-LFM Baseline/LoRA/DoRA 校正重跑 | GPU1 运行中 |

## 损失函数与特征消融

| 报告 | 内容 | 状态 |
|---|---|---|
| [loss_feature_ablation/loss_ablation_summary.md](loss_feature_ablation/loss_ablation_summary.md) | Baseline、WBCE、Focal、CB-Focal，五种子与阈值策略 | 已完成 |
| [loss_feature_ablation/feature_ablation_summary.md](loss_feature_ablation/feature_ablation_summary.md) | 原始 ECG、年龄、性别、中位数 ECG 特征比较 | 已完成 |
| [loss_feature_ablation/feature_association_summary.md](loss_feature_ablation/feature_association_summary.md) | 特征与标签关联分析 | 已完成 |
| [loss_feature_ablation/EXPERIMENT_LOG.md](loss_feature_ablation/EXPERIMENT_LOG.md) | 消融实验时间线和可追溯记录 | 已完成 |
| [loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md](loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md) | 公平比较与泄漏审计 | 已完成 |

## 历史/无效报告（保留用于追溯）

以下报告使用旧的独立 sigmoid 端点，不能与当前互斥 ACS 任务混合解释：

- [ecg_lfm_local_best_lora_dora_optimization_20260828.md](ecg_lfm_local_best_lora_dora_optimization_20260828.md)
- [ecg_lfm_4label_finetuning_20260827.md](ecg_lfm_4label_finetuning_20260827.md)
- [model_comparison_4label_20260828.md](model_comparison_4label_20260828.md)
- [hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md](hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md)

其他历史 ACS/HMoE、阈值、AMI/OMI 和外部平台报告仍保留在本目录；阅读时先
确认报告中的任务定义和 `LEGACY` 标识。

## 指标和外部测试说明

报告分别给出 ACS 四分类、ACS 阳性条件三分类和 OMI 二分类指标。OMI/NSTEMI
不平衡时优先看 Sensitivity、AUPRC、F1、PPV、AUROC，并同时报告 Specificity、
NPV、MCC、Accuracy 和 TP/TN/FP/FN。只有 OMI 阈值在验证 OOF 上确定并冻结；
ACS 使用 argmax，不能产生同时预测多个亚型的结果。

外部平台的聚合返回值、提交哈希和图表位于 `results/`；逐例预测、提交 CSV、
患者级划分、checkpoint、原始波形和原始服务器日志按安全边界不进入 GitHub。
可公开的逐 epoch 脱敏日志、损失/LR、ROC/PR 和混淆矩阵已随对应结果目录
版本化。
