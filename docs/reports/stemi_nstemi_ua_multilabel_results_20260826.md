---
experiment_id: ACS-STEMI-NSTEMI-UA-20260825-GPU0
date: 2026-08-25
status: completed
task: independent multi-label classification of STEMI, NSTEMI and UA
model: XceptionTime
loss: BCEWithLogitsLoss
preprocessing: P3
seed: 2026
device: NVIDIA RTX A6000 GPU0
---

# STEMI / NSTEMI / UA 多标签分类实验结果

## 1. 结论摘要

本次三标签任务已经完成训练、验证集阈值选择、外部测试集提交和平台盲评。模型对三个标签分别输出一个 sigmoid 概率，三个标签之间相互独立，因此这是**多标签**评价，不是 softmax 三分类。

- 验证集宏平均：AUROC **0.7896**，AUPRC **0.4765**，F1 **0.5138**。
- 外部平台盲评逐标签结果：STEMI F1 **0.6647**，NSTEMI F1 **0.3045**，UA F1 **0.5619**。
- 平台返回结果的逐标签宏平均：Sensitivity **0.6407**，Specificity **0.7473**，PPV **0.4411**，NPV **0.9260**，Accuracy **0.7891**，F1 **0.5104**。
- 平台按标签分别计算 one-vs-rest 指标，没有返回 AUROC/AUPRC、micro-F1、exact-match accuracy 或三标签联合混淆矩阵。因此 `0.7891` 只能称为“逐标签 Accuracy 的算术宏平均”，不能称为单一的三分类 Accuracy。

## 2. 数据与划分

- 训练记录：17,960 条；唯一患者：17,018 人。
- 外部 `test.csv`：1,995 条，标签由平台隐藏。
- 输入：12 导联 ECG，采样率 100 Hz，长度 1,000，张量形状 `[N, 1000, 12]`。
- P3：使用项目既有 P2 预处理（0.5–40 Hz 带通及既有标准化），训练集启用在线增强；验证集和外部测试集不增强。
- 历史 patient-level split 保持不变：训练 16,174 条、验证 1,786 条；患者集合互斥。外部 `test.csv` 单独作为最终盲评集。
- 源训练 CSV 的三标签阳性数量：STEMI 1,442（8.03%）、NSTEMI 1,235（6.88%）、UA 6,213（34.59%）。

## 3. 固定训练配置

| 项目 | 配置 |
|---|---|
| 模型 | XceptionTime |
| 输出 | 3 个独立 logits，分别对应 STEMI/NSTEMI/UA |
| 损失 | 原始 `BCEWithLogitsLoss`，无额外 class weight |
| 优化器 | Adam |
| 学习率 | `3e-4` |
| batch size | `256` |
| 最大 epoch | `100` |
| early stopping | patience `15` |
| 梯度裁剪 | `1.0` |
| random seed | `2026` |
| 可训练参数 | `93,379` |
| 最佳 epoch | `20` |
| 训练时间 | `238.05 s` |
| 峰值显存 | `4,956.61 MiB` |

阈值只在验证集按每个标签的 F1 最大化确定，然后原样固定到外部测试集：STEMI `0.30`、NSTEMI `0.33`、UA `0.16`。

## 4. 验证集结果

下表使用固定的验证集阈值；括号内的 TP/TN/FP/FN 是由验证集计数和灵敏度/特异度还原的整数混淆矩阵。

| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Threshold | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9256 | 0.6696 | 0.6748 | 0.9681 | 0.6103 | 0.9758 | 0.6409 | 0.6138 | 0.9479 | 0.30 | 83 | 1610 | 53 | 40 |
| NSTEMI | 0.7839 | 0.2912 | 0.3583 | 0.9442 | 0.3162 | 0.9533 | 0.3359 | 0.2855 | 0.9048 | 0.33 | 43 | 1573 | 93 | 77 |
| UA | 0.6594 | 0.4688 | 0.8776 | 0.3302 | 0.4160 | 0.8322 | 0.5644 | 0.2271 | 0.5230 | 0.16 | 552 | 382 | 775 | 77 |
| Macro mean | 0.7896 | 0.4765 | — | — | — | — | 0.5138 | — | — | — | — | — | — | — |

验证集重点观察：STEMI 区分度最好；NSTEMI 的少数类召回和 F1 较低；UA 阈值偏低使 Sensitivity 较高，但 Specificity 明显下降。

## 5. 外部测试集平台盲评

提交文件按 `data/CSV/test.csv` 的原始顺序重新排列，文件为：

`results/stemi_nstemi_ua_final_gpu0/submission_stemi_nstemi_ua_XceptionTime.csv`

平台返回的外部测试结果如下；每一行都是该标签相对于其他样本的 one-vs-rest 统计，样本数均为 1,995。

| Label | Samples | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 1,995 | 0.7099 | 0.9624 | 0.6250 | 0.9740 | 0.9419 | 0.6647 |
| NSTEMI | 1,995 | 0.3235 | 0.9414 | 0.2876 | 0.9501 | 0.8992 | 0.3045 |
| UA | 1,995 | 0.8886 | 0.3382 | 0.4108 | 0.8538 | 0.5263 | 0.5619 |
| Macro mean | — | 0.6407 | 0.7473 | 0.4411 | 0.9260 | 0.7891 | 0.5104 |

平台结果文件：

- `results/stemi_nstemi_ua_final_gpu0/official_platform_stemi_nstemi_ua_XceptionTime.html`
- `results/stemi_nstemi_ua_final_gpu0/official_test_stemi_nstemi_ua_XceptionTime.csv`

预测结构审计：1,995 条外部记录中有 118 条出现两个或以上预测阳性标签，301 条三个标签均预测为阴性。这说明输出确实是独立 sigmoid 多标签预测，并非强制 one-hot；平台也按多标签 one-vs-rest 方式计算指标。

由于外部测试标签不可见，本地不能重新计算外部 TP/TN/FP/FN、AUROC、AUPRC、micro-F1 或 exact-match accuracy；平台返回表格是该外部集的权威结果。

## 6. GPU1 方案验证的补充结果（AMI/OMI，不与本任务三标签结果混合）

GPU1 已完成 E0–E4 的五折 patient-level 联合分层实验。E4 是计划中的完整 HMoE/KAN 组合：形态专家 + lead/spatial expert + WavelNet WaveletConv frequency expert + task-specific router + KAN heads。

| Variant | AMI AUROC | AMI AUPRC | AMI F1 | OMI AUROC | OMI AUPRC | OMI F1 |
|---|---:|---:|---:|---:|---:|---:|
| E0 baseline | 0.8877 ± 0.0047 | 0.6675 ± 0.0119 | 0.6251 ± 0.0135 | 0.8919 ± 0.0037 | 0.4042 ± 0.0199 | 0.4621 ± 0.0122 |
| E1 KAN head | 0.8842 ± 0.0044 | 0.6568 ± 0.0071 | 0.6124 ± 0.0140 | 0.8904 ± 0.0046 | **0.4056 ± 0.0132** | 0.4547 ± 0.0170 |
| E2 MoE | 0.8824 ± 0.0061 | 0.6565 ± 0.0107 | 0.6225 ± 0.0160 | 0.8918 ± 0.0033 | 0.4039 ± 0.0187 | 0.4553 ± 0.0148 |
| E3 WavelNet + MLP | 0.8840 ± 0.0027 | 0.6600 ± 0.0142 | 0.6177 ± 0.0114 | **0.8934 ± 0.0037** | 0.4036 ± 0.0228 | **0.4637 ± 0.0188** |
| E4 WavelNet + KAN | 0.8867 ± 0.0059 | 0.6604 ± 0.0117 | **0.6279 ± 0.0088** | 0.8926 ± 0.0042 | 0.3940 ± 0.0158 | 0.4548 ± 0.0137 |

E4 相对 E0 的 OMI AUPRC 下降约 `0.0102`，OMI F1 下降约 `0.0073`；因此目前不能宣称 HMoE/KAN/WavelNet 已经改善 OMI。E1 的 OMI AUPRC 仅有约 `0.0014` 的均值增益，仍属于很小的探索性差异，不能作为稳定提升结论。

## 7. 可复现文件与限制

- GPU0 训练结果：`results/stemi_nstemi_ua_final_gpu0/XceptionTime_bce_P3_seed2026.json`
- GPU0 提交预测：`results/stemi_nstemi_ua_final_gpu0/submission_stemi_nstemi_ua_XceptionTime.csv`
- GPU1 E0–E4 汇总：`results/hmoe_kan_github/e0_summary.json` 至 `e4_summary.json`
- 方案文件：`ACS_Xception_HMoE_KAN_experiment_plan.md`
- HMoE 方案报告：`reports/acs_hmoe_kan_feasibility.md`

当前三标签结果是单个 seed（2026）；若用于论文主结果，仍应按相同 patient-level split 运行多个 seed 并给出 mean ± std。外部平台目前只返回离散阈值下的逐标签表格，不能替代完整的多标签概率指标分析。

## 8. 最终判断

本次三标签多标签流程在数据划分、独立 sigmoid 输出、验证集阈值确定、外部测试顺序和平台 one-vs-rest 评价上是闭合的。当前可报告的最终外部结果为：STEMI 表现最好，UA 召回高但特异度低，NSTEMI 是主要短板。GPU1 的 WavelNet/HMoE/KAN 方案已完成可行性验证，但在当前五折结果上尚不足以替换原有 OMI/AMI baseline。
Latest GPU1 E4 three-label CV and external-platform results are recorded in [hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md](hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md).
