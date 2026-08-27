---
experiment_id: ACS-HMOE-WAVELNET-STEMI-NSTEMI-UA-E4-20260826
date: 2026-08-26
status: completed
task: independent multi-label STEMI/NSTEMI/UA classification
variant: E4
model: Morphology + LeadSpatial + WavelNet Frequency + task routers + KAN heads
loss: BCEWithLogitsLoss
seed: 2026
fold_seed: 2026
device: NVIDIA RTX A6000 GPU1
---

# GPU1：WavelNet + HMoE + KAN 三标签分类结果

## 1. 实验设置

本实验将 GPU1 方案中的 E4 完整结构扩展为三个独立 sigmoid 输出，标签为 STEMI、NSTEMI 和 UA。三个标签不使用 softmax，也不强制 one-hot。

- 数据：训练记录 17,960 条，唯一患者 17,018 人；输入 `[N, 1000, 12]`，100 Hz。
- 划分：patient-level 联合标签分层 5-fold，`fold_seed=2026`；每折训练患者和验证患者互斥。
- 预处理：沿用 P2 0.5–40 Hz ECG，按当前 fold 的训练数据拟合 robust scaler；只对训练集做在线增强。
- Loss：原始 `BCEWithLogitsLoss`，没有叠加 `pos_weight` 或采样器。
- 优化器：Adam，学习率 `1e-4`，batch size `128`，最多 100 epochs，patience `15`，梯度裁剪 `1.0`。
- 阈值：每个 fold、每个标签仅在该 fold 验证集上通过 F1 最大化选择。
- 参数量：332,587。
- Frequency Expert：官方 WavelNet `WaveletConv`，16 个共享 Symlet-4 滤波器，kernel=103，fs=100 Hz；没有使用 MVMD/VMD 外部预处理。

## 2. 五折均值 ± 标准差

| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9336 ± 0.0101 | 0.6706 ± 0.0236 | 0.6450 ± 0.0314 | 0.9677 ± 0.0046 | 0.6370 ± 0.0239 | 0.9690 ± 0.0027 | 0.6399 ± 0.0122 | 0.6089 ± 0.0127 | 0.9418 ± 0.0026 |
| NSTEMI | 0.7960 ± 0.0088 | 0.2686 ± 0.0324 | 0.4613 ± 0.0456 | 0.9013 ± 0.0306 | 0.2655 ± 0.0368 | 0.9578 ± 0.0022 | 0.3331 ± 0.0236 | 0.2833 ± 0.0221 | 0.8711 ± 0.0256 |
| UA | 0.6588 ± 0.0058 | 0.4676 ± 0.0114 | 0.8640 ± 0.0160 | 0.3528 ± 0.0194 | 0.4139 ± 0.0028 | 0.8311 ± 0.0114 | 0.5596 ± 0.0032 | 0.2304 ± 0.0098 | 0.5296 ± 0.0078 |

逐标签指标的简单宏平均：AUROC **0.7961**，AUPRC **0.4689**，Sensitivity **0.6570**，Specificity **0.7406**，PPV **0.4388**，NPV **0.9193**，F1 **0.5109**，MCC **0.3742**，Accuracy **0.7808**。这些宏平均是三个 one-vs-rest 标签指标的算术平均。

## 3. 每折宏指标

| Fold | Best epoch | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Train time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 0.8049 | 0.4893 | 0.6685 | 0.7386 | 0.4541 | 0.9260 | 0.5259 | 0.3911 | 0.7846 | 404.21 |
| 1 | 21 | 0.7989 | 0.4617 | 0.6580 | 0.7381 | 0.4295 | 0.9163 | 0.5071 | 0.3674 | 0.7782 | 440.81 |
| 2 | 14 | 0.7891 | 0.4613 | 0.6902 | 0.7258 | 0.4044 | 0.9175 | 0.4967 | 0.3609 | 0.7657 | 355.04 |
| 3 | 20 | 0.7961 | 0.4629 | 0.6244 | 0.7586 | 0.4551 | 0.9189 | 0.5115 | 0.3774 | 0.7920 | 428.78 |
| 4 | 23 | 0.7917 | 0.4694 | 0.6425 | 0.7419 | 0.4507 | 0.9178 | 0.5131 | 0.3741 | 0.7836 | 466.45 |
| **Mean ± SD** | — | **0.7961 ± 0.0062** | **0.4689 ± 0.0119** | **0.6567 ± 0.0250** | **0.7406 ± 0.0118** | **0.4388 ± 0.0218** | **0.9193 ± 0.0039** | **0.5109 ± 0.0105** | **0.3742 ± 0.0114** | **0.7808 ± 0.0098** | — |

The fold-level Sensitivity/Specificity/PPV/NPV/F1/MCC/Accuracy values above use each fold's historical validation F1-maximizing threshold. The later Youden-threshold report uses a single threshold selected from pooled OOF validation predictions and is therefore reported separately.

## 4. 结果判断

- STEMI 表现最好：AUROC 0.9336，F1 0.6399。
- NSTEMI 是主要短板：AUPRC 0.2686，PPV 0.2655，F1 0.3331。
- UA 召回较高（0.8640），但 Specificity 仅 0.3528，说明当前阈值偏向减少漏诊、同时产生较多假阳性。
- 五折之间宏 AUROC 范围为 0.7891–0.8049，结果没有单个异常折，但 NSTEMI 的 AUPRC/F1 波动较大。
- 这次结果只证明 E4 在三标签任务上的可运行性和内部验证表现，不能直接替代 GPU0 的 XceptionTime 外部平台盲评结果。

## 5. 文件位置

服务器结果目录：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4/`

- `e4_summary.json`：五折汇总及全部逐标签指标
- `e4_fold0_seed2026.pt` 至 `e4_fold4_seed2026.pt`：五个 fold 的最佳 checkpoint
- `e4_split.json`：固定的 patient-level 五折划分
- 运行日志：`/data/code/ouyanghao/oyh/ACS-OMI-AMI/logs/hmoe_stemi_nstemi_ua_gpu1.log`

外部测试提交结果见下一节；外部平台只返回离散阈值下的逐标签 one-vs-rest 指标。

## 6. 外部测试平台盲评（已完成）

五个 fold checkpoint 的测试概率取算术平均；阈值使用全部 OOF 验证预测重新按 F1 最大化确定：STEMI `0.28`、NSTEMI `0.25`、UA `0.24`。提交文件严格按 `test.csv` 原始 `ecg_row_record` 顺序排列，共 1,995 条记录。

| Label | Samples | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 1,995 | 0.7407 | 0.9662 | 0.6593 | 0.9768 | 0.9479 | 0.6977 |
| NSTEMI | 1,995 | 0.3088 | 0.9521 | 0.3206 | 0.9496 | 0.9083 | 0.3146 |
| UA | 1,995 | 0.9164 | 0.3062 | 0.4069 | 0.8758 | 0.5148 | 0.5636 |
| 逐标签宏平均 | — | 0.6553 | 0.7415 | 0.4623 | 0.9341 | 0.7903 | 0.5253 |

平台返回的是逐标签 one-vs-rest 指标；平台未提供 AUROC、AUPRC、micro-F1、exact-match accuracy 或联合混淆矩阵。提交预测中 STEMI/NSTEMI/UA 的阳性数量分别为 `182/131/1536`，有 97 条记录预测为多标签阳性，245 条记录预测为全阴性，符合独立 sigmoid 多标签输出。针对 NSTEMI/UA 的预测模式为：NSTEMI-only 61 条、UA-only 1,466 条、NSTEMI+UA 70 条（其中 2 条同时预测为 STEMI），说明模型明显偏向广泛预测 UA。

外部评测文件：

- 提交 CSV：`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4/submission_stemi_nstemi_ua_hmoe_e4.csv`
- 平台 HTML：`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4/official_platform_hmoe_e4.html`
- 推理摘要：`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4/official_test_hmoe_e4_summary.json`
- 提交文件 SHA-256：`55f82394d2417f1b14912a7304d478345e0ae5e740d3e50b4a5435bb4ee00b56`

## 7. 与 GPU0 XceptionTime 外部结果的对照

GPU0 是单次 seed=2026 的 XceptionTime+P3，GPU1 是五 fold E4 集成，因此下表是工程对照，不是严格同预算消融。

| 指标（逐标签宏平均） | GPU0 XceptionTime | GPU1 E4 集成 | 差值（E4−GPU0） |
|---|---:|---:|---:|
| Sensitivity | 0.6407 | 0.6553 | +0.0146 |
| Specificity | 0.7473 | 0.7415 | −0.0058 |
| PPV | 0.4411 | 0.4623 | +0.0212 |
| NPV | 0.9260 | 0.9341 | +0.0081 |
| Accuracy | 0.7891 | 0.7903 | +0.0012 |
| F1 | 0.5104 | 0.5253 | +0.0149 |

逐标签看，E4 的 STEMI F1 为 0.6977（GPU0 为 0.6647），NSTEMI F1 为 0.3146（GPU0 为 0.3045），UA F1 为 0.5636（GPU0 为 0.5619）。E4 的 UA 灵敏度更高，但 UA 特异度进一步降低；NSTEMI 仍然是整体最需要优化的标签。
## 8. 2026-08-26 threshold update

The original external submission used validation-OOF F1-max thresholds (STEMI 0.28, NSTEMI 0.25, UA 0.24). A frozen-checkpoint re-thresholding run now provides the joint Sensitivity/Specificity (Youden J = Sensitivity + Specificity - 1) operating point: STEMI 0.030540, NSTEMI 0.035055, UA 0.323631. The new OOF metrics and submission artifact are recorded in [hmoe_wavelnet_e4_youden_threshold_results_20260826.md](hmoe_wavelnet_e4_youden_threshold_results_20260826.md); the old platform metrics in this report remain historical results for the F1-max submission.
