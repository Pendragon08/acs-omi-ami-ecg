# ECG-LFM ACS–OMI Joint-6 评估报告

## 1. 评估审计结论

- Joint-6 概率顺序为 `OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`。
- ACS 由 `q0, q1+q2, q3+q4, q5` 恢复并使用 `argmax`；没有类别独立阈值。
- OMI 概率为 `q2+q4`；每折阈值只在 inner-validation 用 Youden Index 选择，然后冻结到 outer-test。
- `NSTEMI vs UA` 在真实 NSTEMI/UA 子集内使用 `(q3+q4)/(q3+q4+q5)`。
- `OMI among NSTE-ACS` 沿用真实 NSTEMI+UA 子集和全局 `p_omi=q2+q4`，未重定义 cohort。
- 本次只修改 evaluation、metrics 和 reporting；未修改模型、loss、LoRA、训练配置或划分。
- Joint-6 的某类别若在 Fold 中缺少正样本，其 OvR 指标输出 `NA/null`，不再伪填 0；全六类 Macro 与 supported-class Macro 在机器可读结果中分开保存。

## 2. 完成状态

- `ECG-LFM Partial FT/fold1`
- `ECG-LFM Partial FT/fold2`
- `ECG-LFM Partial FT/fold3`
- `ECG-LFM Partial FT/fold4`
- `InceptionTime/fold0`
- `InceptionTime/fold1`
- `InceptionTime/fold2`
- `InceptionTime/fold3`
- `InceptionTime/fold4`
- `ResNet1D_Wang/fold0`
- `ResNet1D_Wang/fold1`
- `ResNet1D_Wang/fold2`
- `ResNet1D_Wang/fold3`
- `ResNet1D_Wang/fold4`
- `LSTM/fold0`
- `LSTM/fold1`
- `LSTM/fold2`
- `LSTM/fold3`
- `LSTM/fold4`
- `Transformer1D/fold0`
- `Transformer1D/fold1`
- `Transformer1D/fold2`
- `Transformer1D/fold3`
- `Transformer1D/fold4`
- `ViT/fold0`
- `ViT/fold1`
- `ViT/fold2`
- `ViT/fold3`
- `ViT/fold4`

已完成五折的方法：ECG-LFM LoRA。

## 3. Level 1：Joint-6 整体与临床核心指标（五折 mean ± std）

| 方法 | 完成 | Joint-6 Macro AUPRC | Joint-6 Macro F1 | ACS Macro AUPRC | OMI AUPRC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUPRC | OMI Sensitivity | OMI Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 5/5 | 0.3843 ± 0.0056 | 0.3557 ± 0.0095 | 0.5445 ± 0.0067 | 0.4299 ± 0.0258 | 0.5527 ± 0.0363 | 0.2695 ± 0.0599 | 0.8256 ± 0.0608 | 0.8446 ± 0.0195 |
| ECG-LFM Partial FT | 1/5 | 0.3737 | 0.3439 | 0.5291 | 0.3656 | 0.4700 | 0.2073 | 0.8139 | 0.8190 |

Joint-6 的 primary ranking metric 为 Macro AUPRC；不会只凭 Accuracy 选模型。

### Joint-6 六分类完整整体指标

| 方法 | 完成 | Macro AUPRC | Macro AUROC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 5/5 | 0.3843 ± 0.0056 | 0.8122 ± 0.0082 | 0.3557 ± 0.0095 | 0.3461 ± 0.0111 | 0.5505 ± 0.0100 |
| ECG-LFM Partial FT | 1/5 | 0.3737 | 0.7971 | 0.3439 | 0.3416 | 0.5521 |

### 恢复后的 ACS 四分类完整整体指标

| 方法 | 完成 | Macro AUPRC | Macro AUROC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 5/5 | 0.5445 ± 0.0067 | 0.7844 ± 0.0067 | 0.5087 ± 0.0119 | 0.4991 ± 0.0127 | 0.5748 ± 0.0091 |
| ECG-LFM Partial FT | 1/5 | 0.5291 | 0.7735 | 0.4814 | 0.4857 | 0.5760 |

### 恢复后的 OMI 二分类完整指标

所有阈值依赖指标均使用各 Fold 的 inner-validation Youden threshold，冻结后用于 outer-test。

| 方法 | 完成 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV | MCC | Accuracy | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 5/5 | 0.9104 ± 0.0111 | 0.4299 ± 0.0258 | 0.4035 ± 0.0089 | 0.8256 ± 0.0608 | 0.8446 ± 0.0195 | 0.2676 ± 0.0110 | 0.9861 ± 0.0046 | 0.4121 ± 0.0118 | 0.8434 ± 0.0146 | 0.0560 ± 0.0230 |
| ECG-LFM Partial FT | 1/5 | 0.8897 | 0.3656 | 0.3650 | 0.8139 | 0.8190 | 0.2353 | 0.9847 | 0.3731 | 0.8186 | 0.0600 |

| 方法 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| ECG-LFM LoRA | 950 | 2612 | 14197 | 201 |

### NSTE-ACS 子组完整排序指标

| 方法 | 完成 | NSTEMI vs UA AUROC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUROC | OMI among NSTE-ACS AUPRC |
|---|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 5/5 | 0.8377 ± 0.0243 | 0.5527 ± 0.0363 | 0.8564 ± 0.0244 | 0.2695 ± 0.0599 |
| ECG-LFM Partial FT | 1/5 | 0.7874 | 0.4700 | 0.8135 | 0.2073 |

## 4. 五折 95% CI

95% CI 为五折均值的双侧 Student-t 区间；它不是 patient-level bootstrap CI。

| 方法 | Joint-6 Macro AUPRC | ACS Macro AUPRC | OMI AUPRC | NSTEMI vs UA AUPRC | OMI among NSTE-ACS AUPRC |
|---|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | [0.3774, 0.3912] | [0.5361, 0.5528] | [0.3978, 0.4619] | [0.5077, 0.5977] | [0.1951, 0.3439] |

## 5. Level 2：Joint-6 六类与 ACS 四类逐类指标（五折 mean ± std）

粗体类别中 `NSTEMI+nOMI`、`NSTEMI+OMI`、`UA` 是 NSTE-ACS 重点困难类。`support` 是已完成 outer folds 的累计样本数。

| 方法 | 指标族 | 类别 | Support | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | joint6 | OTHER | 9070 | 0.6995 ± 0.0061 | 0.6882 ± 0.0092 | 0.6661 ± 0.0200 | 0.7343 ± 0.0438 | 0.5213 ± 0.0317 | 0.6102 ± 0.0099 |
| ECG-LFM LoRA | joint6 | STEMI+nOMI | 715 | 0.9153 ± 0.0062 | 0.3521 ± 0.0244 | 0.3170 ± 0.0243 | 0.2722 ± 0.0453 | 0.9820 ± 0.0044 | 0.3876 ± 0.0266 |
| ECG-LFM LoRA | joint6 | STEMI+OMI | 727 | 0.9449 ± 0.0063 | 0.4446 ± 0.0357 | 0.4486 ± 0.0358 | 0.4756 ± 0.0881 | 0.9735 ± 0.0076 | 0.4370 ± 0.0407 |
| ECG-LFM LoRA | joint6 | **NSTEMI+nOMI** | 812 | 0.7854 ± 0.0242 | 0.1747 ± 0.0190 | 0.1728 ± 0.0586 | 0.1289 ± 0.0576 | 0.9849 ± 0.0067 | 0.2890 ± 0.0592 |
| ECG-LFM LoRA | joint6 | **NSTEMI+OMI** | 423 | 0.8552 ± 0.0226 | 0.1615 ± 0.0426 | 0.0862 ± 0.0518 | 0.0536 ± 0.0373 | 0.9967 ± 0.0019 | 0.2905 ± 0.0987 |
| ECG-LFM LoRA | joint6 | **UA** | 6213 | 0.6729 ± 0.0033 | 0.4848 ± 0.0101 | 0.4436 ± 0.0317 | 0.4122 ± 0.0559 | 0.7674 ± 0.0413 | 0.4851 ± 0.0167 |
| ECG-LFM LoRA | acs | OTHER | 9070 | 0.6995 ± 0.0061 | 0.6882 ± 0.0092 | 0.6661 ± 0.0188 | 0.7277 ± 0.0426 | 0.5349 ± 0.0338 | 0.6150 ± 0.0101 |
| ECG-LFM LoRA | acs | STEMI | 1442 | 0.9476 ± 0.0065 | 0.7003 ± 0.0316 | 0.6444 ± 0.0288 | 0.6470 ± 0.0417 | 0.9685 ± 0.0047 | 0.6433 ± 0.0335 |
| ECG-LFM LoRA | acs | NSTEMI | 1235 | 0.8178 ± 0.0228 | 0.3046 ± 0.0336 | 0.2819 ± 0.0612 | 0.2158 ± 0.0625 | 0.9786 ± 0.0069 | 0.4308 ± 0.0372 |
| ECG-LFM LoRA | acs | **UA** | 6213 | 0.6729 ± 0.0033 | 0.4848 ± 0.0101 | 0.4424 ± 0.0311 | 0.4060 ± 0.0553 | 0.7765 ± 0.0412 | 0.4915 ± 0.0164 |
| ECG-LFM Partial FT | joint6 | OTHER | 1824 | 0.6990 | 0.6908 | 0.6697 | 0.7582 | 0.4820 | 0.5997 |
| ECG-LFM Partial FT | joint6 | STEMI+nOMI | 140 | 0.9205 | 0.3599 | 0.3713 | 0.4071 | 0.9683 | 0.3413 |
| ECG-LFM Partial FT | joint6 | STEMI+OMI | 148 | 0.9374 | 0.3780 | 0.4014 | 0.3919 | 0.9760 | 0.4113 |
| ECG-LFM Partial FT | joint6 | **NSTEMI+nOMI** | 162 | 0.7255 | 0.1804 | 0.1414 | 0.0864 | 0.9936 | 0.3889 |
| ECG-LFM Partial FT | joint6 | **NSTEMI+OMI** | 83 | 0.8210 | 0.1282 | 0.0460 | 0.0241 | 0.9994 | 0.5000 |
| ECG-LFM Partial FT | joint6 | **UA** | 1249 | 0.6790 | 0.5047 | 0.4334 | 0.3819 | 0.7985 | 0.5011 |
| ECG-LFM Partial FT | acs | OTHER | 1824 | 0.6990 | 0.6908 | 0.6725 | 0.7538 | 0.5006 | 0.6071 |
| ECG-LFM Partial FT | acs | STEMI | 288 | 0.9465 | 0.6584 | 0.6338 | 0.6910 | 0.9575 | 0.5853 |
| ECG-LFM Partial FT | acs | NSTEMI | 245 | 0.7693 | 0.2626 | 0.1847 | 0.1184 | 0.9881 | 0.4203 |
| ECG-LFM Partial FT | acs | **UA** | 1249 | 0.6790 | 0.5047 | 0.4347 | 0.3795 | 0.8057 | 0.5086 |

## 6. Level 3：恢复临床任务，与 factorized LoRA baseline 的描述性对比

| Metric | Baseline | Joint-6 ECG-LFM LoRA（pooled OOF） | Δ |
|---|---:|---:|---:|
| Joint-6 Macro AUPRC | N/A | 0.3761 | — |
| ACS Macro AUPRC | 0.5390 | 0.5405 | 0.0015 |
| OMI AUPRC | 0.4315 | 0.4201 | -0.0115 |
| OMI Sensitivity | 0.8549 | 0.8254 | -0.0295 |
| NSTEMI vs UA AUPRC | 0.5534 | 0.5426 | -0.0108 |
| NSTE-ACS OMI AUPRC | 0.2788 | 0.2498 | -0.0290 |

Baseline 来源为 `results/revalidated/lora/metrics.json`，其协议标记为 `descriptive pooled OOF; no nested checkpoint/threshold selection`。它不是完整 nested checkpoint/threshold selection，因此上述 Δ 仅作描述性参考，特别是阈值依赖的 OMI Sensitivity 不能视为严格配对比较，也不能据此宣称统计显著。

## 7. 五折逐折结果

| 方法 | Fold | Joint-6 Macro AUPRC | Joint-6 Macro F1 | ACS Macro AUPRC | OMI AUPRC | OMI Sensitivity | OMI Specificity | NSTEMI vs UA AUPRC | NSTE-ACS OMI AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM LoRA | 0 | 0.3776 | 0.3657 | 0.5452 | 0.3891 | 0.7619 | 0.8596 | 0.5077 | 0.1930 |
| ECG-LFM LoRA | 1 | 0.3798 | 0.3496 | 0.5440 | 0.4228 | 0.7965 | 0.8547 | 0.5690 | 0.2641 |
| ECG-LFM LoRA | 2 | 0.3876 | 0.3596 | 0.5513 | 0.4368 | 0.8428 | 0.8488 | 0.5534 | 0.2323 |
| ECG-LFM LoRA | 3 | 0.3911 | 0.3613 | 0.5483 | 0.4562 | 0.9214 | 0.8106 | 0.6027 | 0.3313 |
| ECG-LFM LoRA | 4 | 0.3853 | 0.3424 | 0.5336 | 0.4444 | 0.8052 | 0.8495 | 0.5309 | 0.3267 |
| ECG-LFM Partial FT | 0 | 0.3737 | 0.3439 | 0.5291 | 0.3656 | 0.8139 | 0.8190 | 0.4700 | 0.2073 |

## 8. 用户要求的十项结论

1. **Joint-6 Macro AUPRC**：0.3761（LoRA pooled OOF）；五折 mean ± std 为 0.3843 ± 0.0056。
2. **最难 joint class**：NSTEMI+OMI，按五折 class AUPRC 最低判断（0.1615 ± 0.0426）。
3. **NSTEMI+OMI**：AUPRC 0.1615 ± 0.0426；Sensitivity 0.0536 ± 0.0373。
4. **恢复后的 ACS Macro AUPRC 是否优于 baseline**：是，描述性 Δ=0.0015。
5. **OMI AUPRC 是否优于 baseline**：否，描述性 Δ=-0.0115。
6. **OMI Sensitivity 是否下降**：是，描述性 Δ=-0.0295。该项阈值协议不同，只能描述，不能作严格因果结论。
7. **NSTEMI vs UA 是否改善**：否，描述性 Δ=-0.0108。
8. **OMI among NSTE-ACS 是否改善**：否，描述性 Δ=-0.0290。
9. **改善是否跨 Fold 稳定**：现有 factorized baseline 缺少同协议逐折 nested 结果，不能合法计算逐折配对改善率；Joint-6 自身的五折波动已完整列出。
10. **是否值得作为最终主模型**：当前不建议仅凭现有结果把 Joint-6 定为最终主模型：其 ACS 排序性能略升，但 OMI/NSTE 子组未显示一致优势；还应等待全部对比方法完成，并用同一 nested 协议重算 factorized baseline 后再冻结主模型。

## 9. 可追溯输出

- `results/joint6_cv/summary/fold_metrics.csv`
- `results/joint6_cv/summary/summary_metrics.csv`
- `results/joint6_cv/summary/per_class_metrics.csv`
- `results/joint6_cv/summary/per_class_summary.csv`
- `results/joint6_cv/summary/pooled_oof_metrics.json`
- `results/joint6_cv/summary/training_cost.csv`
- `results/joint6_cv/summary/completion_status.json`
- 每个 Fold 的 `joint6_metrics.json`、`per_class_metrics.csv`、`acs_metrics.json`、`omi_metrics.json`、`nste_metrics.json`（由新版 evaluator 生成）。

## 10. 解释限制

- 当前结果采用患者互斥的 nested CV 划分，但 `_aggregate` 后按 ECG record 一票计算 outer-test/pooled OOF 指标；它不是 patient-level 聚合性能，也不是 external test。
- baseline 对比协议并不完全一致，因此未计算配对差异或显著性，也不把描述性差异写成稳定提升。
- 传统模型使用项目既有 P2 输入，ECG-LFM 使用原生预处理；完整对比反映工程方案差异，不是单一网络结构消融。
