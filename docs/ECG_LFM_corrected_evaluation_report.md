# ECG-LFM 修正评估报告

> 本报告由现有 Baseline/LoRA/DoRA 共 15 个 Fold0–4 checkpoint 重新推理生成；没有重新训练。
> 当前仍使用 outer validation 选择 checkpoint，并在同一 pooled OOF 选择阈值，故 hard metrics 是描述性 OOF；nested validation 从下一阶段开始。

## 1. Provenance 与对齐门禁

| 方法 | record-ID join | 重复预测 | 缺失预测 | 缺失标签 | patient overlap | fold coverage | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | record_id | 0 | 0 | 0 | 0 | 100.0% | pass |
| lora | record_id | 0 | 0 | 0 | 0 | 100.0% | pass |
| dora | record_id | 0 | 0 | 0 | 0 | 100.0% | pass |

标签由 prediction `record_id` 与 `master_records.csv`、原始 `train.csv` 显式连接；numeric master index 只作交叉核对，不再作为唯一真值索引。逐例 CSV 含 patient_id，仅保存在受控执行服务器并被 Git 忽略。

## 2. 全队列 ACS 四状态 pooled OOF

| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
| baseline | 0.7826 | 0.5329 | 0.4994 | 0.4954 | 0.5684 |
| lora | 0.7870 | 0.5390 | 0.5103 | 0.5053 | 0.5695 |
| dora | 0.7838 | 0.5369 | 0.5069 | 0.5024 | 0.5680 |

### 2.1 ACS 逐类

| 方法 | 类别 | AUROC | AUPRC | F1 | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| baseline | OTHER | 0.6976 | 0.6873 | 0.6638 | 0.7282 | 0.5246 |
| baseline | STEMI | 0.9467 | 0.6911 | 0.6440 | 0.6706 | 0.9640 |
| baseline | NSTEMI | 0.8141 | 0.2717 | 0.2604 | 0.1976 | 0.9764 |
| baseline | UA | 0.6718 | 0.4816 | 0.4295 | 0.3852 | 0.7841 |
| lora | OTHER | 0.7011 | 0.6908 | 0.6573 | 0.6998 | 0.5618 |
| lora | STEMI | 0.9491 | 0.7017 | 0.6477 | 0.6616 | 0.9667 |
| lora | NSTEMI | 0.8230 | 0.2805 | 0.2813 | 0.2356 | 0.9675 |
| lora | UA | 0.6750 | 0.4830 | 0.4548 | 0.4243 | 0.7665 |
| dora | OTHER | 0.6982 | 0.6878 | 0.6579 | 0.7101 | 0.5422 |
| dora | STEMI | 0.9497 | 0.6972 | 0.6542 | 0.6796 | 0.9653 |
| dora | NSTEMI | 0.8155 | 0.2825 | 0.2748 | 0.2154 | 0.9740 |
| dora | UA | 0.6716 | 0.4799 | 0.4408 | 0.4046 | 0.7719 |

## 3. Conditional ACS（真实 ACS 病例）

连续概率严格按 `P(k|ACS)=P(k)/(P(STEMI)+P(NSTEMI)+P(UA))` 计算。

| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8736 | 0.7030 | 0.6439 | 0.6316 | 0.8055 |
| lora | 0.8792 | 0.7088 | 0.6541 | 0.6400 | 0.8058 |
| dora | 0.8761 | 0.7090 | 0.6522 | 0.6377 | 0.8092 |

## 4. NSTEMI vs UA

NSTEMI 为 positive class，使用连续 `p_NSTEMI/(p_NSTEMI+p_UA)`。

| 方法 | AUROC | AUPRC | F1 | Balanced Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8343 | 0.5373 | 0.4776 | 0.6706 | 0.3887 | 0.9525 |
| lora | 0.8444 | 0.5534 | 0.5094 | 0.6910 | 0.4381 | 0.9440 |
| dora | 0.8374 | 0.5545 | 0.4815 | 0.6731 | 0.3951 | 0.9511 |

## 5. OMI pooled OOF（全队列 Youden 工作点）

| 方法 | AUROC | AUPRC | Threshold | F1 | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9052 | 0.4159 | 0.0500 | 0.3826 | 0.8488 | 0.8228 | 0.2470 | 0.9876 |
| lora | 0.9106 | 0.4315 | 0.0500 | 0.4021 | 0.8549 | 0.8359 | 0.2629 | 0.9883 |
| dora | 0.9094 | 0.4267 | 0.0500 | 0.4054 | 0.8532 | 0.8387 | 0.2658 | 0.9882 |

## 6. OMI among NSTE-ACS（NSTEMI + UA）

使用全队列 pooled OOF Youden threshold，不在子集上再次调阈值。

| 方法 | N | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7448 | 0.8511 | 0.2610 | 0.3060 | 0.7045 | 0.8253 | 0.1954 | 0.9789 |
| lora | 7448 | 0.8576 | 0.2788 | 0.3228 | 0.7116 | 0.8376 | 0.2087 | 0.9797 |
| dora | 7448 | 0.8545 | 0.2773 | 0.3225 | 0.7045 | 0.8396 | 0.2091 | 0.9792 |

## 7. 未校准概率的 calibration audit

当前只测量 calibration，没有拟合 temperature/Platt/isotonic/beta 模型。

| 方法 | ACS multiclass Brier | ACS top-label ECE | OMI Brier | OMI ECE | OMI slope | OMI intercept |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.5246 | 0.0297 | 0.0460 | 0.0123 | 0.8639 | -0.1348 |
| lora | 0.5225 | 0.0286 | 0.0454 | 0.0132 | 0.7988 | -0.2355 |
| dora | 0.5239 | 0.0301 | 0.0452 | 0.0119 | 0.8195 | -0.1253 |

## 8. Phase-1 结论与边界

- 只有三个方法的 alignment status 都为 `pass`，才允许进入 nested split 阶段。
- 本阶段修复 evaluator/provenance，但没有消除旧 checkpoint 由 outer validation 选择造成的乐观偏差；新训练必须使用 inner validation。
- 现有平台集已经参与工程选型，本报告不把它当作 untouched confirmatory test，也不重新提交平台。
- 本阶段没有修改 Loss，没有实现 RDR，更没有实现 RL。

## 9. Source hashes

- `baseline`: evaluator `c398f7d5864b1cd222034bf5083d1845137bfedfe8b4457fe6184798b3b534b4`, task metrics `c02ae24f26f1f2fe133d95272581f99276823e020efe0f200e6522caf8bfb88b`, model source `9487cb27bf48d343b276d6b51cd228b851c75a00f4d94e77027b426eff3f7a68`.
- `lora`: evaluator `c398f7d5864b1cd222034bf5083d1845137bfedfe8b4457fe6184798b3b534b4`, task metrics `c02ae24f26f1f2fe133d95272581f99276823e020efe0f200e6522caf8bfb88b`, model source `9487cb27bf48d343b276d6b51cd228b851c75a00f4d94e77027b426eff3f7a68`.
- `dora`: evaluator `c398f7d5864b1cd222034bf5083d1845137bfedfe8b4457fe6184798b3b534b4`, task metrics `c02ae24f26f1f2fe133d95272581f99276823e020efe0f200e6522caf8bfb88b`, model source `9487cb27bf48d343b276d6b51cd228b851c75a00f4d94e77027b426eff3f7a68`.
