# ECG/年龄/性别/中位数 ECG 特征消融总结（历史 OMI/AMI 任务）

> 本总结只比较旧 OMI/AMI 二分类输入特征，不能作为当前 ACS/OMI 联合任务的
> 特征增益结论。

## 中文摘要

在相同 XceptionTimeFusion 初始化和训练流程下，仅遮蔽/加入输入模态，比较原始
ECG、年龄、性别和中位数 ECG 的增量价值。五种子结果显示：OMI 最好的追加方案
是 `raw_clin`（AUPRC 仅增加约 0.0010，4/5 seed 改善）；AMI 最好的追加方案
是 `raw_age`（AUPRC 增加约 0.0032，5/5 seed 改善）。这些是验证集关联和
增量预测证据，不等同于因果关系，外部测试标签没有用于选择方案。

The controlled neural comparison uses the same XceptionTimeFusion architecture and initialization; only input modalities are masked. Results below use the validation threshold maximizing the F1/Sensitivity harmonic mean.

## Five-seed confirmation

| Task | Feature mode | AUROC | AUPRC | Sensitivity | Specificity | PPV | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OMI | raw_clin | 0.8771 ± 0.0025 | 0.3273 ± 0.0162 | 0.6723 ± 0.0585 | 0.9037 ± 0.0143 | 0.2809 ± 0.0173 | 0.3951 ± 0.0141 | 0.3874 ± 0.0169 |
| OMI | raw | 0.8775 ± 0.0023 | 0.3263 ± 0.0173 | 0.6532 ± 0.0576 | 0.9054 ± 0.0221 | 0.2816 ± 0.0294 | 0.3914 ± 0.0228 | 0.3808 ± 0.0173 |
| AMI | raw | 0.8750 ± 0.0014 | 0.6270 ± 0.0122 | 0.7333 ± 0.0198 | 0.8671 ± 0.0163 | 0.4666 ± 0.0257 | 0.5696 ± 0.0133 | 0.5023 ± 0.0135 |
| AMI | raw_age | 0.8769 ± 0.0014 | 0.6302 ± 0.0124 | 0.7465 ± 0.0386 | 0.8620 ± 0.0348 | 0.4658 ± 0.0463 | 0.5712 ± 0.0258 | 0.5060 ± 0.0242 |

## Added-value verdict

- OMI: best screened addition `raw_clin`; ΔAUPRC=+0.0010; improved in 4/5 seeds — **stable gain**.
- AMI: best screened addition `raw_age`; ΔAUPRC=+0.0032; improved in 5/5 seeds — **stable gain**.

Standalone `median` and `clin` rows quantify predictive association, while raw-plus-feature rows test incremental value beyond raw ECG. External test feedback was not used.

## Clinical association appendix

# Age/sex association and clinical-only probe

All inferential summaries use only the official-training pool and its frozen patient-level train/validation split. The external test was not accessed.

Records: train 16,174; validation 1,786. Missing age: 0; missing sex: 0.

## Associations

| Split | Label | Feature | Correlation | p-value | Effect size | Positive group | Negative group |
|---|---|---|---:|---:|---:|---|---|
| train | OMI | age | -0.0048 | 0.542 | -0.0184 | mean=64.703;median=66.000 | mean=64.917;median=66.000 |
| train | OMI | sex | 0.0981 | 2.21e-38 | 2.5237 | label_rate_sex1=0.085089 | label_rate_sex0=0.035542 |
| train | AMI | age | 0.0109 | 0.164 | 0.0292 | mean=65.189;median=66.000 | mean=64.852;median=66.000 |
| train | AMI | sex | 0.1541 | 3.52e-91 | 2.6883 | label_rate_sex1=0.195458 | label_rate_sex0=0.082881 |
| validation | OMI | age | 0.0030 | 0.9 | 0.0123 | mean=64.957;median=66.000 | mean=64.814;median=66.000 |
| validation | OMI | sex | 0.0565 | 0.017 | 1.7444 | label_rate_sex1=0.062789 | label_rate_sex0=0.036984 |
| validation | AMI | age | 0.0446 | 0.0597 | 0.1215 | mean=66.037;median=67.000 | mean=64.630;median=65.000 |
| validation | AMI | sex | 0.1091 | 2.82e-06 | 2.0250 | label_rate_sex1=0.166205 | label_rate_sex0=0.089616 |

For age, correlation is point-biserial and effect size is standardized mean difference. For sex, correlation is phi/Pearson and effect size is the Fisher-table odds ratio for sex=1 versus sex=0.

## Validation logistic probes

| Label | Features | AUROC | AUPRC | Sens@0.5 | Spec@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| OMI | age | 0.4912 | 0.0606 | 0.0000 | 1.0000 | 0.0000 |
| OMI | sex | 0.5618 | 0.0600 | 0.0000 | 1.0000 | 0.0000 |
| OMI | age_sex | 0.5665 | 0.0638 | 0.0000 | 1.0000 | 0.0000 |
| AMI | age | 0.5425 | 0.1871 | 0.0000 | 1.0000 | 0.0000 |
| AMI | sex | 0.5778 | 0.1584 | 0.0000 | 1.0000 | 0.0000 |
| AMI | age_sex | 0.6039 | 0.2007 | 0.0000 | 1.0000 | 0.0000 |

These probes quantify marginal predictive association; they do not establish causal effects. Added-value conclusions come from the controlled neural feature ablation.
