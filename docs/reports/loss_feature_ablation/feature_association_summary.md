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
