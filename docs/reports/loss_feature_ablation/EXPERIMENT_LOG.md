---
experiment_id: ACS-GPU1-260818-01
date: 2026-08-18
system: ACS-OMI-AMI
experiment_type: loss_and_feature_ablation
device: nlp03_physical_GPU1_RTX_A6000
status: completed
split: historical_patient_level_90_10
external_test_accessed: false
---

# OMI CB-Focal and ECG feature ablation log

## Objective

1. Determine whether CB-Focal improves minority-class OMI performance when loss is the only changed variable.
2. Quantify the association and incremental predictive value of raw ECG, age, sex and median ECG for OMI and AMI.

## Frozen controls

- Train 16,174 ECG / validation 1,786 ECG; fixed patient split and zero patient overlap.
- XceptionTime + P3 + BCE historical baseline; seed-2026 reproduction gate passed exactly.
- Adam, no scheduler, fixed learning rate/batch size by task, 100 epochs, patience 15, clipping 1.0.
- No oversampling, WeightedRandomSampler, SMOTE or additional pos_weight in CB-Focal.
- All loss weights use training labels only. Validation selects hyperparameters and thresholds. External hidden-label test is not used during ablation.

## Reproduction gate

- Baseline best epoch: 13; stop epoch: 28.
- AUROC/AUPRC: 0.8813125597 / 0.3493401437.
- New versus historical checkpoint: all tensors exactly equal; maximum absolute difference 0.

## Execution record

- Unit tests: stable focal equals BCE at gamma=0; extreme logits finite; CB empirical expected weight=1; feature masks cover all eight modes.
- Smoke runs: BCE, CB-Focal and all-feature Fusion each completed one epoch.
- Stage 1 screen session: `acs_ablation_stage1_gpu1`.
- Stage 1 master log: `logs/loss_feature_ablation/stage1_master.log`.
- Stage 2 confirmation: pending stage-1 validation-only selection.

## Known protocol limitation

Historical P3 waveform/median/clinical scalers were fitted on the complete official-training pool, including the internal validation fold. This is preserved identically across ablation groups for comparability, disclosed as leakage, and must be corrected in a future confirmatory cohort/CV protocol.

## Results

# OMI loss ablation summary

Primary operating point: validation threshold maximizing the harmonic mean of F1 and Sensitivity on the 0.01–0.99 grid. Results are mean ± sample SD over seeds [42, 2024, 2025, 2026, 3407].

| Loss | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8751 ± 0.0038 | 0.3270 ± 0.0155 | 0.6043 ± 0.0644 | 0.9252 ± 0.0289 | 0.3264 ± 0.0741 | 0.9769 ± 0.0031 | 0.4160 ± 0.0437 | 0.3974 ± 0.0352 | 0.9083 ± 0.0241 |
| Weighted BCE | 0.8822 ± 0.0053 | 0.3128 ± 0.0151 | 0.7234 ± 0.0677 | 0.8765 ± 0.0292 | 0.2510 ± 0.0348 | 0.9829 ± 0.0035 | 0.3696 ± 0.0258 | 0.3732 ± 0.0143 | 0.8684 ± 0.0242 |
| Focal | 0.8753 ± 0.0037 | 0.3277 ± 0.0195 | 0.6787 ± 0.0231 | 0.8968 ± 0.0169 | 0.2709 ± 0.0347 | 0.9805 ± 0.0013 | 0.3861 ± 0.0338 | 0.3798 ± 0.0308 | 0.8853 ± 0.0156 |
| CB-Focal | 0.8765 ± 0.0036 | 0.3246 ± 0.0135 | 0.6787 ± 0.0836 | 0.8934 ± 0.0261 | 0.2662 ± 0.0303 | 0.9805 ± 0.0044 | 0.3790 ± 0.0163 | 0.3742 ± 0.0104 | 0.8821 ± 0.0205 |

Best CB-Focal: beta=0.99, gamma=1.0; mean per-seed threshold=0.2180.
Paired seed AUPRC improvements: 3/5.
Verdict: **No convincing improvement**.

## CB-Focal minus Baseline (separate bootstrap estimand)

The summary table reports five-seed mean ± sample SD. The paired patient
bootstrap intervals below use the stored pooled validation predictions and
2,000 resamples, so their point estimates are a different estimand rather
than the seed mean. Zero observed bootstrap exceedances are reported as
`p<1/2001`.

- ΔAUROC=-0.00179; paired patient bootstrap 95% CI [-0.0111, +0.0092], p=0.685
- ΔAUPRC=-0.00689; paired patient bootstrap 95% CI [-0.0262, +0.0138], p=0.5
- ΔSensitivity=+0.13897; paired patient bootstrap 95% CI [+0.0723, +0.2118], p<1/2001
- ΔSpecificity=-0.05259; paired patient bootstrap 95% CI [-0.0635, -0.0422], p<1/2001
- ΔPPV=-0.06449; paired patient bootstrap 95% CI [-0.1011, -0.0295], p<1/2001
- ΔF1=-0.04167; paired patient bootstrap 95% CI [-0.0820, +0.0014], p=0.06

External hidden-label test results were not used to select loss parameters or thresholds. This report is validation evidence; one frozen external evaluation may follow after the method is finalized.


# ECG/age/sex/median-ECG feature ablation summary

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


## Integrity audit

# Loss/feature ablation integrity audit

Overall: **WARN** (provisional deterministic audit; no fresh independent semantic reviewer was authorized).

- Parsed and recomputed 63 run JSON files and their validation prediction CSVs.
- Baseline checkpoint exactly reproduces the historical checkpoint: True.
- Ablation runner references the external test: False.
- Final loss seed sets: {'baseline': [42, 2024, 2025, 2026, 3407], 'weighted_bce': [42, 2024, 2025, 2026, 3407], 'focal': [42, 2024, 2025, 2026, 3407], 'cb_focal': [42, 2024, 2025, 2026, 3407]}.
- Deterministic errors: none.

WARN is retained even when deterministic checks pass because the historical scalers include internal-validation data and the required fresh-agent semantic audit could not be performed under the active no-delegation constraint.
