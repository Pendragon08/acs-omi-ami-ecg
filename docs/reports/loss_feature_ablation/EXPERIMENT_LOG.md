---
experiment_id: ACS-GPU1-260818-01
date: 2026-08-18
system: ACS-OMI-AMI
experiment_type: loss_and_feature_ablation
device: nlp03_physical_GPU1_RTX_A6000
status: running
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

Pending completion. Negative results will be retained.
