# OMI 损失函数与 ECG 特征消融计划

> 中文说明：本计划规定 OMI 的 Baseline/WBCE/Focal/CB-Focal 和原始 ECG、
> 年龄、性别、中位数 ECG 对比；超参数只能由训练集/验证集决定，测试集只作
> 最终一次评估。

Status: approved for execution; prior independent reviewer plan incorporated.

## Frozen historical protocol

- Patient-level fixed split: train 16,174 ECG / validation 1,786 ECG; no patient overlap.
- Official external test: 1,995 hidden-label ECGs, not used for tuning.
- Baseline: XceptionTime + P3 + plain BCE, OMI lr=3e-4, batch=256.
- Adam, no scheduler, 100 epochs, patience 15, gradient clipping 1.0.
- P3 augmentation, initialization, shuffle and evaluation are frozen.
- The historical P3 scaler includes the internal validation fold. This known limitation is preserved across all groups for comparability and disclosed; it is not changed inside one ablation.

## Loss study

Screen hyperparameters on the validation set with tuning seed 2026:

- Baseline BCE.
- WBCE with `pos_weight=N_negative/N_positive`, training labels only.
- Focal gamma=2 and alpha in 0.25/0.50/0.75/training-negative-fraction.
- CB-Focal beta in 0.99/0.999/0.9999 and gamma in 1/2/3.

After validation-only selection, rerun Baseline, WBCE, best Focal and best CB-Focal with seeds 42/2024/2025/2026/3407. No sampler or extra `pos_weight` is combined with CB-Focal.

## Feature study

Use one fixed XceptionTimeFusion architecture and mask input branches. This makes the controlled comparison change visible information, not parameter count:

- raw ECG only (controlled fusion reference)
- median ECG only
- age+sex clinical only
- raw+age
- raw+sex
- raw+age+sex
- raw+median ECG
- raw+median ECG+age+sex

Screen both OMI and AMI with seed 2026. Confirm the best raw-plus-feature candidate against controlled raw over all five seeds.

## Outputs

- Per-run JSON, checkpoint, epoch-loss CSV, validation predictions and threshold grid.
- Thresholds 0.01–0.99 plus fixed 0.5, max-F1, Youden, sensitivity >=0.90/0.95, and harmonic mean of F1/Sensitivity.
- AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, F1, MCC, Accuracy, TP/TN/FP/FN.
- Mean±SD and patient-cluster paired bootstrap for final Baseline vs CB-Focal.

## Interpretation rule

Primary evidence is AUPRC, Sensitivity and F1 without a material loss of PPV/Specificity, consistency across seeds, and paired-bootstrap uncertainty. Accuracy alone cannot establish improvement.
