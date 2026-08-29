# ACS experiment report index

Reports are grouped by the task definition used in the run. The current
research endpoint is the corrected joint task: categorical ACS
`OTHER/STEMI/NSTEMI/UA` plus independent binary OMI/NOMI. Any report that
describes three independent ACS sigmoids or four independent sigmoids is
legacy/invalid for the final task and is retained only for audit history.

## Corrected task reports (2026-08-29)

| Report | Scope | Status |
|---|---|---|
| [corrected_task_formulation_20260829.md](corrected_task_formulation_20260829.md) | Label audit, model/loss/split correction and rerun registry | canonical |
| [corrected_joint_e4_20260829.md](corrected_joint_e4_20260829.md) | E4 HMoE/WavelNet/KAN joint ACS categorical + OMI binary five-fold rerun | running; update after completion |
| [corrected_joint_model_comparison_20260829.md](corrected_joint_model_comparison_20260829.md) | InceptionTime, ResNet1D-Wang, LSTM, ViT and Transformer1D under corrected task | queued/running |
| [corrected_ecg_lfm_joint_20260829.md](corrected_ecg_lfm_joint_20260829.md) | ECG-LFM Baseline/LoRA/DoRA with joint criterion | queued/running |

## Legacy ECG-LFM reports (independent-sigmoid task; invalid for current ACS subtype formulation)

| Report | Scope | Status |
|---|---|---|
| [ecg_lfm_local_best_lora_dora_optimization_20260828.md](ecg_lfm_local_best_lora_dora_optimization_20260828.md) | Legacy independent-sigmoid Baseline/LoRA/DoRA | legacy—invalid |
| [ecg_lfm_external_comparison_20260828.md](ecg_lfm_external_comparison_20260828.md) | ECG-LFM Baseline versus historical E4 external-platform comparison | complete |
| [ecg_lfm_baseline_external_platform_20260828.md](ecg_lfm_baseline_external_platform_20260828.md) | Baseline four-label external-platform submission and per-label metrics | complete |
| [ecg_lfm_4label_finetuning_20260827.md](ecg_lfm_4label_finetuning_20260827.md) | Legacy independent-sigmoid ECG-LFM preprocessing/fine-tuning | legacy—invalid |
| [ecg_lfm_hyperparameter_loss_ablation_20260827.md](ecg_lfm_hyperparameter_loss_ablation_20260827.md) | Learning-rate/batch-size checks, loss comparison, training curves and overfitting audit | complete |
| [model_comparison_4label_20260828.md](model_comparison_4label_20260828.md) | Legacy independent-sigmoid architecture comparison | legacy—invalid |

## Loss and feature ablation

| Report | Scope | Status |
|---|---|---|
| [loss_feature_ablation/loss_ablation_summary.md](loss_feature_ablation/loss_ablation_summary.md) | Baseline, automatic weighted BCE, focal and class-balanced focal loss; five seeds and threshold strategies | complete |
| [loss_feature_ablation/feature_ablation_summary.md](loss_feature_ablation/feature_ablation_summary.md) | Raw ECG, age, sex and median-ECG feature comparisons for AMI/OMI | complete |
| [loss_feature_ablation/feature_association_summary.md](loss_feature_ablation/feature_association_summary.md) | Feature/label association analysis and interpretation | complete |
| [loss_feature_ablation/EXPERIMENT_LOG.md](loss_feature_ablation/EXPERIMENT_LOG.md) | Chronological loss/feature ablation run record | complete |
| [loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md](loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md) | Fair-comparison and leakage audit | complete |

## Historical four-label and ACS-subtype reports (legacy unless explicitly marked corrected)

| Report | Scope | Status |
|---|---|---|
| [hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md](hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md) | Historical E4 HMoE/WavelNet/KAN four-label run, OOF thresholds and platform metrics | complete |
| [e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json](e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json) | E4 command, split and aggregate artifact provenance | complete |
| [acs_sparse_hmoe_kan_feasibility_gpu1_20260826.md](acs_sparse_hmoe_kan_feasibility_gpu1_20260826.md) | Revised sparse HMoE/KAN feasibility run | historical |
| [hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md](hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md) | Historical three-label E4 run | historical |
| [stemi_nstemi_ua_multilabel_results_20260826.md](stemi_nstemi_ua_multilabel_results_20260826.md) | Earlier GPU0/GPU1 three-label and AMI/OMI experiments | historical |
| [hmoe_wavelnet_e4_youden_threshold_results_20260826.md](hmoe_wavelnet_e4_youden_threshold_results_20260826.md) | Youden/balanced-accuracy threshold audit | historical |
| [hmoe_wavelnet_e4_four_class_results_20260826.md](hmoe_wavelnet_e4_four_class_results_20260826.md) | Historical mutually exclusive four-class correction | historical |

## Reading the results correctly

The corrected reports separate categorical ACS metrics (four-way and
ACS-positive conditional three-way) from OMI binary metrics. Accuracy is not
the primary criterion for imbalanced OMI/NSTEMI; sensitivity, AUPRC, F1, PPV,
AUROC and specificity are interpreted together. Only the OMI threshold is
selected on validation OOF predictions and frozen before external testing;
ACS uses argmax and cannot produce simultaneous subtype positives. Since the
outer fold also selects its checkpoint on that fold's validation records,
pooled OOF hard metrics are descriptive/non-nested rather than confirmatory
estimates; corrected reports state this limitation explicitly.

## External artefacts and raw logs

The external-platform response files, per-record predictions, checkpoints and
raw server logs remain on the controlled execution environment. Only aggregate
metrics and selected figures are versioned here. See
[`docs/logs/EXPERIMENT_LOG_INDEX.md`](../logs/EXPERIMENT_LOG_INDEX.md) for the
run-by-run log mapping and retention policy.
