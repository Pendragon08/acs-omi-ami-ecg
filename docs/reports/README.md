# ACS experiment report index

Reports are grouped by the task definition used in the run. The current
research endpoint is the four-label independent binary task; older softmax
and three-label experiments are retained for audit history.

## Current ECG-LFM reports

| Report | Scope | Status |
|---|---|---|
| [ecg_lfm_local_best_lora_dora_optimization_20260828.md](ecg_lfm_local_best_lora_dora_optimization_20260828.md) | Local-best Baseline (partial fine-tuning), LoRA and DoRA; five-fold OOF metrics, external LoRA evaluation, parameter efficiency and cost caveats | complete |
| [ecg_lfm_external_comparison_20260828.md](ecg_lfm_external_comparison_20260828.md) | ECG-LFM Baseline versus historical E4 external-platform comparison | complete |
| [ecg_lfm_baseline_external_platform_20260828.md](ecg_lfm_baseline_external_platform_20260828.md) | Baseline four-label external-platform submission and per-label metrics | complete |
| [ecg_lfm_4label_finetuning_20260827.md](ecg_lfm_4label_finetuning_20260827.md) | Official ECG-LFM preprocessing, four-label fine-tuning, gates, five-fold protocol and external submission | complete |
| [ecg_lfm_hyperparameter_loss_ablation_20260827.md](ecg_lfm_hyperparameter_loss_ablation_20260827.md) | Learning-rate/batch-size checks, loss comparison, training curves and overfitting audit | complete |
| [model_comparison_4label_20260828.md](model_comparison_4label_20260828.md) | ECG-LFM versus InceptionTime, ResNet1D-Wang, LSTM, ViT-1D and Transformer1D | complete |

## Loss and feature ablation

| Report | Scope | Status |
|---|---|---|
| [loss_feature_ablation/loss_ablation_summary.md](loss_feature_ablation/loss_ablation_summary.md) | Baseline, automatic weighted BCE, focal and class-balanced focal loss; five seeds and threshold strategies | complete |
| [loss_feature_ablation/feature_ablation_summary.md](loss_feature_ablation/feature_ablation_summary.md) | Raw ECG, age, sex and median-ECG feature comparisons for AMI/OMI | complete |
| [loss_feature_ablation/feature_association_summary.md](loss_feature_ablation/feature_association_summary.md) | Feature/label association analysis and interpretation | complete |
| [loss_feature_ablation/EXPERIMENT_LOG.md](loss_feature_ablation/EXPERIMENT_LOG.md) | Chronological loss/feature ablation run record | complete |
| [loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md](loss_feature_ablation/EXPERIMENT_AUDIT_ABLATION.md) | Fair-comparison and leakage audit | complete |

## Historical four-label and ACS-subtype reports

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

The four-label task reports a metric for each independent label and a macro
summary. Accuracy is not the primary criterion for the imbalanced OMI/NSTEMI
labels; sensitivity, AUPRC, F1, PPV, AUROC and specificity are interpreted
together. Thresholds are selected on validation OOF predictions and frozen
before external testing. External-platform responses generally provide
per-label confusion-derived metrics but not AUROC/AUPRC.

## External artefacts and raw logs

The external-platform response files, per-record predictions, checkpoints and
raw server logs remain on the controlled execution environment. Only aggregate
metrics and selected figures are versioned here. See
[`docs/logs/EXPERIMENT_LOG_INDEX.md`](../logs/EXPERIMENT_LOG_INDEX.md) for the
run-by-run log mapping and retention policy.
