# 校正后的 ACS 亚型 + OMI 任务方案

> 中文执行摘要：本方案是当前所有新实验的唯一标签与评估规范。ACS 的
> STEMI/NSTEMI/UA 使用互斥四分类（含 OTHER），OMI/NOMI 使用独立二分类；
> 所有模型、阈值和外部提交必须遵守该定义。旧的四独立 sigmoid 方案保留为
> `LEGACY/INVALID` 审计记录，不得用于最终性能结论。

**Effective date:** 2026-08-29  
**Status:** canonical protocol for all new runs

## Why the previous four-label task was invalid

The source columns `STEMI`, `NSTEMI` and `UA` describe mutually exclusive ACS
subtypes.  A record cannot be both NSTEMI and UA.  Training three independent
sigmoid/BCE heads nevertheless permits impossible multi-positive predictions
and makes independent threshold selection a different task from ACS subtype
classification.  The old code and metrics are preserved as legacy evidence,
but are not post-hoc reinterpreted.

## Correct target representation

The source four columns are retained for audit.  For each record:

| source state | ACS target | OMI target |
|---|---|---|
| `[0,0,0,0]` | `OTHER` (class 0) | NOMI (0) |
| `[1,0,0,0]` | `STEMI` (class 1) | NOMI (0) |
| `[0,1,0,0]` | `NSTEMI` (class 2) | NOMI (0) |
| `[0,0,1,0]` | `UA` (class 3) | NOMI (0) |
| any subtype above with `OMI=1` | same ACS class | OMI (1) |
| `[0,0,0,1]` | `OTHER` | OMI (1) |

Rows with unknown/non-binary labels are excluded.  `[0,0,0]` is never treated
as missing; it is the explicit non-ACS state.  Any subtype overlap is a hard
data-integrity error.

### Fairseq label-file compatibility

The official ECG-LFM Fairseq validation loop reshapes each batch label to
`model.num_labels`. Therefore the ECG-LFM runner materialises a private
five-column `y_joint_5.npy` file in the execution environment with columns
`[OTHER, STEMI, NSTEMI, UA, OMI]`. This is a one-hot representation of the
same target above, not an additional label or a change to the source data.
The criterion accepts both source `[STEMI, NSTEMI, UA, OMI]` rows (for checks)
and the Fairseq-compatible five-column rows, and rejects any non-binary or
overlapping ACS state before training.

## Model and loss

Every corrected model emits five logits.  The first four are a categorical
ACS head (`OTHER/STEMI/NSTEMI/UA`) trained with `CrossEntropyLoss`; the fifth is
an independent OMI logit trained with numerically stable
`BCEWithLogitsLoss`.  The objective is

`L = L_ACS_CE + lambda_OMI * L_OMI_BCE`.

No OMI loss is allowed to add a second `pos_weight` unless it is a separately
registered ablation.  Any class/positive weighting is computed from the
training fold only.  ACS inference is `argmax(softmax)` and therefore cannot
produce two subtypes.  Only OMI receives a validation-selected threshold
(0.5, max-F1, Youden J, sensitivity-constrained 0.90/0.95 are recorded).

## Split and evaluation

The historical patient-level five-fold assignment is reused when valid;
otherwise a deterministic patient-level iterative split uses the joint
one-hot ACS-state + OMI target.  No patient appears in both train and
validation.  Training-only augmentation and train-fold robust scaling are
unchanged.  Reports separate four-way ACS metrics, ACS-positive conditional
three-way subtype metrics, and OMI binary metrics (AUROC, AUPRC, sensitivity,
specificity, PPV, NPV, F1, MCC, accuracy, TP/TN/FP/FN).  External submissions
encode the ACS argmax back into one-hot STEMI/NSTEMI/UA columns plus OMI.

## Corrected entry points

* `experiments/acs_joint_e4.py` — E4 HMoE/WavelNet/KAN.
* `experiments/acs_joint_model_comparison.py` — InceptionTime, ResNet1D-Wang,
  LSTM, ViT and Transformer1D.
* `experiments/ecg_lfm/run_joint_fold.py` plus `configs/ecg_lfm/corrected_joint_*.yaml` — official ECG-LFM with the joint criterion (current GPU1 queue: `results/corrected_ecg_lfm_joint_bs64_v2/`).
* `third_party/ECG-LFM/fairseq_signals/criterions/acs_joint.py` — stable
  Fairseq criterion; summed loss is returned because Fairseq divides by
  `sample_size` during gradient scaling.

The corrected PyTorch runners write provenance, split/label audits, per-epoch
loss/LR curves, ROC/PR and confusion figures.  ECG-LFM's Fairseq launcher
writes provenance and raw JSONL first; its joint record-level evaluator is a
separate post-training step recorded in the ECG-LFM report.  Raw ECG, patient tables, checkpoints
and row-level predictions stay on the controlled server and are not committed
to GitHub.
