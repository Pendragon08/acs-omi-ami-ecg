# Corrected ACS subtype + OMI task formulation

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
* `experiments/ecg_lfm/run_joint_fold.py` plus `configs/ecg_lfm/corrected_joint_*.yaml` — official ECG-LFM with the joint criterion.
* `third_party/ECG-LFM/fairseq_signals/criterions/acs_joint.py` — stable
  Fairseq criterion; summed loss is returned because Fairseq divides by
  `sample_size` during gradient scaling.

All corrected jobs write provenance, split/label audits, per-epoch loss/LR
curves, ROC/PR and confusion figures.  Raw ECG, patient tables, checkpoints
and row-level predictions stay on the controlled server and are not committed
to GitHub.
