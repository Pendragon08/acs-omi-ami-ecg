# Corrected E4 joint ACS/OMI experiment

**Status:** complete (GPU1, 2026-08-29).  This report is generated from the
versioned aggregate `results/corrected_joint_acs_omi_e4/summary.json`; no
record-level predictions or checkpoints are committed.

## Task and data audit

The source columns are `STEMI`, `NSTEMI`, `UA`, `OMI`.  STEMI/NSTEMI/UA are
encoded as one categorical ACS target (`OTHER`, `STEMI`, `NSTEMI`, `UA`), while
OMI remains an independent binary target.  The explicit `[0,0,0]` state is
retained as `OTHER`; only unknown/non-binary labels would be excluded, and any
ACS-subtype overlap is a hard error.

| Quantity | Value |
|---|---:|
| labelled ECG records | 17,960 |
| patients | 17,018 |
| `OTHER` (`[0,0,0]`) | 9,070 |
| STEMI / NSTEMI / UA | 1,442 / 1,235 / 6,213 |
| OMI positive | 1,151 |
| excluded/invalid rows | 0 |
| overlapping ACS subtype rows | 0 |
| external test records | 1,995 |

The historical patient-level five-fold assignment was reused and validated;
each fold has disjoint patient sets and the five validation folds exactly cover
all 17,960 records.  Each fold fits its robust scaler on training records only
and applies augmentation only to training batches.

## Model, loss and training

E4 is the Xception morphology + lead-spatial GAT + WavelNet frequency expert
with task routers and KAN heads.  It emits five logits: four ACS logits and
one OMI logit.  The loss is

`CrossEntropyLoss(ACS) + BCEWithLogitsLoss(OMI)`

with no additional OMI weighting (`omi_weighting=none`, `lambda=1`).  Training
used Adam, learning rate `1e-4`, batch size `128`, at most 100 epochs,
early-stopping patience 15, gradient clipping 1.0 and no scheduler.  The fold
checkpoint metric was the mean of ACS macro OVR AUROC and OMI AUROC.  Best
epochs for folds 0–4 were 16, 21, 14, 23 and 21.

Loss and learning-rate curves are in
`results/corrected_joint_acs_omi_e4/training_validation_loss_lr.png` and the
underlying epoch table is `training_curves.csv`.  ROC/PR and confusion figures
are also stored in that directory.

## OOF threshold analysis

Thresholds were searched on the 0.01–0.99 grid using pooled validation OOF
probabilities. ACS uses softmax argmax and has no threshold. The table below
keeps AUROC/AUPRC/ACS values fixed across OMI thresholds; only OMI hard metrics
change.

| OMI rule | threshold | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.5 | 0.50 | 0.2911 | 0.9808 | 0.5091 | 0.9528 | 0.3704 | 0.3544 | 0.9366 | 335/16486/323/816 |
| max F1 | 0.32 | 0.4883 | 0.9506 | 0.4037 | 0.9644 | 0.4420 | 0.4020 | 0.9210 | 562/15979/830/589 |
| Youden J (**registered**) | **0.03** | **0.8358** | **0.8041** | **0.2261** | **0.9862** | **0.3559** | **0.3686** | **0.8061** | **962/13516/3293/189** |
| sensitivity ≥0.90 | 0.01 | 0.9288 | 0.6470 | 0.1526 | 0.9925 | 0.2622 | 0.2891 | 0.6650 | 1069/10875/5934/82 |
| sensitivity ≥0.95 | 0.99* | 0.0000 | 1.0000 | 0.0000 | 0.9359 | 0.0000 | 0.0000 | 0.9359 | 0/16809/0/1151 |

`*` No grid threshold achieved sensitivity ≥0.95; the implementation records
the fallback with maximum specificity and flags the constraint as unsatisfied.
The registered operating point is Youden J because it provides high OMI
sensitivity while retaining measurable specificity; the full alternatives
remain available for clinical operating-point selection.

## Pooled OOF results at Youden J

### ACS four-way and ACS-positive conditional three-way

| Task | Accuracy | Macro F1 | Macro AUROC (OVR) | Macro AUPRC (OVR) |
|---|---:|---:|---:|---:|
| ACS (`OTHER` + 3 subtypes) | 0.5601 | 0.4882 | 0.7668 | 0.5061 |
| ACS-positive conditional 3-way | 0.7951 | 0.6271 | — | — |

| ACS class (argmax one-vs-rest) | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OTHER | 0.6814 | 0.6685 | 0.7304 | 0.5220 | 0.6093 | 0.6550 | 0.6644 | 0.2583 | 6625/4641/4249/2445 |
| STEMI | 0.9361 | 0.6431 | 0.6644 | 0.9579 | 0.5792 | 0.9703 | 0.6189 | 0.5847 | 958/15822/696/484 |
| NSTEMI | 0.7966 | 0.2565 | 0.2097 | 0.9715 | 0.3524 | 0.9433 | 0.2629 | 0.2315 | 259/16249/476/976 |
| UA | 0.6531 | 0.4565 | 0.3570 | 0.7890 | 0.4722 | 0.6988 | 0.4066 | 0.1580 | 2218/9268/2479/3995 |

| ACS-positive subtype | Sensitivity | Specificity | PPV | NPV | F1 | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.6907 | 0.9423 | 0.6985 | 0.9402 | 0.6946 | 0.6358 | 996/7018/430/446 |
| NSTEMI | 0.2235 | 0.9568 | 0.4547 | 0.8842 | 0.2997 | 0.2472 | 276/7324/331/959 |
| UA | 0.9329 | 0.6037 | 0.8453 | 0.7949 | 0.8869 | 0.5861 | 5796/1616/1061/417 |

### OMI binary

AUROC `0.8886`, AUPRC `0.3864`, sensitivity `0.8358`, specificity `0.8041`,
PPV `0.2261`, NPV `0.9862`, F1 `0.3559`, MCC `0.3686`, accuracy `0.8061`;
confusion matrix TP/TN/FP/FN = `962/13,516/3,293/189`.

Across folds, the descriptive mean ± sample SD at the pooled Youden threshold
was ACS accuracy `0.5601 ± 0.0118`, ACS macro-F1 `0.4864 ± 0.0193`, ACS macro
AUROC `0.7687 ± 0.0030`, OMI AUROC `0.8925 ± 0.0096`, OMI AUPRC
`0.3965 ± 0.0278`, OMI sensitivity `0.8358 ± 0.0350`, specificity
`0.8041 ± 0.0307`, F1 `0.3578 ± 0.0240`, and MCC `0.3703 ± 0.0194`.

## External platform

The frozen corrected submission was uploaded once to the existing platform;
no external labels were read locally.  Platform metrics (1,995 records) were:

| Label | Sensitivity | Specificity | PPV | NPV | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| STEMI | 0.7778 | 0.9678 | 0.6811 | 0.9801 | 0.7262 | 0.9524 |
| NSTEMI | 0.1544 | 0.9736 | 0.3000 | 0.9403 | 0.2039 | 0.9178 |
| UA | 0.3460 | 0.7944 | 0.4664 | 0.7005 | 0.3973 | 0.6411 |
| OMI | 0.9062 | 0.7600 | 0.2057 | 0.9916 | 0.3353 | 0.7694 |

The platform accepts source one-hot columns, so ACS predictions in the
submission were generated from the categorical argmax and cannot contain
simultaneous STEMI/NSTEMI/UA positives.  The response and submission hash are
in `results/corrected_joint_acs_omi_e4/official_platform_4label_metrics.json`
and `summary.json` respectively.

## Interpretation and limitations

This run fixes the endpoint definition; old independent-sigmoid four-label
results are retained only as `LEGACY—任务定义错误` and are not compared by
post-hoc relabelling.  The fold checkpoint and OMI threshold are selected from
the same validation OOF records used for the summary, so hard OOF metrics are
descriptive/non-nested and likely optimistic.  A nested patient-level
calibration/evaluation is required for a confirmatory paper estimate.
