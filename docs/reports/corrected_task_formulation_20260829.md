# Corrected task audit and rerun registry

This report records the correction requested on 2026-08-29.  The canonical
protocol is described in [`../plans/CORRECTED_TASK_FORMULATION.md`](../plans/CORRECTED_TASK_FORMULATION.md).

## Data audit

The controlled server audit found 17,960 labelled training ECG records from
17,018 patients.  Source counts are STEMI 1,442, NSTEMI 1,235, UA 6,213 and
OMI 1,151.  There are 9,070 records with no ACS subtype (`[0,0,0]`), including
one OMI-positive record.  No record has more than one positive ACS subtype.

The corrected loader excludes rows with unknown/non-binary labels, hard-fails
on any overlapping ACS subtype, and retains the explicit all-zero state as
`OTHER`.  The historical patient-level five-fold assignment is reused and
validated for record partition completeness and patient disjointness.

## Code audit outcome

The former E4 runner, traditional comparison runner, ECG-LFM configs and
external predictor used independent sigmoid/BCE outputs for all four columns.
They are now labelled legacy/invalid in the repository. New entry points use
one categorical ACS head plus one OMI binary head; no ACS threshold vector is
allowed. During the first corrected ECG-LFM launch, Fairseq exposed a
four-column versus five-logit validation reshape mismatch. The runner now
materialises and validates a private five-column
`[OTHER,STEMI,NSTEMI,UA,OMI]` label file; the failed attempt contributes no
metrics.

## Rerun registry

| family | implementation | output directory | status |
|---|---|---|---|
| E4 HMoE/WavelNet/KAN | `experiments/acs_joint_e4.py` | `results/corrected_joint_acs_omi_e4/` | complete; five folds + external platform; [`corrected_joint_e4_20260829.md`](corrected_joint_e4_20260829.md) |
| Traditional architectures | `experiments/acs_joint_model_comparison.py` | `results/corrected_joint_model_comparison/` | complete; five models × five folds + external platform; [`corrected_joint_model_comparison_20260829.md`](corrected_joint_model_comparison_20260829.md) |
| ECG-LFM Baseline/LoRA/DoRA | `experiments/ecg_lfm/run_joint_fold.py` | `results/corrected_ecg_lfm_joint_bs64_v2/` | running on GPU1; [`corrected_ecg_lfm_joint_20260829.md`](corrected_ecg_lfm_joint_20260829.md) |

This file is updated when each job finishes.  It will link the aggregate JSON,
CSV, figures, training logs and (where authorized) external-platform response
hashes.  Until a status changes to **complete**, no corrected metric is
claimed.
