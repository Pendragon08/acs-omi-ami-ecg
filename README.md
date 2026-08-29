# ACS ECG OMI/AMI research archive

This repository is the versioned, reproducible part of the ACS ECG study. It
contains model and data-pipeline code, experiment configurations, research
plans, aggregate metrics, figures, and report/provenance documents. Controlled
ECG waveforms and patient-level files remain outside GitHub.

## Current task and final method (corrected)

The former four-independent-sigmoid formulation was a task-definition error:
`STEMI`, `NSTEMI` and `UA` are mutually exclusive ACS subtypes. The current
task is a joint multi-task model with a categorical ACS head
(`OTHER`, `STEMI`, `NSTEMI`, `UA`) and an independent binary OMI/NOMI head.
`OTHER` is the explicit source state `[0,0,0]` and is retained as a valid
negative. ACS predictions are always one argmax class; only OMI uses a
validation-selected threshold. Three-way ACS metrics are additionally
reported conditional on ACS-positive records.

The corrected implementation is `experiments/acs_joint_e4.py`; the old entry
point now routes there. Pre-correction independent-sigmoid results remain in
their original directories only as invalid legacy diagnostics and must not be
used as current performance claims.

The parameter-count table from the historical local-best ECG-LFM run is kept
below only as an audit record.  That run used the now-invalid four-independent-
sigmoid formulation and its performance must not be interpreted as a result
for the corrected task.  The corrected ECG-LFM Baseline/LoRA/DoRA comparison
is rerun under `results/corrected_ecg_lfm_joint` and will be reported there
after completion.

| Method | Trainable/effective parameters | Relative result |
|---|---:|---|
| Baseline (partial fine-tuning) | 622,479,108 / 622,880,004 (99.94% effective) | **legacy—invalid task** |
| LoRA | 5,902,340 / 628,778,244 (0.94%) | **legacy—invalid task** |
| DoRA | 6,270,980 / 629,146,884 (1.00%) | **legacy—invalid task** |

The Baseline freezes the convolutional ECG frontend (`feature_grad_mult=0`)
while updating the Transformer context/projection/normalization layers and
classification head; it is therefore partial fine-tuning, not literal
all-parameter fine-tuning. LoRA updates about 105.5x fewer effective
parameters than this Baseline and DoRA about 99.3x fewer. This parameter count
reduction should not be interpreted as the same percentage reduction in total
VRAM or wall-clock time because the frozen backbone and activations are still
loaded.

The historical comparison, OOF metrics, external-platform metrics, threshold
choices, and caveats are retained in
[`ecg_lfm_local_best_lora_dora_optimization_20260828.md`](docs/reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md).
It is a legacy record; use
[`corrected_ecg_lfm_joint_20260829.md`](docs/reports/corrected_ecg_lfm_joint_20260829.md)
for the corrected categorical-ACS + OMI task.

## Data and evaluation protocol

- 17,960 ECG records from 17,018 patients.
- Label counts: STEMI 1,442; NSTEMI 1,235; UA 6,213; OMI 1,151.
- Patient-level five-fold split, seed 2026; the historical patient assignment
  is reused when valid and audited for the corrected joint target.
- P2 ECG preprocessing (0.5--40 Hz band-pass, 100 Hz representation); each
  training fold fits its own robust scaler and augmentation is training-only.
- The ECG-LFM pipeline uses the official 500 Hz input contract where noted in
  the corresponding plan/report.
- Main OOF metrics are AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1,
  MCC, accuracy, and per-label confusion matrices.
- Thresholds are chosen on validation OOF predictions only; external labels
  are never used for tuning. Because the current five-fold implementation
  also selects each fold checkpoint on that fold's validation data, pooled OOF
  hard metrics are descriptive (not nested, confirmatory estimates); this
  limitation is stated in each corrected report.

Corrected runs write aggregate results under `results/corrected_*` (the jobs
are being rerun; paths are output targets until their summaries appear).
Pre-correction aggregates remain under their original names but are marked
legacy/invalid. The associated reports are indexed in
[`docs/reports/README.md`](docs/reports/README.md).

## Repository layout

```
acs_utils/                 metrics, thresholds, seeds, and configuration helpers
baselines/                 historical ACS baseline implementations
configs/                   historical and ECG-LFM experiment configurations
data_provider/             loaders and patient-level split utilities
experiments/               HMoE/KAN and ECG-LFM experiment entry points
losses/                    BCE, focal, and asymmetric-loss implementations
models/                    ECG-LFM adapter and parameter-summary code
nets/                      time-series model adapters
preprocessing/             ECG preprocessing and augmentation
scripts/                   historical scripts and scripts/ecg_lfm/ utilities
third_party/ECG-LFM/       pinned upstream source used by the ECG-LFM runs
docs/plans/                research plans and implementation specifications
docs/reports/              experiment reports and provenance summaries
docs/logs/                 curated log index (raw logs are not committed)
results/                   aggregate metrics, tables, and selected figures
```

## Reproduction boundary

The repository intentionally does **not** contain raw `.med`/`.npy`/`.npz`
waveforms, train/test CSV files, patient-level tables, OOF or external
per-record predictions, split-index files, checkpoints, or downloaded model
weights. These are controlled clinical data or large execution artifacts and
must be supplied through the approved local/server data path described in
[`DATASET.md`](DATASET.md). The upstream ECG-LFM source is included as code
only; downloaded checkpoints remain outside Git.

Use the YAML files under `configs/ecg_lfm/` and the entry points under
`experiments/ecg_lfm/`/`scripts/ecg_lfm/` to reproduce the ECG-LFM runs. Paths
to data, checkpoints, and the execution device are environment-specific and
must be supplied locally.

## Logs and traceability

The execution server retains the raw `screen`/training logs and fairseq JSONL
streams. They are not committed to GitHub because they are bulky and may
expose infrastructure paths or record-level context. The repository does
contain a curated experiment log index at
[`docs/logs/EXPERIMENT_LOG_INDEX.md`](docs/logs/EXPERIMENT_LOG_INDEX.md),
aggregate training curves, and report/provenance files that identify the
configuration, seed, GPU, output family, and completion status. Thus the
current GitHub archive has traceability summaries, not every raw log byte.

## License and data use

See [`DATASET.md`](DATASET.md) and [`REPOSITORY_MANIFEST.md`](REPOSITORY_MANIFEST.md)
before sharing this repository. Keep the repository private unless the data
and code-release approvals explicitly allow publication.
