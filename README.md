# ACS ECG OMI/AMI research archive

This repository is the versioned, reproducible part of the ACS ECG study. It
contains model and data-pipeline code, experiment configurations, research
plans, aggregate metrics, figures, and report/provenance documents. Controlled
ECG waveforms and patient-level files remain outside GitHub.

## Current task and final method

The current task predicts four labels independently for every ECG:

`STEMI`, `NSTEMI`, `UA`, and `OMI`.

Each label has its own logit and sigmoid probability. Thresholds are selected
per label from validation OOF predictions and then frozen for external test
submission. This is a four-label multi-label binary task, not a mutually
exclusive softmax three-/four-class task. Records with
`STEMI/NSTEMI/UA = [0, 0, 0]` are retained as explicit non-ACS-subtype
negatives; OMI is an independent label and may co-occur with STEMI or NSTEMI.

The latest local-best comparison uses ECG-LFM with parameter-efficient
adaptation:

| Method | Trainable/effective parameters | Relative result |
|---|---:|---|
| Baseline (partial fine-tuning) | 622,479,108 / 622,880,004 (99.94% effective) | reference |
| LoRA | 5,902,340 / 628,778,244 (0.94%) | selected local-best |
| DoRA | 6,270,980 / 629,146,884 (1.00%) | ablation |

The Baseline freezes the convolutional ECG frontend (`feature_grad_mult=0`)
while updating the Transformer context/projection/normalization layers and
classification head; it is therefore partial fine-tuning, not literal
all-parameter fine-tuning. LoRA updates about 105.5x fewer effective
parameters than this Baseline and DoRA about 99.3x fewer. This parameter count
reduction should not be interpreted as the same percentage reduction in total
VRAM or wall-clock time because the frozen backbone and activations are still
loaded.

The full comparison, OOF metrics, external-platform metrics, threshold
choices, and caveats are in
[`ecg_lfm_local_best_lora_dora_optimization_20260828.md`](docs/reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md).

## Data and evaluation protocol

- 17,960 ECG records from 17,018 patients.
- Label counts: STEMI 1,442; NSTEMI 1,235; UA 6,213; OMI 1,151.
- Patient-level iterative multi-label stratified five-fold split, seed 2026.
- P2 ECG preprocessing (0.5--40 Hz band-pass, 100 Hz representation); each
  training fold fits its own robust scaler and augmentation is training-only.
- The ECG-LFM pipeline uses the official 500 Hz input contract where noted in
  the corresponding plan/report.
- Main OOF metrics are AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1,
  MCC, accuracy, and per-label confusion matrices.
- Thresholds are chosen on validation OOF predictions only; external labels
  are never used for tuning.

The aggregate results currently archived for the ECG-LFM Baseline, LoRA/DoRA,
loss/feature ablations, and traditional architecture comparison are under
[`results/`](results/). The associated reports are indexed in
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
