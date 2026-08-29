# Experiment log index and retention policy

Generated from the execution-server run inventory on 2026-08-29. The paths
below are relative to the private project root on that server. They identify
the raw log files that back the versioned reports and aggregate result tables.

## Important answer about GitHub

The raw files listed below are **not** committed to GitHub. The repository
contains this curated index, the chronological loss/feature log, aggregate
training curves, and report/provenance documents. This is intentional: raw
logs are bulky, can expose infrastructure paths, and fairseq JSONL streams are
execution artifacts. No raw ECG, patient-level table, checkpoint, or
record-level prediction is included in the archive.

## Current ECG-LFM fine-tuning and adaptation

| Run family | GPU / status | Raw server logs | Versioned evidence |
|---|---|---|---|
| Official ECG-LFM preprocessing and baseline gates | GPU1, complete | `logs/ecg_lfm/preprocess_full.log`, `raw_download.log`, `raw_extract.log`, `manifest_and_contracts.log`, `post_checkpoint_gates.log`, `post_checkpoint_gates_storage_fixed.log`, `after_gates.log`, `after_gates_large_batch.log` | `docs/reports/ecg_lfm_4label_finetuning_20260827.md`, `results/ecg_lfm_4label/` |
| Baseline fold-0 and tuning checks | GPU1, complete | `logs/ecg_lfm/l1a_fold0_parallel.log`, `l1a_fold0_large_batch.log`, `l1a_resume.log`, `l1a_resume_repaired.log`, `logs/ecg_lfm_tuning/fold0_gpu1_queue.log`, `logs/ecg_lfm_tuning/stages.log`, `stages_resume.log` | `docs/reports/ecg_lfm_hyperparameter_loss_ablation_20260827.md`, tuning summaries |
| LoRA local-best five-fold run (legacy independent sigmoid) | GPU1, complete | `logs/ecg_lfm_local_best_adapters_gpu1.log`, `logs/ecg_lfm_local_best_finalize.log` | legacy-invalid `docs/reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md` |
| DoRA local-best five-fold run (legacy independent sigmoid) | GPU0, complete | `logs/ecg_lfm_dora_gpu0.log` | legacy-invalid `docs/reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md` |
| External ECG-LFM Baseline submission | complete | `logs/ecg_lfm_baseline_external_submit.log` | `docs/reports/ecg_lfm_baseline_external_platform_20260828.md` |

## Loss and feature ablation

The raw seed-level logs are under `logs/loss_feature_ablation/`, including the
stage-1 grid, stage-2 confirmations for the five seeds, feature comparisons,
CB-Focal beta/gamma sweeps, and the pipeline watcher. The canonical
chronological record is retained as
[`docs/reports/loss_feature_ablation/EXPERIMENT_LOG.md`](../reports/loss_feature_ablation/EXPERIMENT_LOG.md).
The aggregate CSV/JSON tables and figures are under
`results/loss_feature_ablation/`.

Key raw log groups:

- `stage1_master.log`, `stage1_complete.log`, `stage2_master.log`,
  `baseline_master.log`, `baseline_seed2026.log`;
- `feature_*_seed2026.log` for raw ECG, age, sex, median ECG, and clinical
  feature variants;
- `confirm_*_seed*.log` for the five-seed confirmation runs;
- `loss_cb_b*.log`, `loss_focal_*.log`, `loss_wbce.log` for the loss grid;
- `pipeline_watcher.log` for completion monitoring.

## Traditional architecture comparison

The corrected E4 rerun completed on physical GPU1; its raw screen log is
`logs/corrected/acs_joint_e4_gpu1.log` and its aggregate report is
`docs/reports/corrected_joint_e4_20260829.md`.

| Run family | GPU / status | Raw server logs | Versioned evidence |
|---|---|---|---|
| Corrected E4 joint ACS categorical + OMI binary | GPU1, complete | `logs/corrected/acs_joint_e4_gpu1.log` | `docs/reports/corrected_joint_e4_20260829.md`, `results/corrected_joint_acs_omi_e4/` |
| Corrected traditional architectures | GPU1, complete | `logs/corrected_joint_model_comparison/all_gpu1.log` | `docs/reports/corrected_joint_model_comparison_20260829.md`, `results/corrected_joint_model_comparison/` |
| Corrected traditional external submissions | complete | `logs/corrected_joint_model_comparison/platform_*.log`, `external_predict.log` | per-model `official_platform_4label_metrics.json` and provenance JSON |
| Four-label model comparison training (legacy independent sigmoid) | GPU0/GPU1, complete | `logs/model_comparison_4label/model_cmp_gpu0.log`, `model_cmp_gpu0_resume.log`, `model_cmp_gpu1.log`, `model_cmp_gpu1_resume.log`, `queue.log` | legacy-invalid `docs/reports/model_comparison_4label_20260828.md` |
| External-platform submissions | complete | `logs/model_comparison_4label_external.log`, `logs/model_comparison_4label_platform_*.log` | per-model aggregate platform metric JSON files |

## Corrected joint ECG-LFM rerun

| Run family | GPU / status | Raw server logs | Versioned evidence |
|---|---|---|---|
| Corrected ECG-LFM Baseline/LoRA/DoRA (ACS categorical + OMI binary) | GPU1, running | `logs/corrected_ecg_lfm_joint_bs64/master.log` and per-adaptation/fold logs | `docs/reports/corrected_ecg_lfm_joint_20260829.md`; aggregate outputs will be archived after completion |

## Historical ACS/HMoE experiments

Historical runs remain on the server and are summarized in the reports under
`docs/reports/`: `logs/hmoe_kan_github_gpu1.log`,
`logs/hmoe_kan_stemi_nstemi_ua_gpu1.log`, `logs/hmoe_official_test_gpu1.log`,
`logs/stemi_nstemi_ua_gpu0.log`, `logs/protocol_v2/`,
`logs/historical_current_rerun/`, and the earlier `logs/vmd_*`/`logs/xct_*`
families. They are not copied into GitHub as raw files.

## How to retrieve a raw log internally

Use the approved SSH/SFTP access to the private execution project and read the
relative path listed above. Do not paste credentials into scripts or commit
the retrieved files. For a reproducible citation, reference the corresponding
report, configuration, seed, GPU, and aggregate result path in a new report;
do not add record-level output to Git.
