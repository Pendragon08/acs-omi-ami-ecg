#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
PY=/data/code/ouyanghao/miniconda3/envs/oyh/bin/python
$PY scripts/ecg/select_ablation_stage1.py --require_complete \
  > logs/loss_feature_ablation/stage1_selection_stdout.json
$PY scripts/ecg/run_ablation_stage2.py
$PY scripts/ecg/summarize_loss_ablation.py --bootstrap 2000 \
  > logs/loss_feature_ablation/loss_summary_stdout.json
$PY scripts/ecg/summarize_feature_ablation.py \
  > logs/loss_feature_ablation/feature_summary_stdout.json
$PY scripts/ecg/audit_loss_feature_ablation.py \
  > logs/loss_feature_ablation/audit_stdout.json
$PY scripts/ecg/finalize_ablation_log.py
$PY scripts/ecg/update_final_report_ablation.py
printf 'ALL_ABLATIONS_COMPLETE %s\n' "$(date -Iseconds)"
