#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
ENV_DIR=/data/code/ouyanghao/miniconda3/envs/ecg-lfm
RUN_ROOT=results/joint10_cv_comparison
EVAL_ROOT=results/joint10_cv_comparison_evaluation
RUN_NAME=baseline_rerun
MASTER_LOG="$ROOT/logs/joint10_cv_comparison/baseline_rerun_gpu1.log"

mkdir -p "$(dirname "$MASTER_LOG")" "$ROOT/$RUN_ROOT" "$ROOT/$EVAL_ROOT"
exec > >(tee -a "$MASTER_LOG") 2>&1

export CUDA_VISIBLE_DEVICES=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HYDRA_FULL_ERROR=1

cd "$ROOT"
echo "[$(date -Is)] baseline nested five-fold rerun started on physical GPU1"

for fold in 0 1 2 3 4; do
  echo "[$(date -Is)] fold ${fold} training started"
  "$ENV_DIR/bin/python" experiments/ecg_lfm/run_joint_fold.py \
    --fold "$fold" \
    --adaptation corrected_joint_lora \
    --run_name "$RUN_NAME" \
    --manifest_root manifests/ecg_lfm_nested \
    --output_root "$RUN_ROOT" \
    --weighting w0 \
    --seed 2026

  echo "[$(date -Is)] fold ${fold} evaluation started"
  "$ENV_DIR/bin/python" experiments/ecg_lfm/evaluate_nested_fold.py \
    --fold "$fold" \
    --run_name "$RUN_NAME" \
    --run_root "$RUN_ROOT" \
    --manifest_root manifests/ecg_lfm_nested \
    --output_root "$EVAL_ROOT" \
    --device cuda:0 \
    --workers 8 \
    --batch 64
  echo "[$(date -Is)] fold ${fold} completed"
done

touch "$ROOT/$EVAL_ROOT/$RUN_NAME/FIVE_FOLD_COMPLETE"
echo "[$(date -Is)] baseline nested five-fold rerun completed"
