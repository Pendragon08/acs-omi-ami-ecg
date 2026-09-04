#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
ENV_DIR=/data/code/ouyanghao/miniconda3/envs/ecg-lfm
LOG="$ROOT/logs/joint6_cv/smoke_gpu1.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM"
export OMP_NUM_THREADS=8
cd "$ROOT"

for method in lora partial; do
  echo "[$(date -Is)] ECG-LFM ${method} one-update smoke"
  "$ENV_DIR/bin/python" experiments/ecg_lfm/run_joint6_fold.py \
    --fold 0 --method "$method" \
    --output_root results/joint6_cv_smoke/ecg_lfm \
    --audit_output results/joint6_cv/label_audit \
    optimization.max_update=1 optimization.max_epoch=1 \
    checkpoint.no_save=true dataset.batch_size=2 \
    optimization.update_freq=[1] dataset.num_workers=0 \
    dataset.disable_validation=true
done

echo "[$(date -Is)] traditional InceptionTime end-to-end smoke"
"$ENV_DIR/bin/python" experiments/joint6_traditional_nested.py \
  --model InceptionTime --fold 0 \
  --output_root results/joint6_cv_smoke/traditional \
  --epochs 1 --patience 1 --batch_size 128 --num_workers 0 \
  --max_train_batches 1 --device cuda:0

touch "$ROOT/results/joint6_cv_smoke/SMOKE_COMPLETE"
echo "[$(date -Is)] Joint-6 smoke complete"
