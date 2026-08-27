#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
mkdir -p logs/loss_feature_ablation results/loss_feature_ablation
PY=/data/code/ouyanghao/miniconda3/envs/oyh/bin/python
$PY -m exp.exp_acs_ablation \
  --study loss --task omi --loss bce --seed 2026 \
  --lr 3e-4 --batch_size 256 --epochs 100 --patience 15 --clip 1.0 \
  --num_workers 4 --device cuda --exp_name loss_feature_ablation \
  2>&1 | tee logs/loss_feature_ablation/baseline_seed2026.log
