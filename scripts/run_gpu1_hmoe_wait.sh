#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
GPU=1
LOG="$ROOT/logs/hmoe_kan_github_gpu1.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[start] $(date -Is) GPU1 HMoE/KAN feasibility run (shared-node GPU use authorized)"
source /data/code/ouyanghao/miniconda3/etc/profile.d/conda.sh
conda activate oyh
cd "$ROOT"
export PYTHONPATH="$ROOT/third_party/pykan_tmp:$ROOT/third_party/pytorch_geometric:$ROOT/third_party/WavelNet:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
for variant in e0 e1 e2 e3 e4; do
  echo "[start] $(date -Is) ${variant}"
  python experiments/hmoe_kan_github.py \
    --variant "$variant" \
    --out_dir results/hmoe_kan_github --n_folds 5 --fold_seed 2026 \
    --seed 2026 --epochs 100 --patience 15 --batch_size 128 --lr 1e-4 \
    --lambda_balance 0.01 --num_workers 4 --device cuda:0
  echo "[done] $(date -Is) ${variant}"
done
echo "[done] $(date -Is) GPU1 HMoE/KAN feasibility queue finished"
