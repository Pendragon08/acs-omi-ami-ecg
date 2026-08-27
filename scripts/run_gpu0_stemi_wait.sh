#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
GPU=0
LOG="$ROOT/logs/stemi_nstemi_ua_gpu0.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[wait] $(date -Is) GPU0 STEMI/NSTEMI/UA job queued"
while true; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' || echo 99999)
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 500 )) && ! printf '%s\n' "$apps" | grep -Eq '[0-9]'; then
    break
  fi
  echo "[wait] $(date -Is) GPU0 used=${used}MiB apps=${apps:-none}"
  sleep 60
done
echo "[start] $(date -Is) GPU0 is free"
source /data/code/ouyanghao/miniconda3/etc/profile.d/conda.sh
conda activate oyh
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
python exp/exp_acs_stemi_nstemi_ua.py \
  --model XceptionTime --task stemi_nstemi_ua --loss bce --variant P3 \
  --lr 3e-4 --batch_size 256 --seed 2026 --epochs 100 --patience 15 \
  --clip 1.0 --num_workers 4 --exp_name stemi_nstemi_ua_final_gpu0 \
  --device cuda:0
echo "[done] $(date -Is) GPU0 STEMI/NSTEMI/UA job finished"
