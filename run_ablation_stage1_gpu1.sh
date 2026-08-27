#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
mkdir -p logs/loss_feature_ablation results/loss_feature_ablation
PY=/data/code/ouyanghao/miniconda3/envs/oyh/bin/python

run_loss() {
  local tag="$1"
  shift
  $PY -m exp.exp_acs_ablation \
    --study loss --task omi --seed 2026 --lr 3e-4 --batch_size 256 \
    --epochs 100 --patience 15 --clip 1.0 --num_workers 4 \
    --device cuda --exp_name loss_feature_ablation "$@" \
    2>&1 | tee "logs/loss_feature_ablation/${tag}.log"
}

run_loss loss_wbce_seed2026 --loss wbce
for alpha in 0.25 0.50 0.75; do
  run_loss "loss_focal_g2_a${alpha}_seed2026" --loss focal --gamma 2 --focal_alpha "$alpha"
done
run_loss loss_focal_g2_aratio_seed2026 --loss focal --gamma 2 --focal_alpha_mode ratio
for beta in 0.99 0.999 0.9999; do
  for gamma in 1 2 3; do
    run_loss "loss_cb_b${beta}_g${gamma}_seed2026" --loss cb_focal --beta "$beta" --gamma "$gamma"
  done
done

run_feature() {
  local task="$1"
  local mode="$2"
  local lr="$3"
  local bs="$4"
  $PY -m exp.exp_acs_ablation \
    --study feature --task "$task" --loss bce --feature_mode "$mode" \
    --seed 2026 --lr "$lr" --batch_size "$bs" --epochs 100 --patience 15 \
    --clip 1.0 --num_workers 4 --device cuda --exp_name loss_feature_ablation \
    2>&1 | tee "logs/loss_feature_ablation/feature_${mode}_${task}_seed2026.log"
}

for mode in raw median clin raw_age raw_sex raw_clin raw_median all; do
  run_feature omi "$mode" 3e-4 256
  run_feature ami "$mode" 1e-4 128
done

printf 'STAGE1_COMPLETE %s\n' "$(date -Iseconds)" | tee logs/loss_feature_ablation/stage1_complete.log
