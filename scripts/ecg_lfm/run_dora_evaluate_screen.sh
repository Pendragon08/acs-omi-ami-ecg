#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/code/ouyanghao/oyh/ACS-OMI-AMI"
cd "$ROOT"
source /data/code/ouyanghao/miniconda3/etc/profile.d/conda.sh
conda activate ecg-lfm
export CUDA_VISIBLE_DEVICES=0
exec > >(tee -a "$ROOT/logs/corrected_ecg_lfm_joint_bs64_v2/dora_evaluate_master.log") 2>&1
echo "START dora evaluation"
python -u experiments/ecg_lfm/evaluate_joint.py \
  --run_name dora --run_root results/corrected_ecg_lfm_joint_bs64_v2 \
  --device cuda:0 --workers 8 --batch 64
echo "DONE dora evaluation"
