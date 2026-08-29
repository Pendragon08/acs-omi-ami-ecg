#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
CONDA=/data/code/ouyanghao/miniconda3
cd "$ROOT"
source "$CONDA/etc/profile.d/conda.sh"
conda activate ecg-lfm
visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible" || "$visible" == *,* || "$visible" == *" "* ]]; then
  echo "ERROR: exactly one CUDA_VISIBLE_DEVICES value is required" >&2
  exit 2
fi
mkdir -p logs/corrected_ecg_lfm_joint results/corrected_ecg_lfm_joint

for adaptation in corrected_joint_baseline corrected_joint_lora corrected_joint_dora; do
  case "$adaptation" in
    corrected_joint_baseline) run_name=baseline ;;
    corrected_joint_lora) run_name=lora ;;
    corrected_joint_dora) run_name=dora ;;
  esac
  for fold in 0 1 2 3 4; do
    echo "START $run_name fold$fold"
    python -u experiments/ecg_lfm/run_joint_fold.py \
      --fold "$fold" --adaptation "$adaptation" --run_name "$run_name" \
      --output_root results/corrected_ecg_lfm_joint --device cuda:0 --seed 2026 \
      > "logs/corrected_ecg_lfm_joint/${run_name}_fold${fold}.log" 2>&1
  done
  python -u experiments/ecg_lfm/evaluate_joint.py \
    --run_name "$run_name" --run_root results/corrected_ecg_lfm_joint \
    --device cuda:0 --workers 8 --batch 32 \
    > "logs/corrected_ecg_lfm_joint/${run_name}_evaluate.log" 2>&1
done
