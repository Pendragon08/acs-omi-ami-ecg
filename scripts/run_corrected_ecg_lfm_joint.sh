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
OUTPUT_ROOT="${ECG_LFM_OUTPUT_ROOT:-results/corrected_ecg_lfm_joint}"
LOG_DIR="${ECG_LFM_LOG_DIR:-logs/corrected_ecg_lfm_joint}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
# Keep a compact master log in addition to one log per fold/evaluation.  The
# raw logs remain on the server; only aggregate summaries are archived in Git.
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

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
      --output_root "$OUTPUT_ROOT" --seed 2026 \
      > "$LOG_DIR/${run_name}_fold${fold}.log" 2>&1
  done
  python -u experiments/ecg_lfm/evaluate_joint.py \
    --run_name "$run_name" --run_root "$OUTPUT_ROOT" \
    --device cuda:0 --workers 8 --batch 64 \
    > "$LOG_DIR/${run_name}_evaluate.log" 2>&1
done
