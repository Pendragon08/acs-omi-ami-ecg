#!/usr/bin/env bash
set -euo pipefail

# Standalone recovery helper.  The caller must set CUDA_VISIBLE_DEVICES to one
# physical GPU (for example, 1) before launching this script in screen.
ROOT="/data/code/ouyanghao/oyh/ACS-OMI-AMI"
cd "$ROOT"
source /data/code/ouyanghao/miniconda3/etc/profile.d/conda.sh
conda activate ecg-lfm
: "${CUDA_VISIBLE_DEVICES:?set exactly one GPU before launching}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*|*\ *|'') echo "CUDA_VISIBLE_DEVICES must name exactly one GPU" >&2; exit 2 ;;
esac
exec > >(tee -a "$ROOT/logs/corrected_ecg_lfm_joint_bs64_v2/dora_fold4_parallel_master.log") 2>&1
echo "START dora fold4 parallel"
python -u experiments/ecg_lfm/run_joint_fold.py \
  --fold 4 --adaptation corrected_joint_dora --run_name dora \
  --output_root results/corrected_ecg_lfm_joint_bs64_v2_parallel --seed 2026 \
  > "$ROOT/logs/corrected_ecg_lfm_joint_bs64_v2/dora_fold4_parallel.log" 2>&1
echo "DONE dora fold4 parallel"
