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

RUN_ROOT=results/corrected_ecg_lfm_joint_bs64_v2
OUT_ROOT=results/revalidated
LOG_DIR=logs/revalidated
mkdir -p "$OUT_ROOT" "$LOG_DIR"

for method in baseline lora dora; do
  echo "[$(date -Is)] START revalidation $method"
  python -u experiments/ecg_lfm/evaluate_joint.py \
    --run_name "$method" \
    --run_root "$RUN_ROOT" \
    --revalidation_root "$OUT_ROOT" \
    --device cuda:0 \
    --workers 8 \
    --batch 64 \
    --skip_external \
    2>&1 | tee "$LOG_DIR/${method}_phase1.log"
  echo "[$(date -Is)] DONE revalidation $method"
done

python -u experiments/ecg_lfm/summarize_revalidated.py \
  --result_root "$OUT_ROOT" \
  --report docs/ECG_LFM_corrected_evaluation_report.md \
  2>&1 | tee "$LOG_DIR/summary_phase1.log"

echo "[$(date -Is)] PHASE1_COMPLETE"
