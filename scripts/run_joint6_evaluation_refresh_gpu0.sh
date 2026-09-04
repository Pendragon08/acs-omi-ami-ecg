#!/usr/bin/env bash
set -euo pipefail

cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python

if [[ "${CUDA_VISIBLE_DEVICES}" == *,* || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "Exactly one physical GPU must be bound" >&2
  exit 2
fi

for fold in 0 1 2 3 4; do
  echo "[$(date -Is)] refresh ECG-LFM LoRA fold=${fold}"
  PYTHONPATH=. "${PYTHON}" experiments/ecg_lfm/evaluate_joint6_fold.py \
    --method lora \
    --fold "${fold}" \
    --device cuda:0
done

# Partial-FT fold0 was evaluated by the pre-repair evaluator and was already
# skipped by the long-running training queue. Refresh that completed checkpoint
# so the eventual comparison has the same metric/provenance schema.
echo "[$(date -Is)] refresh ECG-LFM Partial FT fold=0"
PYTHONPATH=. "${PYTHON}" experiments/ecg_lfm/evaluate_joint6_fold.py \
  --method partial \
  --fold 0 \
  --device cuda:0

PYTHONPATH=. "${PYTHON}" experiments/ecg_lfm/summarize_joint6_cv.py \
  --root results/joint6_cv \
  --report docs/ECG_LFM_Joint6_Evaluation_Report.md

echo "[$(date -Is)] Joint-6 evaluation refresh complete"
