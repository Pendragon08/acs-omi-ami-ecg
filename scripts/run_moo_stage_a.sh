#!/usr/bin/env bash
set -u
set -o pipefail

cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES="1"
PY=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python
LOG_ROOT=results/multi_objective/logs
mkdir -p "$LOG_ROOT" results/multi_objective_evaluation

for METHOD in linear_sum pcgrad cagrad fairgrad aligned_mtl; do
  echo "START $METHOD $(date -Is)" | tee -a "$LOG_ROOT/stage_a.log"
  "$PY" experiments/ecg_lfm/run_moo_fold.py --method "$METHOD" --fold 0 \
    > "$LOG_ROOT/${METHOD}_train.log" 2>&1
  RC=$?
  echo "TRAIN_RC $METHOD $RC $(date -Is)" | tee -a "$LOG_ROOT/stage_a.log"
  if [ "$RC" -ne 0 ]; then
    exit "$RC"
  fi
  "$PY" experiments/ecg_lfm/evaluate_moo_inner.py --method "$METHOD" \
    --run_root results/multi_objective --output_root results/multi_objective_evaluation \
    --fold 0 --workers 8 --batch 64 \
    > "$LOG_ROOT/${METHOD}_eval.log" 2>&1
  RC=$?
  echo "EVAL_RC $METHOD $RC $(date -Is)" | tee -a "$LOG_ROOT/stage_a.log"
  if [ "$RC" -ne 0 ]; then
    exit "$RC"
  fi
done

echo "STAGE_A_COMPLETE $(date -Is)" | tee -a "$LOG_ROOT/stage_a.log"

