#!/usr/bin/env bash
set -u
set -o pipefail

cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
export CUDA_VISIBLE_DEVICES="1"
PY=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python
LOG_ROOT=results/multi_objective/logs
mkdir -p "$LOG_ROOT" results/multi_objective_evaluation

# The caller must pass only methods already frozen by the Stage-A inner
# validation gate (normally one or two methods).  Outer-test inference is
# intentionally performed only after every requested method has trained.
if [ "$#" -lt 1 ]; then
  echo "usage: $0 METHOD [METHOD ...]" >&2
  exit 2
fi

for METHOD in "$@"; do
  for FOLD in 0 1 2 3 4; do
    echo "START_STAGE_B $METHOD fold$FOLD $(date -Is)" | tee -a "$LOG_ROOT/stage_b.log"
    "$PY" experiments/ecg_lfm/run_moo_fold.py --method "$METHOD" --fold "$FOLD" \
      > "$LOG_ROOT/${METHOD}_fold${FOLD}_train.log" 2>&1
    RC=$?
    echo "TRAIN_RC_STAGE_B $METHOD fold$FOLD $RC $(date -Is)" | tee -a "$LOG_ROOT/stage_b.log"
    if [ "$RC" -ne 0 ]; then exit "$RC"; fi
  done
done

for METHOD in "$@"; do
  for FOLD in 0 1 2 3 4; do
    "$PY" experiments/ecg_lfm/evaluate_nested_fold.py --run_name "$METHOD" --fold "$FOLD" \
      --run_root results/multi_objective --manifest_root manifests/ecg_lfm_nested \
      --output_root results/multi_objective_evaluation \
      > "$LOG_ROOT/${METHOD}_fold${FOLD}_eval.log" 2>&1
    RC=$?
    echo "EVAL_RC_STAGE_B $METHOD fold$FOLD $RC $(date -Is)" | tee -a "$LOG_ROOT/stage_b.log"
    if [ "$RC" -ne 0 ]; then exit "$RC"; fi
  done
done

echo "STAGE_B_COMPLETE $(date -Is)" | tee -a "$LOG_ROOT/stage_b.log"

