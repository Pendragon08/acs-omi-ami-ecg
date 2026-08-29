#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label}"
cd "$PROJECT_ROOT"
mkdir -p logs/ecg_lfm

while true; do
  if [[ -f "$RESULT_ROOT/gates/overfit/overfit_gate.json" ]] && \
     "${PYTHON[@]}" -c 'import json; assert json.load(open("results/ecg_lfm_4label/gates/overfit/overfit_gate.json"))["passed_minimum_10_percent_loss_drop"]'; then
    break
  fi
  printf '%s WAIT_OVERFIT_GATE\n' "$(date -Is)"
  sleep 30
done
printf '%s START_LINEAR_PROBE_FOLD0_GPU1\n' "$(date -Is)"

"$CONDA" run --no-capture-output -n ecg-lfm env \
  GPU=1 ADAPTATION=linear_probe FOLDS=0 FINALIZE=0 \
  bash scripts/run_ecg_lfm_5fold.sh
linear_pid=""
if "${PYTHON[@]}" experiments/ecg_lfm/verify_fold0_gate.py \
    --predictions "$RESULT_ROOT/linear_probe/fold0/valid_record_predictions.csv" \
    --output "$RESULT_ROOT/linear_probe/fold0/fold0_gate.json"; then
  printf '%s LINEAR_PROBE_FOLD0_GATE_PASSED\n' "$(date -Is)"

  # Only a healthy L0 Fold 0 is expanded to five folds. Run it independently
  # on GPU0 so a weak optional linear probe cannot block the primary L1a arm.
  (
    "$CONDA" run --no-capture-output -n ecg-lfm env \
      GPU=0 ADAPTATION=linear_probe "FOLDS=1 2 3 4" FINALIZE=1 \
      bash scripts/run_ecg_lfm_5fold.sh
    "${PYTHON[@]}" scripts/compare_ecg_lfm_to_e4.py \
      --ecg_summary "$RESULT_ROOT/linear_probe/summary/oof_evaluation.json" \
      --ecg_oof "$RESULT_ROOT/linear_probe/summary/oof_predictions.csv" \
      --model_name "ECG-LFM L0 linear probe" \
      --output_dir "$RESULT_ROOT/linear_probe/summary"
    "${PYTHON[@]}" scripts/submit_hmoe_e4_4label.py \
      --run_dir results/ecg_lfm_4label/linear_probe/summary \
      --submission submission_ecg_lfm_4label_youden.csv
  ) > logs/ecg_lfm/linear_probe_full.log 2>&1 &
  linear_pid=$!
else
  printf '%s LINEAR_PROBE_FOLD0_GATE_FAILED_SKIP_FULL_L0\n' "$(date -Is)"
fi

printf '%s START_FINETUNE_PAPER_FOLD0_GPU1\n' "$(date -Is)"
"$CONDA" run --no-capture-output -n ecg-lfm env \
  GPU=1 ADAPTATION=finetune_paper FOLDS=0 FINALIZE=0 \
  bash scripts/run_ecg_lfm_5fold.sh
"${PYTHON[@]}" experiments/ecg_lfm/verify_fold0_gate.py \
  --predictions "$RESULT_ROOT/finetune_paper/fold0/valid_record_predictions.csv" \
  --output "$RESULT_ROOT/finetune_paper/fold0/fold0_gate.json"
printf '%s FINETUNE_PAPER_FOLD0_GATE_PASSED\n' "$(date -Is)"

"$CONDA" run --no-capture-output -n ecg-lfm env \
  GPU=1 ADAPTATION=finetune_paper "FOLDS=1 2 3 4" FINALIZE=1 \
  bash scripts/run_ecg_lfm_5fold.sh
"${PYTHON[@]}" scripts/compare_ecg_lfm_to_e4.py \
  --ecg_summary "$RESULT_ROOT/finetune_paper/summary/oof_evaluation.json" \
  --ecg_oof "$RESULT_ROOT/finetune_paper/summary/oof_predictions.csv" \
  --model_name "ECG-LFM L1a paper fine-tune" \
  --output_dir "$RESULT_ROOT/finetune_paper/summary"
"${PYTHON[@]}" scripts/submit_hmoe_e4_4label.py \
  --run_dir results/ecg_lfm_4label/finetune_paper/summary \
  --submission submission_ecg_lfm_4label_youden.csv
if [[ -n "$linear_pid" ]]; then
  wait "$linear_pid"
fi
printf '%s ECG_LFM_L0_L1A_FIVEFOLD_COMPLETE\n' "$(date -Is)"
