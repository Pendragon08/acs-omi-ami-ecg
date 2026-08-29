#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label}"
PROVENANCE="$RESULT_ROOT/finetune_paper/fold0/launch_provenance.json"
SUBMIT_EXTERNAL="${SUBMIT_EXTERNAL:-0}"
cd "$PROJECT_ROOT"

training_complete() {
  "${PYTHON[@]}" - "$PROVENANCE" 2>/dev/null <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("returncode") == 0 else 1)
PY
}

while ! training_complete; do
  if ! pgrep -u "$USER" -af fairseq-hydra-train | grep -Fq "checkpoint.save_dir=$RESULT_ROOT/finetune_paper/fold0"; then
    printf '%s L1A_FOLD0_NOT_COMPLETE_AND_NOT_RUNNING\n' "$(date -Is)" >&2
    exit 1
  fi
  printf '%s WAIT_L1A_FOLD0_COMPLETION\n' "$(date -Is)"
  sleep 60
done

# Fold locks in run_ecg_lfm_5fold.sh make this safe if the master pipeline
# reaches the same continuation: one worker performs each fold, the other
# observes completed provenance and predictions and skips it.
"$CONDA" run --no-capture-output -n ecg-lfm env \
  GPU=0 ADAPTATION=finetune_paper FOLDS=0 FINALIZE=0 \
  bash scripts/run_ecg_lfm_5fold.sh
"${PYTHON[@]}" experiments/ecg_lfm/verify_fold0_gate.py \
  --predictions "$RESULT_ROOT/finetune_paper/fold0/valid_record_predictions.csv" \
  --output "$RESULT_ROOT/finetune_paper/fold0/fold0_gate.json"
"${PYTHON[@]}" scripts/evaluate_ecg_lfm_fold.py \
  --predictions "$RESULT_ROOT/finetune_paper/fold0/valid_record_predictions.csv" \
  --run_name finetune_paper \
  --fold 0 \
  --output_dir "$RESULT_ROOT/finetune_paper/fold0/evaluation"
"${PYTHON[@]}" scripts/plot_ecg_lfm_training_curves.py \
  --log "$RESULT_ROOT/finetune_paper/fold0/fairseq_train.jsonl" \
  --output_dir "$RESULT_ROOT/finetune_paper/fold0/curves"
"${PYTHON[@]}" scripts/summarize_ecg_lfm_tuning.py
printf '%s L1A_FOLD0_GATE_PASSED_START_REMAINING\n' "$(date -Is)"

"$CONDA" run --no-capture-output -n ecg-lfm env \
  GPU=0 ADAPTATION=finetune_paper "FOLDS=1 2 3 4" FINALIZE=1 \
  bash scripts/run_ecg_lfm_5fold.sh
"${PYTHON[@]}" scripts/compare_ecg_lfm_to_e4.py \
  --ecg_summary "$RESULT_ROOT/finetune_paper/summary/oof_evaluation.json" \
  --ecg_oof "$RESULT_ROOT/finetune_paper/summary/oof_predictions.csv" \
  --model_name "ECG-LFM L1a paper fine-tune" \
  --output_dir "$RESULT_ROOT/finetune_paper/summary"
if [[ "$SUBMIT_EXTERNAL" == "1" ]]; then
  "${PYTHON[@]}" scripts/submit_hmoe_e4_4label.py \
    --run_dir results/ecg_lfm_4label/finetune_paper/summary \
    --submission submission_ecg_lfm_4label_youden.csv
else
  printf '%s EXTERNAL_SUBMISSION_DEFERRED_UNTIL_FINAL_MODEL_LOCKED\n' "$(date -Is)"
fi
printf '%s ECG_LFM_L1A_RESUME_PIPELINE_COMPLETE\n' "$(date -Is)"
