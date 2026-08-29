#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
GPU="${GPU:-0}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
cd "$PROJECT_ROOT"

backfill_fold0() {
  local run_name="$1"
  local fold_dir="$2"
  local predictions="$fold_dir/valid_record_predictions.csv"
  test -f "$fold_dir/checkpoint_best.pt"
  if [[ ! -f "$predictions" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/infer.py \
      --checkpoint "$fold_dir/checkpoint_best.pt" \
      --manifest_dir manifests/ecg_lfm/fold0 \
      --split valid \
      --segment_index_csv manifests/ecg_lfm/fold0/valid_segments.csv \
      --y_all manifests/ecg_lfm/y_all.npy \
      --output_csv "$predictions"
  fi
  "${PYTHON[@]}" scripts/evaluate_ecg_lfm_fold.py \
    --predictions "$predictions" --run_name "$run_name" --fold 0 \
    --output_dir "$fold_dir/evaluation"
  "${PYTHON[@]}" scripts/plot_ecg_lfm_training_curves.py \
    --log "$fold_dir/fairseq_train.jsonl" --output_dir "$fold_dir/curves"
  "${PYTHON[@]}" experiments/ecg_lfm/verify_fold0_gate.py \
    --predictions "$predictions" --output "$fold_dir/fold0_gate.json"
  printf '%s REPAIRED_POST_TRAINING_OUTPUTS run=%s\n' "$(date -Is)" "$run_name"
}

gpu_lock="/tmp/ecg_lfm_${USER}_gpu${GPU}.lock"
exec 9>"$gpu_lock"
flock 9
backfill_fold0 finetune_paper \
  "$PROJECT_ROOT/results/ecg_lfm_4label/finetune_paper/fold0"
backfill_fold0 lr3e7_bce_b256 \
  "$PROJECT_ROOT/results/ecg_lfm_4label_tuning/lr3e7_bce_b256/fold0"
"${PYTHON[@]}" scripts/summarize_ecg_lfm_tuning.py
flock -u 9
exec env SUBMIT_EXTERNAL=0 bash scripts/resume_ecg_lfm_l1a_after_fold0.sh
