#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
GPU="${GPU:-1}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label_tuning}"
RUNS="${RUNS:-lr3e7_bce_b256 lr3e6_bce_b256 lr1e6_bce_b128 lr1e6_bce_b512 lr3e7_wbce_b256 lr1e6_wbce_b256 lr3e6_wbce_b256}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
cd "$PROJECT_ROOT"
mkdir -p "$RESULT_ROOT" logs/ecg_lfm_tuning

training_complete() {
  local provenance="$1"
  "${PYTHON[@]}" - "$provenance" 2>/dev/null <<'PY'
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

candidate_definition() {
  local run_name="$1"
  LOSS="unweighted_bce"
  OVERRIDES=()
  case "$run_name" in
    lr3e7_bce_b256) OVERRIDES+=("optimization.lr=[0.0000003]") ;;
    lr3e6_bce_b256) OVERRIDES+=("optimization.lr=[0.000003]") ;;
    lr1e6_bce_b128) OVERRIDES+=("optimization.update_freq=[4]" "lr_scheduler.lr_deacy_period=224") ;;
    lr1e6_bce_b512) OVERRIDES+=("optimization.update_freq=[16]" "lr_scheduler.lr_deacy_period=56") ;;
    lr3e7_wbce_b256) LOSS="weighted_bce"; OVERRIDES+=("optimization.lr=[0.0000003]") ;;
    lr1e6_wbce_b256) LOSS="weighted_bce" ;;
    lr3e6_wbce_b256) LOSS="weighted_bce"; OVERRIDES+=("optimization.lr=[0.000003]") ;;
    *) printf 'Unknown tuning candidate: %s\n' "$run_name" >&2; return 2 ;;
  esac
}

run_candidate() {
  local run_name="$1"
  candidate_definition "$run_name"
  local run_dir="$RESULT_ROOT/$run_name/fold0"
  local provenance="$run_dir/launch_provenance.json"
  local checkpoint="$run_dir/checkpoint_best.pt"
  local predictions="$run_dir/valid_record_predictions.csv"
  mkdir -p "$run_dir"
  printf '%s START_CANDIDATE run=%s gpu=%s loss=%s overrides=%s\n' \
    "$(date -Is)" "$run_name" "$GPU" "$LOSS" "${OVERRIDES[*]:-none}"
  if ! training_complete "$provenance"; then
    if ! CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/run_fold.py \
      --fold 0 \
      --adaptation finetune_paper \
      --run_name "$run_name" \
      --loss "$LOSS" \
      --output_root "$RESULT_ROOT" \
      "${OVERRIDES[@]}" >> "$run_dir/runner.log" 2>&1; then
      printf '%s TRAINING_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
      return 1
    fi
  fi
  test -f "$checkpoint"
  if [[ ! -f "$predictions" ]]; then
    if ! CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/infer.py \
      --checkpoint "$checkpoint" \
      --manifest_dir manifests/ecg_lfm/fold0 \
      --split valid \
      --segment_index_csv manifests/ecg_lfm/fold0/valid_segments.csv \
      --y_all manifests/ecg_lfm/y_all.npy \
      --output_csv "$predictions" >> "$run_dir/runner.log" 2>&1; then
      printf '%s INFERENCE_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
      return 1
    fi
  fi
  if ! "${PYTHON[@]}" scripts/evaluate_ecg_lfm_fold.py \
    --predictions "$predictions" \
    --run_name "$run_name" \
    --fold 0 \
    --output_dir "$run_dir/evaluation" >> "$run_dir/runner.log" 2>&1; then
    printf '%s EVALUATION_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
    return 1
  fi
  if ! "${PYTHON[@]}" scripts/plot_ecg_lfm_training_curves.py \
    --log "$run_dir/fairseq_train.jsonl" \
    --output_dir "$run_dir/curves" >> "$run_dir/runner.log" 2>&1; then
    printf '%s CURVE_EXPORT_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
    return 1
  fi
  if ! "${PYTHON[@]}" scripts/summarize_ecg_lfm_tuning.py \
    --tuning_root "$RESULT_ROOT" \
    --output_dir "$RESULT_ROOT/summary" >> "$run_dir/runner.log" 2>&1; then
    printf '%s SUMMARY_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
    return 1
  fi
  if ! "${PYTHON[@]}" experiments/ecg_lfm/verify_fold0_gate.py \
    --predictions "$predictions" \
    --output "$run_dir/fold0_gate.json" >> "$run_dir/runner.log" 2>&1; then
    printf '%s GATE_FAILED run=%s\n' "$(date -Is)" "$run_name" >&2
    return 1
  fi
  printf '%s COMPLETE_CANDIDATE run=%s\n' "$(date -Is)" "$run_name"
}

gpu_lock="/tmp/ecg_lfm_${USER}_gpu${GPU}.lock"
exec 9>"$gpu_lock"
printf '%s WAIT_GPU_LOCK gpu=%s\n' "$(date -Is)" "$GPU"
flock 9
printf '%s ACQUIRED_GPU_LOCK gpu=%s\n' "$(date -Is)" "$GPU"

for run_name in $RUNS; do
  if ! run_candidate "$run_name"; then
    printf '%s FAILED_CANDIDATE run=%s; preserved logs and continuing queue\n' "$(date -Is)" "$run_name" >&2
  fi
done
printf '%s ECG_LFM_FOLD0_TUNING_QUEUE_FINISHED\n' "$(date -Is)"
