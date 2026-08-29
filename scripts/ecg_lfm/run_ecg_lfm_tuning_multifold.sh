#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
GPU="${GPU:-1}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label_tuning}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to a registered tuning candidate}"
FOLDS="${FOLDS:-1 2}"
FINALIZE="${FINALIZE:-0}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
cd "$PROJECT_ROOT"

LOSS="unweighted_bce"
OVERRIDES=()
case "$RUN_NAME" in
  lr3e7_bce_b256) OVERRIDES+=("optimization.lr=[0.0000003]") ;;
  lr3e6_bce_b256) OVERRIDES+=("optimization.lr=[0.000003]") ;;
  lr1e6_bce_b128) OVERRIDES+=("optimization.update_freq=[4]" "lr_scheduler.lr_deacy_period=224") ;;
  lr1e6_bce_b512) OVERRIDES+=("optimization.update_freq=[16]" "lr_scheduler.lr_deacy_period=56") ;;
  lr3e7_wbce_b256) LOSS="weighted_bce"; OVERRIDES+=("optimization.lr=[0.0000003]") ;;
  lr1e6_wbce_b256) LOSS="weighted_bce" ;;
  lr3e6_wbce_b256) LOSS="weighted_bce"; OVERRIDES+=("optimization.lr=[0.000003]") ;;
  *) printf 'Unknown registered tuning candidate: %s\n' "$RUN_NAME" >&2; exit 2 ;;
esac

training_complete() {
  "${PYTHON[@]}" - "$1" 2>/dev/null <<'PY'
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

test "$(git -C third_party/ECG-LFM rev-parse HEAD)" = "647369a012dd8d5d362e99e2bf238c91f3e1d03b"
echo "afa8dfc680590948f82aabbdb123bf43  checkpoints/ecg_lfm/pretrain.pt" | md5sum --check --status
mkdir -p "$RESULT_ROOT/$RUN_NAME" logs/ecg_lfm_tuning

gpu_lock="/tmp/ecg_lfm_${USER}_gpu${GPU}.lock"
exec 9>"$gpu_lock"
printf '%s WAIT_GPU_LOCK gpu=%s run=%s\n' "$(date -Is)" "$GPU" "$RUN_NAME"
flock 9
printf '%s ACQUIRED_GPU_LOCK gpu=%s run=%s\n' "$(date -Is)" "$GPU" "$RUN_NAME"

run_fold() {
  local fold="$1"
  local fold_dir="$RESULT_ROOT/$RUN_NAME/fold$fold"
  mkdir -p "$fold_dir"
  if ! training_complete "$fold_dir/launch_provenance.json"; then
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/run_fold.py \
      --fold "$fold" --adaptation finetune_paper --run_name "$RUN_NAME" \
      --loss "$LOSS" --output_root "$RESULT_ROOT" "${OVERRIDES[@]}" \
      >> "$fold_dir/runner.log" 2>&1
  fi
  test -f "$fold_dir/checkpoint_best.pt"
  if [[ ! -f "$fold_dir/valid_record_predictions.csv" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/infer.py \
      --checkpoint "$fold_dir/checkpoint_best.pt" \
      --manifest_dir "manifests/ecg_lfm/fold$fold" --split valid \
      --segment_index_csv "manifests/ecg_lfm/fold$fold/valid_segments.csv" \
      --y_all manifests/ecg_lfm/y_all.npy \
      --output_csv "$fold_dir/valid_record_predictions.csv" >> "$fold_dir/runner.log" 2>&1
  fi
  "${PYTHON[@]}" scripts/evaluate_ecg_lfm_fold.py \
    --predictions "$fold_dir/valid_record_predictions.csv" --run_name "$RUN_NAME" \
    --fold "$fold" --output_dir "$fold_dir/evaluation" >> "$fold_dir/runner.log" 2>&1
  "${PYTHON[@]}" scripts/plot_ecg_lfm_training_curves.py \
    --log "$fold_dir/fairseq_train.jsonl" --output_dir "$fold_dir/curves" \
    >> "$fold_dir/runner.log" 2>&1
}

for fold in $FOLDS; do
  printf '%s START_PROMOTED_FOLD run=%s fold=%s loss=%s\n' "$(date -Is)" "$RUN_NAME" "$fold" "$LOSS"
  run_fold "$fold"
  printf '%s COMPLETE_PROMOTED_FOLD run=%s fold=%s\n' "$(date -Is)" "$RUN_NAME" "$fold"
done

"${PYTHON[@]}" scripts/summarize_ecg_lfm_multifold.py --tuning_root "$RESULT_ROOT" \
  --output_dir "$RESULT_ROOT/summary"

if [[ "$FINALIZE" == "1" ]]; then
  for fold in 0 1 2 3 4; do
    fold_dir="$RESULT_ROOT/$RUN_NAME/fold$fold"
    training_complete "$fold_dir/launch_provenance.json"
    test -f "$fold_dir/valid_record_predictions.csv"
    if [[ ! -f "$fold_dir/external_record_predictions.csv" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON[@]}" experiments/ecg_lfm/infer.py \
        --checkpoint "$fold_dir/checkpoint_best.pt" \
        --manifest_dir "manifests/ecg_lfm/fold$fold" --split external \
        --segment_index_csv "manifests/ecg_lfm/fold$fold/external_segments.csv" \
        --output_csv "$fold_dir/external_record_predictions.csv" >> "$fold_dir/runner.log" 2>&1
    fi
  done
  "${PYTHON[@]}" scripts/evaluate_ecg_lfm.py --fold_predictions \
    "$RESULT_ROOT/$RUN_NAME/fold0/valid_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold1/valid_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold2/valid_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold3/valid_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold4/valid_record_predictions.csv" \
    --output_dir "$RESULT_ROOT/$RUN_NAME/summary"
  "${PYTHON[@]}" scripts/ensemble_ecg_lfm.py --fold_predictions \
    "$RESULT_ROOT/$RUN_NAME/fold0/external_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold1/external_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold2/external_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold3/external_record_predictions.csv" \
    "$RESULT_ROOT/$RUN_NAME/fold4/external_record_predictions.csv" \
    --thresholds "$RESULT_ROOT/$RUN_NAME/summary/thresholds.json" \
    --output_dir "$RESULT_ROOT/$RUN_NAME/summary"
  printf '%s FINAL_INTERNAL_ARTIFACTS_READY_EXTERNAL_NOT_SUBMITTED run=%s\n' "$(date -Is)" "$RUN_NAME"
fi
