#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
ADAPTATION="${ADAPTATION:-finetune_paper}"
GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label}"
FOLDS="${FOLDS:-0 1 2 3 4}"
FINALIZE="${FINALIZE:-1}"
cd "$PROJECT_ROOT"

test "$(git -C third_party/ECG-LFM rev-parse HEAD)" = "647369a012dd8d5d362e99e2bf238c91f3e1d03b"
echo "afa8dfc680590948f82aabbdb123bf43  checkpoints/ecg_lfm/pretrain.pt" | md5sum --check --status
test -f manifests/ecg_lfm/cohort_integrity.json
test -f manifests/ecg_lfm/preprocess_integrity.json
command -v flock >/dev/null

training_complete() {
  local provenance="$1"
  "$PYTHON" - "$provenance" <<'PY'
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

# The active two-GPU pipeline starts L1a Fold 0 on physical GPU 0 while L0
# Fold 0 runs on physical GPU 1.  When the master pipeline later enters the
# L0 folds 1-4 worker assigned to GPU 0, wait for that already-running L1a
# Fold 0 process to release the device.  This is scheduling only: no training
# configuration, seed, split, checkpoint, or metric logic is changed.
if [[ "$GPU" == "0" && "$ADAPTATION" == "linear_probe" && " $FOLDS " != *" 0 "* ]]; then
  l1a_fold0_dir="$RESULT_ROOT/finetune_paper/fold0"
  while pgrep -u "$USER" -af fairseq-hydra-train | grep -Fq "checkpoint.save_dir=$l1a_fold0_dir"; do
    printf '%s WAIT_GPU0_FOR_L1A_FOLD0 path=%s\n' "$(date -Is)" "$l1a_fold0_dir"
    sleep 60
  done
fi

# Serialize future workers on each physical GPU. Existing Fold-0 workers were
# launched before this lock was introduced, so the explicit live-process wait
# above remains necessary for the current resumable run.
gpu_lock="/tmp/ecg_lfm_${USER}_gpu${GPU}.lock"
exec 9>"$gpu_lock"
printf '%s WAIT_GPU_LOCK gpu=%s lock=%s\n' "$(date -Is)" "$GPU" "$gpu_lock"
flock 9
printf '%s ACQUIRED_GPU_LOCK gpu=%s\n' "$(date -Is)" "$GPU"

for fold in $FOLDS; do
  fold_dir="$RESULT_ROOT/$ADAPTATION/fold$fold"
  fold_lock="/tmp/ecg_lfm_${USER}_${ADAPTATION}_fold${fold}.lock"
  exec {fold_lock_fd}>"$fold_lock"
  printf '%s WAIT_FOLD_LOCK adaptation=%s fold=%s\n' "$(date -Is)" "$ADAPTATION" "$fold"
  flock "$fold_lock_fd"
  checkpoint="$fold_dir/checkpoint_best.pt"
  provenance="$fold_dir/launch_provenance.json"
  if [[ -f "$checkpoint" ]] && ! training_complete "$provenance"; then
    while pgrep -u "$USER" -af fairseq-hydra-train | grep -Fq "checkpoint.save_dir=$fold_dir"; do
      printf '%s WAIT_EXISTING_TRAINING adaptation=%s fold=%s\n' "$(date -Is)" "$ADAPTATION" "$fold"
      sleep 60
    done
  fi
  if ! training_complete "$provenance"; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/ecg_lfm/run_fold.py \
      --fold "$fold" --adaptation "$ADAPTATION" --output_root "$RESULT_ROOT"
  fi
  test -f "$checkpoint"
  if [[ ! -f "$RESULT_ROOT/$ADAPTATION/fold$fold/valid_record_predictions.csv" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/ecg_lfm/infer.py \
      --checkpoint "$checkpoint" \
      --manifest_dir "manifests/ecg_lfm/fold$fold" \
      --split valid \
      --segment_index_csv "manifests/ecg_lfm/fold$fold/valid_segments.csv" \
      --y_all manifests/ecg_lfm/y_all.npy \
      --output_csv "$RESULT_ROOT/$ADAPTATION/fold$fold/valid_record_predictions.csv"
  fi
  "$PYTHON" scripts/evaluate_ecg_lfm_fold.py \
    --predictions "$RESULT_ROOT/$ADAPTATION/fold$fold/valid_record_predictions.csv" \
    --run_name "$ADAPTATION" \
    --fold "$fold" \
    --output_dir "$RESULT_ROOT/$ADAPTATION/fold$fold/evaluation"
  "$PYTHON" scripts/plot_ecg_lfm_training_curves.py \
    --log "$RESULT_ROOT/$ADAPTATION/fold$fold/fairseq_train.jsonl" \
    --output_dir "$RESULT_ROOT/$ADAPTATION/fold$fold/curves"
  if [[ ! -f "$RESULT_ROOT/$ADAPTATION/fold$fold/external_record_predictions.csv" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/ecg_lfm/infer.py \
      --checkpoint "$checkpoint" \
      --manifest_dir "manifests/ecg_lfm/fold$fold" \
      --split external \
      --segment_index_csv "manifests/ecg_lfm/fold$fold/external_segments.csv" \
      --output_csv "$RESULT_ROOT/$ADAPTATION/fold$fold/external_record_predictions.csv"
  fi
  flock -u "$fold_lock_fd"
  exec {fold_lock_fd}>&-
done

if [[ "$FINALIZE" != "1" ]]; then
  exit 0
fi

for fold in 0 1 2 3 4; do
  test -f "$RESULT_ROOT/$ADAPTATION/fold$fold/valid_record_predictions.csv"
  test -f "$RESULT_ROOT/$ADAPTATION/fold$fold/external_record_predictions.csv"
done

"$PYTHON" scripts/evaluate_ecg_lfm.py \
  --fold_predictions \
    "$RESULT_ROOT/$ADAPTATION/fold0/valid_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold1/valid_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold2/valid_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold3/valid_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold4/valid_record_predictions.csv" \
  --output_dir "$RESULT_ROOT/$ADAPTATION/summary"

"$PYTHON" scripts/ensemble_ecg_lfm.py \
  --fold_predictions \
    "$RESULT_ROOT/$ADAPTATION/fold0/external_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold1/external_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold2/external_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold3/external_record_predictions.csv" \
    "$RESULT_ROOT/$ADAPTATION/fold4/external_record_predictions.csv" \
  --thresholds "$RESULT_ROOT/$ADAPTATION/summary/thresholds.json" \
  --output_dir "$RESULT_ROOT/$ADAPTATION/summary"
