#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ADAPTATION="${ADAPTATION:?set ADAPTATION}"
BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE}"
UPDATE_FREQ="${UPDATE_FREQ:?set UPDATE_FREQ}"
GPU="${GPU:?set GPU}"
OUTPUT_DIR="$PROJECT_ROOT/results/ecg_lfm_4label/gates/batch_${ADAPTATION}_${BATCH_SIZE}x${UPDATE_FREQ}"
cd "$PROJECT_ROOT"

case "$ADAPTATION" in
  linear_probe|finetune_paper|finetune_fixed) ;;
  *) echo "Unsupported adaptation: $ADAPTATION" >&2; exit 2 ;;
esac
echo "afa8dfc680590948f82aabbdb123bf43  checkpoints/ecg_lfm/pretrain.pt" | md5sum --check --status
mkdir -p "$OUTPUT_DIR"

started=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPU" fairseq-hydra-train \
  "task.data=$PROJECT_ROOT/manifests/ecg_lfm/fold0" \
  "task.label_file=$PROJECT_ROOT/manifests/ecg_lfm/y_all.npy" \
  "model.model_path=$PROJECT_ROOT/checkpoints/ecg_lfm/pretrain.pt" \
  "checkpoint.save_dir=$OUTPUT_DIR" \
  "common.log_file=$OUTPUT_DIR/fairseq_batch_gate.jsonl" \
  optimization.max_epoch=1 \
  optimization.max_update=1 \
  "optimization.update_freq=[$UPDATE_FREQ]" \
  "dataset.batch_size=$BATCH_SIZE" \
  dataset.disable_validation=true \
  checkpoint.no_save=true \
  --config-dir "$PROJECT_ROOT/configs/ecg_lfm" \
  --config-name "$ADAPTATION"
finished=$(date +%s)
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
  > "$OUTPUT_DIR/gpu_after.csv"
printf 'BATCH_GATE_PASSED adaptation=%s physical=%s accumulation=%s effective=%s elapsed_seconds=%s\n' \
  "$ADAPTATION" "$BATCH_SIZE" "$UPDATE_FREQ" "$((BATCH_SIZE * UPDATE_FREQ))" "$((finished - started))"
