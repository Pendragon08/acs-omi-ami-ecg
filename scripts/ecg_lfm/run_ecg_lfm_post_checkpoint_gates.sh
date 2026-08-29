#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-0}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
CHECKPOINT="$PROJECT_ROOT/checkpoints/ecg_lfm/pretrain.pt"
cd "$PROJECT_ROOT"

while [[ ! -f "$CHECKPOINT" ]]; do
  printf '%s WAIT_CHECKPOINT\n' "$(date -Is)"
  sleep 30
done
echo "afa8dfc680590948f82aabbdb123bf43  $CHECKPOINT" | md5sum --check --status
printf '%s CHECKPOINT_GATE_PASSED\n' "$(date -Is)"

CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run -n ecg-lfm python experiments/ecg_lfm/smoke_forward.py \
  --manifest_dir manifests/ecg_lfm \
  --label_file manifests/ecg_lfm/y_all.npy \
  --checkpoint "$CHECKPOINT" \
  --split smoke \
  --device cuda:0
printf '%s FORWARD_GATE_PASSED\n' "$(date -Is)"

CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run -n ecg-lfm python experiments/ecg_lfm/run_overfit_gate.py
printf '%s OVERFIT_RUN_COMPLETE\n' "$(date -Is)"
"$CONDA" run -n ecg-lfm python experiments/ecg_lfm/verify_overfit_gate.py
printf '%s OVERFIT_GATE_PASSED\n' "$(date -Is)"
