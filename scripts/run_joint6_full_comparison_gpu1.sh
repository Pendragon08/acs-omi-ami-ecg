#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
ENV_DIR=/data/code/ouyanghao/miniconda3/envs/ecg-lfm
MASTER_LOG="$ROOT/logs/joint6_cv/full_comparison_gpu1.log"
SMOKE_MARKER="$ROOT/results/joint6_cv_resource_smoke/RESOURCE_SMOKE_COMPLETE"
mkdir -p "$(dirname "$MASTER_LOG")" "$ROOT/results/joint6_cv/resource"
exec > >(tee -a "$MASTER_LOG") 2>&1

export CUDA_VISIBLE_DEVICES=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
cd "$ROOT"

run_profiled() {
  local name="$1"
  shift
  local resource="$ROOT/results/joint6_cv/resource/${name}.csv"
  echo "timestamp,index,memory.used,memory.total,utilization.gpu" > "$resource"
  "$@" &
  local job_pid=$!
  (
    while kill -0 "$job_pid" 2>/dev/null; do
      nvidia-smi -i 1 --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits >> "$resource"
      sleep 30
    done
  ) &
  local monitor_pid=$!
  set +e
  wait "$job_pid"
  local status=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  return "$status"
}

echo "[$(date -Is)] waiting for Joint-6 formal-batch resource smoke gate"
while [[ ! -f "$SMOKE_MARKER" ]]; do
  sleep 30
done
echo "[$(date -Is)] smoke gate passed; full comparison starts"

for method in lora partial; do
  for fold in 0 1 2 3 4; do
    result_dir="$ROOT/results/joint6_cv/evaluation/$method/fold$fold"
    if "$ENV_DIR/bin/python" experiments/ecg_lfm/summarize_joint6_cv.py \
      --validate_dir "$result_dir" --validate_kind ecg_lfm \
      --expected_method "$method" --expected_fold "$fold" >/dev/null 2>&1; then
      echo "[$(date -Is)] skip completed ECG-LFM $method fold$fold"
      continue
    fi
    echo "[$(date -Is)] train ECG-LFM $method fold$fold"
    run_profiled "ecg_lfm_${method}_fold${fold}" \
      "$ENV_DIR/bin/python" experiments/ecg_lfm/run_joint6_fold.py \
        --fold "$fold" --method "$method" \
        --output_root results/joint6_cv/ecg_lfm \
        --audit_output results/joint6_cv/label_audit \
        --seed 2026
    echo "[$(date -Is)] evaluate ECG-LFM $method fold$fold"
    "$ENV_DIR/bin/python" experiments/ecg_lfm/evaluate_joint6_fold.py \
      --fold "$fold" --method "$method" \
      --run_root results/joint6_cv/ecg_lfm \
      --output_root results/joint6_cv/evaluation \
      --manifest_root manifests/ecg_lfm_nested \
      --device cuda:0 --workers 8 --batch 64
  done
done

for model in InceptionTime ResNet1D_Wang LSTM Transformer1D ViT; do
  for fold in 0 1 2 3 4; do
    result_dir="$ROOT/results/joint6_cv/traditional/$model/fold$fold"
    if "$ENV_DIR/bin/python" experiments/ecg_lfm/summarize_joint6_cv.py \
      --validate_dir "$result_dir" --validate_kind traditional \
      --expected_method "$model" --expected_fold "$fold" >/dev/null 2>&1; then
      echo "[$(date -Is)] skip completed $model fold$fold"
      continue
    fi
    echo "[$(date -Is)] train/evaluate $model fold$fold"
    run_profiled "${model}_fold${fold}" \
      "$ENV_DIR/bin/python" experiments/joint6_traditional_nested.py \
        --model "$model" --fold "$fold" \
        --output_root results/joint6_cv/traditional \
        --seed 2026 --epochs 50 --patience 8 --batch_size 128 \
        --lr 0.0001 --weight_decay 0 --num_workers 4 --device cuda:0
  done
done

"$ENV_DIR/bin/python" experiments/ecg_lfm/summarize_joint6_cv.py \
  --root results/joint6_cv \
  --report docs/ECG_LFM_Joint6_CV_Comparison_Report.md \
  --require_complete
touch "$ROOT/results/joint6_cv/FULL_COMPARISON_COMPLETE"
echo "[$(date -Is)] Joint-6 full comparison complete"
