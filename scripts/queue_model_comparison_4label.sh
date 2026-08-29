#!/usr/bin/env bash
set -euo pipefail

# Queue the architecture comparison behind the already registered ECG-LFM
# jobs. The two workers use distinct GPUs and run disjoint model subsets;
# every worker still writes a model-specific summary under the common result
# directory. Do not launch a competing job while any protected session lives.
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
CONDA_ROOT=/data/code/ouyanghao/miniconda3
LOG_DIR="$ROOT/logs/corrected_joint_model_comparison"
OUT_DIR="results/corrected_joint_model_comparison"
mkdir -p "$LOG_DIR"

protected_sessions=(ecg_lfm_l1a_resume ecg_lfm_tune_gpu1 ecg_lfm_tuning_stages)
while :; do
  active=0
  screen_list="$(screen -ls 2>/dev/null || true)"
  for session in "${protected_sessions[@]}"; do
    if printf '%s\n' "$screen_list" | grep -q "\.${session}[[:space:]]"; then
      active=1
      break
    fi
  done
  if [[ "$active" == 0 ]]; then
    break
  fi
  printf '%s waiting for protected ECG-LFM sessions: %s\n' "$(date -u +%FT%TZ)" "${protected_sessions[*]}" >> "$LOG_DIR/queue.log"
  sleep 60
done

# Confirm there is enough free memory for the small traditional models after
# the protected jobs finish.  Other long-lived non-project processes may keep
# a baseline allocation, so requiring memory.used == 0 would deadlock the
# queue; each worker's smoke test needs <3 GiB at batch 128.
while :; do
  free0="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
  free1="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')"
  if [[ "${free0:-0}" =~ ^[0-9]+$ && "${free1:-0}" =~ ^[0-9]+$ ]] && (( free0 > 4000 && free1 > 4000 )); then
    break
  fi
  printf '%s waiting for free GPU memory: gpu0=%s MiB gpu1=%s MiB\n' "$(date -u +%FT%TZ)" "$free0" "$free1" >> "$LOG_DIR/queue.log"
  sleep 30
done

start_worker() {
  local session="$1"
  local gpu="$2"
  local models="$3"
  local log="$LOG_DIR/${session}.log"
  if screen -ls 2>/dev/null | grep -q "\.${session}[[:space:]]"; then
    printf '%s worker already running session=%s\n' "$(date -u +%FT%TZ)" "$session" >> "$LOG_DIR/queue.log"
    return
  fi
  screen -dmS "$session" bash -lc "cd '$ROOT' && source '$CONDA_ROOT/etc/profile.d/conda.sh' && conda activate oyh && export CUDA_VISIBLE_DEVICES='$gpu' && python experiments/model_comparison_4label.py --models $(printf '%s' '$models' | tr ',' ' ') --out_dir '$OUT_DIR' --device cuda:0 --num_workers 0 > '$log' 2>&1"
  printf '%s started %s on physical GPU%s models=%s\n' "$(date -u +%FT%TZ)" "$session" "$gpu" "$models" >> "$LOG_DIR/queue.log"
}

remaining_models() {
  local models="$1"
  local result=""
  local model
  IFS=',' read -r -a model_array <<< "$models"
  for model in "${model_array[@]}"; do
    if [[ ! -f "$ROOT/$OUT_DIR/$model/summary.json" ]]; then
      if [[ -n "$result" ]]; then result+=","; fi
      result+="$model"
    else
      printf '%s skip completed model=%s\n' "$(date -u +%FT%TZ)" "$model" >> "$LOG_DIR/queue.log"
    fi
  done
  printf '%s' "$result"
}

models_gpu0="$(remaining_models 'InceptionTime,ResNet1D_Wang,ViT')"
models_gpu1="$(remaining_models 'LSTM,Transformer1D')"
if [[ -n "$models_gpu0" ]]; then
  start_worker model_cmp_gpu0 0 "$models_gpu0"
fi
if [[ -n "$models_gpu1" ]]; then
  start_worker model_cmp_gpu1 1 "$models_gpu1"
fi

while :; do
  screen_list="$(screen -ls 2>/dev/null || true)"
  if ! printf '%s\n' "$screen_list" | grep -q '\.model_cmp_gpu0[[:space:]]' && \
     ! printf '%s\n' "$screen_list" | grep -q '\.model_cmp_gpu1[[:space:]]'; then
    break
  fi
  sleep 60
done

cd "$ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate oyh
python scripts/write_model_comparison_4label_report.py > "$LOG_DIR/report_generation.log" 2>&1
printf '%s comparison workers finished; report regenerated\n' "$(date -u +%FT%TZ)" >> "$LOG_DIR/queue.log"
