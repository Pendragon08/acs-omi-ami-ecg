#!/usr/bin/env bash
set -euo pipefail

task_mode="${1:?usage: run_single_task_control_fold0.sh <acs|omi>}"
if [[ "$task_mode" != "acs" && "$task_mode" != "omi" ]]; then
  echo "task_mode must be acs or omi" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == *" "* ]]; then
  echo "bind exactly one physical GPU with CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_name="${task_mode}_only_w0"
train_dir="results/single_task_vs_joint/${run_name}/fold0"
eval_dir="results/single_task_vs_joint_evaluation/${run_name}/fold0"
runtime_csv="results/single_task_vs_joint/runtime.csv"
mkdir -p "$train_dir" "$eval_dir"
if [[ ! -f "$runtime_csv" ]]; then
  echo "method,fold,seed,status,start_unix,end_unix,elapsed_seconds,peak_gpu_memory_mib,gpu,launch_provenance_sha256" > "$runtime_csv"
fi

start="$(date +%s)"
peak_file="$(mktemp)"
echo 0 > "$peak_file"
(
  while true; do
    used="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)"
    current="$(cat "$peak_file")"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > current )); then
      echo "$used" > "$peak_file"
    fi
    sleep 10
  done
) &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python experiments/ecg_lfm/run_single_task_fold.py \
  --task_mode "$task_mode" \
  --fold 0 \
  --run_name "$run_name"

python experiments/ecg_lfm/evaluate_single_task_nested_fold.py \
  --task_mode "$task_mode" \
  --fold 0 \
  --run_name "$run_name" \
  --device cuda:0

end="$(date +%s)"
peak="$(cat "$peak_file")"
provenance_sha="$(sha256sum "$train_dir/launch_provenance.json" | awk '{print $1}')"
echo "${run_name},0,2026,completed,${start},${end},$((end-start)),${peak},${CUDA_VISIBLE_DEVICES},${provenance_sha}" >> "$runtime_csv"
echo "COMPLETED task=${task_mode} metrics=${eval_dir}/metrics.json"
