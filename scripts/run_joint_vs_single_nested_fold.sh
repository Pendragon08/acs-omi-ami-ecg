#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: run_joint_vs_single_nested_fold.sh <joint|acs|omi> <fold>}"
fold="${2:?usage: run_joint_vs_single_nested_fold.sh <joint|acs|omi> <fold>}"

if [[ "$mode" != "joint" && "$mode" != "acs" && "$mode" != "omi" ]]; then
  echo "mode must be joint, acs, or omi" >&2
  exit 2
fi
if [[ ! "$fold" =~ ^[0-4]$ ]]; then
  echo "fold must be 0..4" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == *" "* ]]; then
  echo "bind exactly one physical GPU with CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/third_party/ECG-LFM${PYTHONPATH:+:$PYTHONPATH}"

case "$mode" in
  joint)
    run_name="w0"
    train_dir="results/adaptive_weighting/${run_name}/fold${fold}"
    eval_dir="results/adaptive_weighting_evaluation/${run_name}/fold${fold}"
    ;;
  acs|omi)
    run_name="${mode}_only_w0"
    train_dir="results/single_task_vs_joint/${run_name}/fold${fold}"
    eval_dir="results/single_task_vs_joint_evaluation/${run_name}/fold${fold}"
    ;;
esac

provenance="$train_dir/launch_provenance.json"
alignment="$eval_dir/provenance_alignment_report.json"
metrics="$eval_dir/metrics.json"

is_complete() {
  [[ -s "$provenance" && -s "$alignment" && -s "$metrics" ]] &&
    grep -Eq '"returncode"[[:space:]]*:[[:space:]]*0' "$provenance" &&
    grep -Eq '"status"[[:space:]]*:[[:space:]]*"pass"' "$alignment"
}

if is_complete; then
  echo "SKIP completed mode=${mode} fold=${fold} metrics=${metrics}"
  exit 0
fi

if [[ -d "$train_dir" ]] && find "$train_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "ABORT incomplete existing training directory: $train_dir" >&2
  echo "Inspect and rescue explicitly; this launcher will not overwrite partial checkpoints." >&2
  exit 3
fi
if [[ -d "$eval_dir" ]] && find "$eval_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "ABORT incomplete existing evaluation directory: $eval_dir" >&2
  exit 3
fi

mkdir -p "$train_dir" "$eval_dir" "logs/joint_vs_single_nested5fold"
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

echo "START mode=${mode} fold=${fold} gpu=${CUDA_VISIBLE_DEVICES} utc=$(date -u +%FT%TZ)"
if [[ "$mode" == "joint" ]]; then
  python experiments/ecg_lfm/run_joint_fold.py \
    --fold "$fold" \
    --adaptation corrected_joint_lora \
    --run_name "$run_name" \
    --manifest_root manifests/ecg_lfm_nested \
    --output_root results/adaptive_weighting \
    --weighting w0
  python experiments/ecg_lfm/evaluate_nested_fold.py \
    --fold "$fold" \
    --run_name "$run_name" \
    --device cuda:0
else
  python experiments/ecg_lfm/run_single_task_fold.py \
    --task_mode "$mode" \
    --fold "$fold" \
    --run_name "$run_name"
  python experiments/ecg_lfm/evaluate_single_task_nested_fold.py \
    --task_mode "$mode" \
    --fold "$fold" \
    --run_name "$run_name" \
    --device cuda:0
fi

if ! is_complete; then
  echo "ABORT completion gate failed mode=${mode} fold=${fold}" >&2
  exit 4
fi

end="$(date +%s)"
peak="$(cat "$peak_file")"
provenance_sha="$(sha256sum "$provenance" | awk '{print $1}')"
printf '%s\n' \
  'mode,fold,seed,status,start_unix,end_unix,elapsed_seconds,peak_physical_gpu_memory_mib,gpu,launch_provenance_sha256' \
  "${mode},${fold},2026,completed,${start},${end},$((end-start)),${peak},${CUDA_VISIBLE_DEVICES},${provenance_sha}" \
  > "$eval_dir/runtime.csv"
echo "COMPLETED mode=${mode} fold=${fold} elapsed=$((end-start))s peak_physical=${peak}MiB metrics=${metrics}"

