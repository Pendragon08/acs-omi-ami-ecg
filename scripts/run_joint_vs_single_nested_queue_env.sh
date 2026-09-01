#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_joint_vs_single_nested_queue_env.sh <gpu> <fold> [fold ...]}"
shift
if [[ "$gpu" != "0" && "$gpu" != "1" ]]; then
  echo "gpu must be physical GPU 0 or 1" >&2
  exit 2
fi
if [[ "$#" -lt 1 ]]; then
  echo "at least one fold is required" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_bin="/data/code/ouyanghao/miniconda3/bin/conda"
if [[ ! -x "$conda_bin" ]]; then
  echo "conda executable not found: $conda_bin" >&2
  exit 2
fi

cd "$project_root"
mkdir -p logs/joint_vs_single_nested5fold
eval "$("$conda_bin" shell.bash hook)"
conda activate ecg-lfm
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$project_root:$project_root/third_party/ECG-LFM${PYTHONPATH:+:$PYTHONPATH}"

log="logs/joint_vs_single_nested5fold/gpu${gpu}_master.log"
set +e
bash scripts/run_joint_vs_single_nested_queue.sh "$@" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
echo "QUEUE_EXIT gpu=${gpu} folds=$* status=${status} utc=$(date -u +%FT%TZ)" | tee -a "$log"
exit "$status"
