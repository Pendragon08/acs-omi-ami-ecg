#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: run_joint_vs_single_nested_queue.sh <fold> [fold ...]" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == *" "* ]]; then
  echo "bind exactly one physical GPU with CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p logs/joint_vs_single_nested5fold

echo "QUEUE_START gpu=${CUDA_VISIBLE_DEVICES} folds=$* utc=$(date -u +%FT%TZ)"
for fold in "$@"; do
  for mode in joint acs omi; do
    bash scripts/run_joint_vs_single_nested_fold.sh "$mode" "$fold"
  done
done
echo "QUEUE_COMPLETED gpu=${CUDA_VISIBLE_DEVICES} folds=$* utc=$(date -u +%FT%TZ)"
