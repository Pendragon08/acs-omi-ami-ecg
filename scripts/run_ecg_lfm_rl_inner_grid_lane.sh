#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "usage: $0 {0|1}" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "bind exactly one GPU with CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

PYTHON=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python
RUNNER=experiments/ecg_lfm/train_rl_fold.py
CONFIG=configs/ecg_lfm/rl/clinical_rl_joint_w0.yaml

run_candidate() {
  local lr="$1" kl="$2" scale="$3" suffix="$4"
  "$PYTHON" "$RUNNER" --config "$CONFIG" --smoke --max_updates 2 \
    --learning_rate "$lr" --kl_coefficient "$kl" \
    --hard_reward_scale "$scale" --run_suffix "$suffix"
}

if [[ "$1" == "0" ]]; then
  run_candidate 0.000001 0.01 0.5 _lr1e6_k001_s05
  run_candidate 0.000001 0.05 0.5 _lr1e6_k005_s05
  run_candidate 0.000005 0.01 0.5 _lr5e6_k001_s05
  run_candidate 0.000005 0.05 0.5 _lr5e6_k005_s05
else
  run_candidate 0.000001 0.01 1.0 _lr1e6_k001_s10
  run_candidate 0.000001 0.05 1.0 _lr1e6_k005_s10
  run_candidate 0.000005 0.01 1.0 _lr5e6_k001_s10
  run_candidate 0.000005 0.05 1.0 _lr5e6_k005_s10
fi
