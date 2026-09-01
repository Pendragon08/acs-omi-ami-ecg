#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "usage: $0 {0|1}" >&2; exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "bind exactly one GPU with CUDA_VISIBLE_DEVICES" >&2; exit 2
fi
PYTHON=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python
RUNNER=experiments/ecg_lfm/train_rl_v2_head_only.py
CONFIG=configs/ecg_lfm/rl/clinical_rl_v2_head_only.yaml
run_candidate() {
  "$PYTHON" "$RUNNER" --config "$CONFIG" --learning_rate "$1" --target_kl "$2" --run_suffix "$3"
}
if [[ "$1" == "0" ]]; then
  run_candidate 0.000001 0.005 _lr1e6_tkl005
  run_candidate 0.000001 0.01 _lr1e6_tkl010
else
  run_candidate 0.000003 0.005 _lr3e6_tkl005
  run_candidate 0.000003 0.01 _lr3e6_tkl010
fi
