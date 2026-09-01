#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
SESSION=ecg_rdr_gate_queue
cd "$ROOT"
if screen -list 2>/dev/null | grep -q "[.]${SESSION}"; then
  echo "ERROR: ${SESSION} already exists" >&2
  exit 2
fi
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 bash scripts/run_rdr_after_static_gate_gpu1.sh > logs/adaptive_weighting/rdr_gate_queue.log 2>&1"
echo "launched ${SESSION}; it will not use GPU1 until the static completion gate passes"
