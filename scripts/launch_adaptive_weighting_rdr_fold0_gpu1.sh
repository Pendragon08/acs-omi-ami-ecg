#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 RDR_WARMUP_UPDATES" >&2
  exit 2
fi
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
SESSION=ecg_rdr_fold0
if screen -list | grep -q "[.]${SESSION}"; then
  echo "ERROR: screen session ${SESSION} already exists" >&2
  exit 3
fi
cd "$ROOT"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 RDR_WARMUP_UPDATES='$1' bash scripts/run_adaptive_weighting_rdr_fold0.sh > logs/adaptive_weighting/rdr_fold0_master.log 2>&1"
echo "launched ${SESSION} on physical GPU1 with frozen warmup_updates=$1"
