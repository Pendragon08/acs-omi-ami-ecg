#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
marker=logs/loss_feature_ablation/stage1_complete.log
while [[ ! -f "$marker" ]]; do
  if ! screen -ls 2>/dev/null | grep -q acs_ablation_stage1_gpu1; then
    printf 'STAGE1_STOPPED_WITHOUT_MARKER %s\n' "$(date -Iseconds)"
    exit 1
  fi
  sleep 30
done
bash run_ablation_stage2_gpu1.sh > logs/loss_feature_ablation/stage2_master.log 2>&1
