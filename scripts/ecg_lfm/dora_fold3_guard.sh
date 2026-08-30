#!/usr/bin/env bash
set -euo pipefail

# Stop the recovery screen immediately after fold3, before its queued fold4
# command can overwrite the already completed parallel fold4 checkpoint.
LOG="/data/code/ouyanghao/oyh/ACS-OMI-AMI/logs/corrected_ecg_lfm_joint_bs64_v2/dora_recovery_master.log"
while ! grep -q FOLD3_DONE "$LOG"; do
  sleep 5
done
screen -S corrected_dora_recovery -X quit
echo "GUARD_STOPPED_AFTER_FOLD3"
