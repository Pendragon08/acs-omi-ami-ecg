#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/code/ouyanghao/oyh/ACS-OMI-AMI"
LOG_DIR="$ROOT/logs/joint_vs_single_nested5fold"
FINAL_LOG="$LOG_DIR/finalizer.log"
PYTHON="/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python"
mkdir -p "$LOG_DIR"
cd "$ROOT"

complete_count() {
  local count=0 mode fold dir
  for fold in 0 1 2 3 4; do
    for mode in joint acs omi; do
      case "$mode" in
        joint) dir="results/adaptive_weighting_evaluation/w0/fold${fold}" ;;
        acs) dir="results/single_task_vs_joint_evaluation/acs_only_w0/fold${fold}" ;;
        omi) dir="results/single_task_vs_joint_evaluation/omi_only_w0/fold${fold}" ;;
      esac
      if [[ -s "$dir/metrics.json" && -s "$dir/provenance_alignment_report.json" && -s "$dir/outer_test_predictions.csv" ]]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

while true; do
  now="$(date --iso-8601=seconds)"
  completed="$(complete_count)"
  available_gib="$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')"
  printf '[%s] completed=%s/15 data_available_gib=%s\n' "$now" "$completed" "$available_gib" | tee -a "$FINAL_LOG"
  if [[ "$available_gib" -lt 20 ]]; then
    printf '[%s] ABORT: /data safety floor (<20 GiB) reached.\n' "$now" | tee -a "$FINAL_LOG"
    exit 20
  fi
  if [[ "$completed" -eq 15 ]]; then
    break
  fi
  live_queues="$(screen -ls 2>/dev/null | grep -Ec 'ecg_rl_s0_gpu[01]' || true)"
  if [[ "$live_queues" -eq 0 ]]; then
    printf '[%s] ABORT: queues ended before all artifacts completed.\n' "$now" | tee -a "$FINAL_LOG"
    exit 21
  fi
  sleep 300
done

"$PYTHON" experiments/ecg_lfm/summarize_joint_vs_single_nested5fold.py \
  --replicates 1000 --seed 2026 2>&1 | tee -a "$FINAL_LOG"
printf '[%s] FINALIZED nested Joint vs single-task comparison.\n' "$(date --iso-8601=seconds)" | tee -a "$FINAL_LOG"
