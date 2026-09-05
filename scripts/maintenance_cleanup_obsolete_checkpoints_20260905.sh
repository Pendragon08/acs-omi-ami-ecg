#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

PROJECT_ROOT="/data/code/ouyanghao/oyh/ACS-OMI-AMI"
RESULTS_ROOT="$PROJECT_ROOT/results"
LOG_DIR="$PROJECT_ROOT/logs/maintenance"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$LOG_DIR/obsolete_checkpoint_cleanup_${STAMP}.tsv"
MIN_BYTES=$((30 * 1024 * 1024 * 1024))
MAX_BYTES=$((50 * 1024 * 1024 * 1024))

targets=(
  "$RESULTS_ROOT/ecg_lfm_4label/partial_runs/physical8_20260827_1338/linear_probe_fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/linear_probe/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/finetune_paper/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/finetune_paper/fold1/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/finetune_paper/fold2/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/finetune_paper/fold3/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label/finetune_paper/fold4/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr3e6_bce_b256/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr1e6_bce_b512/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr1e6_bce_b128/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr3e7_bce_b256/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr3e6_wbce_b256/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr1e6_wbce_b256/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/ecg_lfm_4label_tuning/lr3e7_wbce_b256/fold0/checkpoint_best.pt"
  "$RESULTS_ROOT/corrected_ecg_lfm_joint_bs64_v2/dora/fold3_interrupted_20260830_1052/checkpoint_best.pt"
  "$RESULTS_ROOT/corrected_ecg_lfm_joint_bs64_v2_parallel/dora/fold4/checkpoint_best.pt"
)

mkdir -p "$LOG_DIR"
printf 'status\tbytes\tpath\n' > "$MANIFEST"

total_bytes=0
existing_count=0
for target in "${targets[@]}"; do
  if [[ ! -e "$target" ]]; then
    printf 'missing\t0\t%s\n' "$target" | tee -a "$MANIFEST"
    continue
  fi
  if [[ ! -f "$target" || -L "$target" ]]; then
    echo "REFUSE: target is not a regular non-symlink file: $target" >&2
    exit 3
  fi
  resolved="$(readlink -f -- "$target")"
  case "$resolved" in
    "$RESULTS_ROOT"/*) ;;
    *) echo "REFUSE: target escapes results root: $resolved" >&2; exit 4 ;;
  esac
  case "$resolved" in
    "$RESULTS_ROOT/joint6_cv"/*)
      echo "REFUSE: Joint-6 artifacts are protected: $resolved" >&2
      exit 5
      ;;
  esac
  bytes="$(stat -c '%s' -- "$resolved")"
  total_bytes=$((total_bytes + bytes))
  existing_count=$((existing_count + 1))
  printf 'selected\t%s\t%s\n' "$bytes" "$resolved" | tee -a "$MANIFEST"
done

gib="$(awk -v bytes="$total_bytes" 'BEGIN {printf "%.2f", bytes/1024/1024/1024}')"
echo "Selected $existing_count files, $total_bytes bytes (${gib} GiB)."
echo "Manifest: $MANIFEST"

if (( total_bytes < MIN_BYTES || total_bytes > MAX_BYTES )); then
  echo "REFUSE: selected size ${gib} GiB is outside the approved 30-50 GiB range." >&2
  exit 6
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY RUN: no files deleted."
  exit 0
fi

for target in "${targets[@]}"; do
  if [[ -f "$target" && ! -L "$target" ]]; then
    rm -f -- "$target"
    if [[ -e "$target" ]]; then
      echo "ERROR: deletion verification failed: $target" >&2
      exit 7
    fi
    printf 'deleted\t0\t%s\n' "$target" | tee -a "$MANIFEST"
  fi
done

echo "EXECUTE COMPLETE: deleted $existing_count files and reclaimed approximately ${gib} GiB."
df -h / "$RESULTS_ROOT"
