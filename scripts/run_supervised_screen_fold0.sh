#!/usr/bin/env bash
set -euo pipefail

scheme="${1:?usage: run_supervised_screen_fold0.sh <focal|omi_asl|label_smoothing|omi_hnm>}"
case "$scheme" in
  focal|omi_asl|label_smoothing|omi_hnm) ;;
  *) echo "unknown scheme: $scheme" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == *" "* ]]; then
  echo "bind exactly one physical GPU with CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_DIR="results/rl_supervised_screening/${scheme}/fold0"
EVAL_DIR="results/rl_supervised_screening_evaluation/${scheme}/fold0"
PROVENANCE="$TRAIN_DIR/launch_provenance.json"
ALIGNMENT="$EVAL_DIR/provenance_alignment_report.json"
METRICS="$EVAL_DIR/metrics.json"

is_complete() {
  [[ -s "$PROVENANCE" && -s "$ALIGNMENT" && -s "$METRICS" ]] &&
    grep -Eq '"returncode"[[:space:]]*:[[:space:]]*0' "$PROVENANCE" &&
    grep -Eq '"status"[[:space:]]*:[[:space:]]*"pass"' "$ALIGNMENT"
}
if is_complete; then
  echo "SKIP completed scheme=$scheme"
  exit 0
fi
if [[ -d "$TRAIN_DIR" ]] && find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "ABORT incomplete existing training directory: $TRAIN_DIR" >&2
  exit 3
fi
if [[ -d "$EVAL_DIR" ]] && find "$EVAL_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "ABORT incomplete existing evaluation directory: $EVAL_DIR" >&2
  exit 3
fi
available_gib="$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')"
if [[ "$available_gib" -lt 24 ]]; then
  echo "ABORT /data has only ${available_gib} GiB free; retain a 20-GiB safety floor plus checkpoint headroom" >&2
  exit 24
fi

mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs/rl_supervised_screening
start="$(date +%s)"
peak_file="$(mktemp)"
echo 0 > "$peak_file"
(
  while true; do
    used="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)"
    current="$(cat "$peak_file")"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > current )); then echo "$used" > "$peak_file"; fi
    sleep 10
  done
) &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  rm -f "$peak_file"
}
trap cleanup EXIT

python experiments/ecg_lfm/run_supervised_screen_fold.py --scheme "$scheme" --fold 0
python experiments/ecg_lfm/evaluate_nested_fold.py \
  --fold 0 --run_name "$scheme" \
  --run_root results/rl_supervised_screening \
  --output_root results/rl_supervised_screening_evaluation \
  --device cuda:0
if ! is_complete; then
  echo "ABORT completion gate failed scheme=$scheme" >&2
  exit 4
fi
end="$(date +%s)"
printf '%s\n' \
  'scheme,fold,seed,status,start_unix,end_unix,elapsed_seconds,peak_physical_gpu_memory_mib,gpu,launch_provenance_sha256' \
  "${scheme},0,2026,completed,${start},${end},$((end-start)),$(cat "$peak_file"),${CUDA_VISIBLE_DEVICES},$(sha256sum "$PROVENANCE" | awk '{print $1}')" \
  > "$EVAL_DIR/runtime.csv"
echo "COMPLETED scheme=$scheme metrics=$METRICS"
