#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
CONDA=/data/code/ouyanghao/miniconda3
cd "$ROOT"
source "$CONDA/etc/profile.d/conda.sh"
conda activate ecg-lfm
visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible" || "$visible" == *,* || "$visible" == *" "* ]]; then
  echo "ERROR: exactly one CUDA_VISIBLE_DEVICES value is required" >&2
  exit 2
fi

mkdir -p logs/adaptive_weighting results/adaptive_weighting results/adaptive_weighting_evaluation
runtime_csv=results/adaptive_weighting/fold0_runtime.csv
if [[ ! -f "$runtime_csv" ]]; then
  echo "method,fold,seed,status,start_unix,end_unix,elapsed_seconds,peak_gpu_memory_mib,gpu,launch_provenance_sha256" > "$runtime_csv"
fi

for method in w0 w1 w2 w3; do
  checkpoint="results/adaptive_weighting/${method}/fold0/checkpoint_best.pt"
  provenance="results/adaptive_weighting/${method}/fold0/launch_provenance.json"
  start=$(date +%s)
  peak=0
  completed=0
  if [[ -s "$checkpoint" && -s "$provenance" ]]; then
    completed=$(python - "$provenance" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(data.get("returncode") == 0 and bool(data.get("finished_at"))))
PY
)
  fi
  if [[ "$completed" == 0 && ( -e "$checkpoint" || -e "$provenance" ) ]]; then
    echo "ERROR: partial or unverified run exists for ${method}; refusing to skip or overwrite" >&2
    exit 3
  fi
  if [[ "$completed" == 0 ]]; then
    echo "START ${method} nested fold0 at $(date -u +%FT%TZ)"
    python -u experiments/ecg_lfm/run_joint_fold.py \
      --fold 0 \
      --adaptation corrected_joint_lora \
      --run_name "$method" \
      --manifest_root manifests/ecg_lfm_nested \
      --output_root results/adaptive_weighting \
      --seed 2026 \
      --weighting "$method" \
      > "logs/adaptive_weighting/${method}_fold0_train.log" 2>&1 &
    train_pid=$!
    while kill -0 "$train_pid" 2>/dev/null; do
      used=$(nvidia-smi --id="$visible" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' || echo 0)
      [[ "$used" =~ ^[0-9]+$ ]] || used=0
      if (( used > peak )); then
        peak=$used
      fi
      sleep 15
    done
    wait "$train_pid"
    end=$(date +%s)
    provenance_sha=$(sha256sum "$provenance" | awk '{print $1}')
    echo "${method},0,2026,completed,${start},${end},$((end-start)),${peak},${visible},${provenance_sha}" >> "$runtime_csv"
  else
    echo "SKIP TRAIN ${method}: successful completion provenance already exists"
  fi
  python -u experiments/ecg_lfm/evaluate_nested_fold.py \
    --run_name "$method" \
    --fold 0 \
    --run_root results/adaptive_weighting \
    --manifest_root manifests/ecg_lfm_nested \
    --output_root results/adaptive_weighting_evaluation \
    --device cuda:0 --workers 8 --batch 64 \
    > "logs/adaptive_weighting/${method}_fold0_evaluate.log" 2>&1
  echo "DONE ${method} nested fold0 at $(date -u +%FT%TZ)"
done
