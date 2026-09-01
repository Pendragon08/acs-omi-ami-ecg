#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
CONDA=/data/code/ouyanghao/miniconda3
cd "$ROOT"
source "$CONDA/etc/profile.d/conda.sh"
conda activate ecg-lfm

visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ "$visible" != "1" ]]; then
  echo "ERROR: this queued gate must bind exactly physical GPU1" >&2
  exit 2
fi

static_master=logs/adaptive_weighting/static_fold0_master.log
echo "WAIT static W0-W3 at $(date -u +%FT%TZ)"
while ! grep -q "DONE w3 nested fold0" "$static_master" 2>/dev/null; do
  if ! screen -list 2>/dev/null | grep -q "[.]ecg_static_fold0"; then
    echo "ERROR: static screen ended without W3 completion marker" >&2
    exit 3
  fi
  sleep 30
done

python - <<'PY'
import json
from pathlib import Path
root = Path("results")
for method in ("w0", "w1", "w2", "w3"):
    provenance = json.loads((root / "adaptive_weighting" / method / "fold0" / "launch_provenance.json").read_text())
    alignment = json.loads((root / "adaptive_weighting_evaluation" / method / "fold0" / "provenance_alignment_report.json").read_text())
    if provenance.get("returncode") != 0 or not provenance.get("finished_at"):
        raise SystemExit(f"{method} training provenance gate failed")
    if alignment.get("status") != "pass" or any(alignment.get("patient_overlap", {}).values()):
        raise SystemExit(f"{method} evaluation alignment gate failed")
print("STATIC_GATE_PASS")
PY

python -u experiments/ecg_lfm/derive_rdr_warmup.py
warmup_updates=$(python - <<'PY'
import json
print(json.load(open("results/adaptive_weighting/rdr_warmup_selection.json"))["frozen_rdr_warmup_updates"])
PY
)
echo "FROZEN warmup_updates=${warmup_updates}"

python -u experiments/ecg_lfm/profile_rdr_1024d.py \
  --checkpoint results/adaptive_weighting/w0/fold0/checkpoint_best.pt \
  --manifest_dir manifests/ecg_lfm_nested/fold0 \
  --output results/adaptive_weighting/rdr_profile_fold0.json \
  --device cuda:0 --batch 64 --workers 8 --small_subset_batches 8

RDR_WARMUP_UPDATES="$warmup_updates" bash scripts/run_adaptive_weighting_rdr_fold0.sh
echo "RDR_FOLD0_GATE_EXPERIMENTS_DONE at $(date -u +%FT%TZ)"
