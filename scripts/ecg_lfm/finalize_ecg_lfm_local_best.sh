#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_local_best}"
cd "$PROJECT_ROOT"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

log "waiting for LoRA and DoRA five-fold OOF artifacts"
while [[ ! -f "$RESULT_ROOT/lora/summary/oof_evaluation.json" || ! -f "$RESULT_ROOT/dora/summary/oof_evaluation.json" ]]; do
  sleep 60
done

for method in lora dora; do
  "$PYTHON" experiments/ecg_lfm/write_adapter_artifacts.py \
    --checkpoint "$RESULT_ROOT/$method/fold0/checkpoint_best.pt" \
    --manifest-dir manifests/ecg_lfm/fold0 \
    --output-dir "$RESULT_ROOT/$method/summary/adapter_audit"
done

"$PYTHON" scripts/write_ecg_lfm_local_best_report.py \
  --project-root "$PROJECT_ROOT"

winner="$($PYTHON -c 'import json,sys; p=sys.argv[1]; names=[("Baseline (PFT)","results/ecg_lfm_4label/finetune_paper/summary/oof_evaluation.json"),("LoRA","results/ecg_lfm_local_best/lora/summary/oof_evaluation.json"),("DoRA","results/ecg_lfm_local_best/dora/summary/oof_evaluation.json")]; vals=[]; root=__import__("pathlib").Path(p); [vals.append((n,json.loads((root/q).read_text())["fivefold_mean_std_at_global_oof_thresholds"]["macro"])) for n,q in names if (root/q).exists()]; print(max(vals,key=lambda x:(x[1]["auprc"]["mean"],x[1]["auroc"]["mean"],x[1]["f1"]["mean"]))[0])' "$PROJECT_ROOT")"
if [[ "$winner" == "LoRA" || "$winner" == "DoRA" ]]; then
  run_dir="$RESULT_ROOT/$(echo "$winner" | tr '[:upper:]' '[:lower:]')/summary"
  if [[ ! -f "$run_dir/official_platform_4label_metrics.json" ]]; then
    "$PYTHON" scripts/submit_hmoe_e4_4label.py \
      --run_dir "$run_dir" \
      --submission submission_ecg_lfm_4label_youden.csv \
      || log "external platform submission failed; raw stderr/HTML remain in $run_dir"
  fi
fi
"$PYTHON" scripts/write_ecg_lfm_local_best_report.py --project-root "$PROJECT_ROOT"
log "finalization complete; winner=$winner"
