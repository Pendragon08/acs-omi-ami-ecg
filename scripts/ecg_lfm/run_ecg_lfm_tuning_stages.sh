#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-/data/code/ouyanghao/miniconda3/bin/conda}"
GPU="${GPU:-1}"
PYTHON=("$CONDA" run --no-capture-output -n ecg-lfm python)
TUNING_ROOT="${TUNING_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label_tuning}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_ROOT/results/ecg_lfm_4label/finetune_paper}"
SUMMARY="$TUNING_ROOT/summary"
cd "$PROJECT_ROOT"
mkdir -p "$SUMMARY" logs/ecg_lfm_tuning

fold0_queue_complete() {
  # The detached launcher can finish all runs before writing its completion
  # marker.  Treat the validated artifacts as the completion contract so a
  # stale or missing screen marker cannot block candidate promotion.
  "${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

root = Path("results/ecg_lfm_4label_tuning")
runs = (
    "lr3e7_bce_b256", "lr3e6_bce_b256", "lr1e6_bce_b128",
    "lr1e6_bce_b512", "lr3e7_wbce_b256", "lr1e6_wbce_b256",
    "lr3e6_wbce_b256",
)
ok = True
for run in runs:
    fold = root / run / "fold0"
    evaluation = fold / "evaluation" / "fold_evaluation.json"
    provenance = fold / "launch_provenance.json"
    gate = fold / "fold0_gate.json"
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ok = False
        break
    if not evaluation.exists() or payload.get("returncode") != 0 or not gate_payload.get("passed"):
        ok = False
        break
raise SystemExit(0 if ok else 1)
PY
}

while ! fold0_queue_complete; do
  if ! screen -ls 2>/dev/null | grep -Fq '.ecg_lfm_tune_gpu1'; then
    printf '%s FOLD0_QUEUE_NOT_RUNNING_AND_NOT_COMPLETE\n' "$(date -Is)" >&2
    exit 1
  fi
  printf '%s WAIT_FOLD0_TUNING_QUEUE\n' "$(date -Is)"
  sleep 60
done

"${PYTHON[@]}" scripts/select_ecg_lfm_candidates.py --mode fold0 \
  --tuning_root "$TUNING_ROOT" --output "$SUMMARY/stage1_selection.json"
readarray -t promoted < <("${PYTHON[@]}" -c \
  "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); print(p['selected']['unweighted_bce']); print(p['selected']['weighted_bce'])" \
  "$SUMMARY/stage1_selection.json")
for run_name in "${promoted[@]}"; do
  printf '%s PROMOTE_TO_FOLDS_1_2 run=%s\n' "$(date -Is)" "$run_name"
  GPU="$GPU" RUN_NAME="$run_name" FOLDS="1 2" FINALIZE=0 \
    bash scripts/run_ecg_lfm_tuning_multifold.sh
done

for fold in 0 1 2; do
  expected="$BASELINE_ROOT/fold$fold/evaluation/fold_evaluation.json"
  while [[ ! -f "$expected" ]]; do
    printf '%s WAIT_BASELINE_EVALUATION fold=%s\n' "$(date -Is)" "$fold"
    sleep 60
  done
done

"${PYTHON[@]}" scripts/select_ecg_lfm_candidates.py --mode threefold \
  --tuning_root "$TUNING_ROOT" --baseline_root "$BASELINE_ROOT" \
  --stage1_selection "$SUMMARY/stage1_selection.json" \
  --output "$SUMMARY/stage2_selection.json"
winner=$("${PYTHON[@]}" -c \
  "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['selected_final'])" \
  "$SUMMARY/stage2_selection.json")
printf '%s LOCKED_THREEFOLD_WINNER run=%s\n' "$(date -Is)" "$winner"

if [[ "$winner" == "finetune_paper" ]]; then
  final_dir="$BASELINE_ROOT/summary"
  while [[ ! -f "$final_dir/submission_ecg_lfm_4label_youden.csv" || ! -f "$final_dir/oof_evaluation.json" ]]; do
    printf '%s WAIT_BASELINE_FIVEFOLD_FINALIZATION\n' "$(date -Is)"
    sleep 60
  done
else
  GPU="$GPU" RUN_NAME="$winner" FOLDS="3 4" FINALIZE=1 \
    bash scripts/run_ecg_lfm_tuning_multifold.sh
  final_dir="$TUNING_ROOT/$winner/summary"
fi

while [[ ! -f "$BASELINE_ROOT/summary/oof_evaluation.json" ]]; do
  printf '%s WAIT_BASELINE_FIVEFOLD_FOR_FAIR_FINAL_DELTA\n' "$(date -Is)"
  sleep 60
done

"${PYTHON[@]}" scripts/write_ecg_lfm_final_tuning_report.py \
  --selection "$SUMMARY/stage2_selection.json" --final_dir "$final_dir"

submission_lock="/tmp/ecg_lfm_${USER}_external_platform_submission.lock"
exec 8>"$submission_lock"
flock 8
receipt_complete() {
  [[ -f "$final_dir/official_platform_4label_metrics.json" ]] && \
    "${PYTHON[@]}" -c \
      "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); raise SystemExit(0 if p.get('curl_returncode') == 0 and not p.get('missing_metric_labels') else 1)" \
      "$final_dir/official_platform_4label_metrics.json"
}
if ! receipt_complete; then
  relative_final=${final_dir#"$PROJECT_ROOT/"}
  "${PYTHON[@]}" scripts/submit_hmoe_e4_4label.py \
    --run_dir "$relative_final" --submission submission_ecg_lfm_4label_youden.csv
fi
"${PYTHON[@]}" scripts/write_ecg_lfm_final_tuning_report.py \
  --selection "$SUMMARY/stage2_selection.json" --final_dir "$final_dir"
printf '%s ECG_LFM_TUNING_EXTERNAL_AND_REPORT_COMPLETE winner=%s\n' "$(date -Is)" "$winner"
