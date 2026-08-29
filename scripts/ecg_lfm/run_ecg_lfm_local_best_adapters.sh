#!/usr/bin/env bash
set -euo pipefail

# LEGACY INVALID TASK: this script runs the historical four-independent-
# sigmoid endpoint. Use scripts/run_corrected_ecg_lfm_joint.sh instead.
# Five-fold, architecture-preserving adaptation ablation for the ECG-LFM
# four-label task.  The historical folds and official 500-Hz pipeline are
# reused verbatim; only model.adapter_method and its prescribed optimizer
# setting differ between lora and dora.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-1}"
PYTHON="${PYTHON:-python}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/ecg_lfm_local_best}"
cd "$PROJECT_ROOT"

test "$(git -C third_party/ECG-LFM rev-parse HEAD)" = "647369a012dd8d5d362e99e2bf238c91f3e1d03b"
echo "afa8dfc680590948f82aabbdb123bf43  checkpoints/ecg_lfm/pretrain.pt" | md5sum --check --status
test -f manifests/ecg_lfm/cohort_integrity.json
test -f manifests/ecg_lfm/preprocess_integrity.json

mkdir -p "$RESULT_ROOT"
printf '%s\n' \
  'ECG-LFM local-best adapter ablation' \
  "started_at=$(date -Is)" \
  "gpu=$GPU" \
  "split=historical patient-level five-fold seed2026" \
  "baseline=results/ecg_lfm_4label/finetune_paper" \
  "loss=binary_cross_entropy_with_logits (unweighted)" \
  "lora=r8 alpha16 dropout0.05 lr1e-4" \
  "dora=r8 alpha16 dropout0.05 lr5e-5" \
  > "$RESULT_ROOT/ablation_manifest.txt"

for adaptation in lora dora; do
  echo "$(date -Is) START $adaptation" | tee -a "$RESULT_ROOT/ablation_manifest.txt"
  GPU="$GPU" PYTHON="$PYTHON" RESULT_ROOT="$RESULT_ROOT" \
    ADAPTATION="$adaptation" FINALIZE=1 \
    bash scripts/run_ecg_lfm_5fold.sh
  test -f "$RESULT_ROOT/$adaptation/summary/oof_evaluation.json"
  test -f "$RESULT_ROOT/$adaptation/summary/thresholds.json"
  "$PYTHON" experiments/ecg_lfm/write_adapter_artifacts.py \
    --checkpoint "$RESULT_ROOT/$adaptation/fold0/checkpoint_best.pt" \
    --manifest-dir manifests/ecg_lfm/fold0 \
    --output-dir "$RESULT_ROOT/$adaptation/summary/adapter_audit"
  echo "$(date -Is) COMPLETE $adaptation" | tee -a "$RESULT_ROOT/ablation_manifest.txt"
done

echo "$(date -Is) COMPLETE all adapter methods" | tee -a "$RESULT_ROOT/ablation_manifest.txt"
