#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
export PATH=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin:$PATH
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$ROOT"
mkdir -p logs/joint6_optimization_20260907
exec > >(tee -a logs/joint6_optimization_20260907/reporting_repair.log) 2>&1
python -m experiments.ecg_lfm.repair_joint6_reporting
python -m experiments.ecg_lfm.summarize_joint6_cv --root results/joint6_cv --report docs/ECG_LFM_Joint6_CV_Comparison_Report.md --require_complete
cp docs/ECG_LFM_Joint6_CV_Comparison_Report.md docs/ECG_LFM_Joint6_Evaluation_Report.md
