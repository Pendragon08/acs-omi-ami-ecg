#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/code/ouyanghao/oyh/ACS-OMI-AMI
GPU=${1:?physical GPU is required}
CANDIDATE=${2:?candidate is required}
case "$GPU:$CANDIDATE" in 0:lr5e5|1:lr3e5) ;; *) echo "unexpected GPU/candidate pair"; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="$GPU"
export PATH=/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin:$PATH
export PYTHONPATH="$ROOT:$ROOT/third_party/ECG-LFM"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HYDRA_FULL_ERROR=1
cd "$ROOT"
mkdir -p logs/joint6_optimization_20260907
exec > >(tee -a "logs/joint6_optimization_20260907/${CANDIDATE}.log") 2>&1
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( USED >= 500 )); then echo "GPU $GPU no longer idle; refusing to launch"; exit 3; fi
python -m experiments.ecg_lfm.joint6_optimization run --candidate "$CANDIDATE"
