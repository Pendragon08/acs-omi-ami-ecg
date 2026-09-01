#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/ACS-OMI-AMI
exec env CUDA_VISIBLE_DEVICES=1 bash scripts/run_adaptive_weighting_static_fold0.sh \
  >> logs/adaptive_weighting/static_fold0_master.log 2>&1
