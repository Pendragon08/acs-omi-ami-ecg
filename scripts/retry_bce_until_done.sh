#!/usr/bin/env bash
# Patiently re-submit the bce summary until every entry succeeds.
# The platform intermittently rate-limits; loop retry_official_test.py with
# backoff, checking the summary between passes.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=/data/code/ouyanghao/miniconda3/envs/oyh/bin/python
SUMMARY=results/model_comparison/official_test_summary_bce.json

for i in $(seq 1 30); do
  echo "=== pass $i: $(date '+%F %T') ==="
  "$PY" scripts/ecg/retry_official_test.py \
      --exp_name model_comparison --summary official_test_summary_bce.json \
      --attempts 2 --sleep_base 15
  left=$("$PY" -c "import json;r=json.load(open('$SUMMARY'));print(sum(1 for x in r if not x.get('ok')))")
  echo "remaining=$left"
  if [ "$left" = "0" ]; then
    echo "ALL_DONE"
    break
  fi
  sleep 60
done
