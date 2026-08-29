#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
url="https://zenodo.org/records/20388950/files/pretrain.pt?download=1"
expected_md5="afa8dfc680590948f82aabbdb123bf43"
total=7522279891
count="${PARALLEL_CONNECTIONS:-32}"
chunk=$(((total + count - 1) / count))
request_bytes="${RANGE_REQUEST_BYTES:-16777216}"
part_dir="checkpoints/ecg_lfm/parts_20260827"
mkdir -p "$part_dir" checkpoints/ecg_lfm logs/ecg_lfm

date -Is
for i in $(seq 0 $((count - 1))); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if [[ "$end" -ge "$total" ]]; then end=$((total - 1)); fi
  expected=$((end - start + 1))
  part=$(printf "%s/part_%02d" "$part_dir" "$i")
  (
    if [[ -f "$part" ]] && [[ "$(stat -c %s "$part")" -eq "$expected" ]]; then
      printf 'PART_OK %02d %s reused\n' "$i" "$expected"
      exit 0
    fi
    tmp="${part}.tmp"
    touch "$tmp"
    have=$(stat -c %s "$tmp")
    if [[ "$have" -gt "$expected" ]]; then
      printf 'PART_OVERSIZE %02d have=%s expected=%s\n' "$i" "$have" "$expected" >&2
      exit 1
    fi
    while [[ "$have" -lt "$expected" ]]; do
      absolute_start=$((start + have))
      absolute_end=$((absolute_start + request_bytes - 1))
      if [[ "$absolute_end" -gt "$end" ]]; then absolute_end="$end"; fi
      piece="${part}.piece"
      : > "$piece"
      set +e
      curl -L --fail --connect-timeout 30 \
        --max-time 900 -r "${absolute_start}-${absolute_end}" "$url" -o "$piece"
      curl_rc=$?
      set -e
      piece_size=$(stat -c %s "$piece")
      remaining=$((expected - have))
      if [[ "$piece_size" -gt "$remaining" ]]; then
        printf 'RANGE_IGNORED %02d piece=%s remaining=%s\n' "$i" "$piece_size" "$remaining" >&2
        exit 1
      fi
      if [[ "$piece_size" -gt 0 ]]; then
        cat "$piece" >> "$tmp"
        have=$((have + piece_size))
        printf 'PART_PROGRESS %02d %s/%s curl_rc=%s\n' "$i" "$have" "$expected" "$curl_rc"
      else
        printf 'PART_RETRY %02d %s/%s curl_rc=%s\n' "$i" "$have" "$expected" "$curl_rc" >&2
        sleep 2
      fi
    done
    if [[ "$(stat -c %s "$tmp")" -eq "$expected" ]]; then
      mv "$tmp" "$part"
    fi
    printf 'PART_OK %02d %s\n' "$i" "$expected"
  ) > "${part}.log" 2>&1 &
done
wait

combined="checkpoints/ecg_lfm/pretrain.pt.part.parallel"
: > "$combined"
for i in $(seq 0 $((count - 1))); do
  part=$(printf "%s/part_%02d" "$part_dir" "$i")
  test -f "$part"
  cat "$part" >> "$combined"
done
[[ "$(stat -c %s "$combined")" -eq "$total" ]]
actual_md5=$(md5sum "$combined" | cut -d' ' -f1)
printf 'CHECKPOINT_MD5=%s\n' "$actual_md5"
[[ "$actual_md5" = "$expected_md5" ]]
mv "$combined" checkpoints/ecg_lfm/pretrain.pt
printf 'CHECKPOINT_BYTES=%s\n' "$(stat -c %s checkpoints/ecg_lfm/pretrain.pt)"
date -Is
