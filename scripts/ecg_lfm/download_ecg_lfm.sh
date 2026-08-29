#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/data/raw_acs_figshare_500hz}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints/ecg_lfm}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/ecg_lfm}"
mkdir -p "$RAW_DIR" "$RAW_DIR/extracted" "$CHECKPOINT_DIR" "$LOG_DIR"

RAW_URL="https://ndownloader.figshare.com/files/62951395"
RAW_MD5="acea6ca86a2d0b937ecfe7d1df6d30a2"
CHECKPOINT_URL="https://zenodo.org/records/20388950/files/pretrain.pt?download=1"
CHECKPOINT_MD5="afa8dfc680590948f82aabbdb123bf43"

download_and_verify() {
  local url="$1" destination="$2" expected_md5="$3"
  local part="${destination}.part"
  curl -L --fail --continue-at - --retry 12 --retry-delay 5 --connect-timeout 30 \
    -A "Mozilla/5.0 ECG-LFM-research/1.0" "$url" -o "$part"
  local actual_md5
  actual_md5=$(md5sum "$part" | cut -d' ' -f1)
  printf '%s  %s\n' "$actual_md5" "$part"
  test "$actual_md5" = "$expected_md5"
  mv "$part" "$destination"
}

if [[ ! -f "$RAW_DIR/ECG_row_data.zip" ]]; then
  download_and_verify "$RAW_URL" "$RAW_DIR/ECG_row_data.zip" "$RAW_MD5"
fi
echo "$RAW_MD5  $RAW_DIR/ECG_row_data.zip" | md5sum --check --status

if [[ ! -f "$CHECKPOINT_DIR/pretrain.pt" ]]; then
  download_and_verify "$CHECKPOINT_URL" "$CHECKPOINT_DIR/pretrain.pt" "$CHECKPOINT_MD5"
fi
echo "$CHECKPOINT_MD5  $CHECKPOINT_DIR/pretrain.pt" | md5sum --check --status

if ! find "$RAW_DIR/extracted" -type f -name '*.hea' -print -quit | grep -q .; then
  unzip -q "$RAW_DIR/ECG_row_data.zip" -d "$RAW_DIR/extracted"
fi

hea_count=$(find "$RAW_DIR/extracted" -type f -name '*.hea' | wc -l)
dat_count=$(find "$RAW_DIR/extracted" -type f -name '*.dat' | wc -l)
test "$hea_count" -eq 19955
test "$dat_count" -eq 19955
printf 'RAW_WFDB_OK hea=%s dat=%s\n' "$hea_count" "$dat_count"
printf 'ECG_LFM_CHECKPOINT_OK md5=%s\n' "$CHECKPOINT_MD5"
