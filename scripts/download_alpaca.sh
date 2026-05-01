#!/usr/bin/env bash
# download_alpaca.sh — fetch real SFT data on the rental.
#
# Pulls Stanford-Alpaca-format JSON for english + chinese + LIMA + math QA.
# Multi-mirror with first-success-wins; verifies size + jq parses.
#
# Total ~60MB. Targets /workspace/data/sft/.
#
# Usage:
#   bash scripts/download_alpaca.sh                # do all 4
#   bash scripts/download_alpaca.sh --help         # show this and exit
#   bash scripts/download_alpaca.sh --only alpaca_en  # one file only

set -u
set -o pipefail

OUT_DIR="${OUT_DIR:-/workspace/data/sft}"
MIN_BYTES="${MIN_BYTES:-1048576}"   # 1MB floor for "valid" download
TIMEOUT="${TIMEOUT:-120}"           # per-mirror seconds
ONLY=""

usage() {
  sed -n '2,14p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --only)    ONLY="$2"; shift 2 ;;
    --out)     OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- mirror table -----------------------------------------------------------
# Each entry: <name>|<dest_filename>|<url1>|<url2>|<url3>
MIRRORS=(
  "alpaca_en|alpaca_en.json|https://hf-mirror.com/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet|https://hf-mirror.com/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json|https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
  "alpaca_zh|alpaca_zh.json|https://hf-mirror.com/datasets/silk-road/alpaca-data-gpt4-chinese/resolve/main/Chinese_alpaca_data.json|https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json|https://hf-mirror.com/datasets/c-s-ale/alpaca-gpt4-data-zh/resolve/main/alpaca_gpt4_data_zh.json"
  "lima|lima.json|https://hf-mirror.com/datasets/GAIR/lima/resolve/main/train.jsonl|https://hf-mirror.com/datasets/64bits/lima_vicuna_format/resolve/main/lima_vicuna.json|https://raw.githubusercontent.com/64bits/lima/main/train.jsonl"
  "math_qa|math_qa.json|https://hf-mirror.com/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json|https://hf-mirror.com/datasets/TIGER-Lab/MathInstruct/resolve/main/MathInstruct.json|https://hf-mirror.com/datasets/microsoft/orca-math-word-problems-200k/resolve/main/data/train-00000-of-00001.parquet"
)

mkdir -p "$OUT_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

# Validate JSON structurally if jq available, else just ensure non-empty.
verify_file() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local size; size=$(wc -c < "$path" 2>/dev/null || echo 0)
  if (( size < MIN_BYTES )); then
    warn "$path too small ($size B < $MIN_BYTES B)"
    return 1
  fi
  if command -v jq >/dev/null 2>&1; then
    # Try array first, then JSONL line-by-line.
    if ! head -c 65536 "$path" | jq -e . >/dev/null 2>&1; then
      if ! head -n 5 "$path" | jq -e . >/dev/null 2>&1; then
        warn "$path failed jq parse"
        return 1
      fi
    fi
  fi
  return 0
}

fetch_one() {
  local name="$1" dst="$2"; shift 2
  local urls=("$@")
  local out="$OUT_DIR/$dst"

  if verify_file "$out"; then
    log "$name: already present ($(wc -c < "$out") B), skip"
    return 0
  fi

  for url in "${urls[@]}"; do
    log "$name: trying $url"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$url" 2>/dev/null; then
      mv -f "$out.tmp" "$out"
      if verify_file "$out"; then
        log "$name: OK ($(wc -c < "$out") B from $url)"
        return 0
      fi
    fi
    rm -f "$out.tmp"
    warn "$name: failed $url"
  done
  warn "$name: all mirrors failed; leaving $out absent"
  return 1
}

ok=0; fail=0
for entry in "${MIRRORS[@]}"; do
  IFS='|' read -r name dst u1 u2 u3 <<<"$entry"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then continue; fi
  if fetch_one "$name" "$dst" "$u1" "$u2" "$u3"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
  fi
done

log "done: $ok ok, $fail fail; out=$OUT_DIR"
[[ "$fail" -eq 0 ]]
