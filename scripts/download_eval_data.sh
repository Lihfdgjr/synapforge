#!/usr/bin/env bash
# download_eval_data.sh — fetch eval datasets for synapforge/bench/*.
#
# HumanEval, MBPP, MMLU, GSM8K, HellaSwag, LAMBADA — totals ~100MB.
# These bench scripts already fall back to `datasets.load_dataset` but on the
# rental we want the data co-located so eval doesn't burn HF API per run.
#
# Usage:
#   bash scripts/download_eval_data.sh                  # all 6
#   bash scripts/download_eval_data.sh --only mmlu      # one only
#   bash scripts/download_eval_data.sh --help

set -u
set -o pipefail

OUT="${OUT:-/workspace/data/eval}"
TIMEOUT="${TIMEOUT:-180}"
ONLY=""

usage() {
  sed -n '2,11p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --only)    ONLY="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

# Fetcher: try mirrors in order; first one yielding >MIN bytes wins.
fetch() {
  local name="$1" dst="$2" min="$3"; shift 3
  local urls=("$@")
  local sub="$OUT/$name"; mkdir -p "$sub"
  local out="$sub/$dst"
  if [[ -f "$out" && $(wc -c < "$out") -gt "$min" ]]; then
    log "$name: $out already present ($(wc -c < "$out") B), skip"
    return 0
  fi
  for u in "${urls[@]}"; do
    log "$name: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$u" 2>/dev/null; then
      mv "$out.tmp" "$out"
      local size; size=$(wc -c < "$out")
      if (( size > min )); then
        log "$name: OK ${size} B from $u"
        return 0
      fi
    fi
    rm -f "$out.tmp"
    warn "$name: failed $u"
  done
  warn "$name: all mirrors failed for $dst"
  return 1
}

ok=0; fail=0

run_one() {
  local name="$1"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  if "$@"; then ok=$((ok+1)); else fail=$((fail+1)); fi
}

# ------------------------------------------------------------------ HumanEval
do_humaneval() {
  fetch humaneval HumanEval.jsonl.gz 100000 \
    "https://hf-mirror.com/datasets/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/openai/openai_humaneval/resolve/main/data/test-00000-of-00001.parquet" \
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
}

# ------------------------------------------------------------------------ MBPP
do_mbpp() {
  fetch mbpp mbpp.jsonl 1000000 \
    "https://hf-mirror.com/datasets/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/google/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet" \
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
}

# ------------------------------------------------------------------------ MMLU
do_mmlu() {
  # MMLU is 57 subjects; HF mirror ships a single combined parquet.
  fetch mmlu mmlu_test.parquet 5000000 \
    "https://hf-mirror.com/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/lukaemon/mmlu/resolve/main/test/test.parquet" \
    "https://hf-mirror.com/datasets/hails/mmlu_no_train/resolve/main/all/test-00000-of-00001.parquet"
  # also grab dev (5-shot examples)
  fetch mmlu mmlu_dev.parquet 100000 \
    "https://hf-mirror.com/datasets/cais/mmlu/resolve/main/all/dev-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/lukaemon/mmlu/resolve/main/dev/dev.parquet" \
    "https://hf-mirror.com/datasets/hails/mmlu_no_train/resolve/main/all/dev-00000-of-00001.parquet" || true
}

# ------------------------------------------------------------------------ GSM8K
do_gsm8k() {
  fetch gsm8k gsm8k_test.jsonl 500000 \
    "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet" \
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
}

# ----------------------------------------------------------------- HellaSwag
do_hellaswag() {
  fetch hellaswag hellaswag_val.parquet 5000000 \
    "https://hf-mirror.com/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/hellaswag/resolve/main/data/validation-00000-of-00001.parquet" \
    "https://github.com/rowanz/hellaswag/raw/master/data/hellaswag_val.jsonl"
}

# ------------------------------------------------------------------- LAMBADA
do_lambada() {
  fetch lambada lambada_test.jsonl 1000000 \
    "https://hf-mirror.com/datasets/EleutherAI/lambada_openai/resolve/main/data/lambada_test.jsonl" \
    "https://hf-mirror.com/datasets/lambada/resolve/main/plain_text/test-00000-of-00001.parquet" \
    "https://hf-mirror.com/datasets/cimec/lambada/resolve/main/data/test.jsonl"
}

run_one humaneval do_humaneval
run_one mbpp      do_mbpp
run_one mmlu      do_mmlu
run_one gsm8k     do_gsm8k
run_one hellaswag do_hellaswag
run_one lambada   do_lambada

# Summary manifest.
SUMMARY="$OUT/SUMMARY.txt"
{
  echo "synapforge eval data summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  for d in humaneval mbpp mmlu gsm8k hellaswag lambada; do
    if [[ -d "$OUT/$d" ]]; then
      sz=$(du -sb "$OUT/$d" 2>/dev/null | awk '{print $1}')
      printf "  %-12s %10s bytes  %s\n" "$d" "$sz" "$OUT/$d"
    else
      printf "  %-12s %10s  %s\n" "$d" "MISSING" "$OUT/$d"
    fi
  done
} | tee "$SUMMARY"

log "done: $ok ok, $fail fail; out=$OUT"
[[ "$fail" -eq 0 ]]
