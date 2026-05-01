#!/usr/bin/env bash
# download_eval_extended.sh — fetch extended eval datasets for synapforge bench.
#
# Eight datasets covering: zh exam (cmmlu, ceval, agieval-zh), competition math
# (math500), truthfulness (truthfulqa), translation (wmt23 en<->zh), long ctx
# (niah haystacks), computer-use (osworld), web agent (webarena).
#
# Designed to be idempotent + cache-friendly: re-runs skip files >MIN bytes.
# Each dataset has 3+ mirror URLs (HF mirror -> HF main -> GitHub raw).
#
# Output: $OUT/<name>/{...}.json,.parquet,.jsonl
# Default OUT: /workspace/data/eval (override with --out / OUT env).
#
# Usage:
#   bash scripts/download_eval_extended.sh                    # all 8
#   bash scripts/download_eval_extended.sh --only math500     # one only
#   bash scripts/download_eval_extended.sh --size smoke       # tiny shards
#   bash scripts/download_eval_extended.sh --size full        # everything
#   bash scripts/download_eval_extended.sh --help

set -u
set -o pipefail

OUT="${OUT:-/workspace/data/eval}"
TIMEOUT="${TIMEOUT:-240}"
ONLY=""
SIZE="full"

usage() {
  sed -n '2,22p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --only)    ONLY="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --size)    SIZE="$2"; shift 2 ;;
    --smoke)   SIZE="smoke"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$SIZE" in
  smoke|full) : ;;
  *) echo "--size must be smoke|full (got $SIZE)" >&2; exit 2 ;;
esac

mkdir -p "$OUT"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

# Optional jq sanity check (some mirrors return HTML on failure).
have_jq=0
if command -v jq >/dev/null 2>&1; then have_jq=1; fi

# Smoke / full size table (bytes). Smoke = scrape/test split only.
min_bytes() {
  local name="$1"
  if [[ "$SIZE" == "smoke" ]]; then
    case "$name" in
      cmmlu)      echo 50000 ;;
      ceval)      echo 50000 ;;
      agieval_zh) echo 50000 ;;
      math500)    echo 30000 ;;
      truthfulqa) echo 50000 ;;
      wmt23)      echo 100000 ;;
      niah)       echo 30000 ;;
      osworld)    echo 30000 ;;
      webarena)   echo 30000 ;;
      *)          echo 10000 ;;
    esac
  else
    case "$name" in
      cmmlu)      echo 5000000 ;;
      ceval)      echo 1000000 ;;
      agieval_zh) echo 500000 ;;
      math500)    echo 200000 ;;
      truthfulqa) echo 200000 ;;
      wmt23)      echo 5000000 ;;
      niah)       echo 100000 ;;
      osworld)    echo 200000 ;;
      webarena)   echo 100000 ;;
      *)          echo 100000 ;;
    esac
  fi
}

# fetch <name> <dst-filename> <min-bytes> <url1> [url2 ...]
fetch() {
  local name="$1" dst="$2" min="$3"; shift 3
  local urls=("$@")
  local sub="$OUT/$name"; mkdir -p "$sub"
  local out="$sub/$dst"
  if [[ -f "$out" && $(wc -c < "$out") -gt "$min" ]]; then
    log "$name: $out present ($(wc -c < "$out") B), skip"
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

# Validate JSON/JSONL/parquet shapes if jq is available.
validate_json() {
  local f="$1"
  [[ "$have_jq" == 1 && -f "$f" && "$f" == *.json ]] || return 0
  if ! jq -e '.' "$f" >/dev/null 2>&1; then
    warn "validate_json: $f is malformed"
    return 1
  fi
  return 0
}

ok=0; fail=0

run_one() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  if "$@"; then ok=$((ok+1)); else fail=$((fail+1)); fi
}

# --------------------------------------------------------------------- CMMLU
do_cmmlu() {
  # CMMLU is 67 subjects; HF mirror has them as parquet; we grab the test split.
  fetch cmmlu cmmlu_test.parquet "$(min_bytes cmmlu)" \
    "https://hf-mirror.com/datasets/haonan-li/cmmlu/resolve/main/test/anatomy.parquet" \
    "https://hf-mirror.com/datasets/haonan-li/cmmlu/resolve/main/data/cmmlu.zip" \
    "https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/test/anatomy.parquet"
}

# --------------------------------------------------------------------- C-Eval
do_ceval() {
  fetch ceval ceval_val.parquet "$(min_bytes ceval)" \
    "https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/val/computer_network.parquet" \
    "https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/data/ceval-exam.zip" \
    "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/val/computer_network.parquet"
}

# ---------------------------------------------------------------- AGIEval ZH
do_agieval_zh() {
  fetch agieval_zh gaokao_chinese.jsonl "$(min_bytes agieval_zh)" \
    "https://hf-mirror.com/datasets/microsoft/AGIEval/resolve/main/v1/gaokao-chinese.jsonl" \
    "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/gaokao-chinese.jsonl" \
    "https://hf-mirror.com/datasets/hails/agieval/resolve/main/gaokao-chinese.jsonl"
}

# -------------------------------------------------------------------- MATH-500
do_math500() {
  # MATH-500 = OpenAI's reduced 500-problem competition-math benchmark
  # (subset of full Hendrycks MATH).
  fetch math500 math500.jsonl "$(min_bytes math500)" \
    "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl" \
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl" \
    "https://raw.githubusercontent.com/openai/prm800k/main/prm800k/math_splits/test.jsonl"
}

# ---------------------------------------------------------------- TruthfulQA
do_truthfulqa() {
  fetch truthfulqa truthfulqa_mc.jsonl "$(min_bytes truthfulqa)" \
    "https://hf-mirror.com/datasets/truthful_qa/resolve/main/multiple_choice/validation-00000-of-00001.parquet" \
    "https://huggingface.co/datasets/truthful_qa/resolve/main/multiple_choice/validation-00000-of-00001.parquet" \
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
  fetch truthfulqa truthfulqa_gen.jsonl "$(min_bytes truthfulqa)" \
    "https://hf-mirror.com/datasets/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet" \
    "https://huggingface.co/datasets/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet" \
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/finetune_truth.jsonl" || true
}

# -------------------------------------------------------------------- WMT-23
do_wmt23() {
  # General WMT-23 en<->zh test set (newstest2023).
  fetch wmt23 wmt23_en_zh.tsv "$(min_bytes wmt23)" \
    "https://hf-mirror.com/datasets/haoranxu/WMT23-Test/resolve/main/en-zh/test.tsv" \
    "https://raw.githubusercontent.com/wmt-conference/wmt23-news-systems/master/txt/sources/newstest2023.en-zh.src.en" \
    "https://hf-mirror.com/datasets/wmt/wmt23/resolve/main/en-zh/test.parquet"
  fetch wmt23 wmt23_zh_en.tsv "$(min_bytes wmt23)" \
    "https://hf-mirror.com/datasets/haoranxu/WMT23-Test/resolve/main/zh-en/test.tsv" \
    "https://raw.githubusercontent.com/wmt-conference/wmt23-news-systems/master/txt/sources/newstest2023.zh-en.src.zh" \
    "https://hf-mirror.com/datasets/wmt/wmt23/resolve/main/zh-en/test.parquet" || true
}

# --------------------------------------------------------------------- NIAH
do_niah() {
  # Needle in a Haystack: original is a Greg Kamradt blog rig; HF mirrors
  # the curated "ruler" + "long-context-arena" question packs.
  fetch niah niah_paul_graham.txt "$(min_bytes niah)" \
    "https://hf-mirror.com/datasets/gkamradt/needle-in-haystack/resolve/main/PaulGrahamEssays.txt" \
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/needlehaystack/PaulGrahamEssays/articles_used.txt" \
    "https://hf-mirror.com/datasets/RMT-team/babilong/resolve/main/data/needles.json"
  fetch niah niah_questions.json "$(min_bytes niah)" \
    "https://hf-mirror.com/datasets/RMT-team/babilong/resolve/main/data/qa1.json" \
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/needlehaystack/needle_questions.json" || true
}

# -------------------------------------------------------------------- OSWorld
do_osworld() {
  # OSWorld (Xie et al. 2024, arxiv 2404.07972) — 369 real-OS tasks.
  fetch osworld osworld_tasks.json "$(min_bytes osworld)" \
    "https://raw.githubusercontent.com/xlang-ai/OSWorld/main/evaluation_examples/test_all.json" \
    "https://hf-mirror.com/datasets/xlangai/osworld/resolve/main/test_all.json" \
    "https://huggingface.co/datasets/xlangai/osworld/resolve/main/test_all.json"
  fetch osworld osworld_apps.json "$(min_bytes osworld)" \
    "https://raw.githubusercontent.com/xlang-ai/OSWorld/main/evaluation_examples/test_small.json" \
    "https://hf-mirror.com/datasets/xlangai/osworld/resolve/main/test_small.json" || true
}

# ------------------------------------------------------------------- WebArena
do_webarena() {
  # WebArena (Zhou et al. 2023, arxiv 2307.13854) — 800 web tasks.
  fetch webarena webarena_test.json "$(min_bytes webarena)" \
    "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json" \
    "https://hf-mirror.com/datasets/webarena/webarena/resolve/main/test.raw.json" \
    "https://huggingface.co/datasets/webarena/webarena/resolve/main/test.raw.json"
}

run_one cmmlu      do_cmmlu
run_one ceval      do_ceval
run_one agieval_zh do_agieval_zh
run_one math500    do_math500
run_one truthfulqa do_truthfulqa
run_one wmt23      do_wmt23
run_one niah       do_niah
run_one osworld    do_osworld
run_one webarena   do_webarena

# Validate any *.json files we produced.
if [[ "$have_jq" == 1 ]]; then
  for f in "$OUT"/*/*.json; do
    [[ -f "$f" ]] || continue
    validate_json "$f" || true
  done
fi

# Summary manifest.
SUMMARY="$OUT/SUMMARY_EXTENDED.txt"
{
  echo "synapforge eval-extended summary $(date -u +%Y-%m-%dT%H:%M:%SZ) size=$SIZE"
  echo
  for d in cmmlu ceval agieval_zh math500 truthfulqa wmt23 niah osworld webarena; do
    if [[ -d "$OUT/$d" ]]; then
      sz=$(du -sb "$OUT/$d" 2>/dev/null | awk '{print $1}')
      printf "  %-12s %12s bytes  %s\n" "$d" "${sz:-0}" "$OUT/$d"
    else
      printf "  %-12s %12s         %s\n" "$d" "MISSING" "$OUT/$d"
    fi
  done
} | tee "$SUMMARY"

log "done: $ok ok, $fail fail; out=$OUT size=$SIZE"
[[ "$fail" -eq 0 ]]
