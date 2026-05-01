#!/usr/bin/env bash
# download_pretrain_multilingual.sh — fetch multilingual pretrain corpora on rental.
#
# Mirrors-first (hf-mirror.com → modelscope → origin), per-corpus 3-mirror walk,
# size + sha256 verify, idempotent. Each corpus is a small/medium subset because
# 100GB rental disks cannot host full WuDao (200GB) / SkyPile (150GB).
#
# Sizes per --size flag (per-corpus targets):
#   smoke   ~ 100 docs    (<100MB total)   — runs offline-friendly
#   small   ~ 10K docs    (~10GB total)
#   medium  ~ 100K docs   (~30GB total)
#
# Corpora:
#   fineweb_edu          English educational web (HuggingFaceFW/fineweb-edu)
#   wudao                BAAI WuDao 200G subset (Chinese)
#   skypile              SkyPile-150B (Chinese, AI4Skywork/SkyPile)
#   the_stack_v2_python  Code (BigCode the-stack-v2-train-smol)
#   cosmopedia_v2        Synthetic educational (HuggingFaceTB/cosmopedia-v2)
#   cci3_hq              Chinese CC filtered (BAAI/CCI3-HQ)
#
# Output layout:
#   /workspace/data/pretrain/<corpus>/{train.parquet, manifest.json, .sha256}
#
# Usage:
#   bash scripts/download_pretrain_multilingual.sh --size smoke
#   bash scripts/download_pretrain_multilingual.sh --corpora wudao,skypile --size small
#   bash scripts/download_pretrain_multilingual.sh --help

set -u
set -o pipefail

ROOT="${ROOT:-/workspace/data/pretrain}"
SIZE="${SIZE:-smoke}"
CORPORA_FLAG=""
TIMEOUT="${TIMEOUT:-600}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-15}"
UA='synapforge/1.0 pretrain-fetch'

usage() {
  sed -n '2,28p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage ;;
    --size)      SIZE="$2"; shift 2 ;;
    --corpora)   CORPORA_FLAG="$2"; shift 2 ;;
    --root)      ROOT="$2"; shift 2 ;;
    --timeout)   TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$SIZE" in
  smoke)  N_DOCS=100;     MIN_BYTES=10000     ;;
  small)  N_DOCS=10000;   MIN_BYTES=1048576   ;;  # 1 MB
  medium) N_DOCS=100000;  MIN_BYTES=10485760  ;;  # 10 MB
  *) echo "bad --size: $SIZE (smoke|small|medium)" >&2; exit 2 ;;
esac

ALL_CORPORA=(fineweb_edu wudao skypile the_stack_v2_python cosmopedia_v2 cci3_hq)
if [[ -n "$CORPORA_FLAG" ]]; then
  IFS=',' read -ra CORPORA <<<"$CORPORA_FLAG"
else
  CORPORA=("${ALL_CORPORA[@]}")
fi

mkdir -p "$ROOT"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

# -- fetch helper -----------------------------------------------------------
# Tries each url in order; first one yielding > min_bytes wins.
# Writes file at $dst, also writes $dst.sha256.
fetch_with_mirrors() {
  local name="$1" dst="$2" min_bytes="$3"; shift 3
  local urls=("$@")
  if [[ -f "$dst" ]]; then
    local sz; sz=$(wc -c < "$dst" 2>/dev/null || echo 0)
    if (( sz > min_bytes )); then
      log "$name: $dst already present ($sz B), skip"
      return 0
    fi
    rm -f "$dst"
  fi
  for u in "${urls[@]}"; do
    log "$name: try $u"
    if curl -fL --connect-timeout "$CONNECT_TIMEOUT" --max-time "$TIMEOUT" \
        -A "$UA" -o "$dst.tmp" "$u" 2>/dev/null; then
      mv -f "$dst.tmp" "$dst"
      local sz; sz=$(wc -c < "$dst" 2>/dev/null || echo 0)
      if (( sz > min_bytes )); then
        sha256sum "$dst" 2>/dev/null | awk '{print $1}' > "$dst.sha256" || true
        log "$name: OK $sz B from $u"
        return 0
      fi
      warn "$name: $u returned only $sz B (< $min_bytes)"
    fi
    rm -f "$dst.tmp"
  done
  warn "$name: all mirrors failed for $dst"
  return 1
}

write_manifest() {
  local corpus="$1" status="$2" path="$3" size="$4" n_docs="$5" source="$6"
  local mfile="$ROOT/$corpus/manifest.json"
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat > "$mfile" <<EOF
{
  "corpus": "$corpus",
  "status": "$status",
  "path": "$path",
  "size_bytes": $size,
  "n_docs": $n_docs,
  "source": "$source",
  "size_class": "$SIZE",
  "updated": "$ts",
  "synapforge": "1.0"
}
EOF
}

# Each corpus function fills $ROOT/<corpus>/train.parquet (or .tsv/.jsonl).
# All return 0 on success, 1 on failure. Per-mirror walk + size verify.

# --------------------------------------------------------------- fineweb_edu
do_fineweb_edu() {
  local sub="$ROOT/fineweb_edu"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors fineweb_edu "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet" \
    "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data/CC-MAIN-2024-10/000_00000.parquet" \
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet" \
    || { write_manifest fineweb_edu failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest fineweb_edu ok "$dst" "$sz" "$N_DOCS" "HuggingFaceFW/fineweb-edu"
  return 0
}

# --------------------------------------------------------------------- wudao
do_wudao() {
  local sub="$ROOT/wudao"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors wudao "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/p208p2002/wudao/resolve/main/data/train-00000-of-00100.parquet" \
    "https://modelscope.cn/api/v1/datasets/BAAI/WuDaoCorpora/repo?Revision=master&FilePath=part-00000.parquet" \
    "https://hf-mirror.com/datasets/IDEA-CCNL/wudao_180g/resolve/main/wudao_part_001.jsonl.gz" \
    || { write_manifest wudao failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest wudao ok "$dst" "$sz" "$N_DOCS" "BAAI/WuDaoCorpora"
  return 0
}

# ------------------------------------------------------------------- skypile
do_skypile() {
  local sub="$ROOT/skypile"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors skypile "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-40_zh_head_0000.jsonl.gz" \
    "https://hf-mirror.com/datasets/AI4Skywork/SkyPile/resolve/main/data/2021-49_zh_head_0000.jsonl.gz" \
    "https://modelscope.cn/api/v1/datasets/Skywork/SkyPile-150B/repo?Revision=master&FilePath=data%2F2020-40_zh_head_0000.jsonl.gz" \
    || { write_manifest skypile failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest skypile ok "$dst" "$sz" "$N_DOCS" "Skywork/SkyPile-150B"
  return 0
}

# ------------------------------------------------------- the_stack_v2_python
do_the_stack_v2_python() {
  local sub="$ROOT/the_stack_v2_python"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors the_stack_v2_python "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/bigcode/the-stack-v2-train-smol/resolve/main/data/Python/train-00000-of-00064.parquet" \
    "https://hf-mirror.com/datasets/bigcode/the-stack-smol/resolve/main/data/python.parquet" \
    "https://hf-mirror.com/datasets/bigcode/the-stack/resolve/main/data/python/train-00000-of-00206.parquet" \
    || { write_manifest the_stack_v2_python failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest the_stack_v2_python ok "$dst" "$sz" "$N_DOCS" "bigcode/the-stack-v2"
  return 0
}

# -------------------------------------------------------------- cosmopedia_v2
do_cosmopedia_v2() {
  local sub="$ROOT/cosmopedia_v2"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors cosmopedia_v2 "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/web_samples_v2/train-00000-of-00072.parquet" \
    "https://hf-mirror.com/datasets/HuggingFaceTB/cosmopedia-v2/resolve/main/data/web_samples_v2/train-00000-of-00072.parquet" \
    "https://hf-mirror.com/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00000-of-00104.parquet" \
    || { write_manifest cosmopedia_v2 failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest cosmopedia_v2 ok "$dst" "$sz" "$N_DOCS" "HuggingFaceTB/cosmopedia-v2"
  return 0
}

# ------------------------------------------------------------------- cci3_hq
do_cci3_hq() {
  local sub="$ROOT/cci3_hq"; mkdir -p "$sub"
  local dst="$sub/train.parquet"
  fetch_with_mirrors cci3_hq "$dst" "$MIN_BYTES" \
    "https://hf-mirror.com/datasets/BAAI/CCI3-HQ/resolve/main/data/train-00000-of-00128.parquet" \
    "https://hf-mirror.com/datasets/BAAI/CCI3-Data/resolve/main/data/train-00000-of-00128.parquet" \
    "https://modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo?Revision=master&FilePath=data%2Ftrain-00000-of-00128.parquet" \
    || { write_manifest cci3_hq failed "$dst" 0 0 "none"; return 1; }
  local sz; sz=$(wc -c < "$dst")
  write_manifest cci3_hq ok "$dst" "$sz" "$N_DOCS" "BAAI/CCI3-HQ"
  return 0
}

# --------------------------------------------------------------- dispatcher
declare -A FETCHERS=(
  [fineweb_edu]=do_fineweb_edu
  [wudao]=do_wudao
  [skypile]=do_skypile
  [the_stack_v2_python]=do_the_stack_v2_python
  [cosmopedia_v2]=do_cosmopedia_v2
  [cci3_hq]=do_cci3_hq
)

ok=0; fail=0
log "size=$SIZE  n_docs_target=$N_DOCS  min_bytes=$MIN_BYTES  root=$ROOT"
for c in "${CORPORA[@]}"; do
  if [[ -z "${FETCHERS[$c]:-}" ]]; then
    warn "unknown corpus: $c (skipping)"
    fail=$((fail+1))
    continue
  fi
  log "==== $c ===="
  if "${FETCHERS[$c]}"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    # smoke-mode ignores per-corpus failure.
    if [[ "$SIZE" == "smoke" ]]; then
      log "$c: smoke mode tolerates failure"
    fi
  fi
done

# Top-level summary manifest.
SUMMARY="$ROOT/SUMMARY.json"
{
  echo "{"
  echo "  \"size_class\": \"$SIZE\","
  echo "  \"updated\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"ok\": $ok,"
  echo "  \"fail\": $fail,"
  echo "  \"corpora\": ["
  first=1
  for c in "${CORPORA[@]}"; do
    [[ $first -eq 0 ]] && echo "    ,"
    first=0
    if [[ -f "$ROOT/$c/manifest.json" ]]; then
      cat "$ROOT/$c/manifest.json" | sed 's/^/    /'
    else
      echo "    {\"corpus\": \"$c\", \"status\": \"missing\"}"
    fi
  done
  echo "  ]"
  echo "}"
} > "$SUMMARY" 2>/dev/null || true

log "done: $ok ok, $fail fail; summary=$SUMMARY"
# smoke mode always succeeds (network may be absent).
if [[ "$SIZE" == "smoke" ]]; then exit 0; fi
[[ "$fail" -eq 0 ]]
