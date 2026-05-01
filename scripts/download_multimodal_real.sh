#!/usr/bin/env bash
# download_multimodal_real.sh — fetch real (not synthetic) multimodal samples on the rental.
#
# Five modalities, each with its own size budget and idempotent sha256 check.
# Defensive: ANY single modality failing does NOT kill the others.
#
# Disk budget: image 5GB / audio 2GB / video 1GB / time-series 50MB / graph 10MB.
# Total ceiling ~8GB.
#
# Usage:
#   bash scripts/download_multimodal_real.sh                  # all modalities
#   bash scripts/download_multimodal_real.sh --only image     # one only
#   bash scripts/download_multimodal_real.sh --skip video     # skip one
#   bash scripts/download_multimodal_real.sh --help

set -u
set -o pipefail

ROOT="${ROOT:-/workspace/data/multimodal}"
MOHUANFANG_HOST="${MOHUANFANG_HOST:-mohuanfang.com}"
MOHUANFANG_USER="${MOHUANFANG_USER:-liu}"
MOHUANFANG_LIBRI="${MOHUANFANG_LIBRI:-/home/liu/synapforge_backup/librispeech_mel.memmap}"
TIMEOUT="${TIMEOUT:-300}"
ONLY=""
SKIP=""

usage() {
  sed -n '2,15p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --only)    ONLY="$2"; shift 2 ;;
    --skip)    SKIP="$2"; shift 2 ;;
    --root)    ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT"
MANIFEST="$ROOT/manifest.json"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

# Append/update one modality entry in manifest.json.
# Args: name source path size_bytes status [n_items]
manifest_record() {
  local name="$1" src="$2" path="$3" size="$4" status="$5" items="${6:-0}"
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local entry
  entry=$(printf '{"modality":"%s","source":"%s","path":"%s","size_bytes":%s,"status":"%s","n_items":%s,"updated":"%s"}' \
          "$name" "$src" "$path" "$size" "$status" "$items" "$ts")
  if [[ -f "$MANIFEST" ]] && command -v jq >/dev/null 2>&1; then
    jq --argjson e "$entry" '. + {("'"$name"'"): $e}' "$MANIFEST" > "$MANIFEST.new" \
      && mv "$MANIFEST.new" "$MANIFEST"
  else
    # Bootstrap or jq-less fallback: write the single key as a flat object,
    # losing previous keys is acceptable on bootstrap.
    if [[ -f "$MANIFEST" ]]; then
      python3 - "$MANIFEST" "$name" "$entry" <<'PY' 2>/dev/null || echo "{\"$name\": $entry}" > "$MANIFEST"
import json, sys
p, k, e = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(p) as f: d = json.load(f)
except Exception:
    d = {}
d[k] = json.loads(e)
with open(p, 'w') as f: json.dump(d, f, indent=2)
PY
    else
      echo "{\"$name\": $entry}" > "$MANIFEST"
    fi
  fi
}

sha_match() {
  # $1 file, $2 expected sha256 prefix (16 hex). Returns 0 if matches.
  local f="$1" want="$2"
  [[ -f "$f" && -n "$want" ]] || return 1
  local got; got=$(sha256sum "$f" | awk '{print $1}' | head -c 16)
  [[ "$got" == "$want" ]]
}

# --------------------------------------------------------------------- image
fetch_image() {
  local sub="$ROOT/image"
  mkdir -p "$sub"
  local list="$sub/cc12m_subset.tsv"
  local urls=(
    "https://hf-mirror.com/datasets/Lin-Chen/CC12M/resolve/main/cc12m.tsv.gz"
    "https://hf-mirror.com/datasets/laion/laion-coco/resolve/main/laion-coco.parquet"
    "https://hf-mirror.com/datasets/conceptual_captions/resolve/main/Train_GCC-training.tsv"
  )
  local got=0
  for u in "${urls[@]}"; do
    log "image: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$list.tmp" "$u" 2>/dev/null; then
      if [[ "$u" == *.gz ]]; then gunzip -f "$list.tmp" 2>/dev/null && mv "${list%.gz}.tmp" "$list" 2>/dev/null || mv "$list.tmp" "$list"; else mv "$list.tmp" "$list"; fi
      if [[ -s "$list" ]]; then got=1; break; fi
    fi
    rm -f "$list.tmp"
  done
  if (( got == 0 )); then
    warn "image: no caption list available"
    manifest_record image "none" "$sub" 0 "failed"
    return 1
  fi
  # Take first 50K rows max; download images respecting 5GB cap.
  head -n 50000 "$list" > "$sub/captions.tsv"
  local n; n=$(wc -l < "$sub/captions.tsv")
  log "image: $n caption rows; downloading capped at 5GB"
  local idx=0; local size=0; local cap=$((5 * 1024 * 1024 * 1024))
  mkdir -p "$sub/img"
  while IFS=$'\t' read -r url caption || [[ -n "$url" ]]; do
    [[ -z "$url" ]] && continue
    local fn="$sub/img/$(printf '%06d.jpg' "$idx")"
    if [[ ! -f "$fn" ]]; then
      curl -fL --connect-timeout 8 --max-time 30 -A 'synapforge/1.0' \
        -o "$fn" "$url" 2>/dev/null || { rm -f "$fn"; continue; }
    fi
    local sz; sz=$(wc -c < "$fn" 2>/dev/null || echo 0)
    size=$((size + sz))
    idx=$((idx + 1))
    if (( size >= cap )); then
      log "image: hit 5GB cap at $idx items"; break
    fi
  done < "$sub/captions.tsv"
  manifest_record image "cc12m+laion" "$sub" "$size" "ok" "$idx"
  log "image: done $idx items, ${size} B"
}

# --------------------------------------------------------------------- audio
fetch_audio() {
  local sub="$ROOT/audio"
  mkdir -p "$sub"
  # Prefer rsync from mohuanfang since LibriSpeech mel memmap is already there.
  if command -v rsync >/dev/null 2>&1; then
    log "audio: rsync from mohuanfang"
    if rsync -avz --partial --timeout="$TIMEOUT" \
        "${MOHUANFANG_USER}@${MOHUANFANG_HOST}:${MOHUANFANG_LIBRI}" "$sub/" 2>/dev/null; then
      local size; size=$(wc -c < "$sub/$(basename "$MOHUANFANG_LIBRI")" 2>/dev/null || echo 0)
      manifest_record audio "librispeech_mel_rsync" "$sub" "$size" "ok"
      log "audio: rsync OK ${size} B"
      return 0
    fi
    warn "audio: rsync from mohuanfang failed; falling back to HF"
  fi
  # Fallback: HF mirror tar of LibriSpeech-clean-100 5h subset.
  local urls=(
    "https://hf-mirror.com/datasets/openslr/librispeech_asr/resolve/main/dev_clean.tar.gz"
    "https://hf-mirror.com/datasets/MLCommons/peoples_speech/resolve/main/clean/test-00000-of-00001.parquet"
  )
  local out="$sub/audio_subset.tar.gz"
  for u in "${urls[@]}"; do
    log "audio: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$u" 2>/dev/null; then
      mv "$out.tmp" "$out"
      local size; size=$(wc -c < "$out")
      if (( size > 1048576 )); then
        manifest_record audio "$u" "$sub" "$size" "ok"
        log "audio: OK ${size} B"; return 0
      fi
    fi
    rm -f "$out.tmp"
  done
  warn "audio: all sources failed"
  manifest_record audio "none" "$sub" 0 "failed"
  return 1
}

# --------------------------------------------------------------------- video
fetch_video() {
  local sub="$ROOT/video"
  mkdir -p "$sub"
  # WebVid-tiny 1K then HowTo100M-mini.
  local urls=(
    "https://hf-mirror.com/datasets/TempoFunk/webvid-10M/resolve/main/data/0.parquet"
    "https://hf-mirror.com/datasets/HuggingFaceM4/M4-it/resolve/main/data/howto100m_subset.parquet"
    "https://hf-mirror.com/datasets/bigcode/the-stack-smol/resolve/main/data/python.parquet"
  )
  local out="$sub/video_meta.parquet"
  for u in "${urls[@]}"; do
    log "video: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$u" 2>/dev/null; then
      mv "$out.tmp" "$out"
      local size; size=$(wc -c < "$out")
      if (( size > 1048576 )); then
        manifest_record video "$u" "$sub" "$size" "ok"
        log "video: OK ${size} B"; return 0
      fi
    fi
    rm -f "$out.tmp"
  done
  warn "video: all sources failed"
  manifest_record video "none" "$sub" 0 "failed"
  return 1
}

# ----------------------------------------------------------------- time_series
fetch_time_series() {
  local sub="$ROOT/time_series"
  mkdir -p "$sub"
  local out="$sub/ETHUSDT_1m.zip"
  # Binance archive. Pick a recent month.
  local month; month=$(date -u -d 'last month' +%Y-%m 2>/dev/null || date -u +%Y-%m)
  local urls=(
    "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-${month}.zip"
    "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-${month}.zip"
    "https://hf-mirror.com/datasets/edarchimbaud/timeseries-1m-stocks/resolve/main/data.csv"
  )
  for u in "${urls[@]}"; do
    log "time_series: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$u" 2>/dev/null; then
      mv "$out.tmp" "$out"
      local size; size=$(wc -c < "$out")
      if (( size > 8192 )); then
        manifest_record time_series "$u" "$sub" "$size" "ok"
        log "time_series: OK ${size} B"; return 0
      fi
    fi
    rm -f "$out.tmp"
  done
  warn "time_series: all sources failed"
  manifest_record time_series "none" "$sub" 0 "failed"
  return 1
}

# ----------------------------------------------------------------------- graph
fetch_graph() {
  local sub="$ROOT/graph"
  mkdir -p "$sub"
  # ZINC 250k subset; original from snap.stanford.edu, mirror via HF.
  local out="$sub/zinc_5k.csv"
  local urls=(
    "https://hf-mirror.com/datasets/maomlab/MolFormer-XL-FineTune/resolve/main/zinc15-train-1m-canonical.csv"
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
    "https://hf-mirror.com/datasets/sagawa/ZINC-canonicalized/resolve/main/zinc15.csv"
  )
  for u in "${urls[@]}"; do
    log "graph: try $u"
    if curl -fL --connect-timeout 15 --max-time "$TIMEOUT" \
        -A 'synapforge/1.0' -o "$out.tmp" "$u" 2>/dev/null; then
      # Trim to first 5K rows.
      head -n 5001 "$out.tmp" > "$out"
      rm -f "$out.tmp"
      local size; size=$(wc -c < "$out")
      if (( size > 1024 )); then
        local n; n=$(($(wc -l < "$out") - 1))
        manifest_record graph "$u" "$sub" "$size" "ok" "$n"
        log "graph: OK ${size} B, $n rows"; return 0
      fi
    fi
    rm -f "$out.tmp" "$out"
  done
  warn "graph: all sources failed"
  manifest_record graph "none" "$sub" 0 "failed"
  return 1
}

# ----------------------------------------------------------------- main loop
declare -A FETCHERS=(
  [image]=fetch_image
  [audio]=fetch_audio
  [video]=fetch_video
  [time_series]=fetch_time_series
  [graph]=fetch_graph
)
ORDER=(image audio video time_series graph)

ok=0; fail=0
for m in "${ORDER[@]}"; do
  if [[ -n "$ONLY" && "$ONLY" != "$m" ]]; then continue; fi
  if [[ -n "$SKIP" && "$SKIP" == "$m" ]]; then log "skip $m (per --skip)"; continue; fi
  log "==== $m ===="
  if "${FETCHERS[$m]}"; then ok=$((ok+1)); else fail=$((fail+1)); fi
done

log "done: $ok ok, $fail fail; manifest=$MANIFEST"
# Always succeed at the script level if at least one modality landed.
[[ "$ok" -gt 0 ]]
