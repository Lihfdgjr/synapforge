#!/usr/bin/env bash
# setup_mohuanfang_warehouse.sh — one-shot bootstrap for tiered data storage.
#
# Goal: free up rental SSD by moving the canonical training corpus to
# mohuanfang (1.5 TB, 1.2 TB free) and turning the rental's
# /workspace/data into a bounded LRU cache that lazy-fetches shards on
# first access.
#
# After this script runs:
#   * mohuanfang holds /home/liu/synapforge_data/<dataset>/*.parquet for
#     each dataset under /workspace/data/.
#   * rental keeps the original tree at /workspace/data_archived/ as a
#     local fallback (move + symlink — no `rm` of the source).
#   * /workspace/data is a symlink to /workspace/data_cache, the new
#     bounded warehouse cache.  Existing trainer paths (--data-glob
#     /workspace/data/...) keep working — the first read of any shard
#     pulls it from mohuanfang into the cache.
#
# This is RECOVERABLE: if you decide later that warehouse mode isn't
# right, `rm /workspace/data && mv /workspace/data_archived /workspace/data`
# restores the original layout in seconds.
#
# Usage:
#   bash scripts/setup_mohuanfang_warehouse.sh                 # full setup
#   bash scripts/setup_mohuanfang_warehouse.sh --dry-run       # show plan
#   bash scripts/setup_mohuanfang_warehouse.sh --skip-upload   # rental-only
#                                                              # (corpus is
#                                                              # already
#                                                              # remote)
#   bash scripts/setup_mohuanfang_warehouse.sh --datasets a,b  # subset
#
# Env overrides: REMOTE_HOST, REMOTE_BASE, LOCAL_DATA, LOCAL_CACHE,
#                LOCAL_ARCHIVE.

set -u
set -o pipefail

REMOTE_HOST="${REMOTE_HOST:-liu@mohuanfang.com}"
REMOTE_BASE="${REMOTE_BASE:-/home/liu/synapforge_data}"
LOCAL_DATA="${LOCAL_DATA:-/workspace/data}"
LOCAL_CACHE="${LOCAL_CACHE:-/workspace/data_cache}"
LOCAL_ARCHIVE="${LOCAL_ARCHIVE:-/workspace/data_archived}"

DRY_RUN=0
SKIP_UPLOAD=0
DATASETS=""

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)        usage ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --skip-upload)    SKIP_UPLOAD=1; shift ;;
    --datasets)       DATASETS="$2"; shift 2 ;;
    --remote-host)    REMOTE_HOST="$2"; shift 2 ;;
    --remote-base)    REMOTE_BASE="$2"; shift 2 ;;
    --local-data)     LOCAL_DATA="$2"; shift 2 ;;
    --local-cache)    LOCAL_CACHE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }
run()  {
  if (( DRY_RUN )); then
    echo "DRY: $*"
  else
    eval "$@"
  fi
}

# 1. SSH probe — fail fast if mohuanfang is unreachable.
log "probing ${REMOTE_HOST}"
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "${REMOTE_HOST}" 'echo ok' >/dev/null 2>&1; then
  warn "ssh probe to ${REMOTE_HOST} failed"
  warn "verify your ~/.ssh/config or pass --remote-host user@host"
  exit 3
fi
log "ssh ok"

# 2. Discover datasets — first-level subdirs of LOCAL_DATA, OR explicit list.
if [[ -n "$DATASETS" ]]; then
  IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
else
  if [[ ! -d "$LOCAL_DATA" ]]; then
    warn "$LOCAL_DATA does not exist; nothing to bootstrap"
    exit 0
  fi
  mapfile -t DATASET_LIST < <(
    find "$LOCAL_DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | sort
  )
fi
if [[ ${#DATASET_LIST[@]} -eq 0 ]]; then
  warn "no datasets discovered under $LOCAL_DATA"
  exit 0
fi
log "datasets: ${DATASET_LIST[*]}"

# 3. Create remote dataset dirs (idempotent).
remote_mkdir_cmd="mkdir -p"
for ds in "${DATASET_LIST[@]}"; do
  remote_mkdir_cmd="$remote_mkdir_cmd \"${REMOTE_BASE}/${ds}\""
done
log "creating remote dataset dirs"
run "ssh '${REMOTE_HOST}' '${remote_mkdir_cmd}'"

# 4. Upload each dataset (rsync, idempotent — re-runs are cheap).
if (( SKIP_UPLOAD == 0 )); then
  for ds in "${DATASET_LIST[@]}"; do
    src="${LOCAL_DATA}/${ds}/"
    dst="${REMOTE_HOST}:${REMOTE_BASE}/${ds}/"
    if [[ ! -d "$src" ]]; then
      warn "skip $ds: $src missing"
      continue
    fi
    log "rsync $src -> $dst"
    run "rsync -avz --partial --inplace --timeout=600 \
         --include='*.parquet' --include='*/' --exclude='*' \
         '$src' '$dst'"
  done
else
  log "--skip-upload: assuming corpus already on mohuanfang"
fi

# 5. Local pivot: move data -> archive, create cache, symlink data ->
#    cache.  This is the bit that frees up rental disk only AFTER you
#    later evict the archive — but the archive is still there as a fast
#    fallback if mohuanfang goes offline mid-run.
if [[ -L "$LOCAL_DATA" ]]; then
  log "$LOCAL_DATA already a symlink; assume previous setup. skipping pivot."
elif [[ -d "$LOCAL_DATA" ]]; then
  if [[ -e "$LOCAL_ARCHIVE" ]]; then
    warn "$LOCAL_ARCHIVE already exists; refusing to overwrite. abort pivot."
    warn "(remove or rename it manually if you want to redo the setup)"
    exit 4
  fi
  log "mv $LOCAL_DATA -> $LOCAL_ARCHIVE"
  run "mv '$LOCAL_DATA' '$LOCAL_ARCHIVE'"
  log "mkdir -p $LOCAL_CACHE"
  run "mkdir -p '$LOCAL_CACHE'"
  log "ln -s $LOCAL_CACHE $LOCAL_DATA"
  run "ln -s '$LOCAL_CACHE' '$LOCAL_DATA'"
  # Pre-seed the cache symlink-style for each dataset so existing
  # --data-glob /workspace/data/<ds>/*.parquet patterns keep working
  # at the directory level (the first parquet read inside lazy-fetches
  # the file).
  for ds in "${DATASET_LIST[@]}"; do
    run "mkdir -p '$LOCAL_CACHE/$ds'"
  done
fi

log "done."
log "next: launch trainer with --remote-warehouse-host '${REMOTE_HOST}'"
log "      --remote-warehouse-base '${REMOTE_BASE}'"
log "      --remote-warehouse-dataset <one of: ${DATASET_LIST[*]}>"
log "      --cache-max-gb 30.0"
log
log "to roll back:"
log "  rm '$LOCAL_DATA' && mv '$LOCAL_ARCHIVE' '$LOCAL_DATA'"
