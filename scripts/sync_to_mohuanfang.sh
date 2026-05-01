#!/usr/bin/env bash
# sync_to_mohuanfang.sh — defense-in-depth backup loop.
#
# Continuously rsyncs /workspace/runs/<run>/ from rental → mohuanfang at
# /home/liu/synapforge_backup/<run>/. This is REDUNDANT with
# triple_backup_daemon.py by design — explicit + simpler, runs even if
# the python daemon dies.
#
# Drives 3 rsyncs per cycle: ckpts (best-only), train.log, skill_log.
# If 3 consecutive cycles fail, posts a Slack/email notification.
#
# Usage:
#   bash scripts/sync_to_mohuanfang.sh                       # default loop
#   bash scripts/sync_to_mohuanfang.sh --run v24h_qwen
#   bash scripts/sync_to_mohuanfang.sh --interval 600        # 10min
#   bash scripts/sync_to_mohuanfang.sh --once                # single cycle, exit
#   bash scripts/sync_to_mohuanfang.sh --help

set -u
set -o pipefail

RUN="${RUN:-v24h_qwen}"
LOCAL_BASE="${LOCAL_BASE:-/workspace/runs}"
REMOTE_HOST="${REMOTE_HOST:-mohuanfang.com}"
REMOTE_USER="${REMOTE_USER:-liu}"
REMOTE_BASE="${REMOTE_BASE:-/home/liu/synapforge_backup}"
INTERVAL="${INTERVAL:-600}"      # 10 min
ONCE=0
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"  # optional
ALERT_EMAIL="${ALERT_EMAIL:-}"      # optional, requires `mail` cmd
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"

usage() {
  sed -n '2,16p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage ;;
    --run)       RUN="$2"; shift 2 ;;
    --interval)  INTERVAL="$2"; shift 2 ;;
    --once)      ONCE=1; shift ;;
    --local)     LOCAL_BASE="$2"; shift 2 ;;
    --remote)    REMOTE_BASE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

LOCAL="$LOCAL_BASE/$RUN/"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${RUN}/"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN $*" >&2; }

notify() {
  local msg="$1"
  warn "NOTIFY: $msg"
  if [[ -n "$SLACK_WEBHOOK" ]]; then
    curl -fsS -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"sync_to_mohuanfang: $msg\"}" \
      "$SLACK_WEBHOOK" >/dev/null 2>&1 \
      || warn "slack webhook post failed"
  fi
  if [[ -n "$ALERT_EMAIL" ]] && command -v mail >/dev/null 2>&1; then
    echo "$msg" | mail -s "synapforge sync failure: $RUN" "$ALERT_EMAIL" \
      || warn "email send failed"
  fi
}

# Reachability probe — fast 5s timeout. Returns 0 if SSH up.
probe() {
  ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
      "${REMOTE_USER}@${REMOTE_HOST}" 'true' >/dev/null 2>&1
}

# One backup cycle: returns 0 if all rsyncs succeeded, else nonzero.
sync_once() {
  if [[ ! -d "$LOCAL" ]]; then
    warn "local source $LOCAL missing; nothing to back up yet"
    return 0  # not a failure — trainer hasn't started
  fi
  if ! probe; then
    warn "ssh probe to ${REMOTE_HOST} failed; will retry next cycle"
    return 1
  fi

  local rc=0
  # 1. ckpts: best-* + step_* (preserve attrs, partial-resume, exclude tmp)
  rsync -avz --partial --inplace --timeout=120 \
    --include='best*' --include='step_*' --include='*.pt' \
    --include='*.json' --include='*/' --exclude='*' \
    "$LOCAL" "$REMOTE" 2>&1 | tail -n 5 || rc=$?

  # 2. logs (always full)
  if [[ -f "$LOCAL/train.log" ]]; then
    rsync -avz --partial --timeout=60 \
      "$LOCAL/train.log" "${REMOTE}train.log" 2>&1 | tail -n 3 || rc=$?
  fi

  # 3. skill log directory (NeuroMCP)
  if [[ -d "$LOCAL/skill_log" ]]; then
    rsync -avz --partial --timeout=60 \
      "$LOCAL/skill_log/" "${REMOTE}skill_log/" 2>&1 | tail -n 3 || rc=$?
  fi

  return $rc
}

run_loop() {
  local fails=0
  while true; do
    log "cycle: rsync $LOCAL -> $REMOTE"
    if sync_once; then
      log "cycle ok"
      fails=0
    else
      fails=$((fails+1))
      warn "cycle FAILED ($fails consecutive)"
      if (( fails >= FAIL_THRESHOLD )); then
        notify "rsync failed $fails cycles in a row for run=$RUN host=$REMOTE_HOST"
        fails=0  # reset so we don't spam every cycle
      fi
    fi
    (( ONCE == 1 )) && break
    sleep "$INTERVAL"
  done
}

if (( ONCE == 1 )); then
  sync_once; exit $?
fi

# Trap cleanly so cron/systemd see exit code.
trap 'log "shutdown"; exit 0' INT TERM

run_loop
