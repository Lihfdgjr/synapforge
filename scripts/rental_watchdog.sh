#!/usr/bin/env bash
# rental_watchdog.sh — runs on mohuanfang via cron every 5 min.
#
# After the 2026-04-30 v4.1 disaster (rental sshd saturated, 7h training
# lost), this watchdog ensures we KNOW within 15 minutes when the rental
# is unreachable, and we EVACUATE best*.pt while ssh still partially works.
#
# Why on mohuanfang and not on the rental: the rental can't watch itself.
# When sshd saturates the box stops accepting connections but the GPU may
# still be alive — we need an external observer.
#
# Failure escalation:
#   1 fail   → log only
#   2 fails  → log + count
#   3 fails  → log + emergency scp of best*.pt while link may still work
#   4 fails  → log
#   5 fails  → write /tmp/.rental_dead_ts + loud WARN to stderr+log
#
# Required env (read from watchdog.env or process env):
#   WATCHDOG_RENTAL_HOST   e.g. 117.74.66.77
#   WATCHDOG_RENTAL_PORT   e.g. 41614
#   WATCHDOG_RENTAL_USER   e.g. root
#   WATCHDOG_BACKUP_DIR    e.g. /home/liu/synapforge_backup
#   WATCHDOG_REMOTE_GLOB   e.g. /workspace/runs/v24h_qwen/best*.pt
#   WATCHDOG_SLACK_URL     optional Slack webhook
#   WATCHDOG_NOTIFY_EMAIL  optional, for `mail -s ...`
#
# NOTE: -uo pipefail (NOT -e) — we deliberately continue past failures.

set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source env file if it sits next to this script
if [ -f "$SCRIPT_DIR/watchdog.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/watchdog.env"
fi

RENTAL_HOST="${WATCHDOG_RENTAL_HOST:-}"
RENTAL_PORT="${WATCHDOG_RENTAL_PORT:-22}"
RENTAL_USER="${WATCHDOG_RENTAL_USER:-root}"
BACKUP_DIR="${WATCHDOG_BACKUP_DIR:-/home/liu/synapforge_backup}"
REMOTE_GLOB="${WATCHDOG_REMOTE_GLOB:-/workspace/runs/v24h_qwen/best*.pt}"
SLACK_URL="${WATCHDOG_SLACK_URL:-}"
NOTIFY_EMAIL="${WATCHDOG_NOTIFY_EMAIL:-}"

STATE_FILE="/tmp/synapforge_rental_health.json"
DEAD_FLAG="/tmp/.rental_dead_ts"
HEARTBEAT_FILE="$BACKUP_DIR/.healthy"
LOG_FILE="${WATCHDOG_LOG_FILE:-/var/log/synapforge_watchdog.log}"

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"

# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------

if [ -z "$RENTAL_HOST" ]; then
    echo "[$(date -Is)] FATAL: WATCHDOG_RENTAL_HOST unset, refusing to run" >&2
    exit 2
fi

mkdir -p "$BACKUP_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    # log to file AND stdout (stdout is captured by cron's redirect)
    echo "[$(date -Is)] $*"
}

notify() {
    local level="$1"
    local msg="$2"
    log "NOTIFY[$level] $msg"
    if [ -n "$SLACK_URL" ]; then
        curl --max-time 5 --silent -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\":rotating_light: synapforge watchdog [$level] $msg\"}" \
            "$SLACK_URL" >/dev/null 2>&1 || log "slack post failed"
    fi
    if [ -n "$NOTIFY_EMAIL" ] && command -v mail >/dev/null 2>&1; then
        echo "$msg" | mail -s "synapforge rental [$level]" "$NOTIFY_EMAIL" \
            >/dev/null 2>&1 || log "mail send failed"
    fi
}

read_fail_count() {
    if [ -f "$STATE_FILE" ]; then
        # crude jq-free parse, robust against missing key
        sed -n 's/.*"consecutive_failures"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' \
            "$STATE_FILE" | head -n1
    fi
}

write_state() {
    local fails="$1"
    local last_status="$2"
    local now
    now="$(date -Is)"
    cat > "$STATE_FILE" <<EOF
{
  "consecutive_failures": $fails,
  "last_status": "$last_status",
  "last_check": "$now",
  "rental_host": "$RENTAL_HOST",
  "rental_port": $RENTAL_PORT
}
EOF
}

ssh_ping() {
    # exit 0 = reachable, nonzero = unreachable
    timeout 12 ssh $SSH_OPTS -p "$RENTAL_PORT" "$RENTAL_USER@$RENTAL_HOST" \
        'echo pong' >/dev/null 2>&1
}

emergency_scp() {
    # Best-effort. Even partial transfers are better than nothing.
    local dest="$BACKUP_DIR/emergency_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dest" 2>/dev/null || true
    log "emergency: scp $REMOTE_GLOB -> $dest/"
    timeout 600 scp $SSH_OPTS -P "$RENTAL_PORT" \
        "$RENTAL_USER@$RENTAL_HOST:$REMOTE_GLOB" "$dest/" 2>&1 \
        | while IFS= read -r line; do log "  scp: $line"; done
    if compgen -G "$dest/best*" > /dev/null; then
        log "emergency: at least one best*.pt evacuated to $dest/"
        notify "EVACUATED" "best*.pt evacuated to mohuanfang:$dest"
        return 0
    else
        log "emergency: nothing evacuated (link fully dead)"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

prev_fails="$(read_fail_count)"
prev_fails="${prev_fails:-0}"

if ssh_ping; then
    # success — reset everything
    if [ "$prev_fails" -gt 0 ]; then
        log "RECOVERED after $prev_fails consecutive failures"
        notify "RECOVERED" "rental ssh back up after $prev_fails fails"
    fi
    write_state 0 "ok"
    : > "$HEARTBEAT_FILE"
    date -Is > "$HEARTBEAT_FILE"
    rm -f "$DEAD_FLAG" 2>/dev/null || true
    log "ok ($RENTAL_HOST:$RENTAL_PORT reachable)"
    exit 0
fi

# failure path
fails=$((prev_fails + 1))
write_state "$fails" "fail"
log "FAIL #$fails — ssh ping to $RENTAL_HOST:$RENTAL_PORT failed"

case "$fails" in
    1|2)
        log "(escalation: log only at fail count $fails)"
        ;;
    3)
        notify "WARN" "rental SSH unreachable 3x — running emergency scp of best*.pt"
        emergency_scp || true
        ;;
    4)
        log "(escalation: holding at fail count $fails — escalation on next fail)"
        ;;
    5)
        date -Is > "$DEAD_FLAG"
        echo "rental_host=$RENTAL_HOST"  >> "$DEAD_FLAG"
        echo "rental_port=$RENTAL_PORT"  >> "$DEAD_FLAG"
        log "==================================================================="
        log "RENTAL DEAD — $fails consecutive ssh failures"
        log "Manual escalation required:"
        log "  1) check 算力牛 web console for instance status"
        log "  2) if SSH unreachable >24h, file refund ticket"
        log "  3) once ckpts saved, spin up new instance + warmstart"
        log "  see docs/RENTAL_RECOVERY.md"
        log "==================================================================="
        notify "DEAD" "rental DEAD after $fails ssh failures — manual escalation required"
        ;;
    *)
        # 6+ — we've already declared dead, just log
        log "(rental still dead, fail count $fails)"
        ;;
esac

# Always exit 0 from cron's perspective so cron doesn't email errors;
# health is communicated via STATE_FILE/DEAD_FLAG.
exit 0
