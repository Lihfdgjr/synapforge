#!/usr/bin/env bash
# install_watchdog_cron.sh — one-shot installer, run on mohuanfang.com.
#
# Idempotent: safe to re-run; will not double-install the cron line and
# will overwrite the deployed watchdog with the latest source.
#
# Usage (on mohuanfang):
#     git clone https://github.com/Lihfdgjr/synapforge && cd synapforge
#     sudo bash scripts/install_watchdog_cron.sh
#
# What it does:
#   1) creates /home/liu/synapforge_backup/watchdog/
#   2) copies rental_watchdog.sh + writes watchdog.env (template if absent)
#   3) creates /var/log/synapforge_watchdog.log with correct perms
#   4) adds `*/5 * * * * .../rental_watchdog.sh >> .../log 2>&1` to crontab if missing
#   5) prints final crontab so user can verify

set -uo pipefail

LIU_HOME="${LIU_HOME:-/home/liu}"
INSTALL_DIR="$LIU_HOME/synapforge_backup/watchdog"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_SCRIPT="$SRC_DIR/rental_watchdog.sh"
LOG_FILE="/var/log/synapforge_watchdog.log"
CRON_LINE="*/5 * * * * $INSTALL_DIR/rental_watchdog.sh >> $LOG_FILE 2>&1"

if [ ! -f "$SRC_SCRIPT" ]; then
    echo "FATAL: $SRC_SCRIPT not found — run from a synapforge checkout" >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ] && [ ! -w /var/log ]; then
    echo "WARN: not root and /var/log not writable; will fall back to $INSTALL_DIR/watchdog.log"
    LOG_FILE="$INSTALL_DIR/watchdog.log"
    CRON_LINE="*/5 * * * * $INSTALL_DIR/rental_watchdog.sh >> $LOG_FILE 2>&1"
fi

echo "[install] target dir: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

echo "[install] copying rental_watchdog.sh"
install -m 0755 "$SRC_SCRIPT" "$INSTALL_DIR/rental_watchdog.sh"

ENV_FILE="$INSTALL_DIR/watchdog.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[install] writing template $ENV_FILE (edit before first cron run!)"
    cat > "$ENV_FILE" <<'EOF'
# synapforge rental watchdog config
# Loaded by rental_watchdog.sh if present.

# Required
WATCHDOG_RENTAL_HOST=117.74.66.77
WATCHDOG_RENTAL_PORT=41614
WATCHDOG_RENTAL_USER=root

# Where to mirror best.pt on mohuanfang
WATCHDOG_BACKUP_DIR=/home/liu/synapforge_backup
WATCHDOG_REMOTE_GLOB=/workspace/runs/v24h_qwen/best*.pt

# Optional: alerting
# WATCHDOG_SLACK_URL=https://hooks.slack.com/services/...
# WATCHDOG_NOTIFY_EMAIL=demery_guernsey032@mail.com
EOF
    chmod 0600 "$ENV_FILE"
else
    echo "[install] $ENV_FILE already present; leaving as-is"
fi

echo "[install] ensuring $LOG_FILE exists"
touch "$LOG_FILE" 2>/dev/null || {
    echo "WARN: cannot create $LOG_FILE — make sure cron user can write it"
}
chmod 0644 "$LOG_FILE" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Cron line install (idempotent)
# ---------------------------------------------------------------------------

CRON_USER="${SUDO_USER:-$USER}"
echo "[install] installing crontab line for user: $CRON_USER"

current_crontab="$(crontab -u "$CRON_USER" -l 2>/dev/null || true)"
# strip stale watchdog lines (different paths) before adding new
filtered_crontab="$(printf '%s\n' "$current_crontab" \
    | grep -v 'rental_watchdog\.sh' || true)"

if printf '%s\n' "$current_crontab" | grep -qF "$CRON_LINE"; then
    echo "[install] cron line already present; skipping"
else
    echo "[install] adding cron line"
    {
        printf '%s\n' "$filtered_crontab"
        echo "$CRON_LINE"
    } | grep -v '^$' | crontab -u "$CRON_USER" -
fi

echo
echo "================================================"
echo "Final crontab for $CRON_USER:"
crontab -u "$CRON_USER" -l 2>/dev/null || echo "(empty)"
echo "================================================"
echo "[install] done. Edit $ENV_FILE if needed, then wait 5 min for first tick."
echo "[install] tail -f $LOG_FILE   # watch the watchdog"
