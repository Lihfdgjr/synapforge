#!/bin/bash
# launch_train_systemd.sh — SynapForge trainer launcher with systemd-run watchdog.
#
# Why this exists (P8, MASTER_PLAN.md §6 — RESOLVED 2026-05-01):
#   MCP `proc_exec` shell on rental boxes sometimes kills nohup'd children when
#   the SSH session expires. `setsid bash -c '...' </dev/null & disown` mostly
#   survives, but lacks (a) auto-restart on crash, (b) clean kill via systemctl,
#   (c) status check via systemctl. This wrapper picks the strongest available
#   isolation primitive on the host and launches the trainer through it.
#
# Memory cross-refs:
#   feedback_mcp_remote_ssh_quirks.md
#   feedback_mcp_nohup_hangs_use_systemd_run.md
#   feedback_no_polling_loops_even_bg.md
#
# Usage:
#   bash scripts/launch_train_systemd.sh \
#     --name v24h_qwen3 \
#     --warmstart /workspace/runs/v24h_qwen/step_002250_plif_reinit.pt \
#     --out /workspace/runs/v24h_qwen3 \
#     --steps 30000 \
#     -- \
#     --teacher Qwen/Qwen2.5-0.5B \
#     --backend triton_block \
#     --batch-size 64 \
#     --kd-every 4 \
#     --phase-aware
#
# Anything after a literal `--` is passed verbatim to train_100m_kd.py.

set -euo pipefail

UNIT_NAME=""
WARMSTART=""
OUT_DIR=""
STEPS=""
EXTRA_ARGS=()

# ---- arg parse ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)       UNIT_NAME="$2"; shift 2;;
    --warmstart)  WARMSTART="$2"; shift 2;;
    --out)        OUT_DIR="$2"; shift 2;;
    --steps)      STEPS="$2"; shift 2;;
    --)           shift; EXTRA_ARGS=("$@"); break;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      # Unrecognized flag → assume it's a trainer flag, pass through.
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$UNIT_NAME" ]]; then
  echo "ERROR: --name is required (e.g. v24h_qwen3)" >&2
  exit 2
fi
if [[ -z "$OUT_DIR" ]]; then
  echo "ERROR: --out is required (e.g. /workspace/runs/v24h_qwen3)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"

# ---- assemble trainer argv ----
TRAINER_ARGS=(--out "$OUT_DIR")
[[ -n "$WARMSTART" ]] && TRAINER_ARGS+=(--warmstart "$WARMSTART")
[[ -n "$STEPS" ]]     && TRAINER_ARGS+=(--steps "$STEPS")
TRAINER_ARGS+=("${EXTRA_ARGS[@]}")

WORKDIR="${SYNAPFORGE_ROOT:-/workspace/synapforge_git}"
if [[ ! -d "$WORKDIR" ]]; then
  echo "ERROR: SYNAPFORGE_ROOT=$WORKDIR does not exist" >&2
  exit 3
fi

# ---- launch path A: systemd-run --user (preferred) ----
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  echo "[launch_train_systemd] using systemd-run --user --unit=$UNIT_NAME"
  systemd-run --user \
    --unit="$UNIT_NAME" \
    --description="SynapForge trainer $UNIT_NAME" \
    --working-directory="$WORKDIR" \
    --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --setenv=OMP_NUM_THREADS=8 \
    --setenv=TOKENIZERS_PARALLELISM=false \
    --property=StandardOutput="append:$LOG" \
    --property=StandardError="append:$LOG" \
    -- python3 -u train_100m_kd.py "${TRAINER_ARGS[@]}"

  # Pull MainPID for the user's report.
  PID=$(systemctl --user show -p MainPID --value "$UNIT_NAME" 2>/dev/null || echo "?")
  echo
  echo "  unit:    $UNIT_NAME (transient, no auto-restart)"
  echo "  pid:     $PID"
  echo "  log:     $LOG"
  echo "  status:  systemctl --user status $UNIT_NAME"
  echo "  follow:  journalctl --user -u $UNIT_NAME -f"
  echo "  stop:    systemctl --user stop $UNIT_NAME"
  echo
  echo "  For auto-restart-on-crash, install the template unit instead:"
  echo "    cp scripts/synapforge-trainer.service.template \\"
  echo "       ~/.config/systemd/user/synapforge-trainer@.service"
  echo "    systemctl --user daemon-reload"
  echo "    systemctl --user enable --now synapforge-trainer@$UNIT_NAME.service"
  echo "  See docs/RENTAL_OPS.md for the full decision tree."
  exit 0
fi

# ---- launch path B: setsid + disown fallback ----
echo "[launch_train_systemd] WARNING: systemd-run --user unavailable; falling back" >&2
echo "[launch_train_systemd] WARNING: setsid+disown — no auto-restart, kill via 'kill <pid>'" >&2

cd "$WORKDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# setsid detaches from controlling terminal so SSH/MCP exit can't SIGHUP it.
# disown removes the job from the shell's job table.
setsid bash -c "exec python3 -u train_100m_kd.py ${TRAINER_ARGS[*]@Q} > $LOG 2>&1" </dev/null &
PID=$!
disown "$PID" 2>/dev/null || true

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: child process died immediately; check $LOG" >&2
  exit 4
fi

echo
echo "  unit:    none (fallback mode)"
echo "  pid:     $PID"
echo "  log:     $LOG"
echo "  status:  ps -p $PID -o pid,cmd  /  tail -f $LOG"
echo "  stop:    kill $PID  (or kill -9 $PID if stuck)"
echo "  NOTE:    no auto-restart — install the template unit for production runs."
echo
exit 0
