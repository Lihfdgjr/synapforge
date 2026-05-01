#!/usr/bin/env bash
# relaunch_loop.sh -- outer wrapper that catches train_100m_kd.py exit 101
# and re-spawns the trainer with the new phase's flags appended.
#
# Without this wrapper, train_100m_kd.py's --phase-aware flag is dead code:
# the trainer cleanly exits with code 101 on a phase change but nothing
# ever picks up the .phase JSON and re-launches. This script is the
# missing link.
#
# See docs/PHASE_RELAUNCH.md for the full design + recovery flow.
# Cross-reference: docs/RELIABILITY.md sec. 3 (silent killer #3).
#
# Usage:
#   bash scripts/relaunch_loop.sh \
#       --out /workspace/runs/v24h_qwen \
#       --warmstart /workspace/runs/v24h_qwen/best_step_001250.pt \
#       --steps 8000 --batch-size 32 --backend triton_block \
#       --teacher gpt2 --kd-weight 0.7 --kd-every 4
#
# Constants you can tune via env (no CLI -- keep CLI clean for trainer flags):
#   RL_MAX_TRANSITIONS  default 10  (hard cap; same-phase-twice also breaks)
#   RL_MAX_HOURS        default 25  (wall-clock total budget)
#   RL_PYTHON           default python3
#   RL_KD_TRAINER       default train_100m_kd.py
#   RL_SFT_TRAINER      default train_100m_sft.py
#   RL_NW_TRAINER       default scripts/train_neural_web.py

set -uo pipefail
# We deliberately do NOT set -e: we need to inspect $? after each python run.

# ---------------------------------------------------------------------------
# Defaults / env knobs.
# ---------------------------------------------------------------------------
MAX_TRANSITIONS="${RL_MAX_TRANSITIONS:-10}"
MAX_HOURS="${RL_MAX_HOURS:-25}"
PYTHON="${RL_PYTHON:-python3}"
KD_TRAINER="${RL_KD_TRAINER:-train_100m_kd.py}"
SFT_TRAINER="${RL_SFT_TRAINER:-train_100m_sft.py}"
NW_TRAINER="${RL_NW_TRAINER:-scripts/train_neural_web.py}"

START_TS=$(date +%s)
DEADLINE=$((START_TS + MAX_HOURS * 3600))

# ---------------------------------------------------------------------------
# Argv pre-scan: lift --out so we know where .phase lives. We do NOT consume
# it; we just peek so the rest passes through to the trainer untouched.
# ---------------------------------------------------------------------------
OUT_DIR=""
for ((i = 1; i <= $#; i++)); do
    if [[ "${!i}" == "--out" ]]; then
        j=$((i + 1))
        OUT_DIR="${!j:-}"
        break
    fi
done
if [[ -z "$OUT_DIR" ]]; then
    echo "[relaunch] ERROR: --out <dir> not found in argv; cannot locate .phase" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/relaunch.log"
PHASE_STATE_FILE="$OUT_DIR/.current_phase"
TRANSITIONS_FILE="$OUT_DIR/.relaunch_transitions"

log() {
    local msg="$*"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] [relaunch] $msg" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Phase id -> trainer + extra flags lookup. Mirrors phase_manager.py PHASES.
# Keep these in sync if PHASES is ever extended.
# ---------------------------------------------------------------------------
flags_for_phase() {
    case "$1" in
        0) echo "" ;;
        1) echo "--curiosity-weight 0.05 --self-learn-ttt --self-learn-k 8" ;;
        2) echo "--curiosity-weight 0.05 --self-learn-ttt --self-learn-k 8 --modal-list image,audio --modal-data-dir /workspace/data/multimodal --modal-alpha 0.05" ;;
        3) echo "" ;;  # phase 3 swaps trainer, flags handled separately
        4) echo "" ;;
        *) echo "" ;;
    esac
}

trainer_for_phase() {
    case "$1" in
        3) echo "$SFT_TRAINER" ;;
        *) echo "$KD_TRAINER" ;;
    esac
}

# ---------------------------------------------------------------------------
# Resume logic: if a prior loop crashed mid-phase the .current_phase file
# tells us where it was. Otherwise we start at 0.
# ---------------------------------------------------------------------------
if [[ -f "$PHASE_STATE_FILE" ]]; then
    CURRENT_PHASE=$(cat "$PHASE_STATE_FILE" 2>/dev/null || echo 0)
    log "resuming from .current_phase=$CURRENT_PHASE"
else
    CURRENT_PHASE=0
    echo 0 > "$PHASE_STATE_FILE"
    log "no prior .current_phase; starting at phase 0"
fi

# Track recent phase ids to detect anti-thrash (same phase twice in a row).
PREV_PHASE=-1
TRANSITION_COUNT=0
: > "$TRANSITIONS_FILE"

# ---------------------------------------------------------------------------
# Optional sidecar: in phase 4 we additionally launch train_neural_web.py
# alongside the RL trainer (RL + neural-web learning run concurrently per
# the phased-training plan).
# ---------------------------------------------------------------------------
NW_PID=""
spawn_neural_web_sidecar() {
    if [[ -z "$NW_PID" ]] && [[ -f "$NW_TRAINER" ]]; then
        log "spawning neural_web sidecar: $PYTHON $NW_TRAINER --out $OUT_DIR/neural_web.json"
        # shellcheck disable=SC2086
        nohup $PYTHON "$NW_TRAINER" --out "$OUT_DIR/neural_web.json" \
            --headless --no-real >> "$OUT_DIR/neural_web.log" 2>&1 &
        NW_PID=$!
        log "neural_web sidecar pid=$NW_PID"
    fi
}

cleanup_sidecar() {
    if [[ -n "$NW_PID" ]] && kill -0 "$NW_PID" 2>/dev/null; then
        log "stopping neural_web sidecar pid=$NW_PID"
        kill "$NW_PID" 2>/dev/null || true
    fi
}
trap cleanup_sidecar EXIT INT TERM

# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------
log "loop start: out=$OUT_DIR phase=$CURRENT_PHASE max_transitions=$MAX_TRANSITIONS deadline=+${MAX_HOURS}h"
log "argv (pass-through): $*"

while true; do
    # Wall-clock budget guard.
    NOW=$(date +%s)
    if (( NOW >= DEADLINE )); then
        log "DEADLINE reached (${MAX_HOURS}h budget exhausted); breaking out of loop"
        break
    fi

    # Hard cap on transitions to avoid runaway loops on a wedged trainer.
    if (( TRANSITION_COUNT >= MAX_TRANSITIONS )); then
        log "MAX_TRANSITIONS=$MAX_TRANSITIONS reached; breaking"
        break
    fi

    # Anti-thrash: if we just transitioned into the same phase we left,
    # something's broken (probably a stale .phase file or trainer ignoring
    # consume_phase). Bail rather than spin.
    if (( CURRENT_PHASE == PREV_PHASE )) && (( TRANSITION_COUNT > 0 )); then
        log "ERROR: same phase twice in a row ($CURRENT_PHASE); bailing to avoid thrash"
        break
    fi

    # Record current phase and pick trainer + flags.
    echo "$CURRENT_PHASE" > "$PHASE_STATE_FILE"
    TRAINER=$(trainer_for_phase "$CURRENT_PHASE")
    EXTRA_FLAGS=$(flags_for_phase "$CURRENT_PHASE")

    # Phase 4 spawns the neural-web sidecar concurrently.
    if (( CURRENT_PHASE == 4 )); then
        spawn_neural_web_sidecar
    fi

    # The trainer itself needs --phase-aware so it actually polls .phase
    # and exits 101 on transitions. We always inject it (trainer ignores
    # duplicate flags anyway).
    log "spawning trainer (phase=$CURRENT_PHASE): $PYTHON $TRAINER $* $EXTRA_FLAGS --phase-aware"
    # shellcheck disable=SC2086
    $PYTHON "$TRAINER" "$@" $EXTRA_FLAGS --phase-aware
    RC=$?

    case "$RC" in
        0)
            log "trainer exited cleanly (rc=0); loop done"
            break
            ;;
        101)
            log "trainer exited 101 (phase change); reading $OUT_DIR/.phase"
            # The trainer's consume_phase atomically renamed .phase to
            # .phase.consumed.<ts>; pick the newest one.
            CONSUMED=$(ls -1t "$OUT_DIR"/.phase.consumed.* 2>/dev/null | head -n 1)
            if [[ -z "$CONSUMED" || ! -f "$CONSUMED" ]]; then
                log "WARN: no .phase.consumed file found after exit 101; aborting"
                break
            fi
            NEW_PHASE=$($PYTHON -c "import json,sys;print(json.load(open(sys.argv[1])).get('phase_id','?'))" "$CONSUMED" 2>/dev/null || echo "?")
            if [[ "$NEW_PHASE" == "?" ]]; then
                log "ERROR: could not parse phase_id from $CONSUMED; aborting"
                break
            fi

            # Find the newest phase_change_step_*.pt for warmstart logging.
            NEW_CKPT=$(ls -1t "$OUT_DIR"/phase_change_step_*.pt 2>/dev/null | head -n 1 || true)
            log "transition $CURRENT_PHASE -> $NEW_PHASE; warmstart_ckpt=${NEW_CKPT:-<none>}"
            echo "ts=$(date +%s) old=$CURRENT_PHASE new=$NEW_PHASE ckpt=${NEW_CKPT:-NONE}" >> "$TRANSITIONS_FILE"

            PREV_PHASE=$CURRENT_PHASE
            CURRENT_PHASE=$NEW_PHASE
            TRANSITION_COUNT=$((TRANSITION_COUNT + 1))
            ;;
        *)
            log "trainer exited rc=$RC (not 0, not 101); breaking"
            break
            ;;
    esac
done

cleanup_sidecar
log "loop end: transitions=$TRANSITION_COUNT final_phase=$CURRENT_PHASE wallclock=$(( $(date +%s) - START_TS ))s"
