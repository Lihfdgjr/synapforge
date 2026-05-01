#!/usr/bin/env bash
# phase_auto_relauncher.sh -- external watchdog that polls <run_dir>/.phase
# and restarts the trainer with the new phase's flags appended.
#
# Counterpart to scripts/relaunch_loop.sh (which wraps the trainer and catches
# exit code 101). Use one or the other, not both. See docs/RENTAL_OPS.md
# "Auto-relauncher" section and docs/MASTER_PLAN.md §6 P22.
#
# Usage:
#   bash scripts/phase_auto_relauncher.sh --run-dir /workspace/runs/v24h_qwen3 --interval 60
#
# Env knobs: PAR_DRY_RUN (default 0), PAR_KILL_GRACE (10), PAR_THRASH_SEC (300),
# PAR_REPO_DIR (cwd at launch). PHASES table mirrors scripts/phase_manager.py.
#
# Smoke locally: bash -n scripts/phase_auto_relauncher.sh

set -uo pipefail

DRY_RUN="${PAR_DRY_RUN:-0}"
REPO_DIR="${PAR_REPO_DIR:-$(pwd)}"
KILL_GRACE="${PAR_KILL_GRACE:-10}"
THRASH_SEC="${PAR_THRASH_SEC:-300}"

# Phase id -> flags, mirror of scripts/phase_manager.py PHASES (P19 audit).
flags_for_phase() {
    case "$1" in
        0) echo "" ;;
        1) echo "--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware" ;;
        2) echo "--modal-list image,audio,time_series --phase-aware" ;;
        3) echo "--sft-data /workspace/data/alpaca_zh/alpaca_zh.json --response-only-loss --lr 1e-4 --phase-aware" ;;
        4) echo "--rl-grpo --rl-verifier sympy --rl-rollouts 8 --phase-aware" ;;
        *) echo "--phase-aware" ;;
    esac
}

RUN_DIR=""; INTERVAL=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)  RUN_DIR="${2:-}"; shift 2 ;;
        --interval) INTERVAL="${2:-60}"; shift 2 ;;
        --help|-h)  sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "[par] unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$RUN_DIR" ]] && { echo "[par] ERROR: --run-dir required" >&2; exit 2; }
[[ ! -d "$RUN_DIR" ]] && { echo "[par] ERROR: run-dir does not exist: $RUN_DIR" >&2; exit 2; }

PHASE_FILE="$RUN_DIR/.phase"
LAST_PHASE_FILE="$RUN_DIR/.par_last_phase"
RESTART_LOG="$RUN_DIR/phase_restart.log"
LAST_RESTART_TS=0

log() {
    local ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] [par] $*" | tee -a "$RESTART_LOG"
}

find_trainer_pid() {
    [[ -n "${PAR_FAKE_PID:-}" ]] && { echo "${PAR_FAKE_PID}"; return 0; }
    pgrep -f "train_100m_(kd|sft)\.py.* $RUN_DIR( |$)" 2>/dev/null | head -n 1
}

pick_warmstart_ckpt() {
    local ckpt
    ckpt=$(ls -1t "$RUN_DIR"/phase_change_step_*.pt 2>/dev/null | head -n 1)
    [[ -z "$ckpt" ]] && ckpt=$(ls -1t "$RUN_DIR"/step_*.pt 2>/dev/null | head -n 1)
    echo "$ckpt"
}

# Strip optim_state inline; stale Adam momentum cripples warmstart after
# vocab/arch change (lesson from v4.0/v4.1). Output: <ckpt>_no_optim.pt.
# Tests can short-circuit the torch path with PAR_SKIP_STRIP=1.
strip_optim_state() {
    local src="$1" dst="${1%.pt}_no_optim.pt"
    if [[ -f "$dst" ]]; then echo "$dst"; return 0; fi
    if [[ "${PAR_SKIP_STRIP:-0}" == "1" ]]; then
        cp "$src" "$dst" 2>/dev/null || : > "$dst"
        echo "$dst"; return 0
    fi
    python3 - "$src" "$dst" <<'PYEOF' 2>>"$RESTART_LOG"
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu")
if isinstance(ck, dict):
    for k in ("optim_state", "optimizer", "optimizer_state", "scheduler_state"):
        ck.pop(k, None)
torch.save(ck, dst)
PYEOF
    echo "$dst"
}

# Read /proc/<pid>/cmdline; drop --no-warmstart; replace --warmstart value
# with new ckpt; append new phase flags. Tests can override via
# PAR_FAKE_CMDLINE_FILE (a file with space-separated argv).
build_new_argv() {
    local pid="$1" new_ckpt="$2" new_flags="$3" raw rebuilt="" saw_warm=0 i=0 src
    if [[ -n "${PAR_FAKE_CMDLINE_FILE:-}" ]] && [[ -r "${PAR_FAKE_CMDLINE_FILE}" ]]; then
        raw=$(cat "${PAR_FAKE_CMDLINE_FILE}")
    else
        [[ -z "$pid" ]] || [[ ! -r "/proc/$pid/cmdline" ]] && { echo ""; return 1; }
        raw=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    fi
    # shellcheck disable=SC2206
    local toks=($raw)
    while [[ $i -lt ${#toks[@]} ]]; do
        local t="${toks[$i]}"
        if [[ "$t" == "--no-warmstart" ]]; then i=$((i + 1)); continue; fi
        if [[ "$t" == "--warmstart" ]]; then
            rebuilt="$rebuilt --warmstart $new_ckpt"; saw_warm=1; i=$((i + 2)); continue
        fi
        rebuilt="$rebuilt $t"; i=$((i + 1))
    done
    [[ "$saw_warm" -eq 0 ]] && [[ -n "$new_ckpt" ]] && rebuilt="$rebuilt --warmstart $new_ckpt"
    echo "$rebuilt $new_flags" | sed 's/  */ /g'
}

do_relaunch() {
    local old_phase="$1" new_phase="$2" now pid ckpt stripped new_flags new_argv waited new_pid
    now=$(date +%s)
    if (( now - LAST_RESTART_TS < THRASH_SEC )); then
        log "skip: only $((now - LAST_RESTART_TS))s since last restart (< ${THRASH_SEC}s anti-thrash)"
        return 1
    fi
    pid=$(find_trainer_pid)
    if [[ -z "$pid" ]]; then
        log "trainer not running; not auto-relaunching. old=$old_phase new=$new_phase"
        return 1
    fi
    ckpt=$(pick_warmstart_ckpt)
    if [[ -z "$ckpt" ]]; then log "no step_*.pt found in $RUN_DIR; abort"; return 1; fi
    stripped=$(strip_optim_state "$ckpt")
    new_flags=$(flags_for_phase "$new_phase")
    new_argv=$(build_new_argv "$pid" "$stripped" "$new_flags") || { log "no /proc/$pid/cmdline; abort"; return 1; }
    log "relaunch: pid=$pid old=$old_phase new=$new_phase ckpt=$stripped flags=$new_flags"
    log "  argv=$new_argv"

    # T6.5: append a phase-transition entry to CHANGELOG.md BEFORE killing
    # the old trainer. If the helper or python is missing, log + continue;
    # the relaunch itself must not depend on the changelog write succeeding.
    # Set PAR_SKIP_CHANGELOG=1 to disable (used by the dry-run test suite
    # so we don't spam the repo's real CHANGELOG).
    if [[ "${PAR_SKIP_CHANGELOG:-0}" != "1" ]]; then
        local helper="$REPO_DIR/scripts/changelog_helper.py" cl_step cl_val
        if [[ -f "$helper" ]] && command -v python3 >/dev/null 2>&1; then
            cl_step=$(echo "$stripped" | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | head -1)
            cl_val=$(grep -oE 'val_ppl_holdout=[0-9.]+' "$RUN_DIR/train.log" 2>/dev/null \
                | tail -1 | cut -d= -f2)
            python3 "$helper" \
                --phase-id "$new_phase" \
                ${cl_val:+--val-ppl "$cl_val"} \
                ${cl_step:+--step "$cl_step"} \
                --ckpt-path "$stripped" \
                --flags "$new_flags" \
                --changelog "$REPO_DIR/CHANGELOG.md" \
                >>"$RESTART_LOG" 2>&1 \
                && log "  changelog appended: phase=$new_phase step=${cl_step:-?} val=${cl_val:-?}" \
                || log "  changelog append failed (non-fatal); continuing relaunch"
        else
            log "  changelog helper not found or python3 missing; skipping append"
        fi
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN=1; not killing or spawning"; LAST_RESTART_TS=$now; return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < KILL_GRACE )); do sleep 1; waited=$((waited + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        log "  SIGTERM grace expired; SIGKILL pid=$pid"; kill -KILL "$pid" 2>/dev/null || true; sleep 1
    fi
    # setsid + disown per feedback_mcp_remote_ssh_quirks.md (no nohup).
    # shellcheck disable=SC2086
    setsid bash -c "cd $REPO_DIR && exec $new_argv >> $RUN_DIR/train.log 2>&1" </dev/null &
    new_pid=$!
    disown "$new_pid" 2>/dev/null || true
    log "relaunched: new_pid=$new_pid"
    LAST_RESTART_TS=$now
    return 0
}

log "start: run_dir=$RUN_DIR interval=${INTERVAL}s thrash=${THRASH_SEC}s dry_run=$DRY_RUN"
LAST_PHASE=-1
[[ -f "$LAST_PHASE_FILE" ]] && LAST_PHASE=$(cat "$LAST_PHASE_FILE" 2>/dev/null || echo -1)
TEST_MODE="${PAR_TEST_MODE:-0}"  # exit after first relaunch attempt

while true; do
    if [[ -f "$PHASE_FILE" ]]; then
        if command -v jq >/dev/null 2>&1; then
            CUR_PHASE=$(jq -r '.phase_id // empty' "$PHASE_FILE" 2>/dev/null || echo "")
        elif command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
            CUR_PHASE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('phase_id',''))" "$PHASE_FILE" 2>/dev/null || echo "")
        elif command -v python >/dev/null 2>&1; then
            CUR_PHASE=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('phase_id',''))" "$PHASE_FILE" 2>/dev/null || echo "")
        else
            # last-ditch grep+regex (json must have phase_id on its own line)
            CUR_PHASE=$(grep -oE '"phase_id"[[:space:]]*:[[:space:]]*[0-9]+' "$PHASE_FILE" | grep -oE '[0-9]+$' | head -1)
        fi
        if [[ -n "$CUR_PHASE" ]] && [[ "$CUR_PHASE" =~ ^[0-9]+$ ]] && [[ "$CUR_PHASE" -gt "$LAST_PHASE" ]]; then
            log "phase change detected: $LAST_PHASE -> $CUR_PHASE"
            if do_relaunch "$LAST_PHASE" "$CUR_PHASE"; then
                echo "$CUR_PHASE" > "$LAST_PHASE_FILE"
                LAST_PHASE="$CUR_PHASE"
            fi
            [[ "$TEST_MODE" == "1" ]] && { log "TEST_MODE=1; exit after first phase event"; exit 0; }
        fi
    fi
    [[ "$TEST_MODE" == "1" ]] && { log "TEST_MODE=1; no phase event; exit"; exit 0; }
    sleep "$INTERVAL"
done
