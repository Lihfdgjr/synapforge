#!/usr/bin/env bash
# cpu_utilization_jobs.sh — run useful background CPU work while the GPU
# trainer is busy. Designed to run ON the rental box (A800 / A100 / H800).
#
# Why: the GPU trainer (train_100m_kd.py with --backend triton_block) keeps
# the GPU around 75-90% but only uses ~25GB RAM and 4-8 dataloader workers.
# Out of 16-32 cores and ~200GB RAM, ~half is idle. We fill that idle time
# with PRE-PROCESSING + EVAL work that the operator would otherwise have
# to do later in serial:
#
#   (a) synthesize + tokenize phase-1 zh pretrain corpus (~1h CPU)
#   (b) tokenize alpaca-zh SFT data for phase 3 (~30 min)
#   (c) heavy bench (mmlu/hellaswag) on the latest ckpt (~30 min, then idle)
#   (d) pre-warm the Qwen tokenizer cache to /dev/shm
#
# Constraints:
#   * MUST not steal GPU. We only run if a trainer is alive AND we cap to
#     `nice -n 19 ionice -c 3` (idle priority).
#   * MUST pin to CPUs >=8 (taskset) so cores 0-7 stay free for the trainer's
#     dataloader workers + system.
#   * Idempotent: if output already exists, skip.
#   * All children launched as `nohup ... &` and disowned, so the SSH/MCP
#     channel can close without killing them. See docs/RENTAL_OPS.md (P8).
#
# Usage (run on rental):
#     bash scripts/cpu_utilization_jobs.sh \
#         --run-dir /workspace/runs/v24h_qwen3 \
#         --data-dir /workspace/data
#
# Logs: $RUN_DIR/cpu_jobs.log
# Cross-refs: docs/MONITOR_AND_CPU_JOBS.md, docs/RENTAL_OPS.md, docs/AUTO_EVAL.md.

set -uo pipefail

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

RUN_DIR=""
DATA_DIR="/workspace/data"
REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-0.5B}"
PIN_CPUS="${PIN_CPUS:-8-31}"  # leave 0-7 free
N_PHASE1="${N_PHASE1:-50000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)   RUN_DIR="$2"; shift 2;;
        --data-dir)  DATA_DIR="$2"; shift 2;;
        --repo-dir)  REPO_DIR="$2"; shift 2;;
        --teacher)   TEACHER="$2"; shift 2;;
        --pin-cpus)  PIN_CPUS="$2"; shift 2;;
        --n-phase1)  N_PHASE1="$2"; shift 2;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: --run-dir is required (e.g. /workspace/runs/v24h_qwen3)" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$DATA_DIR"
LOG="$RUN_DIR/cpu_jobs.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

# Build the priority + CPU pin prefix. Falls back gracefully:
#   1. cpulimit if installed (lets us cap %CPU explicitly)
#   2. else nice -n 19 ionice -c 3
# Then taskset pins to CPUs PIN_CPUS.
make_prefix() {
    local prefix=""
    if command -v cpulimit >/dev/null 2>&1; then
        # cpulimit needs to wrap the FINAL command, so it's added by the caller
        prefix=""
    fi
    prefix="nice -n 19"
    if command -v ionice >/dev/null 2>&1; then
        prefix="$prefix ionice -c 3"
    fi
    if command -v taskset >/dev/null 2>&1; then
        prefix="$prefix taskset -c $PIN_CPUS"
    fi
    echo "$prefix"
}

# Check trainer alive. Return 0 if alive.
trainer_alive() {
    pgrep -f 'train_100m_kd.py|train_100m_sft.py' >/dev/null 2>&1
}

# Spawn a job in the background, fully detached, and log its PID.
# Args: <name> <cmd...>
spawn() {
    local name="$1"; shift
    local outfile="$RUN_DIR/cpu_job_${name}.log"
    log "spawning ${name} -> $outfile"
    # shellcheck disable=SC2068
    nohup setsid bash -c "$*" </dev/null >"$outfile" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    log "  ${name} pid=${pid}"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

log "================================================================"
log "cpu_utilization_jobs starting"
log "RUN_DIR=$RUN_DIR DATA_DIR=$DATA_DIR REPO_DIR=$REPO_DIR"
log "PIN_CPUS=$PIN_CPUS  N_PHASE1=$N_PHASE1  TEACHER=$TEACHER"

if ! trainer_alive; then
    log "WARN: no trainer process detected. Refusing to start CPU jobs."
    log "      (run \`pgrep -af train_100m\` to verify.)"
    exit 1
fi
log "trainer is alive, proceeding"

if [[ ! -d "$REPO_DIR" ]]; then
    log "ERROR: REPO_DIR=$REPO_DIR does not exist"
    exit 1
fi
cd "$REPO_DIR"

PREFIX=$(make_prefix)
log "priority prefix: $PREFIX"

# ---------------------------------------------------------------------------
# Job (a): synth zh pretrain + mix phase 1 corpus
# ---------------------------------------------------------------------------

ZH_OUT="$DATA_DIR/synth_zh_phase1.parquet"
MIX_OUT="$DATA_DIR/mix_phase1.parquet"
WIKI_EN="$DATA_DIR/wiki_en_subset.parquet"

if [[ -f "$MIX_OUT" ]]; then
    log "(a) skip: $MIX_OUT already exists"
else
    CMD_A="$PREFIX python scripts/synth_chinese_pretrain.py --n $N_PHASE1 --out '$ZH_OUT'"
    if [[ -f "$WIKI_EN" ]]; then
        CMD_A="$CMD_A && $PREFIX python scripts/mix_pretrain_corpora.py --corpora '$ZH_OUT,$WIKI_EN' --out '$MIX_OUT'"
    else
        log "(a) note: $WIKI_EN not found, will only synth zh — mix step will be re-run later"
    fi
    spawn "synth_zh_mix" "$CMD_A"
fi

# ---------------------------------------------------------------------------
# Job (b): tokenize alpaca-zh SFT data for phase 3
# ---------------------------------------------------------------------------

ALPACA_IN="$DATA_DIR/alpaca_zh.json"
ALPACA_OUT="$DATA_DIR/alpaca_zh_qwen_tokenized.parquet"

if [[ -f "$ALPACA_OUT" ]]; then
    log "(b) skip: $ALPACA_OUT already exists"
elif [[ ! -f "$ALPACA_IN" ]]; then
    log "(b) skip: $ALPACA_IN not found (run scripts/download_alpaca.sh first)"
else
    CMD_B="$PREFIX python scripts/prep_alpaca_qwen.py --in '$ALPACA_IN' --out '$ALPACA_OUT'"
    spawn "alpaca_tokenize" "$CMD_B"
fi

# ---------------------------------------------------------------------------
# Job (c): heavy bench every 30 min on the latest ckpt
# ---------------------------------------------------------------------------

BENCH_LOOP_FLAG="$RUN_DIR/.cpu_bench_loop_started"
if [[ -f "$BENCH_LOOP_FLAG" ]]; then
    # Check whether the process is still alive
    bench_pid=$(cat "$BENCH_LOOP_FLAG" 2>/dev/null || echo "")
    if [[ -n "$bench_pid" ]] && kill -0 "$bench_pid" 2>/dev/null; then
        log "(c) skip: bench loop already running pid=$bench_pid"
    else
        log "(c) bench loop flag exists but pid dead, restarting"
        rm -f "$BENCH_LOOP_FLAG"
    fi
fi

if [[ ! -f "$BENCH_LOOP_FLAG" ]]; then
    BENCH_LOOP_CMD="while true; do \
        latest=\$(ls -t $RUN_DIR/step_*.pt 2>/dev/null | head -1); \
        if [[ -n \"\$latest\" ]]; then \
            $PREFIX python scripts/auto_eval_daemon.py \
                --bench-heavy mmlu_full,hellaswag_full --once \
                --ckpt \"\$latest\" \
                --watch $RUN_DIR \
                --tokenizer $TEACHER || true; \
        fi; \
        sleep 1800; \
    done"
    spawn "bench_loop" "$BENCH_LOOP_CMD"
    # Capture the most recent pid we spawned and persist it
    pgrep -f "auto_eval_daemon.*--once" | tail -n 1 > "$BENCH_LOOP_FLAG" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Job (d): pre-warm Qwen tokenizer cache to /dev/shm
# ---------------------------------------------------------------------------

TOK_PKL="/dev/shm/qwen_tok.pkl"
if [[ -f "$TOK_PKL" ]]; then
    log "(d) skip: $TOK_PKL already cached"
else
    PREWARM_PY=$(cat <<'PY'
import os, pickle, sys
try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not installed", file=sys.stderr)
    sys.exit(0)
teacher = os.environ.get("TEACHER", "Qwen/Qwen2.5-0.5B")
out = os.environ.get("TOK_PKL", "/dev/shm/qwen_tok.pkl")
tok = AutoTokenizer.from_pretrained(teacher, use_fast=True, trust_remote_code=True)
with open(out, "wb") as f:
    pickle.dump({"name_or_path": teacher, "tok": tok}, f)
print(f"cached {teacher} -> {out}", flush=True)
PY
    )
    CMD_D="TEACHER='$TEACHER' TOK_PKL='$TOK_PKL' $PREFIX python -c \"$PREWARM_PY\""
    spawn "prewarm_tok" "$CMD_D"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log "================================================================"
log "all jobs spawned. tail logs:"
log "  tail -f $LOG"
log "  tail -f $RUN_DIR/cpu_job_*.log"
log "  pgrep -af 'synth_chinese_pretrain|mix_pretrain|prep_alpaca_qwen|auto_eval_daemon|qwen_tok'"
log "================================================================"
