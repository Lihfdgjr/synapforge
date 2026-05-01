#!/bin/bash
# launch_plan_c_cpu.sh — Plan C insurance LoRA train, CPU-only, parallel to GPU trainer.
#
# Why this exists (P5 in MASTER_PLAN.md §6):
#   The rental A800 is 100% occupied by Synap-1 (PID 18653, train_100m_kd.py).
#   Plan C is the v0 demo frontend (Qwen 0.5B + LoRA) — insurance if Synap-1
#   doesn't reach phase 3 (ppl ≤ 60). CPU on rental is mostly idle (5/110 GB
#   RAM, 14 cores Xeon). We can run Plan C on CPU in parallel without
#   touching the GPU trainer at all.
#
# Resource discipline:
#   - CUDA_VISIBLE_DEVICES="" forces CPU. (Trainer auto-detects via
#     torch.cuda.is_available(); it has no --device flag.)
#   - taskset -c 8-13 pins to CPU cores 8-13. Cores 0-7 stay free for the
#     GPU trainer's dataloader workers / NCCL / Python overhead.
#   - nice -n 19 + ionice -c 3 (idle) — yields CPU and disk instantly when
#     anything else (especially GPU trainer dataloader) wakes up.
#   - OMP_NUM_THREADS=6 + MKL_NUM_THREADS=6 cap intra-op parallelism to the
#     6 cores we own.
#
# Process discipline:
#   - setsid + disown so MCP shell exit doesn't kill us. (See
#     feedback_mcp_nohup_hangs_use_systemd_run.md — MCP kills nohup'd kids.)
#   - All output goes to $RUN_DIR/train.log. Foreground prints PID + path
#     and exits in <1s.
#
# DO NOT RUN until operator decides:
#   - Either (a) GPU trainer phase 3 has missed its window, OR
#   - (b) GPU trainer crashed and no recovery is possible.
#
# Memory cross-refs:
#   docs/PLAN_C_RUNBOOK.md (4-step verification flow)
#   docs/PLAN_C_CPU_NOTES.md (this script's ETA + decision tree)
#   docs/MASTER_PLAN.md §6 P5 (tracking item)
#   feedback_mcp_remote_ssh_quirks.md (why setsid+disown and not nohup)
#
# Usage on rental:
#   bash /workspace/synapforge_git/scripts/launch_plan_c_cpu.sh

set -u

# ---------- config (override via env) ----------
RUN_DIR="${RUN_DIR:-/workspace/runs/plan_c_cpu}"
DATA_PARQUET="${DATA_PARQUET:-/workspace/data/alpaca_zh_qwen_tokenized.parquet}"
QWEN_BASE="${QWEN_BASE:-/workspace/teachers/qwen2.5-0.5b}"
SCRIPT_DIR="${SCRIPT_DIR:-/workspace/synapforge_git/scripts}"
STEPS="${STEPS:-200}"
BS="${BS:-4}"
MAX_SEQ="${MAX_SEQ:-512}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-2e-4}"
CORES="${CORES:-8-13}"
THREADS="${THREADS:-6}"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"

# ---------- preflight (cheap, no model load) ----------
if [ ! -f "$DATA_PARQUET" ]; then
  echo "FATAL: data parquet missing: $DATA_PARQUET" | tee -a "$LOG"
  exit 2
fi
if [ ! -d "$QWEN_BASE" ]; then
  echo "FATAL: Qwen base dir missing: $QWEN_BASE" | tee -a "$LOG"
  exit 2
fi
if [ ! -f "$SCRIPT_DIR/train_qwen_lora.py" ]; then
  echo "FATAL: trainer missing: $SCRIPT_DIR/train_qwen_lora.py" | tee -a "$LOG"
  exit 2
fi

# Don't accidentally clobber a successful run.
if [ -f "$RUN_DIR/final.pt" ]; then
  echo "WARNING: $RUN_DIR/final.pt already exists. Aborting to avoid clobber." | tee -a "$LOG"
  echo "         Move the old run aside or set RUN_DIR=/workspace/runs/plan_c_cpu_$(date +%s)" | tee -a "$LOG"
  exit 3
fi

# Sanity: GPU trainer should still be alive. Plan C only runs *alongside* it.
if ! pgrep -fa "train_100m_kd.py" >/dev/null; then
  echo "WARNING: GPU trainer (train_100m_kd.py) not found running." | tee -a "$LOG"
  echo "         If GPU trainer is dead and you want Plan C as the *only* demo," | tee -a "$LOG"
  echo "         consider running on GPU instead (rm CUDA_VISIBLE_DEVICES line)." | tee -a "$LOG"
  echo "         Continuing anyway — Plan C is fine alongside or alone." | tee -a "$LOG"
fi

# ---------- launch ----------
echo "[plan-c-cpu $(date)] launching" | tee -a "$LOG"
echo "  RUN_DIR     = $RUN_DIR" | tee -a "$LOG"
echo "  DATA        = $DATA_PARQUET" | tee -a "$LOG"
echo "  QWEN_BASE   = $QWEN_BASE" | tee -a "$LOG"
echo "  STEPS       = $STEPS" | tee -a "$LOG"
echo "  BS          = $BS" | tee -a "$LOG"
echo "  MAX_SEQ     = $MAX_SEQ" | tee -a "$LOG"
echo "  LORA r/a    = $LORA_R / $LORA_ALPHA" | tee -a "$LOG"
echo "  CORES       = $CORES (taskset)" | tee -a "$LOG"
echo "  THREADS     = $THREADS (OMP/MKL)" | tee -a "$LOG"

# Build the inner command. taskset may fail in some rental containers (cap
# CAP_SYS_NICE missing) — fall through to no-affinity if it does.
INNER="env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  TOKENIZERS_PARALLELISM=false \
  nice -n 19 ionice -c 3 \
  python3 $SCRIPT_DIR/train_qwen_lora.py \
    --base-path $QWEN_BASE \
    --data $DATA_PARQUET \
    --output-dir $RUN_DIR \
    --steps $STEPS \
    --bs $BS \
    --max-seq $MAX_SEQ \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lr $LR \
    --warmup 50 \
    --log-every 10 \
    --sample-every 50 \
    --save-every 100"

setsid bash -c "
  if taskset -c $CORES true 2>/dev/null; then
    taskset -c $CORES bash -c \"$INNER\" >> '$LOG' 2>&1
  else
    echo '[plan-c-cpu] taskset denied; running without affinity' >> '$LOG'
    bash -c \"$INNER\" >> '$LOG' 2>&1
  fi
" </dev/null >/dev/null 2>&1 &
disown

CHILD_PID=$!
sleep 1
echo "[plan-c-cpu] spawned PID=$CHILD_PID; logs -> $LOG" | tee -a "$LOG"
echo "[plan-c-cpu] tail -f $LOG  # to follow"
echo "[plan-c-cpu] pkill -f train_qwen_lora.py  # to stop"
exit 0
