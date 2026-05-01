#!/bin/bash
# Minimal CPU/RAM utilization for rental box — kept simple so it runs without
# git pull (rental network often times out to github.com).
#
# Fills idle CPU cores + idle RAM with USEFUL pre-work while GPU trainer runs:
#   (A) synth_chinese_pretrain.py  — 50K rows for phase 1 backbone
#   (B) prep_alpaca_qwen.py        — phase 3 SFT data, only if input exists
#   (C) Qwen tokenizer pre-warm    — to /dev/shm for next-restart speedup
#
# Each job runs as `setsid + nice -n 19 + ionice -c 3` so it yields CPU/IO
# instantly when trainer dataloader needs cores. `taskset -c 8-13` would be
# ideal but rental container often denies CPU affinity (Invalid argument);
# fall through silently when it does.
#
# Idempotent: skips if the output file already exists.
#
# Usage (on rental):
#   bash scripts/cpu_jobs_minimal.sh
# That's it. Logs to /workspace/runs/v24h_qwen3/cpu_jobs.log.
set -u
RUN_DIR="${RUN_DIR:-/workspace/runs/v24h_qwen3}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
TOK_PATH="${TOK_PATH:-/workspace/teachers/qwen2.5-0.5b}"
SCRIPTS="${SCRIPTS:-/workspace/synapforge_git/scripts}"
LOG="$RUN_DIR/cpu_jobs.log"
mkdir -p "$RUN_DIR"
echo "[cpu-jobs $(date +%T)] starting (RUN_DIR=$RUN_DIR)" >> "$LOG"

# --------- Job A: synthesize 50K Chinese pretrain rows ---------
if [ ! -f "$DATA_DIR/synth_zh_phase1.parquet" ]; then
  setsid bash -c "
    nice -n 19 ionice -c 3 taskset -c 8-11 \
      python3 $SCRIPTS/synth_chinese_pretrain.py \
      --n 50000 --out $DATA_DIR/synth_zh_phase1.parquet \
      >> $LOG 2>&1 \
    || nice -n 19 ionice -c 3 \
      python3 $SCRIPTS/synth_chinese_pretrain.py \
      --n 50000 --out $DATA_DIR/synth_zh_phase1.parquet \
      >> $LOG 2>&1
  " </dev/null >/dev/null 2>&1 &
  disown
  echo "[cpu-jobs] (A) synth_chinese_pretrain n=50K -> $DATA_DIR/synth_zh_phase1.parquet" >> "$LOG"
fi

# --------- Job B: pre-tokenize alpaca-zh for phase 3 SFT ---------
# Try several common locations for the JSON.
ALPACA_JSON=""
for cand in "$DATA_DIR/alpaca_zh.json" "$DATA_DIR/alpaca_zh/alpaca_zh.json" "$DATA_DIR/alpaca-zh/alpaca_zh.json"; do
  if [ -f "$cand" ]; then ALPACA_JSON="$cand"; break; fi
done
if [ -n "$ALPACA_JSON" ] && [ ! -f "$DATA_DIR/alpaca_zh_qwen_tokenized.parquet" ]; then
  setsid bash -c "
    nice -n 19 ionice -c 3 taskset -c 12-13 \
      python3 $SCRIPTS/prep_alpaca_qwen.py \
      --in $ALPACA_JSON \
      --out $DATA_DIR/alpaca_zh_qwen_tokenized.parquet \
      --tokenizer $TOK_PATH \
      >> $LOG 2>&1 \
    || nice -n 19 ionice -c 3 \
      python3 $SCRIPTS/prep_alpaca_qwen.py \
      --in $ALPACA_JSON \
      --out $DATA_DIR/alpaca_zh_qwen_tokenized.parquet \
      --tokenizer $TOK_PATH \
      >> $LOG 2>&1
  " </dev/null >/dev/null 2>&1 &
  disown
  echo "[cpu-jobs] (B) prep_alpaca_qwen $ALPACA_JSON -> alpaca_zh_qwen_tokenized.parquet" >> "$LOG"
fi

# --------- Job C: pre-warm Qwen tokenizer to /dev/shm ---------
if [ ! -f /dev/shm/qwen_tok.pkl ]; then
  python3 -c "
import pickle
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$TOK_PATH', trust_remote_code=True)
with open('/dev/shm/qwen_tok.pkl', 'wb') as f: pickle.dump(tok, f)
print('qwen tok cached to /dev/shm/qwen_tok.pkl')
" >> "$LOG" 2>&1
fi

echo "[cpu-jobs $(date +%T)] all jobs spawned" >> "$LOG"
