#!/bin/bash
# Launch Synap-Mini (~30M LNN+SNN, fast training) as the small-model insurance
# path for investor demo / paper.  See docs/INSURANCE_NATIVE.md Option A and
# docs/ANTI_LORA.md for why this is the only "smaller fallback" path we accept
# (still 100% LNN+SNN -- no LoRA-on-transformer).
#
# Architecture (must satisfy ANTI_LORA constraints -- no transformer):
#   vocab        151936  (Qwen vocab; same tokenizer as Synap-1)
#   d            256     (hidden size; ~1/2 of Synap-1's 512)
#   n_layers     6       (HybridBlock count; ~3/5 of Synap-1's 10)
#   loop_depth   1       (Ouro recursion -- keep at 1 for fast train)
#   ffn_ratio    4       (FFN expansion; smaller than Synap-1's 8)
#
# Param breakdown at vocab=151936 / d=256 / n_layers=6 / ffn_ratio=4:
#   tok_embed + lm_head (tied):       151936 *  256 ~  39M  (53% of total)
#   per-layer (ATT block + FFN + PLIF + LayerNorm) at d=256 ffn=1024:
#                                     ~     1M each  *  6  ~  6M
#   total                             ~ 30-50M depending on residual heads
#
# Training budget:
#   bs=128, lr=2e-4, steps=30000, KD on Qwen2.5-0.5B (same teacher as Synap-1).
#   On A800-80GB this runs at ~30k tok/s, so 30k steps * 128 bs * 256 seq /
#   30k tok/s  ~= 6-12h wall time -- well inside an A800 rental window.
#   bs=128 fits because (B*T, V, fp32) = 128*256*151936*4 = ~19 GB even with
#   sparse z-loss, BUT chunked KD splits that across forwards; total peak
#   memory is < 50 GB (Synap-1 sweet spot was bs=80 due to teacher overhead).
#
# Decision criterion to actually launch this (see docs/INSURANCE_NATIVE.md
# §Option A "Decision criterion"):
#   * Synap-1 (100M) hasn't tripped phase 1 (val ppl <= 250) within 8h, AND
#   * rental still has 6+ GPU-h on the clock.
# Synap-Mini does not block Synap-1 GPU time -- bs=128 + small body fits in
# free VRAM even while Synap-1 is running, but in practice we launch on a
# separate rental window or after Synap-1 hits a save point.
#
# Acceptance: val ppl <= 80 on alpaca-zh-eval (= "weakly chat-coherent",
# not "GPT-grade").  This is honest about what 30M can do.
#
# Usage on rental:
#     setsid bash scripts/launch_synap_mini.sh > /workspace/runs/synap_mini/train.log 2>&1 &
#     disown
#
# Anti-stuck: per feedback_mcp_remote_ssh_quirks.md, MCP shell hangs on
# nohup; setsid+disown is the proven escape.  Returns to caller in <1s.

set -e
cd /workspace/synapforge_git

# Must be set or KD chunks blow up at vocab=151936; same as Synap-1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Output dir; mkdir -p so re-launch after crash doesn't fail.
OUT_DIR="${OUT_DIR:-/workspace/runs/synap_mini}"
mkdir -p "$OUT_DIR"

exec python3 -u train_100m_kd.py \
  --no-warmstart \
  --teacher Qwen/Qwen2.5-0.5B \
  --teacher-fallback-ckpt "" \
  --backend triton_block \
  --vocab 151936 \
  --d 256 \
  --n-layers 6 \
  --loop-depth 1 \
  --ffn-ratio 4 \
  --batch-size 128 \
  --lr 2e-4 \
  --steps 30000 \
  --kd-every 4 \
  --save-every 1000 \
  --eval-every 500 \
  --out "$OUT_DIR"
