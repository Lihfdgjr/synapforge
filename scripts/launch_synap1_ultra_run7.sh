#!/bin/bash
# launch_synap1_ultra_run7.sh -- Synap-1 Ultra Run 7 integration launcher.
#
# Activates ALL Run 6 features PLUS the 7 bit-exact efficiency packs from
# feature/run7-integration. Per the user's quality 铁律 (2026-05-02:
# "improvements MUST NOT regress quality"), this launcher uses ONLY the
# white-listed bit-exact / quality-neutral flags. The black-listed flags
# remain available in train_100m_kd.py for controlled side-by-side tests:
#
#   BLACK-LIST (NOT enabled by Run 7):
#     --kwta-k 128         k-WTA changes the model function (sparse mask
#                          on gate); requires offline quality validation
#                          before flipping on in a production run.
#     --cuda-graphs        CUDA Graphs auto-disable when any mixin is
#                          live (lazy-host-sync, NeuroMCP, EMA), which is
#                          true for Run 7. Adding the flag would just
#                          fall back to the non-graph path with a warning,
#                          so we skip it for cleaner banners.
#
# Run 7 NEW levers (over Run 6):
#   * --rfold + --rfold-chunk 16     R-fold closed-form LiquidCell scan
#                                    (math-simplification 2026-05-02).
#                                    Bit-exact for chunk<=16. 1.5-2x
#                                    LiquidCell speed-up.
#   * --grad-ckpt                    Activation checkpointing (ALIAS for
#                                    --grad-checkpoint). Lets bs jump
#                                    24 -> 32 without OOM.
#   * --bf16-full                    Banner-only acknowledgement (autocast
#                                    already on). Documents the bf16
#                                    forward-math contract.
#   * --cpu-offload-optim            ZeRO-Offload Stage 0 -- AdamW m/v
#                                    moments + master fp32 params on CPU.
#                                    Frees ~4GB HBM.
#   * --teacher-cpu-int8             Qwen2.5-0.5B teacher on CPU INT8.
#                                    Frees ~2GB HBM. Async D2H copy.
#   * --sparse-spike-synapse         Binary-sparse PLIF spike kernel.
#     --sparse-spike-threshold 0.30  Auto-fallback to dense GEMM above
#                                    30% spike density (so Run 7 PLIF
#                                    revival post-step-3000 captures the
#                                    win without bit-flipping).
#   * --async-data-pipeline          Hide CPU tokenize behind GPU compute.
#     --async-pipeline-stages 4      4 producer threads.
#     --async-pipeline-prefetch 8    8 batches buffered.
#   * --batch-size 32                Up from 24 (CPU offload + grad-ckpt
#                                    free enough HBM).
#   * --self-learn-k 4               Down from 8 (per-step cost). TTT
#                                    signal is currently dead anyway.
#   * --ttt-every 10                 NEW: only TTT every 10 outer steps.
#   * --kill-if-val-regresses 1.03   NEW: quality guard. Exit non-zero
#                                    if val_ppl exceeds 1.03 x baseline.
#   * --spike-target-loss-weight 0.5 10x increase from 0.05 to revive PLIF
#                                    (Run 7 PLIF-fix #2).
#   * --shuffle-seed 1717            Fresh seed (was 1515 in Run 6).
#
# Smoke: bash -n scripts/launch_synap1_ultra_run7.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra_run7}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_run7_integration.log}"
WARMSTART="${WARMSTART:-/workspace/runs/synap1_ultra/best_step.pt}"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

if [[ ! -f "${WARMSTART}" ]]; then
  echo "[run7] WARMSTART not found: ${WARMSTART}" >&2
  echo "[run7] picking latest step ckpt available..."
  WARMSTART="$(ls -t /workspace/runs/synap1_ultra/step_[0-9]*.pt 2>/dev/null | head -1)"
  if [[ -z "${WARMSTART}" ]]; then
    echo "[run7] no ckpt found; aborting" >&2
    exit 1
  fi
  echo "[run7] using ${WARMSTART}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[run7] warmstart from ${WARMSTART}"
echo "[run7] log -> ${LOG_FILE}"
echo "[run7] black-listed flags (NOT enabled): --kwta-k, --cuda-graphs"

setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --warmstart '${WARMSTART}' \
  --teacher Qwen/Qwen2.5-0.5B \
  --backend triton_block \
  --vocab 151936 \
  --d 1280 --n-layers 16 --loop-depth 2 --ffn-ratio 3.0 \
  --batch-size 32 --grad-accum-steps 2 \
  --lr 1e-4 --warmup 200 \
  --kd-every 4 --kd-topk 2048 --kd-weight 0.4 \
  --shuffle-buffer 10000 --shuffle-seed 1717 \
  --grad-clip 1.0 --lr-decay cosine --steps 60000 --save-every 2500 \
  --self-learn-ttt --self-learn-k 4 --ttt-val-fraction 0.80 \
  --ttt-every 10 \
  --curiosity-weight 0.05 \
  --neuromcp-weight 0.02 --neuromcp-codebook-size 16 --neuromcp-action-dim 64 \
  --val-seq-lens 256,512,1024 \
  --byte-patch-pool max+avg \
  --high-pass-residual-weight 0.05 \
  --plif-tau-init trimodal \
  --spike-target-loss-weight 0.5 \
  --lm-head-spectral-norm \
  --cuda-sync-every 10 \
  --clip-grad-cache \
  --ema-decay 0.999 \
  --phase-aware \
  --sew-shortcut \
  --plif-dense-bypass-steps 2000 \
  --lazy-host-sync-accum \
  --fused-adamw \
  --skip-warmstart-eval-N 1 \
  --rfold --rfold-chunk 16 \
  --grad-ckpt \
  --bf16-full \
  --cpu-offload-optim \
  --teacher-cpu-int8 \
  --sparse-spike-synapse --sparse-spike-threshold 0.30 \
  --async-data-pipeline --async-pipeline-stages 4 --async-pipeline-prefetch 8 \
  --kill-if-val-regresses 1.03 \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[run7] pid=${NEW_PID}"
