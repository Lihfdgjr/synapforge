#!/bin/bash
# launch_synap1_ultra_run6.sh -- Synap-1 Ultra full-stack Run 6.
#
# Activates EVERY new feature shipped in the last 6 hours:
#   * SelfLearn (TTT)            -- proven helpful in Run 5
#   * Curiosity (ICM)            -- proven helpful in Run 5
#   * NeuroMCP (T9.2)            -- new this round, action loss + plasticity
#   * Multi-seq val (T9.3)       -- measure long-context monotonic quality
#   * Max-Pool byte-patch (A1)   -- NeurIPS 25 Fang et al. fix
#   * High-pass residual (A2)    -- Fang et al. fix to revive PLIF
#   * Tri-modal tau init (A3)    -- Fang et al. fix
#   * EMA weights (T8.4)         -- inference smoothing
#   * cuda-sync-every 10 (perf)  -- +3-6% throughput
#   * clip-grad-cache (perf)     -- +0.5-1% throughput
#   * Spectral norm (T2.6)       -- z-loss bound
#   * Spike-target-loss (T2.5)   -- PLIF revival aux
#   * SEW shortcut (PLIF revive) -- fix 16/16 dead PLIF in lite_mixin
#   * dense bypass 2000 steps    -- warmup window for PLIF gradient
#   * lazy-host-sync + fused-adamw + skip-1st-eval -- 3-flag speedup pack
#
# Resumes from latest Run 5 ckpt (step_010000.pt or whatever's freshest).
#
# WARNING: combining 8+ aux loss terms can confuse small models. We damp via
# small weights: NeuroMCP 0.02, curiosity 0.05, modal 0.0 (no synth data yet).
# Self-learn is OK because it's eval-time-only weight restore.
#
# Smoke: bash -n scripts/launch_synap1_ultra_run6.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_run6_fullstack.log}"
WARMSTART="${WARMSTART:-${RUN_DIR}/step_010000.pt}"

cd "${REPO_DIR}"

if [[ ! -f "${WARMSTART}" ]]; then
  echo "[run6] WARMSTART not found: ${WARMSTART}" >&2
  echo "[run6] picking latest step ckpt available..."
  WARMSTART="$(ls -t "${RUN_DIR}"/step_[0-9]*.pt 2>/dev/null | head -1)"
  if [[ -z "${WARMSTART}" ]]; then
    echo "[run6] no ckpt found; aborting" >&2
    exit 1
  fi
  echo "[run6] using ${WARMSTART}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[run6] warmstart from ${WARMSTART}"
echo "[run6] log -> ${LOG_FILE}"

setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --warmstart '${WARMSTART}' \
  --teacher Qwen/Qwen2.5-0.5B \
  --backend triton_block \
  --vocab 151936 \
  --d 1280 --n-layers 16 --loop-depth 2 --ffn-ratio 3.0 \
  --batch-size 24 --grad-accum-steps 2 \
  --lr 1e-4 --warmup 200 \
  --kd-every 4 --kd-topk 2048 --kd-weight 0.4 \
  --shuffle-buffer 10000 --shuffle-seed 1515 \
  --grad-clip 1.0 --lr-decay cosine --steps 60000 --save-every 2500 \
  --self-learn-ttt --self-learn-k 8 --ttt-val-fraction 0.80 \
  --curiosity-weight 0.05 \
  --neuromcp-weight 0.02 --neuromcp-codebook-size 16 --neuromcp-action-dim 64 \
  --val-seq-lens 256,512,1024 \
  --byte-patch-pool max+avg \
  --high-pass-residual-weight 0.05 \
  --plif-tau-init trimodal \
  --spike-target-loss-weight 0.05 \
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
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[run6] pid=${NEW_PID}"
