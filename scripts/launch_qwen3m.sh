#!/bin/bash
# launch_qwen3m.sh -- Run 3m: T2.3 + T2.5 + T2.6 dead-PLIF revival package.
#
# DOES NOT START AUTOMATICALLY. User invokes manually OR phase autopilot
# triggers via scripts/build_next_launch.py when val ppl crosses 250.
#
# Build context (2026-05-02):
#   * Run 3l (currently running) was launched WITHOUT T2.3/T2.5/T2.6 fixes
#     because they landed after launch. Run 3l VAL trajectory may still
#     recover; this script is the pre-staged next-gen launch ready to fire.
#   * T2.3 (commit 240624c): surrogate gradient width anneal 10 -> 1 over
#     5000 steps. PLIFCell.surrogate_width buffer divides alpha at forward
#     time so width=10 gives a wide / smooth surrogate that can flow
#     gradient through dead PLIF cells (P25 dead 10/10).
#   * T2.5 (commit 2b086ec / 34eb832): spike-rate-target auxiliary loss.
#     Quadratic penalty when PLIF spike rate falls outside [low, high]
#     band. Backprops through surrogate gradient to threshold/log_tau.
#     Default weight 0.001; we use 0.05 (50x default, 5x relative to "5x"
#     phrasing in spec) for aggressive PLIF revival.
#   * T2.6 (commit 16f5de5): LM head spectral norm. Bounds Lipschitz so
#     logsumexp(logits) -- the z-loss term -- cannot drift unboundedly
#     during long training. Default OFF; opt-in here. T2.6 docstring lists
#     ONE caveat ("bf16 quirks: power-iter buffers stay fp32") -- this is
#     PyTorch's standard spectral_norm behaviour and is NOT incompatible
#     with --backend triton_block or --kd-topk 2048. Tied weights case
#     wraps tok_embed which is the live path.
#
# New-vs-3l flags (sorted by appearance below):
#   --warmstart                LATEST=$(ls -t step_[0-9]*.pt | head -1)  (was step_002250_plif_reinit.pt)
#   --kd-topk            2048  (proven safe per Run 3i; matches argparse default)
#   --shuffle-seed       411   (was 311 in 3l; rotate to avoid identical
#                              data ordering of the prior run -- belt &
#                              suspenders on top of P24 shuffle-buffer 10000)
#   --spike-target-loss-weight 0.05   (T2.5; default is 0.001 = 50x dial up)
#   --surrogate-anneal-start   10.0   (T2.3)
#   --surrogate-anneal-target  1.0    (T2.3)
#   --surrogate-anneal-steps   5000   (T2.3; 0 disables)
#   --lm-head-spectral-norm    (T2.6 boolean opt-in)
#
# T2.2 (--triton-fused-backward) is NOT enabled here. It is currently a
# stub (commit 5a3ecef, formal kernel 3dab79c on a separate worktree).
# Enable it only after main lands the kernel + it passes 7-test pytest
# suite on A800.
#
# Use systemd-run --user OR setsid+disown to survive MCP shell closure
# (per CLAUDE.md memory feedback_mcp_nohup_hangs_use_systemd_run +
# feedback_mcp_remote_ssh_quirks). NO nohup.
#
# Smoke: bash -n scripts/launch_qwen3m.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/v24h_qwen3}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_run3m.log}"

cd "${REPO_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# 2026-05-02 00:55 -- Run 3l diverged catastrophically (step 5000 val=1864,
# step 5500 val=2522). Per docs/RUN3L_DIAGNOSIS.md decision matrix, this is
# Run-3c-class divergence; relaunch at lower LR + longer warmup + lower KD
# alternation frequency. Warmstart explicitly from step_002000.pt (the last
# known-good ckpt; step_004000.pt is already on the divergent trajectory at
# val=569).
WARMSTART_CKPT="${WARMSTART_CKPT:-${RUN_DIR}/step_002000.pt}"
if [[ ! -f "${WARMSTART_CKPT}" ]]; then
  echo "[launch_qwen3m] ERROR: warmstart ckpt missing: ${WARMSTART_CKPT}" >&2
  exit 1
fi
echo "[launch_qwen3m] warmstart from: ${WARMSTART_CKPT} (last known-good)"
echo "[launch_qwen3m] log -> ${LOG_FILE}"

# setsid + disown -- survives MCP ssh channel close.
setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --warmstart '${WARMSTART_CKPT}' \
  --teacher Qwen/Qwen2.5-0.5B \
  --teacher-fallback-ckpt '' \
  --backend triton_block \
  --batch-size 64 \
  --lr 5e-5 \
  --warmup 500 \
  --kd-every 8 \
  --kd-topk 2048 \
  --shuffle-buffer 10000 \
  --shuffle-seed 411 \
  --grad-clip 0.5 \
  --spike-target-loss-weight 0.05 \
  --surrogate-anneal-start 10.0 \
  --surrogate-anneal-target 1.0 \
  --surrogate-anneal-steps 5000 \
  --lm-head-spectral-norm \
  --lr-decay cosine \
  --steps 30000 \
  --out '${RUN_DIR}' \
  --phase-aware \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[launch_qwen3m] pid=${NEW_PID}"
