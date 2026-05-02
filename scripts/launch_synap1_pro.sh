#!/bin/bash
# launch_synap1_pro.sh -- Synap-1 Pro 300M from-scratch launch.
#
# DOES NOT START AUTOMATICALLY. Wait for Run 3o (100M baseline) to terminate
# before invoking. Run 3o continues on the 100M architecture; Pro is a
# separate next-gen run with d=1024 / n_layers=14 (vs 100M's d=512 / n=10).
#
# Architecture context (sister agent ships configs/synap1.py SYNAP1_PRO):
#   Pro shape:   vocab=151936 / d=1024 / n_layers=14 / loop_depth=1 / ffn_ratio=8
#   Pro params:  ~300M total; tok_embed eats ~155M (vocab*d), so backbone is
#                ~145M of useful capacity vs ~25M in 100M (5.8x more).
#   Teacher:     Qwen2.5-0.5B (494M).  student/teacher param ratio = 0.61 (vs
#                0.20 at 100M).  At this ratio the student is competitive in
#                capacity with the teacher and can absorb more nuanced
#                distribution -- so KD weight goes DOWN (don't over-pull) and
#                KD frequency goes UP (extract more signal).  See
#                docs/SYNAP1_PRO_PLAN.md §KD-math (DistilBERT, MiniLM).
#
# WHY FROM SCRATCH (no warmstart):
#   Architecture changed (d=512 -> 1024, n_layers=10 -> 14).  Old 100M ckpts
#   are dimensionally incompatible at every linear weight; loading them with
#   strict=False would only pick up a handful of name-matching shapes
#   (LayerNorm gains, scalar buffers).  The cost of "false-warmstart bounce"
#   (Run 3l at step 40 jumping ce 5 -> 9.1, see PROGRESS.md) is worse than a
#   clean random-init at this scale.  --no-warmstart is mandatory.
#
# WHY EACH NON-DEFAULT FLAG (deltas vs 100M Run 3n/3o defaults):
#   --batch-size 32       VRAM at d=1024 / n=14 doubles per-token activation
#                         vs d=512 / n=10.  bs=64 would push past 80 GB on
#                         A800 with KD on.  bs=32 leaves ~10 GB headroom.
#                         If OOM, drop to 24 or set grad-accum-steps 2.
#   --lr 3e-4             100M ran at 5e-5 (Run 3m onward) because the small
#                         backbone diverged at higher LR.  Pro's 5.8x bigger
#                         backbone has a much smoother loss surface and per
#                         GPT-3 / Chinchilla scaling can take 6x higher LR.
#                         3e-4 is the canonical AdamW-pretrain default.
#   --warmup 1000         5x the 100M warmup (200).  Bigger model has more
#                         capacity for the optimizer to mis-step in the first
#                         few hundred updates; longer warmup runway dampens
#                         that.  1000 ~ 2% of the 50000-step budget, matches
#                         GPT-2 / Chinchilla recipe.
#   --kd-every 4          100M used 8.  Closer student/teacher capacity ratio
#                         (0.61 vs 0.20) means KD signal is more useful per
#                         step -- worth paying ~12% extra step time for it.
#   --kd-topk 2048        Same as 100M.  Sparse softmax keeps KD VRAM bounded
#                         regardless of student size; no reason to change.
#   --kd-weight 0.5       100M used 0.7 (default).  Pro has its own
#                         representation power -- over-weighting teacher
#                         logits at this capacity ratio collapses the student
#                         onto teacher manifold and limits emergent quality.
#                         0.5 keeps CE and KD balanced.  See SYNAP1_PRO_PLAN
#                         §KD-math.
#   --shuffle-buffer 10000  P24 default; required to break data-ordering
#                         lockstep.  Same as 100M.
#   --shuffle-seed 1011   Rotate from 100M's last seed (911 in Run 3o).
#                         Fresh ordering eliminates any P24-style coupling.
#   --grad-clip 1.0       100M used 0.5.  Bigger model has bigger natural
#                         gradient norm; 0.5 over-clips and slows
#                         convergence.  1.0 is GPT-3 / Llama default.
#   --lr-decay cosine     Same as 100M.
#   --steps 50000         100M plateaued around step 30000 at val ~3700.
#                         Pro has 5.8x backbone -- needs proportionally more
#                         optimization steps to reach its capacity.  50000
#                         steps * ~0.55s/step = 7.5h on A800.
#   --spike-target-loss-weight 0.05    PLIF revival aux loss; T2.5 default
#                         is 0.001 (50x dial up).  Same as Run 3m onward;
#                         keeps PLIF spike rate inside [0.05, 0.20].
#   --lm-head-spectral-norm    T2.6.  ENABLED FROM STEP 0 (vs Run 3m which
#                         tried to enable mid-run -- triggered weight_orig /
#                         weight_v / weight_u state-dict mismatch + lm-head
#                         reset, lost 5h.  See feedback_spectral_norm_warmstart_cost.)
#   --phase-aware         Same as 100M.  Trainer reads .phase signal file.
#
# T2.2 (--triton-fused-backward) NOT enabled -- still a stub on this branch.
#
# Use setsid + disown to survive MCP ssh channel close (per
# feedback_mcp_nohup_hangs_use_systemd_run + feedback_mcp_remote_ssh_quirks).
#
# Smoke: bash -n scripts/launch_synap1_pro.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_pro}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_synap1_pro.log}"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[launch_synap1_pro] from-scratch (no warmstart) -- architecture change d=512->1024 n=10->14"
echo "[launch_synap1_pro] log -> ${LOG_FILE}"

# setsid + disown -- survives MCP ssh channel close.
setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --no-warmstart \
  --teacher Qwen/Qwen2.5-0.5B \
  --teacher-fallback-ckpt '' \
  --backend triton_block \
  --vocab 151936 \
  --d 1024 \
  --n-layers 14 \
  --loop-depth 1 \
  --ffn-ratio 8 \
  --batch-size 32 \
  --lr 3e-4 \
  --warmup 1000 \
  --kd-every 4 \
  --kd-topk 2048 \
  --kd-weight 0.5 \
  --shuffle-buffer 10000 \
  --shuffle-seed 1011 \
  --grad-clip 1.0 \
  --lr-decay cosine \
  --steps 50000 \
  --spike-target-loss-weight 0.05 \
  --lm-head-spectral-norm \
  --phase-aware \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[launch_synap1_pro] pid=${NEW_PID}"
