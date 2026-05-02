#!/bin/bash
# launch_synap1_ultra.sh -- Synap-1 Ultra 500M from-scratch launch.
#
# DOES NOT START AUTOMATICALLY. User explicitly skipped Pro and is going
# straight to Ultra. Wait for any active Run 3o (100M) / Synap-1 Pro run
# to terminate before invoking. Ultra is a separate next-gen run with
# d=1280 / n_layers=16 / ffn_ratio=3.0 / loop_depth=2 (vs Pro's
# d=1024 / n=14 / ffn=2.5 / loop=2).
#
# Architecture context (sister agent ships configs/synap1.py SYNAP1_ULTRA):
#   Ultra shape:  vocab=151936 / d=1280 / n_layers=16 / loop_depth=2 / ffn=3.0
#   Ultra params: ~535.8M total; tok_embed eats ~194.5M (vocab*d), so
#                 backbone is **~341.3M** of useful capacity vs ~25M in 100M
#                 (13.7x more), or ~2.0x Pro's ~169M backbone.
#   Teacher:      Qwen2.5-0.5B (494M).  student/teacher param ratio = 1.08
#                 (vs Pro's 0.61, 100M's 0.20).  Student is now SLIGHTLY
#                 LARGER than the teacher in totals (though backbone is
#                 still 0.71 of teacher full size).  At this ratio KD
#                 weight goes DOWN further (don't over-pull onto teacher
#                 manifold) and KD frequency stays at every-4 to keep the
#                 distillation pipe extracting useful signal without
#                 dominating the loss.  See docs/SCALING_PLAN.md SS5.
#
# WHY FROM SCRATCH (no warmstart):
#   Architecture changed (d=1024 -> 1280, n_layers=14 -> 16).  Old Pro
#   ckpts are dimensionally incompatible at every linear weight; loading
#   them with strict=False would only pick up a handful of name-matching
#   shapes (LayerNorm gains, scalar buffers).  The cost of "false-warmstart
#   bounce" (Run 3l at step 40 jumping ce 5 -> 9.1, see PROGRESS.md) is
#   worse than a clean random-init at this scale.  --no-warmstart is
#   mandatory.
#
# WHY EACH NON-DEFAULT FLAG (deltas vs Pro launch_synap1_pro.sh):
#   --batch-size 24       VRAM at d=1280 / n=16 jumps another ~50% over
#                         Pro's d=1024 / n=14 (per-token activation footprint
#                         scales as d * n_layers * ffn_ratio).  bs=32 (Pro
#                         setting) extrapolates to ~75GB on A800 80GB with
#                         KD on -- no headroom for the teacher fwd pass.
#                         bs=24 leaves ~10-15 GB headroom; if OOM, drop to
#                         16 + grad-accum-steps 3.
#   --grad-accum-steps 2  Effective batch size 48 (vs Pro's 32).  Compensates
#                         for the smaller per-step bs and gives a smoother
#                         gradient estimate at the higher-capacity model
#                         (Chinchilla: bigger model wants bigger effective
#                         batch).
#   --lr 2e-4             Pro ran at 3e-4.  Ultra's backbone is 2x bigger
#                         AND the embedding is 1.25x wider, so the gradient
#                         scale at lm_head + tok_embed is meaningfully
#                         higher.  Per Chinchilla / GPT-3 scaling LR should
#                         drop ~sqrt(2) at this size class -- 2e-4 is the
#                         conservative landing.  If train ce stays >7 past
#                         step 1500, escalate to 2.5e-4.
#   --warmup 1500         50% longer than Pro's 1000.  Bigger model + lower
#                         LR -> needs more steps for the optimizer state
#                         (Adam m_t / v_t) to stabilise before the cosine
#                         decay kicks in.  ~2.5% of the 60000-step budget.
#   --kd-every 4          Same as Pro.  Closer student/teacher capacity
#                         ratio (1.08 totals) means KD signal is per-step
#                         valuable; no reason to drop the frequency.
#   --kd-topk 2048        Same as Pro / 100M.  Sparse softmax keeps KD VRAM
#                         bounded regardless of student size.
#   --kd-weight 0.4       Pro used 0.5.  Ultra has its own representation
#                         power AT TEACHER SCALE -- over-weighting teacher
#                         logits at this ratio collapses the student onto
#                         teacher manifold and limits emergent quality.
#                         0.4 keeps CE dominant (0.6 weight on student CE)
#                         while still extracting the teacher's soft-label
#                         distribution.  Per DistilBERT/MiniLM the optimal
#                         alpha_kd at student >= teacher capacity is 0.3-0.5;
#                         0.4 is the midpoint.  See SCALING_PLAN SS5.
#   --shuffle-buffer 10000  P24 default; required to break data-ordering
#                         lockstep.  Same as Pro.
#   --shuffle-seed 1212   Rotate from Pro's 1011.  Fresh ordering eliminates
#                         any P24-style coupling with Pro / 100M training.
#   --grad-clip 1.0       Same as Pro.  GPT-3 / Llama default, appropriate
#                         for d>=1024 backbones.
#   --lr-decay cosine     Same as Pro / 100M.  Cosine to lr_min over the
#                         full step budget.
#   --steps 60000         Pro ran 50000.  Ultra has 2x backbone -- needs
#                         proportionally more optimization steps to reach
#                         its capacity.  60000 steps * ~0.85 s/step (est.
#                         based on Pro's 0.55 s/step at 1.55x activation
#                         cost) = ~14 h on A800.  Sub-linear because the
#                         triton_block kernel is bandwidth-bound.
#   --spike-target-loss-weight 0.05    PLIF revival aux loss; T2.5 default
#                         is 0.001 (50x dial up).  Same as Pro / Run 3m
#                         onward; keeps PLIF spike rate inside [0.05, 0.20].
#   --lm-head-spectral-norm    T2.6.  ENABLED FROM STEP 0 (vs Run 3m which
#                         tried to enable mid-run -- triggered weight_orig /
#                         weight_v / weight_u state-dict mismatch + lm-head
#                         reset, lost 5h.  See feedback_spectral_norm_warmstart_cost.)
#   --phase-aware         Same as Pro / 100M.  Trainer reads .phase signal file.
#
# T2.2 (--triton-fused-backward) NOT enabled -- still a stub on this branch.
#
# Use setsid + disown to survive MCP ssh channel close (per
# feedback_mcp_nohup_hangs_use_systemd_run + feedback_mcp_remote_ssh_quirks).
#
# Smoke: bash -n scripts/launch_synap1_ultra.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_synap1_ultra.log}"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[launch_synap1_ultra] from-scratch (no warmstart) -- architecture change d=1024->1280 n=14->16 ffn=2.5->3.0"
echo "[launch_synap1_ultra] log -> ${LOG_FILE}"

# setsid + disown -- survives MCP ssh channel close.
setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --no-warmstart \
  --teacher Qwen/Qwen2.5-0.5B \
  --teacher-fallback-ckpt '' \
  --backend triton_block \
  --vocab 151936 \
  --d 1280 \
  --n-layers 16 \
  --loop-depth 2 \
  --ffn-ratio 3.0 \
  --batch-size 24 \
  --grad-accum-steps 2 \
  --lr 2e-4 \
  --warmup 1500 \
  --kd-every 4 \
  --kd-topk 2048 \
  --kd-weight 0.4 \
  --shuffle-buffer 10000 \
  --shuffle-seed 1212 \
  --grad-clip 1.0 \
  --lr-decay cosine \
  --steps 60000 \
  --spike-target-loss-weight 0.05 \
  --lm-head-spectral-norm \
  --phase-aware \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[launch_synap1_ultra] pid=${NEW_PID}"
