#!/bin/bash
# launch_synap1_ultra_full_mixin.sh -- Synap-1 Ultra 500M with ALL MIXINS ON.
#
# User explicit decision 2026-05-02 10:50: stop gating self-learn / curiosity /
# multimodal behind val-ppl thresholds. Enable from step 0 with whatever VAL
# happens to be. The phase autopilot's "val<=250 -> phase 1" gate was strangling
# the headline features behind a barrier this 100% LNN+SNN small model would
# never cross on raw LM data alone (per ETA analysis: ppl 50 floor without
# SFT). Just train everything together; let the optimizer figure it out.
#
# Resumes from step_002500.pt (last saved Ultra ckpt before disk-full crash).
#
# Mixin flags ON (vs vanilla launch_synap1_ultra.sh):
#   --self-learn-ttt              T8.x test-time-training inner loop on val rows
#                                 mod 5 in [0,1,2,3] (holdout = mod 5 = 4 stays
#                                 honest). Trains intrinsic.SelfLearnHead each
#                                 step on a tiny 8-step inner loop.
#   --self-learn-k 8              8 inner steps per outer step (cheap; ~5%
#                                 throughput hit, +real continual-learning).
#   --ttt-val-fraction 0.80       80% of val data goes to TTT replay, 20%
#                                 reserved for honest eval. Already default.
#   --curiosity-weight 0.05       intrinsic.CuriosityModule ICM-style forward+
#                                 inverse loss on hidden state h_prev/h_next.
#                                 0.05 keeps signal subordinate to LM but
#                                 graph-attached so it backprops through CfC.
#   --modal-list image,audio,time_series   Three byte-patch modalities. Each
#                                 reuses the Qwen embed by mapping bytes ->
#                                 raw token IDs via a small adapter. Generated
#                                 data via scripts/synth_image_pretrain.py +
#                                 synth_audio_pretrain.py + synth_timeseries.
#   --modal-data-dir /workspace/data/modal/    where adapter loads patches
#   --modal-alpha 0.10            modal contrastive loss weight; 10% of LM
#                                 signal so text remains primary.
#
# Risks (real, not theoretical):
#   - VRAM: each mixin adds 0.5-1 GB. Currently 60.5 GB / 80 GB on Ultra
#     vanilla -> with all mixins might hit 65-68 GB. Drop bs to 20 if OOM.
#   - Throughput: TTT inner loop = +5-10%; curiosity hidden-state Jacobian
#     + 5%; modal contrastive +3-5%. Total ~15-20% slower wall clock.
#   - Convergence: combining 5 loss terms can confuse a small model.
#     Mitigation: weights chosen so LM-CE dominates (KD 40% + LM 60% within
#     the base loss, then modal_alpha 0.10 + curiosity 0.05 are aux). If
#     val ppl rises by >50% in 1000 steps, kill and revert.
#
# Backup discipline (post step_002500 disk-full lesson):
#   --save-every 2000   Same as resume launch. ckpt every 2000 steps not 250.
#   Sister daemon (PID from earlier setsid loop) does sync-then-cleanup.
#
# Smoke: bash -n scripts/launch_synap1_ultra_full_mixin.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_synap1_ultra_fullmix.log}"
WARMSTART="${WARMSTART:-${RUN_DIR}/step_002500.pt}"
MODAL_DATA_DIR="${MODAL_DATA_DIR:-/workspace/data/modal}"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}" "${MODAL_DATA_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[launch_ultra_fullmix] warmstart from: ${WARMSTART}"
echo "[launch_ultra_fullmix] modal data dir: ${MODAL_DATA_DIR}"
echo "[launch_ultra_fullmix] log -> ${LOG_FILE}"

# Generate synth modal data if missing (idempotent; agents already shipped
# the synth scripts but they need to be RUN to produce parquet files).
if [ ! -f "${MODAL_DATA_DIR}/synth_image_50k.parquet" ]; then
  echo "[launch_ultra_fullmix] generating synth image data (50K)..."
  python3 scripts/synth_image_pretrain.py --n 50000 --out "${MODAL_DATA_DIR}/synth_image_50k.parquet" 2>&1 | tail -5 || echo "(image synth failed, continuing without)"
fi
if [ ! -f "${MODAL_DATA_DIR}/synth_audio_50k.parquet" ]; then
  echo "[launch_ultra_fullmix] generating synth audio data (50K)..."
  python3 scripts/synth_audio_pretrain.py --n 50000 --out "${MODAL_DATA_DIR}/synth_audio_50k.parquet" 2>&1 | tail -5 || echo "(audio synth failed, continuing without)"
fi
if [ ! -f "${MODAL_DATA_DIR}/synth_timeseries_100k.parquet" ]; then
  echo "[launch_ultra_fullmix] generating synth timeseries data (100K)..."
  python3 scripts/synth_timeseries_pretrain.py --n 100000 --out "${MODAL_DATA_DIR}/synth_timeseries_100k.parquet" 2>&1 | tail -5 || echo "(ts synth failed, continuing without)"
fi

setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --warmstart '${WARMSTART}' \
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
  --warmup 200 \
  --kd-every 4 \
  --kd-topk 2048 \
  --kd-weight 0.4 \
  --shuffle-buffer 10000 \
  --shuffle-seed 1414 \
  --grad-clip 1.0 \
  --lr-decay cosine \
  --steps 60000 \
  --save-every 2000 \
  --self-learn-ttt \
  --self-learn-k 8 \
  --ttt-val-fraction 0.80 \
  --curiosity-weight 0.05 \
  --modal-list image,audio,time_series \
  --modal-data-dir '${MODAL_DATA_DIR}' \
  --modal-alpha 0.10 \
  --spike-target-loss-weight 0.05 \
  --lm-head-spectral-norm \
  --phase-aware \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[launch_ultra_fullmix] pid=${NEW_PID}"
