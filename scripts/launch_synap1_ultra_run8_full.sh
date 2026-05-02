#!/bin/bash
# launch_synap1_ultra_run8_full.sh -- Synap-1 Ultra Run 8 FULL launcher
#
# This launcher EXTENDS the Run 8 base (`launch_synap1_ultra_run8.sh`,
# owned by the integration agent) with the activations from
# `feature/activate-decorative` (2026-05-02):
#
#   1. STDP-only optimizer routing  -- already enabled implicitly by the
#      `_sf_grad_source=['stdp']` tag set at construction time on
#      ``SparseSynapticLayer.weight``. Toggling NeuroMCP on (Run 8 base
#      already does this with `--neuromcp-weight`) is enough; no extra
#      flag needed.
#
#   2. Multimodal byte-patch (image / audio / time_series) -- runs
#      `MultimodalMixin` from synapforge.trainer_mixins. The mixin only
#      activates when BOTH ``--modal-list`` AND ``--modal-data-dir`` are
#      passed. Phase-aware gating (`--phase-aware`) defers the actual
#      forward-pass cost until val_ppl <= 100 (Phase 2 in
#      scripts/phase_manager.py).
#
#   3. Web daemon -> trainer pipe -- the continual-learning daemon
#      (scripts/launch_continual_daemon.py + the new
#      scripts/web_self_learn_daemon.py) writes filtered web text into
#      ``/workspace/data/web_self_learn/web_*.parquet`` (rotates every
#      hour). The trainer consumes the rolling glob via ``--data-files``
#      with weight 0.10. Pollution gates (TRAK + 7-gate from
#      ``synapforge.learn.continual_daemon``) guarantee
#      ``quality_score >= 0.5`` per row before writing.
#
#   4. R-fold in TRAINING -- ``--rfold --rfold-chunk 16`` is bit-exact
#      with the legacy sequential path within fp32 round-off
#      (tests/cells/test_rfold_equivalence.py covers grad flow); already
#      passed by Run 7.
#
#   5. Ternary BitNet QAT -- NOT enabled (deferred; needs offline
#      validation against the LM-head reset risk per
#      `feedback_spectral_norm_warmstart_cost.md`). Documented for
#      transparency.
#
# Hard contract:
#   * Unchanged from Run 8 base: black-listed `--kwta-k`, `--cuda-graphs`.
#   * Additive only: every new flag has its mixin's "off" path tested.
#   * Smoke: `bash -n scripts/launch_synap1_ultra_run8_full.sh` passes.
#
# Usage (rental):
#   bash scripts/launch_synap1_ultra_run8_full.sh
#
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra_run8_full}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_run8_full.log}"
WARMSTART="${WARMSTART:-/workspace/runs/synap1_ultra_run7/best_step.pt}"

# Extra-data inputs (decorative -> real activations).
MODAL_DATA_DIR="${MODAL_DATA_DIR:-/workspace/data/modal}"
WEB_DAEMON_GLOB="${WEB_DAEMON_GLOB:-/workspace/data/web_self_learn/web_*.parquet}"
WEB_DAEMON_WEIGHT="${WEB_DAEMON_WEIGHT:-0.10}"

# Modal control flags. Default ON; pass MODAL_LIST="" to disable.
MODAL_LIST="${MODAL_LIST:-image,audio,time_series}"
MODAL_ALPHA="${MODAL_ALPHA:-0.10}"

mkdir -p "${RUN_DIR}"
mkdir -p "$(dirname "${WEB_DAEMON_GLOB}")"
cd "${REPO_DIR}"

# --- Warmstart resolution (same logic as Run 7) -----------------------------
if [[ ! -f "${WARMSTART}" ]]; then
  echo "[run8-full] WARMSTART not found at ${WARMSTART}; falling back" >&2
  for cand in \
      /workspace/runs/synap1_ultra_run7/best_step.pt \
      /workspace/runs/synap1_ultra/best_step.pt \
      $(ls -t /workspace/runs/synap1_ultra_run7/step_[0-9]*.pt 2>/dev/null | head -1) \
      $(ls -t /workspace/runs/synap1_ultra/step_[0-9]*.pt 2>/dev/null | head -1); do
    if [[ -f "${cand}" ]]; then
      WARMSTART="${cand}"
      break
    fi
  done
  if [[ ! -f "${WARMSTART}" ]]; then
    echo "[run8-full] no warmstart ckpt found; aborting" >&2
    exit 1
  fi
  echo "[run8-full] resolved warmstart -> ${WARMSTART}"
fi

# --- Modal + web-daemon flag assembly ---------------------------------------
MODAL_ARGS=""
if [[ -n "${MODAL_LIST}" ]]; then
  if [[ -d "${MODAL_DATA_DIR}" ]]; then
    MODAL_ARGS="--modal-list ${MODAL_LIST} --modal-data-dir ${MODAL_DATA_DIR} --modal-alpha ${MODAL_ALPHA}"
    echo "[run8-full] modal: ${MODAL_LIST} dir=${MODAL_DATA_DIR} alpha=${MODAL_ALPHA}"
  else
    echo "[run8-full] modal: SKIPPED (no ${MODAL_DATA_DIR})" >&2
  fi
else
  echo "[run8-full] modal: SKIPPED (MODAL_LIST=\"\")"
fi

# Web-daemon parquet glob (only included when at least one shard exists, so
# the trainer doesn't crash on empty glob during pre-daemon bootstrap).
DATA_FILES_BASE="/workspace/data/kd_distill_v1_text.parquet:0.40,/workspace/data/fineweb_edu/000_00000.parquet:0.40,/workspace/data/wt103_raw/train-00000.parquet:0.18,/workspace/data/wt103_raw/validation.parquet:0.02"
DATA_FILES="${DATA_FILES_BASE}"
if compgen -G "${WEB_DAEMON_GLOB}" >/dev/null; then
  DATA_FILES="${DATA_FILES_BASE},${WEB_DAEMON_GLOB}:${WEB_DAEMON_WEIGHT}"
  echo "[run8-full] web-daemon corpus glob hit -- adding ${WEB_DAEMON_GLOB}@${WEB_DAEMON_WEIGHT}"
else
  echo "[run8-full] web-daemon corpus empty -- skipping ${WEB_DAEMON_GLOB}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[run8-full] warmstart from ${WARMSTART}"
echo "[run8-full] log -> ${LOG_FILE}"
echo "[run8-full] black-listed flags (NOT enabled): --kwta-k, --cuda-graphs, --weight-quant ternary"

# --- Spawn ------------------------------------------------------------------
setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --warmstart '${WARMSTART}' \
  --untie-lm-head \
  --teacher Qwen/Qwen2.5-0.5B \
  --backend triton_block \
  --vocab 151936 \
  --d 1280 --n-layers 16 --loop-depth 2 --ffn-ratio 3.0 \
  --batch-size 32 --grad-accum-steps 2 \
  --lr 1e-4 --warmup 200 \
  --kd-every 4 --kd-topk 2048 --kd-weight 0.7 \
  --data-files \"${DATA_FILES}\" \
  --shuffle-buffer 10000 --shuffle-seed 1818 \
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
  --cuda-sync-every 10 \
  --clip-grad-cache \
  --ema-decay 0.999 \
  --phase-aware \
  --sew-shortcut \
  --plif-dense-bypass-steps 4000 \
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
  ${MODAL_ARGS} \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[run8-full] pid=${NEW_PID}"
echo "[run8-full] tail -f ${LOG_FILE}"
