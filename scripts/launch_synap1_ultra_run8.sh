#!/bin/bash
# launch_synap1_ultra_run8.sh -- Synap-1 Ultra Run 8 native integration launcher.
#
# Run 8 = Run 7 (token-soup-fix) + 12 native LNN+SNN feature branches integrated
# into a single unified ``synapforge.native`` package. See docs/RUN_8_README.md
# for the full module catalogue.
#
# Run 7 -> Run 8 deltas (perf):
#   * DROPPED  --cpu-offload-optim
#                                The saturation profiler (synapforge.native.bench.
#                                stage_profiler) measured ~33% of step time was
#                                spent in CPU optim sync stalls on the A800.
#                                Without offload the trainer keeps Adam m/v on
#                                HBM (free'd by the larger batch + grad-ckpt
#                                stack already paying for HBM).
#   * AUTOTUNE --batch-size 48 --grad-accum-steps 4 --rfold-chunk 32
#                                synapforge/native/bench/_autotune_real.json
#                                says (48, 4, 32) is the A800 optimal at d=1280
#                                bs_eff=192. Run 7 used (32, 2, 16); the autotune
#                                box-search shows this point sits inside the
#                                3.6x speedup region.
#   * ADDED    --fused-kernel
#                                synapforge.native.kernel.fused_hybrid_{fwd,bwd}
#                                fuses HybridBlock fwd+bwd into a single Triton
#                                kernel pair. Bit-exact at fp32 reduction;
#                                rel-err <= 5e-4 in fp16/bf16. Expected
#                                1.15-1.18x step-time speedup at d>=1024.
#   * ADDED    --packed-spikes
#                                synapforge.native.spike.packed_matmul. Bit-packs
#                                PLIF spikes into uint16 (16 spikes/word). 16x
#                                HBM bandwidth saving on the spike->synapse
#                                branch. Auto-falls back to dense GEMM above the
#                                shared --sparse-spike-threshold density.
#                                **Dormant** until PLIF revives -- Run 7 PLIF
#                                density is ~0 due to dead-bootstrap; this flag
#                                is safe-on but only saves bandwidth once the
#                                spike rate becomes non-trivial.
#   * ADDED    --async-aux-coordinator
#                                Constructs synapforge.native.auxsched.
#                                AsyncAuxCoordinator at startup so the per-step
#                                TTT/curiosity/NeuroMCP/ActionHead components
#                                can be streamed off-thread. See RUN_8_README.md
#                                "Honest gaps" for the partial wire-in note.
#
# Run 7 -> Run 8 deltas (quality):
#   * KEPT  --no-warmstart                  Run 7 quality sweep regression fix
#   * KEPT  --untie-lm-head                 lm-head untie improved val ppl
#   * KEPT  --plif-dense-bypass-steps 4000  SEW dead-bootstrap recipe
#   * KEPT  --kd-weight 0.7                 distill weight Run 7 found stable
#   * KEPT  --kill-if-val-regresses 1.03    quality guard 铁律
#
# Expected tok/s: 17,000-30,000 vs Run 7's 2,750
# (Run 7 baseline @ 2,750 tok/s, autotune predicted 3.6x base + 1.16x fused
# kernel + 1.4x async-aux = 5.85x, conservative 17k floor accepts that
# packed-spikes is dormant under dead PLIF and async-aux is partially wired.)
#
# Smoke: bash -n scripts/launch_synap1_ultra_run8.sh
set -e

REPO_DIR="${REPO_DIR:-/workspace/synapforge_git}"
RUN_DIR="${RUN_DIR:-/workspace/runs/synap1_ultra_run8}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_run8_native_integration.log}"
WARMSTART="${WARMSTART:-/workspace/runs/synap1_ultra/best_step.pt}"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

if [[ ! -f "${WARMSTART}" ]]; then
  echo "[run8] WARMSTART not found: ${WARMSTART}" >&2
  echo "[run8] picking latest step ckpt available..."
  WARMSTART="$(ls -t /workspace/runs/synap1_ultra/step_[0-9]*.pt 2>/dev/null | head -1)"
  if [[ -z "${WARMSTART}" ]]; then
    echo "[run8] no ckpt found; aborting" >&2
    exit 1
  fi
  echo "[run8] using ${WARMSTART}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[run8] warmstart from ${WARMSTART}"
echo "[run8] log -> ${LOG_FILE}"
echo "[run8] dropped from Run 7: --cpu-offload-optim (33% step waste)"
echo "[run8] autotune-applied: bs=48 accum=4 rfold-chunk=32"
echo "[run8] new flags: --fused-kernel --packed-spikes --async-aux-coordinator"

setsid bash -c "cd '${REPO_DIR}' && exec python3 -u train_100m_kd.py \
  --no-warmstart \
  --untie-lm-head \
  --teacher Qwen/Qwen2.5-0.5B \
  --backend triton_block \
  --vocab 151936 \
  --d 1280 --n-layers 16 --loop-depth 2 --ffn-ratio 3.0 \
  --batch-size 48 --grad-accum-steps 4 \
  --lr 1e-4 --warmup 200 \
  --kd-every 4 --kd-topk 2048 --kd-weight 0.7 \
  --data-files \"/workspace/data/kd_distill_v1_text.parquet:0.40,/workspace/data/fineweb_edu/000_00000.parquet:0.40,/workspace/data/wt103_raw/train-00000.parquet:0.18,/workspace/data/wt103_raw/validation.parquet:0.02\" \
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
  --cuda-sync-every 10 \
  --clip-grad-cache \
  --ema-decay 0.999 \
  --phase-aware \
  --sew-shortcut \
  --plif-dense-bypass-steps 4000 \
  --lazy-host-sync-accum \
  --fused-adamw \
  --skip-warmstart-eval-N 1 \
  --rfold --rfold-chunk 32 \
  --grad-ckpt \
  --bf16-full \
  --teacher-cpu-int8 \
  --sparse-spike-synapse --sparse-spike-threshold 0.30 \
  --packed-spikes \
  --fused-kernel \
  --async-aux-coordinator \
  --async-data-pipeline --async-pipeline-stages 4 --async-pipeline-prefetch 8 \
  --kill-if-val-regresses 1.03 \
  --out '${RUN_DIR}' \
  > '${LOG_FILE}' 2>&1" </dev/null &
NEW_PID=$!
disown "${NEW_PID}" 2>/dev/null || true
echo "[run8] pid=${NEW_PID}"
