#!/usr/bin/env bash
# tests/integration/test_data_pipeline_smoke.sh -- P9 E2E data pipeline smoke.
#
# Verifies that the chain  synth -> mix -> 50-step train  works on a fresh
# machine. Pitch claim "EN+ZH chat" hinges on this entire pipeline being
# runnable; before this script that was a code path, not a verified claim.
#
# Pipeline:
#   1.  synth_chinese_pretrain.py --n 200  -> tiny synthetic Chinese parquet
#   2.  mix_pretrain_corpora.py --corpora <synth>  -> single-corpus filter+dedup
#   3.  train_100m_kd.py --steps 50 --backend gpu_dense  -> 50 trainer steps,
#       on a 1.5M-param tiny model so we fit the <=5min CPU budget. The
#       trainer emits step_000050.pt because we pass --save-every 50.
#
# Asserts:
#   (a) ckpt step_000050.pt exists in $TMP/runs/smoke/
#   (b) ckpt has the new "config" dict (P12 architecture round-trip)
#   (c) train.log contains a "step 50" line
#   (d) ce decreases over the 50 steps (last log line ce < first)
#
# Cleans up the temp dir on PASS, leaves it for inspection on FAIL.
#
# Total runtime budget: <=5 minutes on CPU. Requires:  bash, python3, torch,
# pyarrow, transformers (gpt2 tokenizer is auto-downloaded on first run).

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script is at tests/integration/, repo root is two up).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Temp dir. Default mktemp; allow override via $P9_SMOKE_TMP for debugging.
# We DO NOT auto-cleanup on failure so the caller can inspect train.log
# and ckpts when something goes wrong.
# ---------------------------------------------------------------------------
TMP="${P9_SMOKE_TMP:-$(mktemp -d -t p9_smoke.XXXXXX)}"
echo "[smoke] tmp dir: ${TMP}"
mkdir -p "${TMP}/data" "${TMP}/runs/smoke"

# Capture failure for "leave on FAIL, clean on PASS" behaviour.
SMOKE_FAILED=1
on_exit() {
  local rc=$?
  if [[ "${SMOKE_FAILED}" == "0" ]]; then
    echo "[smoke] PASS -- removing ${TMP}"
    rm -rf "${TMP}"
  else
    echo "[smoke] FAIL (rc=${rc}) -- preserving ${TMP} for inspection"
    echo "        train.log:    ${TMP}/runs/smoke/train.log"
    echo "        ckpts:        ${TMP}/runs/smoke/"
  fi
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# 1. synth: 200 Chinese articles via the deterministic synthesizer.
# ---------------------------------------------------------------------------
echo "[smoke] step 1/3: synth 200 zh articles"
python3 scripts/synth_chinese_pretrain.py \
    --n 200 \
    --seed 42 \
    --out "${TMP}/data/synth_zh.parquet"

if [[ ! -f "${TMP}/data/synth_zh.parquet" ]]; then
  echo "[smoke] FAIL: synth output missing"
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. mix: single-corpus run via the new --corpora flag (no per-corpus
#    directory layout needed). Output is the parquet the trainer reads.
# ---------------------------------------------------------------------------
echo "[smoke] step 2/3: mix (single-corpus filter+dedup)"
python3 scripts/mix_pretrain_corpora.py \
    --corpora "${TMP}/data/synth_zh.parquet" \
    --out "${TMP}/data/pretrain_mix.parquet" \
    --target-rows 200 \
    --strict

if [[ ! -f "${TMP}/data/pretrain_mix.parquet" ]]; then
  echo "[smoke] FAIL: mix output missing"
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. train: 50 steps on a tiny model (d=64, n_layers=2, vocab=gpt2 50257).
#    --backend gpu_dense so we don't need triton compilation. --kd-weight 0
#    so we don't try to download a teacher. --no-warmstart because there's
#    no rental ckpt to warm from. --save-every 50 so step_000050.pt lands.
# ---------------------------------------------------------------------------
echo "[smoke] step 3/3: train 50 steps (tiny model, gpu_dense, no KD, no warmstart)"
LOG="${TMP}/runs/smoke/train.log"
# stdbuf so the line-by-line tail later is reliable on a buffered stdout.
# We redirect both stderr+stdout into train.log AND tee to the console so
# CI logs surface the trainer's progress.
stdbuf -oL -eL python3 train_100m_kd.py \
    --out "${TMP}/runs/smoke" \
    --backend gpu_dense \
    --no-warmstart \
    --batch-size 4 \
    --steps 50 \
    --warmup 5 \
    --save-every 50 \
    --eval-every 50 \
    --log-every 5 \
    --seq-len 32 \
    --vocab 50257 \
    --d 64 \
    --n-layers 2 \
    --loop-depth 1 \
    --ffn-ratio 4.0 \
    --sparsity 0.5 \
    --lr 1e-3 \
    --kd-weight 0 \
    --tokenizer-name gpt2 \
    --data-glob "${TMP}/data/pretrain_mix.parquet" \
    --val-glob "${TMP}/data/pretrain_mix.parquet" \
    --no-honest-eval \
    2>&1 | tee "${LOG}"

# ---------------------------------------------------------------------------
# Assertions.
# ---------------------------------------------------------------------------
echo "[smoke] asserts:"

# (a) ckpt exists
CKPT="${TMP}/runs/smoke/step_000050.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "[smoke] FAIL (a): missing ${CKPT}"
  ls -la "${TMP}/runs/smoke/" || true
  exit 1
fi
echo "[smoke]   (a) OK: ${CKPT}"

# (b) ckpt has "config" dict (P12). Inline torch.load + key check.
python3 - <<PY
import sys, torch
ck = torch.load("${CKPT}", map_location="cpu")
if not isinstance(ck, dict):
    print(f"[smoke] FAIL (b): ckpt is not a dict, got {type(ck)}")
    sys.exit(1)
if "config" not in ck:
    print(f"[smoke] FAIL (b): ckpt missing 'config' key; keys={list(ck)}")
    sys.exit(1)
cfg = ck["config"]
required = {"vocab","d","n_layers","loop_depth","max_seq",
            "ffn_ratio","sparsity","dropout","tie_lm_head"}
missing = required - set(cfg)
if missing:
    print(f"[smoke] FAIL (b): config missing keys: {sorted(missing)}")
    sys.exit(1)
# Round-trip sanity: shapes must match what we passed on the CLI.
if int(cfg["d"]) != 64:        sys.exit(f"[smoke] FAIL (b): d={cfg['d']} != 64")
if int(cfg["n_layers"]) != 2:  sys.exit(f"[smoke] FAIL (b): n_layers={cfg['n_layers']} != 2")
if int(cfg["vocab"]) != 50257: sys.exit(f"[smoke] FAIL (b): vocab={cfg['vocab']} != 50257")
if int(cfg["max_seq"]) != 32:  sys.exit(f"[smoke] FAIL (b): max_seq={cfg['max_seq']} != 32")
print("[smoke]   (b) OK: ckpt['config'] =", cfg)
PY

# (c) train.log contains a "step 50" line. The trainer emits e.g.
#     "[HH:MM:SS] step    50 loss=... ce=... kd=... z=..."
if ! grep -q -E 'step[[:space:]]+50[[:space:]]+loss=' "${LOG}"; then
  echo "[smoke] FAIL (c): no 'step 50 loss=' line in ${LOG}"
  echo "        last 30 lines:"
  tail -30 "${LOG}" || true
  exit 1
fi
echo "[smoke]   (c) OK: 'step 50' line present"

# (d) ce decreases over the 50 steps. Compare first and last ce= log line.
#     Use awk so we don't need python again. ce= is space-delimited.
FIRST_CE=$(grep -oE 'ce=[0-9.]+' "${LOG}" | head -n 1 | cut -d= -f2)
LAST_CE=$(grep -oE  'ce=[0-9.]+' "${LOG}" | tail -n 1 | cut -d= -f2)
if [[ -z "${FIRST_CE}" || -z "${LAST_CE}" ]]; then
  echo "[smoke] FAIL (d): no ce= lines found in ${LOG}"
  exit 1
fi
# bash arithmetic is integer-only; compare via awk.
if awk -v f="${FIRST_CE}" -v l="${LAST_CE}" 'BEGIN{exit !(l<f)}'; then
  echo "[smoke]   (d) OK: ce ${FIRST_CE} -> ${LAST_CE} (decreased)"
else
  echo "[smoke] FAIL (d): ce did NOT decrease: first=${FIRST_CE} last=${LAST_CE}"
  exit 1
fi

# All asserts passed.
SMOKE_FAILED=0
echo "[smoke] PASS"
