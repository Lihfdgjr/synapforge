#!/usr/bin/env bash
# P20 + P21 — Long-context validation harness runner.
#
# Designed to fire-and-forget on rental once Run 2 (v24h_qwen) has its
# first checkpoint. Two test files are run with the ``slow`` marker
# enabled, both are JSON-tee'd to a timestamped artefact under
# ``/workspace/runs/v24h_qwen/`` so we can attach the report directly
# to MASTER_PLAN §11 / INVESTOR.md.
#
# Usage (rental, after `ssh -p 41614 root@117.74.66.77`):
#   bash scripts/run_long_context_validation.sh
#
# To run on a smaller GPU (no megacontext), the harness self-skips L >= 1M
# when total VRAM < 70 GB; the smaller lengths (1K / 10K / 100K) still
# produce a meaningful signal.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-v24h_qwen}"
OUT_ROOT="${OUT_ROOT:-/workspace/runs/${RUN_NAME}}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${OUT_ROOT}/long_ctx_validation_${TS}.json"
OUT_LOG="${OUT_ROOT}/long_ctx_validation_${TS}.log"

mkdir -p "${OUT_ROOT}"

# ---------------------------------------------------------------------------
# venv activation (best-effort; fall through if not present)
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
elif [[ -f "/workspace/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "/workspace/.venv/bin/activate"
fi

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Run with slow marker enabled, write per-test JSON via pytest's --report-log.
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] starting long-context validation" | tee "${OUT_LOG}"
echo "  repo:    ${REPO_ROOT}"                              | tee -a "${OUT_LOG}"
echo "  run:    ${RUN_NAME}"                                | tee -a "${OUT_LOG}"
echo "  out:    ${OUT_JSON}"                                | tee -a "${OUT_LOG}"

# Capture both stdout and JSON-style summary in two files.
PYTEST_ADDOPTS="" \
python -m pytest \
    -m slow \
    -v --tb=short \
    --report-log="${OUT_JSON}" \
    tests/integration/test_long_context_50m.py \
    tests/integration/test_long_context_monotonic_quality.py \
    2>&1 | tee -a "${OUT_LOG}"

RC=${PIPESTATUS[0]}

echo "[$(date -Iseconds)] done rc=${RC}" | tee -a "${OUT_LOG}"
echo "" | tee -a "${OUT_LOG}"
echo "Artefacts:" | tee -a "${OUT_LOG}"
echo "  ${OUT_LOG}"  | tee -a "${OUT_LOG}"
echo "  ${OUT_JSON}" | tee -a "${OUT_LOG}"

exit "${RC}"
