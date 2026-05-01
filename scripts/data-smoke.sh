#!/usr/bin/env bash
# scripts/data-smoke.sh -- shortcut to run the P9 data-pipeline smoke test.
#
# Equivalent to:
#   bash tests/integration/test_data_pipeline_smoke.sh
#
# Usage:
#   bash scripts/data-smoke.sh           # run smoke (auto-tmp, cleans on PASS)
#   P9_SMOKE_TMP=/tmp/foo bash scripts/data-smoke.sh   # pin tmp dir
#
# Runtime budget: <=5 minutes on CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec bash "${REPO_ROOT}/tests/integration/test_data_pipeline_smoke.sh" "$@"
