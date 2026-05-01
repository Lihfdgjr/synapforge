#!/usr/bin/env bash
# tests/integration/run_smokes.sh — entrypoint for the integration suite.
#
# What it does:
#   - Runs the 9 cross-module integration tests under tests/integration/.
#   - Skips tests that need missing optional deps (torch / playwright /
#     pyarrow) gracefully — `pytest.mark.skipif` + the conftest hooks
#     handle this; we just print a summary at the end.
#   - Suitable for CI: a non-zero exit code only on real failures, never
#     on a missing-optional-dep skip.
#
# Usage:
#   bash tests/integration/run_smokes.sh
#   bash tests/integration/run_smokes.sh -k 'phase or skill'   # subset
#
# Env knobs:
#   PYTEST       which pytest binary (default: python -m pytest)
#   EXTRA_FLAGS  appended to the pytest argv (e.g. "-x --tb=short")
set -uo pipefail

# Resolve repo root (parent of tests/) so this script works from anywhere.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

PYTEST="${PYTEST:-python -m pytest}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

echo "== synapforge integration smoke =="
echo "  repo:   $REPO_ROOT"
echo "  pytest: $PYTEST"
echo

# Detect optional deps so the summary is honest.
have() {
    python -c "import $1" >/dev/null 2>&1 && echo "yes" || echo "no"
}
TORCH=$(have torch)
PYARROW=$(have pyarrow)
PLAYWRIGHT=$(have playwright)
echo "  torch installed:      $TORCH"
echo "  pyarrow installed:    $PYARROW"
echo "  playwright installed: $PLAYWRIGHT"
echo

# Run the suite. -ra surfaces skips/xfails so CI sees them.
# Capture exit code separately so we can print a friendly summary even
# when pytest itself failed.
set +e
$PYTEST tests/integration/ -v -ra "$@" $EXTRA_FLAGS
RC=$?
set -e

echo
echo "== summary =="
if [[ $RC -eq 0 ]]; then
    echo "  PASS (or skipped on missing optional dep)"
else
    echo "  FAIL (exit code $RC) — see traceback above"
fi
exit $RC
