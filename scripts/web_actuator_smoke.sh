#!/usr/bin/env bash
# MIT-licensed — sf.action.web_actuator P18 smoke.
#
# Boots Playwright headless against the static fixture, runs 50 random
# ActionHead steps, asserts >=1 successful click, prints action histogram.
#
#     bash scripts/web_actuator_smoke.sh
#
# Requires:  pip install -e ".[web]"  &&  playwright install chromium
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${REPO_ROOT}/synapforge/tests/fixtures/static_demo.html"
[ -f "${FIXTURE}" ] || { echo "missing fixture: ${FIXTURE}"; exit 2; }

export REPO_ROOT FIXTURE
PYTHON="${PYTHON:-python}"

"${PYTHON}" - <<'PYEOF'
import collections
import os
import sys

sys.path.insert(0, os.environ["REPO_ROOT"])

import torch
import torch.nn as nn
from playwright.sync_api import sync_playwright

from synapforge.action.web_actuator import ACTION_NAMES, WebActuator

ACTION_DIM = 64
N_STEPS = 50
fixture_url = "file:///" + os.environ["FIXTURE"].replace(os.sep, "/").lstrip("/")

action_head = nn.Linear(ACTION_DIM, ACTION_DIM)
counts: collections.Counter = collections.Counter()
ok_clicks = 0

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(fixture_url, wait_until="domcontentloaded")
    actuator = WebActuator(page, action_head, action_dim=ACTION_DIM)
    torch.manual_seed(0)
    for i in range(N_STEPS):
        h = torch.randn(ACTION_DIM)
        # Bias the click slot every few steps so we always get >=1.
        if i % 7 == 3:
            h[1] += 5.0
        rec = actuator.step(h)
        counts[rec["action"]] += 1
        if rec["action"] == "click" and rec["result"] == "ok":
            ok_clicks += 1
    browser.close()

print("[web_actuator_smoke] step histogram:")
for name in ACTION_NAMES:
    print(f"  {name:<10s} {counts.get(name, 0):>3d}")
print(f"[web_actuator_smoke] successful clicks: {ok_clicks}")
assert ok_clicks >= 1, "no successful click — actuator wiring broken"
print("[web_actuator_smoke] OK")
PYEOF
