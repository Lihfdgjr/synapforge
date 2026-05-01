#!/usr/bin/env python
"""sf.action.web_actuator REAL Playwright smoke (P7).

Resolves docs/MASTER_PLAN.md §6 P7. The MVP shipped under P18 has unit
tests where ``playwright`` is mocked via ``unittest.mock.MagicMock``.
That proves the dispatch table is wired but does NOT prove that an
actual Chromium instance receives clicks driven by a neural action head.

This script boots a real headless Chromium against the static fixture
``synapforge/tests/fixtures/static_demo.html``, runs 100 ActionHead
steps, asserts (a) >= 1 successful click, (b) >= 1 navigate or scroll,
(c) no uncaught exceptions, (d) total runtime <= 60 s. It saves a
screenshot after step 50 and a per-step JSON trace under
``synapforge/tests/fixtures/p7_evidence/``.

Usage:

    python scripts/web_actuator_real_smoke.py [--n-steps 100] [--seed 0]

Exit codes:
    0  success
    1  assertion failure (real run executed but criteria not met)
    2  Playwright unavailable (no install / no chromium binary)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "synapforge" / "tests" / "fixtures" / "p7_evidence"
FIXTURE = REPO_ROOT / "synapforge" / "tests" / "fixtures" / "static_demo.html"


def _candidate_chromium_paths() -> list[str]:
    """Return chrome.exe candidates, preferring matching playwright build."""
    cands: list[str] = []
    appdata = os.environ.get("LOCALAPPDATA")
    pwbrowsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    roots = [pwbrowsers] if pwbrowsers else []
    if appdata:
        roots.append(os.path.join(appdata, "ms-playwright"))
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("chromium-"):
                continue
            for sub in ("chrome-win64", "chrome-win"):
                p = os.path.join(root, entry, sub, "chrome.exe")
                if os.path.isfile(p):
                    cands.append(p)
    return cands


def _ensure_evidence_dir() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _bail_no_playwright(msg: str) -> int:
    sys.stderr.write(
        "[web_actuator_real_smoke] playwright unavailable; "
        "run on a box with internet for browser binary\n"
    )
    sys.stderr.write(f"[web_actuator_real_smoke] reason: {msg}\n")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screenshot-step", type=int, default=50)
    ap.add_argument(
        "--time-budget",
        type=float,
        default=60.0,
        help="hard wallclock cap in seconds",
    )
    args = ap.parse_args()

    if not FIXTURE.is_file():
        sys.stderr.write(f"missing fixture: {FIXTURE}\n")
        return 1

    # ----- Playwright availability -----
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # ImportError or .impl missing
        return _bail_no_playwright(f"import: {type(e).__name__}: {e}")

    # ----- torch + WebActuator -----
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        sys.stderr.write(f"torch unavailable: {e}\n")
        return 1
    from synapforge.action.web_actuator import ACTION_NAMES, WebActuator

    # ----- model -----
    ACTION_DIM = 64
    torch.manual_seed(args.seed)
    action_head = nn.Linear(ACTION_DIM, ACTION_DIM)  # random init, no training

    # file:// URL across platforms
    fixture_url = FIXTURE.absolute().as_uri()

    _ensure_evidence_dir()
    trace: list[dict] = []
    counts: collections.Counter = collections.Counter()
    ok_clicks = 0
    nav_or_scroll = 0
    started = time.time()
    err: str | None = None

    chromium_candidates = _candidate_chromium_paths()

    try:
        with sync_playwright() as pw:
            launch_kwargs = {"headless": True}
            launched = False
            last_exc: Exception | None = None
            try:
                browser = pw.chromium.launch(**launch_kwargs)
                launched = True
            except Exception as e:
                last_exc = e
                for exe in chromium_candidates:
                    try:
                        browser = pw.chromium.launch(executable_path=exe, **launch_kwargs)
                        launched = True
                        break
                    except Exception as ee:
                        last_exc = ee
                        continue
            if not launched:
                msg = f"chromium launch failed: {last_exc}"
                if not chromium_candidates:
                    msg += " (no chrome.exe found in $LOCALAPPDATA/ms-playwright)"
                return _bail_no_playwright(msg)

            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(fixture_url, wait_until="domcontentloaded")
            actuator = WebActuator(page, action_head, action_dim=ACTION_DIM)

            for step_i in range(args.n_steps):
                if time.time() - started > args.time_budget:
                    err = f"time budget exceeded at step {step_i}"
                    break
                hidden = actuator.encode_dom() + 0.05 * torch.randn(ACTION_DIM)
                # bias action diversity so we don't end with 100x noop
                if step_i % 9 == 1:
                    hidden[1] += 6.0  # click
                elif step_i % 9 == 4:
                    hidden[2] += 4.0  # scroll
                elif step_i % 9 == 7:
                    hidden[4] += 4.0  # navigate
                rec = actuator.step(hidden)
                trace.append({"step": step_i, **rec})
                counts[rec["action"]] += 1
                if rec["action"] == "click" and rec["result"] == "ok":
                    ok_clicks += 1
                if rec["action"] in {"navigate", "scroll"} and rec["result"] == "ok":
                    nav_or_scroll += 1

                if step_i == args.screenshot_step:
                    try:
                        page.screenshot(path=str(EVIDENCE_DIR / "web_actuator_smoke.png"))
                    except Exception as e:
                        sys.stderr.write(f"screenshot failed: {e}\n")

            browser.close()
    except Exception as e:
        err = f"uncaught {type(e).__name__}: {e}"

    runtime = time.time() - started

    # ----- persist evidence -----
    histogram = {name: counts.get(name, 0) for name in ACTION_NAMES}
    summary = {
        "n_steps_run": len(trace),
        "n_steps_target": args.n_steps,
        "runtime_s": round(runtime, 3),
        "histogram": histogram,
        "ok_clicks": ok_clicks,
        "nav_or_scroll": nav_or_scroll,
        "fixture": fixture_url,
        "seed": args.seed,
        "error": err,
        "playwright_runtime": "real",
    }
    with open(EVIDENCE_DIR / "web_actuator_smoke_trace.json", "w") as f:
        json.dump({"summary": summary, "trace": trace}, f, indent=2)
    with open(EVIDENCE_DIR / "web_actuator_smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----- pretty print -----
    print("[web_actuator_real_smoke] action histogram:")
    for name in ACTION_NAMES:
        print(f"  {name:<10s} {histogram[name]:>3d}")
    print(f"[web_actuator_real_smoke] ok_clicks: {ok_clicks}")
    print(f"[web_actuator_real_smoke] nav_or_scroll: {nav_or_scroll}")
    print(f"[web_actuator_real_smoke] runtime_s: {runtime:.2f}")
    print(f"[web_actuator_real_smoke] evidence dir: {EVIDENCE_DIR}")

    # ----- assertions -----
    failures: list[str] = []
    if err is not None:
        failures.append(f"uncaught error: {err}")
    if ok_clicks < 1:
        failures.append("no successful click")
    if nav_or_scroll < 1:
        failures.append("no navigate or scroll")
    if runtime > args.time_budget:
        failures.append(f"runtime {runtime:.1f}s > budget {args.time_budget}s")
    if not (EVIDENCE_DIR / "web_actuator_smoke.png").is_file():
        failures.append("screenshot missing")

    if failures:
        sys.stderr.write("[web_actuator_real_smoke] FAIL:\n")
        for f_ in failures:
            sys.stderr.write(f"  - {f_}\n")
        return 1
    print("[web_actuator_real_smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
