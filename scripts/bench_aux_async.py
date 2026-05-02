"""bench_aux_async.py -- benchmark async-aux speedup vs sequential.

Synopsis
--------
Runs a synthetic main+aux workload to measure end-to-end step time
with the ``synapforge.native.auxsched.AsyncAuxCoordinator`` vs a
sequential reference (everything on the main thread, in order).

Synthetic workload
------------------
Each "outer step" simulates::

    main fwd+bwd:   main_ms  (default 80 ms)  -- a numpy compute spin
    curiosity:      cur_ms   (default 8 ms)
    TTT k=8:        ttt_ms   (default 25 ms / inner step  -> 200 ms total)
    NeuroMCP tick:  nm_ms    (default 1 ms)
    Tool exec:      tool_ms  (default 100 ms; runs on its own pool)

Sequential model: total = main + cur + ttt*k + nm + tool   (default 397 ms)
Streamed model:   total = max(main, ttt*k_async, ...)       (~200 ms)

Numbers depend on host: on a CPU-only dev box "main fwd+bwd" is just a
busy-wait; the speedup is bounded by Python+threading overhead. On the
A800 rental with real GPU compute, the measured speedup will be larger
because the streamed math actually runs on independent CUDA streams.

Usage
-----
.. code-block:: bash

    python scripts/bench_aux_async.py
    python scripts/bench_aux_async.py --steps 50 --ttt-k 8 --inline-k 2
    python scripts/bench_aux_async.py --quick   # 10 steps, smaller msec budget

Output
------
Writes a JSON dict to stdout with the per-mode wallclock and the
computed speedup. Used by CI smoke tests + release notes.

Hard constraint
---------------
**No ``import torch``.**
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUX_DIR = _REPO_ROOT / "synapforge" / "native" / "auxsched"


def _load_auxsched() -> Dict[str, Any]:
    """Load auxsched modules without going through synapforge.__init__."""
    cached_keys = [k for k in sys.modules if k.startswith("synapforge.native.auxsched")]
    if cached_keys:
        return _modules_dict()

    for name in ("synapforge", "synapforge.native", "synapforge.native.auxsched"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    streams = _load("synapforge.native.auxsched.streams", _AUX_DIR / "streams.py")
    future = _load("synapforge.native.auxsched.future", _AUX_DIR / "future.py")
    parent = sys.modules["synapforge.native.auxsched"]
    for k in dir(streams):
        if not k.startswith("_"):
            setattr(parent, k, getattr(streams, k))
    for k in dir(future):
        if not k.startswith("_"):
            setattr(parent, k, getattr(future, k))
    cur = _load("synapforge.native.auxsched.curiosity_async", _AUX_DIR / "curiosity_async.py")
    ttt = _load("synapforge.native.auxsched.ttt_async", _AUX_DIR / "ttt_async.py")
    nm = _load("synapforge.native.auxsched.neuromcp_cpu", _AUX_DIR / "neuromcp_cpu.py")
    act = _load("synapforge.native.auxsched.action_async", _AUX_DIR / "action_async.py")
    coord = _load("synapforge.native.auxsched.coordinator", _AUX_DIR / "coordinator.py")
    for m in (cur, ttt, nm, act):
        for k in dir(m):
            if not k.startswith("_"):
                setattr(parent, k, getattr(m, k))
    return _modules_dict()


def _modules_dict() -> Dict[str, Any]:
    return {
        "streams": sys.modules["synapforge.native.auxsched.streams"],
        "future": sys.modules["synapforge.native.auxsched.future"],
        "curiosity": sys.modules["synapforge.native.auxsched.curiosity_async"],
        "ttt": sys.modules["synapforge.native.auxsched.ttt_async"],
        "neuromcp": sys.modules["synapforge.native.auxsched.neuromcp_cpu"],
        "action": sys.modules["synapforge.native.auxsched.action_async"],
        "coordinator": sys.modules["synapforge.native.auxsched.coordinator"],
    }


# ---------------------------------------------------------------------------
# Synthetic workload pieces
# ---------------------------------------------------------------------------


def _busy_wait(ms: float) -> None:
    """Burn ``ms`` milliseconds. Uses ``time.sleep`` because that's what
    a GIL-yielding GPU compute would do (our threads can run in parallel).
    """
    time.sleep(ms / 1000.0)


def _main_fwd_bwd(main_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Pretend to do the main forward+backward. Returns ``(h_prev, h_next)``."""
    _busy_wait(main_ms)
    h_prev = np.zeros((4, 32), dtype=np.float32)
    h_next = np.ones((4, 32), dtype=np.float32)
    return h_prev, h_next


def _make_curiosity_fn(cur_ms: float, mods: Dict[str, Any]):
    CR = mods["curiosity"].CuriosityResult

    def fn(payload):
        _busy_wait(cur_ms)
        return CR(step_idx=payload.step_idx, loss=0.1)

    return fn


def _make_ttt_inner(per_step_ms: float):
    def fn(state, vx, vy, i):
        _busy_wait(per_step_ms)
        return state + 1, 1.0 / (1.0 + i)
    return fn


def _make_neuromcp_tick(nm_ms: float, mods: Dict[str, Any]):
    PR = mods["neuromcp"].PlasticityResult

    def fn(stats, prev_mask):
        _busy_wait(nm_ms)
        return PR(step_idx=stats.step_idx)

    return fn


def _make_tool_fn(tool_ms: float, mods: Dict[str, Any]):
    TO = mods["action"].ToolObservation

    def fn(call):
        _busy_wait(tool_ms)
        return TO(step_idx=call.step_idx, tool_id=call.tool_id, success=True)

    return fn


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


def run_sequential(
    *,
    steps: int,
    main_ms: float,
    cur_ms: float,
    ttt_per_step_ms: float,
    ttt_k: int,
    nm_ms: float,
    tool_ms: float,
) -> Dict[str, float]:
    """Strict sequential: main, then cur, then full ttt loop, then nm, then tool."""
    t0 = time.time()
    state = 0
    for s in range(steps):
        h_prev, h_next = _main_fwd_bwd(main_ms)
        # curiosity (in-line)
        _busy_wait(cur_ms)
        # ttt full inner loop in-line
        for i in range(ttt_k):
            state += 1
            _busy_wait(ttt_per_step_ms)
        # neuromcp tick in-line
        _busy_wait(nm_ms)
        # tool exec in-line (worst case: blocks)
        _busy_wait(tool_ms)
    elapsed = time.time() - t0
    return {
        "wallclock_s": elapsed,
        "steps": steps,
        "step_avg_ms": (elapsed / steps) * 1000.0,
    }


def run_async(
    *,
    steps: int,
    main_ms: float,
    cur_ms: float,
    ttt_per_step_ms: float,
    ttt_k: int,
    ttt_inline_k: int,
    nm_ms: float,
    tool_ms: float,
    mods: Dict[str, Any],
) -> Dict[str, float]:
    Coord = mods["coordinator"].AsyncAuxCoordinator
    SS = mods["neuromcp"].SpikeStats
    TC = mods["action"].ToolCall

    cur_fn = _make_curiosity_fn(cur_ms, mods)
    inner_fn = _make_ttt_inner(ttt_per_step_ms)
    nm_fn = _make_neuromcp_tick(nm_ms, mods)
    tool_fn = _make_tool_fn(tool_ms, mods)

    t0 = time.time()
    with Coord(
        curiosity_compute_fn=cur_fn,
        ttt_inner_step_fn=inner_fn,
        ttt_total_k=ttt_k,
        ttt_inline_k=ttt_inline_k,
        neuromcp_tick_fn=nm_fn,
        action_execute_fn=tool_fn,
    ) as aux:
        state = 0
        for s in range(steps):
            h_prev, h_next = _main_fwd_bwd(main_ms)
            aux.submit_curiosity(s, h_prev, h_next)
            state, _ = aux.submit_ttt(s, None, None, state)
            aux.submit_spike_stats(SS(
                step_idx=s, spike_rate=np.zeros(1),
                proto_sim=np.zeros(1),
                proto_used=np.zeros(1, dtype=np.uint8),
            ))
            aux.submit_tool_call(TC(step_idx=s, tool_id=0, arg_payload=None))
            # Drain finished tool obs every step (cheap, non-blocking).
            _ = aux.drain_observations()
        # End-of-loop: wait for any in-flight aux work to finish so the
        # bench number includes the tail.
        aux.wait_aux(timeout=10.0)
        # Drain the final tool obs.
        _ = aux.drain_observations()
        cm = aux.metrics()
    elapsed = time.time() - t0
    return {
        "wallclock_s": elapsed,
        "steps": steps,
        "step_avg_ms": (elapsed / steps) * 1000.0,
        "curiosity_completed": cm.curiosity.get("completed", 0),
        "curiosity_dropped": cm.curiosity.get("dropped_stale", 0),
        "ttt_completed": cm.ttt.get("completed", 0),
        "ttt_dropped": cm.ttt.get("dropped_stale", 0),
        "neuromcp_ticked": cm.neuromcp.get("ticked", 0),
        "neuromcp_dropped": cm.neuromcp.get("dropped", 0),
        "action_completed": cm.action.get("completed", 0),
        "action_dropped": cm.action.get("dropped_full_queue", 0),
        "wait_aux_s": cm.wait_aux_total_s,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20, help="outer steps")
    ap.add_argument("--main-ms", type=float, default=80.0,
                    help="main fwd+bwd time (ms)")
    ap.add_argument("--cur-ms", type=float, default=8.0,
                    help="curiosity compute time (ms)")
    ap.add_argument("--ttt-per-step-ms", type=float, default=25.0,
                    help="time per TTT inner step (ms)")
    ap.add_argument("--ttt-k", type=int, default=8, help="TTT total inner k")
    ap.add_argument("--inline-k", type=int, default=2,
                    help="how many TTT inner steps to run inline")
    ap.add_argument("--nm-ms", type=float, default=1.0,
                    help="neuromcp tick time (ms)")
    ap.add_argument("--tool-ms", type=float, default=100.0,
                    help="tool exec time (ms; sequential blocks here)")
    ap.add_argument("--quick", action="store_true",
                    help="small budget for CI smoke")
    ap.add_argument("--json", dest="json_out", action="store_true",
                    help="output JSON only (no human header)")
    args = ap.parse_args()

    if args.quick:
        args.steps = max(1, args.steps // 2)
        args.main_ms = min(20.0, args.main_ms / 4)
        args.cur_ms = min(2.0, args.cur_ms / 4)
        args.ttt_per_step_ms = min(6.0, args.ttt_per_step_ms / 4)
        args.nm_ms = min(0.5, args.nm_ms)
        args.tool_ms = min(20.0, args.tool_ms / 5)

    mods = _load_auxsched()

    seq = run_sequential(
        steps=args.steps,
        main_ms=args.main_ms,
        cur_ms=args.cur_ms,
        ttt_per_step_ms=args.ttt_per_step_ms,
        ttt_k=args.ttt_k,
        nm_ms=args.nm_ms,
        tool_ms=args.tool_ms,
    )
    asy = run_async(
        steps=args.steps,
        main_ms=args.main_ms,
        cur_ms=args.cur_ms,
        ttt_per_step_ms=args.ttt_per_step_ms,
        ttt_k=args.ttt_k,
        ttt_inline_k=args.inline_k,
        nm_ms=args.nm_ms,
        tool_ms=args.tool_ms,
        mods=mods,
    )
    speedup = seq["wallclock_s"] / max(asy["wallclock_s"], 1e-6)

    out = {
        "config": {
            "steps": args.steps,
            "main_ms": args.main_ms,
            "cur_ms": args.cur_ms,
            "ttt_per_step_ms": args.ttt_per_step_ms,
            "ttt_k": args.ttt_k,
            "inline_k": args.inline_k,
            "nm_ms": args.nm_ms,
            "tool_ms": args.tool_ms,
        },
        "sequential": seq,
        "async": asy,
        "speedup_x": speedup,
    }

    if args.json_out:
        print(json.dumps(out, indent=2))
        return

    # Human-readable header.
    print("=" * 70)
    print("bench_aux_async -- AsyncAuxCoordinator vs sequential reference")
    print("=" * 70)
    print(f"  steps={args.steps}  main={args.main_ms}ms  cur={args.cur_ms}ms"
          f"  ttt={args.ttt_per_step_ms}ms x {args.ttt_k}  inline_k={args.inline_k}"
          f"  nm={args.nm_ms}ms  tool={args.tool_ms}ms")
    print()
    print(f"  sequential:  {seq['wallclock_s']:7.3f} s   "
          f"step_avg={seq['step_avg_ms']:6.1f} ms")
    print(f"  async:       {asy['wallclock_s']:7.3f} s   "
          f"step_avg={asy['step_avg_ms']:6.1f} ms")
    print()
    print(f"  speedup:     {speedup:.2f}x")
    print()
    print(f"  driver metrics (async path):")
    print(f"    curiosity completed/dropped: "
          f"{asy['curiosity_completed']}/{asy['curiosity_dropped']}")
    print(f"    ttt completed/dropped:       "
          f"{asy['ttt_completed']}/{asy['ttt_dropped']}")
    print(f"    neuromcp ticked/dropped:     "
          f"{asy['neuromcp_ticked']}/{asy['neuromcp_dropped']}")
    print(f"    action completed/dropped:    "
          f"{asy['action_completed']}/{asy['action_dropped']}")
    print(f"    wait_aux total:              {asy['wait_aux_s']*1000:.1f} ms")
    print()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
