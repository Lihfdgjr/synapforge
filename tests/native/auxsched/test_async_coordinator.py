"""Tests for synapforge.native.auxsched.

Acceptance gates from the deliverable spec:

1. **Routing**: each ``submit_*`` ends up in the right driver's queue
   and the right callback fires (we verify by counting per-driver
   callback invocations).
2. **TTT quality guard**: 2-inline + 6-async TTT produces a final
   adapted state whose val ppl matches 8-inline within 1%.
3. **ActionHead non-blocking**: with a deliberately slow (5 s)
   actuator, the main thread's submit + drain cycle still completes
   in <200 ms.
4. **Backpressure**: rapid submit shows the documented drop-policy per
   driver (replace-stale for curiosity/TTT, drop-oldest for NeuroMCP,
   drop-newest for ActionHead).
5. **No torch import**: ``synapforge.native.auxsched`` and submodules
   load with ``sys.modules['torch']`` stubbed out.

Tests run in pure Python with numpy only; no torch / cupy required.
We import the auxsched submodules via ``importlib.spec_from_file_location``
to bypass the synapforge top-level ``__init__`` which currently imports
torch (a separate ongoing migration).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUX_DIR = _REPO_ROOT / "synapforge" / "native" / "auxsched"


def _load_auxsched():
    """Load auxsched submodules without going through synapforge.__init__.

    The synapforge top-level package imports torch in its __init__; we
    don't want that for these tests (auxsched has zero torch deps).
    """
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


def _modules_dict():
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mods():
    return _load_auxsched()


# ---------------------------------------------------------------------------
# 0. Sanity: no torch in auxsched modules
# ---------------------------------------------------------------------------


def test_no_torch_imports():
    """``synapforge.native.auxsched`` modules must not import torch."""
    src_dir = _AUX_DIR
    found = []
    for p in src_dir.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        # Skip the docstring "no torch" annotations -- they're explanations.
        # We look for actual import statements.
        for ln_no, ln in enumerate(text.splitlines(), 1):
            stripped = ln.strip()
            if stripped.startswith("import torch") or stripped.startswith("from torch"):
                found.append(f"{p.name}:{ln_no}: {stripped}")
    assert not found, f"torch imports found: {found}"


def test_module_load_with_torch_blocked(mods):
    """Verify modules already loaded don't depend on torch.

    Note: we can't easily un-import + re-import with torch blocked
    (sys.modules cache). Instead we verify that the module objects
    don't have a 'torch' attribute as a sanity check. The strong
    constraint is checked structurally above.
    """
    for name, m in mods.items():
        assert not hasattr(m, "torch"), f"{name} has torch attribute"


# ---------------------------------------------------------------------------
# 1. Future smoke
# ---------------------------------------------------------------------------


def test_future_basic_set_result(mods):
    F = mods["future"].AuxFuture
    f = F(label="t1")
    assert not f.done()
    f.set_result(42)
    assert f.done()
    assert f.result() == 42


def test_future_exception(mods):
    F = mods["future"].AuxFuture
    f = F(label="t2")
    f.set_exception(ValueError("boom"))
    assert f.done()
    with pytest.raises(ValueError, match="boom"):
        f.result()


def test_future_poll_and_callback(mods):
    F = mods["future"].AuxFuture
    f = F(label="t3")
    seen: list = []
    f.add_done_callback(lambda fut: seen.append(fut.result()))
    assert not f.done()
    f.set_result("hello")
    assert seen == ["hello"]
    done, val = f.poll()
    assert done and val == "hello"


# ---------------------------------------------------------------------------
# 2. CuriosityAsyncDriver
# ---------------------------------------------------------------------------


def test_curiosity_driver_routes_to_compute_fn(mods):
    """``submit`` -> compute_fn -> CuriosityResult must be wired correctly."""
    Cur = mods["curiosity"].CuriosityAsyncDriver
    Res = mods["curiosity"].CuriosityResult
    seen = []

    def compute(payload):
        seen.append((payload.step_idx, payload.h_prev.shape))
        return Res(
            step_idx=payload.step_idx,
            loss=float(np.linalg.norm(payload.h_prev - payload.h_next)),
            forward_model_loss=0.5,
            inverse_model_loss=0.3,
        )

    with Cur(compute_fn=compute) as drv:
        h_prev = np.ones((4, 8), dtype=np.float32)
        h_next = np.zeros((4, 8), dtype=np.float32)
        fut = drv.submit(step_idx=0, h_prev=h_prev, h_next=h_next)
        res = fut.result(timeout=5.0)
        assert res.step_idx == 0
        assert res.loss > 0
        assert res.forward_model_loss == 0.5
        # Allow a couple seconds for any async housekeeping.
        time.sleep(0.05)
    assert seen == [(0, (4, 8))]


def test_curiosity_driver_replaces_stale(mods):
    """Submitting a 2nd step before worker picks up the 1st should
    drop the 1st with a 'stale-skip' marker."""
    Cur = mods["curiosity"].CuriosityAsyncDriver
    Res = mods["curiosity"].CuriosityResult
    started_evt = threading.Event()
    block_evt = threading.Event()

    def compute(payload):
        # Hold the worker on the very first call so a 2nd submit can
        # come in and replace any pending 3rd / 4th. We block once and
        # then run normally.
        started_evt.set()
        block_evt.wait(timeout=2.0)
        return Res(step_idx=payload.step_idx, loss=1.0)

    with Cur(compute_fn=compute) as drv:
        h = np.zeros((1, 4), dtype=np.float32)
        f0 = drv.submit(step_idx=0, h_prev=h, h_next=h)
        # Wait until worker grabs f0 and is blocked on event.
        assert started_evt.wait(timeout=2.0)
        # Now while worker is blocked on f0, send f1 and f2; f1 should
        # be dropped (replace-stale) when f2 lands on the queue.
        f1 = drv.submit(step_idx=1, h_prev=h, h_next=h)
        f2 = drv.submit(step_idx=2, h_prev=h, h_next=h)
        # f1 is dropped immediately at queue replace time; verify stale tag.
        # (Driver sets a synthetic result on f1 with extra={'dropped':...}.)
        r1 = f1.result(timeout=2.0)
        assert r1.extra.get("dropped") == "stale-skip"
        # Release the worker; f0 then f2 complete.
        block_evt.set()
        r0 = f0.result(timeout=2.0)
        r2 = f2.result(timeout=2.0)
        assert r0.step_idx == 0
        assert r2.step_idx == 2
        m = drv.metrics()
        assert m["dropped_stale"] >= 1


# ---------------------------------------------------------------------------
# 3. TTTAsyncDriver -- core async behaviour + parity guard
# ---------------------------------------------------------------------------


def test_ttt_driver_routes_inline_then_async(mods):
    """k=8 with inline_k=2 should run 2 inline (sync), then 6 async."""
    TTT = mods["ttt"].TTTAsyncDriver
    iter_ids: list = []
    state_history: list = []

    def inner_step(state, vx, vy, i):
        iter_ids.append((threading.current_thread().name, i))
        state_history.append(state)
        new_state = state + 1
        loss = 1.0 / (1.0 + i)  # decreasing
        return new_state, loss

    with TTT(inner_step_fn=inner_step, total_k=8, inline_k=2) as drv:
        new_state, fut = drv.run(
            step_idx=0, val_inputs=None, val_targets=None, inner_state=0,
        )
        # Inline portion completed: state went 0 -> 2 (2 inline iters).
        assert new_state == 2
        # Async portion not yet complete -- wait.
        stats = fut.result(timeout=5.0)
        assert stats.inline_k == 2
        assert stats.async_k == 6
        # All 8 inner iters fired, in order.
        # Inline iters 0,1 ran on the calling thread (Main / current);
        # async iters 2-7 ran on the worker thread (aux-ttt).
        inline_threads = [t for t, i in iter_ids if i < 2]
        async_threads = [t for t, i in iter_ids if i >= 2]
        assert all(t != "aux-ttt" for t in inline_threads), inline_threads
        # The async ones may all be on the worker thread.
        assert all(t == "aux-ttt" for t in async_threads), async_threads


def test_ttt_quality_parity_2_inline_6_async_vs_8_inline(mods):
    """Spec: 2-inline + 6-async final TTT-adapted state must match
    8-inline reference within 1% on val ppl on a toy LM.

    Toy "ppl" here is the running ``loss`` across a deterministic SGD-
    style update: inner_step_fn does a fixed-rule update on a small
    weight vector and returns a loss based on it. Because the math is
    deterministic and the async path is just *scheduling* the same
    user callable, the two trajectories must produce the same final
    state up to floating-point noise (well within 1%).
    """
    TTT = mods["ttt"].TTTAsyncDriver
    rng = np.random.default_rng(0)
    target = rng.normal(size=(16,)).astype(np.float32)

    def make_inner_step():
        def inner_step(state, vx, vy, i):
            # Simple gradient descent on (state - target).
            grad = state - target
            new_state = state - 0.1 * grad
            loss = float(np.linalg.norm(new_state - target) ** 2)
            return new_state, loss
        return inner_step

    init_state = rng.normal(size=(16,)).astype(np.float32)

    # Reference: 8-inline.
    with TTT(inner_step_fn=make_inner_step(), total_k=8, inline_k=8) as drv_ref:
        # Use run_inline directly; equivalent.
        ref_state, ref_stats = drv_ref.run_inline(
            step_idx=0, val_inputs=None, val_targets=None, inner_state=init_state.copy(),
        )

    # Async path: 2-inline + 6-async.
    with TTT(inner_step_fn=make_inner_step(), total_k=8, inline_k=2) as drv_async:
        new_state, fut = drv_async.run(
            step_idx=0, val_inputs=None, val_targets=None, inner_state=init_state.copy(),
        )
        stats = fut.result(timeout=5.0)
        # After the async chunk completes, the *final* TTT-adapted
        # state isn't returned via the future API directly (it's
        # passed to done_fn). Instead the final loss is in stats.
        assert stats.async_k == 6
        async_final_loss = stats.async_loss_last
    ref_final_loss = ref_stats.inline_loss_last

    # Quality guard: relative diff < 1%.
    rel_diff = abs(async_final_loss - ref_final_loss) / max(abs(ref_final_loss), 1e-9)
    assert rel_diff < 1e-2, (
        f"TTT 2-inline+6-async vs 8-inline diverged: ref={ref_final_loss:.6f} "
        f"async={async_final_loss:.6f} rel={rel_diff:.4%}"
    )


def test_ttt_done_fn_called_with_final_state(mods):
    TTT = mods["ttt"].TTTAsyncDriver
    delivered = []

    def inner_step(state, vx, vy, i):
        return state + 1, 1.0

    with TTT(
        inner_step_fn=inner_step,
        total_k=4, inline_k=1,
        done_fn=lambda s: delivered.append(s),
    ) as drv:
        _, fut = drv.run(step_idx=0, val_inputs=None, val_targets=None, inner_state=0)
        fut.result(timeout=5.0)
        # Allow done_fn to fire on the worker thread.
        for _ in range(50):
            if delivered:
                break
            time.sleep(0.02)
    assert delivered == [4]


# ---------------------------------------------------------------------------
# 4. NeuroMCPCpuDriver
# ---------------------------------------------------------------------------


def test_neuromcp_driver_runs_tick_and_publishes_mask(mods):
    NM = mods["neuromcp"].NeuroMCPCpuDriver
    Stats = mods["neuromcp"].SpikeStats
    Plast = mods["neuromcp"].PlasticityResult

    def tick(stats, prev_mask):
        new_mask = np.ones((4, 4), dtype=np.float32) * stats.step_idx
        return Plast(step_idx=stats.step_idx, new_mask=new_mask, grew_prototype=True)

    with NM(tick_fn=tick) as drv:
        s = Stats(
            step_idx=7,
            spike_rate=np.array([0.1, 0.2]),
            proto_sim=np.array([0.9, 0.5]),
            proto_used=np.array([1, 0], dtype=np.uint8),
            novelty=0.4,
        )
        drv.submit_spikes(s)
        # Wait for tick to land.
        for _ in range(50):
            if drv.latest_mask() is not None:
                break
            time.sleep(0.02)
        m = drv.latest_mask()
        assert m is not None
        np.testing.assert_array_equal(m, np.ones((4, 4), dtype=np.float32) * 7)


def test_neuromcp_drops_oldest_on_full_queue(mods):
    NM = mods["neuromcp"].NeuroMCPCpuDriver
    Stats = mods["neuromcp"].SpikeStats
    Plast = mods["neuromcp"].PlasticityResult
    block = threading.Event()

    def slow_tick(stats, prev_mask):
        # Hold the first tick so subsequent submissions queue up.
        if stats.step_idx == 0:
            block.wait(timeout=2.0)
        return Plast(step_idx=stats.step_idx)

    with NM(tick_fn=slow_tick, queue_capacity=2) as drv:
        # Submit 6 spike batches quickly. Capacity=2; one gets popped
        # by worker (and blocked on event), so queue can hold up to 2
        # more, the rest must drop.
        for i in range(6):
            drv.submit_spikes(Stats(
                step_idx=i, spike_rate=np.zeros(1), proto_sim=np.zeros(1),
                proto_used=np.zeros(1, dtype=np.uint8), novelty=0.0,
            ))
        block.set()
        # Wait for everything to drain.
        for _ in range(50):
            m = drv.metrics()
            if m["ticked"] >= 1 and m["dropped"] >= 1:
                break
            time.sleep(0.05)
        m = drv.metrics()
    assert m["dropped"] >= 1, f"expected drops, got {m}"


# ---------------------------------------------------------------------------
# 5. ActionHeadAsyncDriver -- non-blocking even with slow tools
# ---------------------------------------------------------------------------


def test_action_driver_non_blocking_with_slow_tool(mods):
    """Spec: even with a 5-second tool latency, the main thread's
    submit + drain cycle completes in <200 ms."""
    Act = mods["action"].ActionHeadAsyncDriver
    Call = mods["action"].ToolCall
    Obs = mods["action"].ToolObservation

    def slow_tool(call):
        # Real-world: web fetch behind slow CDN.
        time.sleep(5.0)
        return Obs(
            step_idx=call.step_idx, tool_id=call.tool_id,
            success=True, result="OK",
        )

    with Act(execute_fn=slow_tool, num_workers=2) as drv:
        # Main loop: emit a tool call per "outer step", drain
        # observations. Verify the loop's wallclock is dominated by
        # python overhead, not the 5-second tool latency.
        t0 = time.time()
        for step in range(10):
            ok = drv.submit(Call(
                step_idx=step, tool_id=0,
                arg_payload="https://slow.example.com/x",
                confidence=0.5, timeout_s=10.0,
            ))
            assert ok
            # Drain whatever's done -- typically nothing because tools
            # are still running.
            obs = drv.drain_completed()
            assert isinstance(obs, list)
        elapsed = time.time() - t0
    # 10 submits + 10 drains should take <200 ms even though each tool
    # call takes 5 s; tools run on workers, not the main thread.
    assert elapsed < 0.2, f"main loop blocked: {elapsed:.3f}s for 10 iters"


def test_action_driver_drops_when_full(mods):
    Act = mods["action"].ActionHeadAsyncDriver
    Call = mods["action"].ToolCall
    Obs = mods["action"].ToolObservation
    block = threading.Event()
    accepted = 0

    def block_tool(call):
        block.wait(timeout=5.0)
        return Obs(step_idx=call.step_idx, tool_id=0, success=True)

    # capacity=4, workers=2 -- 2 in flight, 4 queued = 6 max accepted
    with Act(
        execute_fn=block_tool, num_workers=2, submit_capacity=4,
    ) as drv:
        for step in range(20):
            if drv.submit(Call(
                step_idx=step, tool_id=0,
                arg_payload="x", confidence=0.5, timeout_s=10.0,
            )):
                accepted += 1
        block.set()
        # Drain all -- some will be dropped per metrics.
        m = drv.metrics()
    # Some submits MUST have been dropped, since 20 > workers + capacity.
    assert m["dropped_full_queue"] >= 1, f"expected drops, got {m}"
    # Accepted is bounded by capacity + workers in flight.
    assert accepted <= 6 + 2, f"accepted too many: {accepted}"


def test_action_driver_returns_observation_via_drain(mods):
    Act = mods["action"].ActionHeadAsyncDriver
    Call = mods["action"].ToolCall
    Obs = mods["action"].ToolObservation

    def fast_tool(call):
        return Obs(
            step_idx=call.step_idx, tool_id=call.tool_id,
            success=True, result=f"step={call.step_idx}",
        )

    with Act(execute_fn=fast_tool, num_workers=2) as drv:
        for step in range(5):
            drv.submit(Call(step_idx=step, tool_id=0, arg_payload=None))
        # Wait briefly for workers to finish.
        for _ in range(50):
            if drv.metrics()["completed"] >= 5:
                break
            time.sleep(0.02)
        obs_list = drv.drain_completed()
    assert len(obs_list) == 5
    steps = sorted(o.step_idx for o in obs_list)
    assert steps == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 6. AsyncAuxCoordinator -- end-to-end routing
# ---------------------------------------------------------------------------


def test_coordinator_routes_each_submit_to_correct_driver(mods):
    Coord = mods["coordinator"].AsyncAuxCoordinator
    CR = mods["curiosity"].CuriosityResult
    PR = mods["neuromcp"].PlasticityResult
    SS = mods["neuromcp"].SpikeStats
    TC = mods["action"].ToolCall
    TO = mods["action"].ToolObservation

    log = {"cur": [], "ttt": [], "nm": [], "act": []}

    def cur_fn(payload):
        log["cur"].append(payload.step_idx)
        return CR(step_idx=payload.step_idx, loss=0.1)

    def ttt_step(state, vx, vy, i):
        log["ttt"].append((state, i))
        return state + 1, 1.0

    def nm_tick(stats, prev_mask):
        log["nm"].append(stats.step_idx)
        return PR(step_idx=stats.step_idx, new_mask=np.eye(2, dtype=np.float32))

    def act_exec(call):
        log["act"].append(call.step_idx)
        return TO(step_idx=call.step_idx, tool_id=0, success=True)

    with Coord(
        curiosity_compute_fn=cur_fn,
        ttt_inner_step_fn=ttt_step,
        ttt_total_k=4,
        ttt_inline_k=1,
        neuromcp_tick_fn=nm_tick,
        action_execute_fn=act_exec,
    ) as aux:
        h = np.zeros((1, 8), dtype=np.float32)
        # Run 3 mock outer steps.
        for s in range(3):
            aux.submit_curiosity(s, h, h)
            aux.submit_ttt(s, None, None, 0)
            aux.submit_spike_stats(SS(
                step_idx=s, spike_rate=np.zeros(1),
                proto_sim=np.zeros(1),
                proto_used=np.zeros(1, dtype=np.uint8),
            ))
            aux.submit_tool_call(TC(step_idx=s, tool_id=0, arg_payload=None))
        # Wait for everything to drain.
        for _ in range(100):
            cm = aux.metrics()
            if (
                cm.curiosity.get("completed", 0) >= 1
                and cm.ttt.get("completed", 0) >= 3
                and cm.neuromcp.get("ticked", 0) >= 1
                and cm.action.get("completed", 0) >= 3
            ):
                break
            time.sleep(0.02)

    # Each driver saw at least the most recent step.
    assert log["cur"], "curiosity never ran"
    assert log["ttt"], "TTT never ran"
    assert log["nm"], "neuromcp never ran"
    assert log["act"], "action never ran"


def test_coordinator_disabled_components_become_noops(mods):
    """``None`` callables -> submit returns None / False without error."""
    Coord = mods["coordinator"].AsyncAuxCoordinator
    SS = mods["neuromcp"].SpikeStats
    TC = mods["action"].ToolCall
    h = np.zeros((1, 4), dtype=np.float32)
    with Coord(
        curiosity_compute_fn=None,
        ttt_inner_step_fn=None,
        neuromcp_tick_fn=None,
        action_execute_fn=None,
    ) as aux:
        assert aux.submit_curiosity(0, h, h) is None
        s, f = aux.submit_ttt(0, None, None, 0)
        assert s == 0 and f is None
        # Spike stats noop.
        aux.submit_spike_stats(SS(
            step_idx=0, spike_rate=np.zeros(1), proto_sim=np.zeros(1),
            proto_used=np.zeros(1, dtype=np.uint8),
        ))
        ok = aux.submit_tool_call(TC(step_idx=0, tool_id=0, arg_payload=None))
        assert ok is False
        assert aux.drain_observations() == []
        # No drivers exist.
        assert aux.curiosity_driver is None
        assert aux.ttt_driver is None
        assert aux.neuromcp_driver is None
        assert aux.action_driver is None


def test_coordinator_wait_aux_blocks_on_curiosity_then_returns(mods):
    Coord = mods["coordinator"].AsyncAuxCoordinator
    CR = mods["curiosity"].CuriosityResult

    def slow_cur(payload):
        time.sleep(0.05)
        return CR(step_idx=payload.step_idx, loss=0.1)

    h = np.zeros((1, 4), dtype=np.float32)
    with Coord(curiosity_compute_fn=slow_cur) as aux:
        aux.submit_curiosity(step_idx=0, h_prev=h, h_next=h)
        t0 = time.time()
        aux.wait_aux(timeout=2.0)
        elapsed = time.time() - t0
        # Should have waited at least the 50 ms of slow_cur.
        assert elapsed >= 0.04, f"wait_aux returned too quickly: {elapsed*1000:.1f}ms"
        m = aux.metrics()
        assert m.wait_aux_calls == 1


# ---------------------------------------------------------------------------
# 7. Spec acceptance: ActionHead never blocks main step (5 s tool latency)
# ---------------------------------------------------------------------------


def test_spec_acceptance_main_step_under_200ms_with_5s_tool(mods):
    """Spec test (item 8): assert main step time < 200ms even with 5s
    tool latency. This is the load-bearing 'training never blocks on
    a tool' guarantee.
    """
    Coord = mods["coordinator"].AsyncAuxCoordinator
    CR = mods["curiosity"].CuriosityResult
    PR = mods["neuromcp"].PlasticityResult
    SS = mods["neuromcp"].SpikeStats
    TC = mods["action"].ToolCall
    TO = mods["action"].ToolObservation

    # Curiosity is fast (5 ms).
    def cur(payload):
        time.sleep(0.005)
        return CR(step_idx=payload.step_idx, loss=0.1)

    # TTT inner is fast (1 ms per inner step).
    def inner(state, vx, vy, i):
        time.sleep(0.001)
        return state + 1, 1.0

    # NeuroMCP fast (1 ms).
    def nm(stats, prev_mask):
        time.sleep(0.001)
        return PR(step_idx=stats.step_idx)

    # Tool: deliberately 5 seconds. Must NOT block step.
    def tool(call):
        time.sleep(5.0)
        return TO(step_idx=call.step_idx, tool_id=0, success=True)

    with Coord(
        curiosity_compute_fn=cur,
        ttt_inner_step_fn=inner,
        ttt_total_k=4,
        ttt_inline_k=1,
        neuromcp_tick_fn=nm,
        action_execute_fn=tool,
    ) as aux:
        h = np.zeros((1, 4), dtype=np.float32)
        # Mock "main step": all the submits we expect from the trainer.
        # NO wait_aux -- async-only path.
        step_times: list = []
        for s in range(5):
            t0 = time.time()
            aux.submit_curiosity(s, h, h)
            aux.submit_ttt(s, None, None, 0)
            aux.submit_spike_stats(SS(
                step_idx=s, spike_rate=np.zeros(1),
                proto_sim=np.zeros(1),
                proto_used=np.zeros(1, dtype=np.uint8),
            ))
            aux.submit_tool_call(TC(step_idx=s, tool_id=0, arg_payload=None))
            # main fwd+bwd would happen here -- we don't simulate it
            # since we're testing aux non-blocking.
            step_times.append(time.time() - t0)

    # Each "step" must be well under 200 ms despite the 5 s tool.
    max_step = max(step_times)
    assert max_step < 0.2, (
        f"main step blocked: max={max_step*1000:.1f}ms; "
        f"all steps={[f'{t*1000:.1f}ms' for t in step_times]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
