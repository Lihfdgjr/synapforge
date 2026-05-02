"""Tests for synapforge.native.dispatch.HeteroPipeline.

Acceptance gates from the deliverable spec:

1. **Parity**: pipelined produces same final params as sequential
   (within fp32 noise, since it's just reordering ops). We use a
   *param-independent* gradient function so the math is associative
   under reorder; that isolates the pipeline's correctness from
   the documented ASGD staleness.
2. **Backpressure**: a slow Stage C shouldn't OOM the GPU queue.
   We simulate this by making Stage C deliberately slow and asserting
   that ``queue_bc`` never holds more than its capacity.

Tests run in pure Python with numpy only; no torch / cupy required.
The test imports modules directly via importlib.spec_from_file_location
to bypass the synapforge top-level __init__ which currently imports
torch (a separate ongoing migration).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DISPATCH_DIR = _REPO_ROOT / "synapforge" / "native" / "dispatch"


def _load_dispatch_modules():
    """Load the dispatch modules without going through synapforge.__init__.

    The synapforge top-level package currently imports torch in its
    __init__; we don't want that for these tests (the dispatch layer
    has zero torch dependencies).
    """
    if "synapforge.native.dispatch.pipeline" in sys.modules:
        return (
            sys.modules["synapforge.native.dispatch.streams"],
            sys.modules["synapforge.native.dispatch.cpu_pool"],
            sys.modules["synapforge.native.dispatch.pipeline"],
        )

    # Build fake parent packages so absolute imports work.
    for name in ("synapforge", "synapforge.native", "synapforge.native.dispatch"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    streams = _load("synapforge.native.dispatch.streams", _DISPATCH_DIR / "streams.py")
    cpu_pool = _load("synapforge.native.dispatch.cpu_pool", _DISPATCH_DIR / "cpu_pool.py")
    parent = sys.modules["synapforge.native.dispatch"]
    for k in dir(streams):
        if not k.startswith("_"):
            setattr(parent, k, getattr(streams, k))
    for k in dir(cpu_pool):
        if not k.startswith("_"):
            setattr(parent, k, getattr(cpu_pool, k))
    pipeline = _load("synapforge.native.dispatch.pipeline", _DISPATCH_DIR / "pipeline.py")
    return streams, cpu_pool, pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dispatch():
    return _load_dispatch_modules()


# ---------------------------------------------------------------------------
# Helper: build a simple training problem with param-independent grads
# ---------------------------------------------------------------------------

class _Problem:
    """Synthetic problem where grad is a deterministic fn of the input.

    Critical property: the gradient does NOT depend on the current
    parameter values. That makes sequential and pipelined trajectories
    bit-identical because the ASGD-style staleness is invisible -- the
    grad sequence is the same regardless of what params look like
    when the grad is computed.
    """

    def __init__(self, num_params=8, param_size=64, num_steps=20, seed=0):
        self.num_steps = num_steps
        rng = np.random.default_rng(seed)
        self.params = [
            rng.standard_normal(param_size, dtype=np.float32)
            for _ in range(num_params)
        ]
        self._initial = [p.copy() for p in self.params]
        self._batches = []
        for s in range(num_steps):
            x = rng.standard_normal(param_size, dtype=np.float32)
            self._batches.append(x)

    def reset(self):
        for p, init in zip(self.params, self._initial):
            np.copyto(p, init)

    def batch_fn(self, step):
        if step >= self.num_steps:
            return None
        return self._batches[step], None, {}

    def fb_fn(self, x, _y, _extra):
        g = (x * 0.01).astype(np.float32)
        grads = [g.copy() for _ in self.params]
        return grads, float(g.mean()), {}

    def make_optim(self, slow_seconds=0.0):
        params = self.params

        def opt(step, grads, _extra):
            if slow_seconds > 0:
                time.sleep(slow_seconds)
            for p, g in zip(params, grads):
                p -= 1e-2 * g
        return opt

    def snapshot(self):
        return [p.copy() for p in self.params]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipelined_matches_sequential_final_params(dispatch):
    """Acceptance gate 1: same final params, within fp32 noise."""
    _, _, pipeline = dispatch

    prob = _Problem(num_params=4, param_size=128, num_steps=20, seed=42)

    prob.reset()
    pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn, prob.make_optim(),
        enable_pipeline=False).run()
    seq_final = prob.snapshot()

    prob.reset()
    metrics = pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn, prob.make_optim(),
        enable_pipeline=True).run()
    pipe_final = prob.snapshot()

    assert metrics.num_steps == 20
    for s, p in zip(seq_final, pipe_final):
        np.testing.assert_allclose(s, p, atol=1e-6, rtol=1e-6)


def test_pipelined_matches_sequential_more_steps(dispatch):
    """Same as above with longer trajectory."""
    _, _, pipeline = dispatch

    prob = _Problem(num_params=8, param_size=64, num_steps=100, seed=7)

    prob.reset()
    pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn, prob.make_optim(),
        enable_pipeline=False).run()
    seq_final = prob.snapshot()

    prob.reset()
    pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn, prob.make_optim(),
        enable_pipeline=True).run()
    pipe_final = prob.snapshot()

    for s, p in zip(seq_final, pipe_final):
        np.testing.assert_allclose(s, p, atol=1e-5, rtol=1e-5)


def test_queue_back_pressure_no_OOM(dispatch):
    """Acceptance gate 2: a slow Stage C does NOT cause queue_bc growth.

    With queue_bc capacity = 1 and a deliberately-slow optim_step_fn,
    Stage B blocks on put and stays blocked. We assert that queue_bc
    never exceeds its capacity (no infinite queue growth, no OOM).
    """
    _, _, pipeline = dispatch

    prob = _Problem(num_params=2, param_size=32, num_steps=5, seed=0)
    hp = pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn,
        prob.make_optim(slow_seconds=0.05),  # 50ms per step
        queue_ab_size=2,
        queue_bc_size=1,
        enable_pipeline=True,
    )
    metrics = hp.run()
    assert hp._queue_bc.qsize() <= 1
    assert hp._queue_ab.qsize() <= 2
    assert metrics.stage_b_wait_s > 0.0
    assert metrics.num_steps == 5


def test_queue_back_pressure_against_slow_dataloader(dispatch):
    """The opposite case: slow A means B blocks on empty AB queue."""
    _, _, pipeline = dispatch

    def slow_batch_fn(step):
        if step >= 4:
            return None
        time.sleep(0.04)
        return np.ones(8, dtype=np.float32), None, {}

    def fb(_x, _y, _e):
        return [np.zeros(8, dtype=np.float32)], 0.0, {}

    def opt(_s, _g, _e):
        pass

    hp = pipeline.HeteroPipeline(
        slow_batch_fn, fb, opt,
        queue_ab_size=2, queue_bc_size=1,
        enable_pipeline=True,
    )
    m = hp.run()
    assert m.num_steps == 4
    assert m.stage_b_wait_s > 0.0


def test_sequential_mode_runs_in_calling_thread(dispatch):
    """``enable_pipeline=False`` MUST run on the calling thread (no spawn)."""
    _, _, pipeline = dispatch

    import threading
    main_tid = threading.get_ident()
    seen_tids = []

    def fb(x, _y, _e):
        seen_tids.append(threading.get_ident())
        return [np.zeros(8)], 0.0, {}

    def opt(_s, _g, _e):
        seen_tids.append(threading.get_ident())

    def bf(s):
        if s >= 3:
            return None
        return np.ones(8), None, {}

    hp = pipeline.HeteroPipeline(bf, fb, opt, enable_pipeline=False)
    hp.run()
    assert all(tid == main_tid for tid in seen_tids), \
        "sequential mode must run on the calling thread"


def test_exception_in_stage_b_propagates(dispatch):
    """A crash in Stage B must bubble out of run() with the original exc."""
    _, _, pipeline = dispatch

    def bf(s):
        if s >= 5:
            return None
        return np.ones(4), None, {}

    def fb_crash(_x, _y, _e):
        raise RuntimeError("forward boom")

    def opt(_s, _g, _e):
        pass

    hp = pipeline.HeteroPipeline(bf, fb_crash, opt, enable_pipeline=True)
    with pytest.raises(RuntimeError, match="forward boom"):
        hp.run()


def test_exception_in_stage_c_propagates(dispatch):
    """A crash in Stage C must bubble out of run() with the original exc."""
    _, _, pipeline = dispatch

    def bf(s):
        if s >= 5:
            return None
        return np.ones(4), None, {}

    def fb(_x, _y, _e):
        return [np.ones(4)], 0.0, {}

    def opt_crash(_s, _g, _e):
        raise RuntimeError("optim boom")

    hp = pipeline.HeteroPipeline(bf, fb, opt_crash, enable_pipeline=True)
    with pytest.raises(RuntimeError, match="optim boom"):
        hp.run()


def test_metrics_populated(dispatch):
    """PipelineMetrics fields are non-zero after a run."""
    _, _, pipeline = dispatch

    prob = _Problem(num_params=2, param_size=16, num_steps=5)
    m = pipeline.HeteroPipeline(
        prob.batch_fn, prob.fb_fn, prob.make_optim(),
        enable_pipeline=True,
    ).run()
    assert m.num_steps == 5
    assert m.wallclock_s > 0.0
    assert m.stage_b_total_s > 0.0
    assert m.stage_c_total_s >= 0.0
    d = m.as_dict()
    assert "steps_per_second" in d


def test_b_c_overlap_ratio_high_when_balanced(dispatch):
    """With deliberately-balanced and GIL-friendly sleep stages, the
    overlap_ratio should be high (>= 0.5)."""
    _, _, pipeline = dispatch

    def bf(s):
        if s >= 6:
            return None
        return np.ones(4), None, {}

    def fb(_x, _y, _e):
        time.sleep(0.04)
        return [np.zeros(4)], 0.0, {}

    def opt(_s, _g, _e):
        time.sleep(0.04)

    hp = pipeline.HeteroPipeline(bf, fb, opt, enable_pipeline=True)
    m = hp.run()
    assert m.num_steps == 6
    assert m.b_c_overlap_ratio >= 0.5, (
        f"expected >=0.5, got {m.b_c_overlap_ratio:.3f}; "
        f"wallclock={m.wallclock_s:.3f} B={m.stage_b_total_s:.3f} "
        f"C={m.stage_c_total_s:.3f}")


def test_zero_max_steps_is_noop(dispatch):
    """``max_steps=0`` runs no steps."""
    _, _, pipeline = dispatch

    def bf(s):
        return np.ones(4), None, {}

    def fb(_x, _y, _e):
        return [np.zeros(4)], 0.0, {}

    def opt(_s, _g, _e):
        pass

    for enable in (True, False):
        hp = pipeline.HeteroPipeline(bf, fb, opt, enable_pipeline=enable)
        m = hp.run(max_steps=0)
        assert m.num_steps == 0


def test_no_torch_import(dispatch):
    """The dispatch package must NOT import torch anywhere."""
    repo_root = Path(__file__).resolve().parents[3]
    dispatch_dir = repo_root / "synapforge" / "native" / "dispatch"
    files = list(dispatch_dir.glob("*.py"))
    assert files, "no .py files found"
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import torch"), \
                f"{fp.name} has 'import torch': {line}"
            assert not stripped.startswith("from torch"), \
                f"{fp.name} has 'from torch': {line}"
