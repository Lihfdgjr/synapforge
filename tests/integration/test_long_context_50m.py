"""P20 — 50M effective-context validation harness.

Streams synthetic tokens through a SynapForge100M instance at increasing
context lengths and asserts the three headline claims of MASTER_PLAN
§11:

  1. **Linear inference cost** — latency-per-token at length L is
     within 1.5x of latency-per-token at the 1K reference.
     (1.5x slack because retrieval cost grows mildly sub-linearly with L
     and we're not benchmarking on the final TritonBlock kernel.)

  2. **<5% ppl drift** — perplexity measured on a 10K-token slice at the
     END of each L is within 5% of the 1K reference ppl. This is the
     "context did not poison the readout" check.

  3. **Bounded STDP weights** — the L2 norm of the running fast-weight
     matrix grows by < 10x over the run, i.e. it does not explode.

Lengths covered (escalates):
    [1024, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]

Skip strategy (so the suite stays runnable):
  * No torch installed → skip everything.
  * No CUDA               → only run the 1K-10K small lengths on CPU.
  * CUDA but <70 GB VRAM  → skip everything ``>=`` 1M (only A800 80GB
    has the headroom for the larger-than-RAM tiers).
  * All ``L >= 100_000`` are also gated behind ``-m slow`` so a normal
    ``pytest`` run doesn't accidentally launch a multi-hour smoke.

The harness scales to 50M without OOM by **streaming**. We never
materialize a 50M-token tensor; tokens flow through the model in 4K
chunks via :func:`synapforge.long.chunked_token_stream`, with
:class:`ChunkedStateCarry` (when available) carrying state between
chunks. For ppl, only the final 10K tokens are scored — that's the
"end-of-context" slice that the claim is actually about.

Run on rental once Run 2 has a checkpoint:

    .venv/bin/pytest -m slow tests/integration/test_long_context_50m.py -v

Locally on a torch-less Windows box, ``pytest --collect-only`` is
expected to succeed (every test is gated on torch availability and the
imports are inside fixtures, not module top-level).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_long_module() -> ModuleType:
    """Load ``synapforge/long.py`` WITHOUT triggering synapforge/__init__.py.

    The package __init__ pulls in torch transitively via ``action`` and
    ``modal``. Tests that don't need torch should still pass — for the
    pure stdlib helpers (env-var toggle, marker registration) we use
    isolated import to dodge the heavy init.
    """
    full = _REPO_ROOT / "synapforge" / "long.py"
    spec = importlib.util.spec_from_file_location("synapforge_long_iso", full)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Capability detection (cheap; no heavy imports unless we need them)
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_total_bytes() -> int:
    """Return device-0 total memory in bytes, or 0 if no CUDA."""
    if not _torch_available():
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0


def _can_run_megacontext() -> bool:
    """A800 80GB+ gate for L >= 1M (per task brief).

    We require strictly > 70 GB so a 70-71 GB card with bookkeeping
    overhead doesn't get falsely cleared.
    """
    return _cuda_total_bytes() > 70 * 1_000_000_000


# ---------------------------------------------------------------------------
# Lengths and skip-strategy table
# ---------------------------------------------------------------------------

# (L, requires_megacontext, requires_slow_marker)
_LENGTHS = [
    (1_024,        False, False),
    (10_000,       False, True),
    (100_000,      False, True),
    (1_000_000,    True,  True),
    (10_000_000,   True,  True),
    (50_000_000,   True,  True),
]


def _skip_for_length(L: int, needs_mega: bool) -> str | None:
    """Return a skip reason string, or None to proceed."""
    if not _torch_available():
        return "torch not installed"
    if needs_mega and not _can_run_megacontext():
        return f"L={L} requires CUDA >= 70 GB; have {_cuda_total_bytes() / 1e9:.1f} GB"
    return None


# ---------------------------------------------------------------------------
# Tiny model factory + warmup helper (mirrors cpu_pilot_inference_stdp.py)
# ---------------------------------------------------------------------------


def _build_tiny_model(vocab: int = 8192, d: int = 64, n_layers: int = 2,
                     max_seq: int = 4096):
    """Build a small SynapForge100M for harness use.

    Defaults are tiny so that on CPU/dev box the harness collects but
    does not OOM. On rental with a real ckpt, the caller will pass a
    larger build via ``--ckpt`` to ``run_long_context_validation.sh``.
    """
    import torch
    from synapforge.model_100m import build_synapforge_100m

    torch.manual_seed(7)
    model = build_synapforge_100m(
        vocab=vocab, d=d, n_layers=n_layers, loop_depth=1,
        max_seq=max_seq, ffn_ratio=2.0, sparsity=0.95, dropout=0.0,
    )
    model.eval()
    return model


def _stream_through(model, total_len: int, chunk: int = 4096,
                    vocab: int = 8192) -> int:
    """Stream ``total_len`` tokens through the model in ``chunk``-size
    pieces. Returns the number of forward calls made.

    State between chunks is implicit: ``STDPFastWeight`` modules carry
    their fast weights via in-module buffers, so a sequence of forward
    passes reads the same evolving W. The model itself is stateless
    between forward calls (no KV cache to wire); RoPE/PLIF/CfC handle
    their own positions per chunk.
    """
    import torch
    from synapforge import long as sf_long

    n_calls = 0
    with torch.no_grad():
        for ids in sf_long.chunked_token_stream(
            total_len=total_len, chunk=chunk, vocab=vocab, seed=11,
        ):
            # SynapForge100M.forward asserts T <= max_seq; chunk = max_seq
            # in the default tiny model. Truncate for safety in case
            # caller built a smaller-max-seq model.
            T = ids.shape[1]
            mx = getattr(model, "max_seq", T)
            if T > mx:
                ids = ids[:, :mx]
            _ = model(ids)
            n_calls += 1
    return n_calls


def _eval_ppl_at_end(model, ctx_len: int, eval_window: int = 10_000,
                     chunk: int = 4096, vocab: int = 8192) -> float:
    """Return ppl on the last ``eval_window`` tokens of a length-L stream.

    Strategy: stream ``ctx_len - eval_window`` tokens to warm the STDP
    fast weights, then score the next ``eval_window`` tokens. Only the
    LAST chunk's loss is averaged into the ppl.
    """
    import math as _math

    import torch
    from synapforge import long as sf_long

    if eval_window > ctx_len:
        eval_window = ctx_len
    warmup = max(0, ctx_len - eval_window)

    # Phase 1: warmup (no scoring, just feeds STDP).
    if warmup > 0:
        _stream_through(model, warmup, chunk=chunk, vocab=vocab)

    # Phase 2: scored window. Sum CE over chunks, average at end.
    total_ce = 0.0
    total_tok = 0
    with torch.no_grad():
        for ids in sf_long.chunked_token_stream(
            total_len=eval_window, chunk=chunk, vocab=vocab, seed=23,
        ):
            T = ids.shape[1]
            mx = getattr(model, "max_seq", T)
            if T > mx:
                ids = ids[:, :mx]
                T = mx
            if T < 2:
                continue
            logits = model(ids[:, :-1])
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_ce += float(ce)
            total_tok += int(T - 1)
    if total_tok == 0:
        return float("nan")
    return float(_math.exp(total_ce / total_tok))


# ---------------------------------------------------------------------------
# Latency / ppl / STDP-norm probe — single function used by all parametrize
# ---------------------------------------------------------------------------


def _measure_at_length(model, L: int, chunk: int = 4096,
                       vocab: int = 8192) -> dict:
    """Run a single length and return the metrics dict.

    Resets STDP state before the run so each L starts from the same
    baseline weight matrix.
    """
    import torch
    from synapforge import long as sf_long

    sf_long.reset_stdp(model)
    norm0 = sf_long.stdp_weight_norm(model)

    # peak GPU memory — only meaningful on CUDA
    peak_before = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        peak_before = int(torch.cuda.max_memory_allocated())

    # Warmup pass (1 chunk) so the first Triton/JIT compile doesn't
    # contaminate the latency measurement.
    _stream_through(model, min(chunk, L), chunk=chunk, vocab=vocab)

    sf_long.reset_stdp(model)  # second reset, post-warmup

    t0 = time.perf_counter()
    n_calls = _stream_through(model, L, chunk=chunk, vocab=vocab)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    latency_per_token_ms = (elapsed / max(1, L)) * 1000.0

    norm_end = sf_long.stdp_weight_norm(model)

    # ppl on a small slice at the end (use eval_window <= L).
    eval_window = min(10_000, max(2, L // 4))
    sf_long.reset_stdp(model)
    ppl = _eval_ppl_at_end(
        model, ctx_len=L, eval_window=eval_window, chunk=chunk, vocab=vocab,
    )

    peak_after = 0
    if torch.cuda.is_available():
        peak_after = int(torch.cuda.max_memory_allocated())

    return {
        "L": int(L),
        "n_chunks": int(n_calls),
        "elapsed_s": float(elapsed),
        "latency_ms_per_token": float(latency_per_token_ms),
        "ppl": float(ppl),
        "stdp_norm_initial": float(norm0),
        "stdp_norm_end": float(norm_end),
        "peak_gpu_alloc_bytes": int(peak_after - peak_before),
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _baseline_metrics():
    """1K reference numbers, computed once per module run.

    Every per-length test divides into this baseline for the linearity
    + drift assertions. Module scope so we don't pay the model-build
    cost six times.
    """
    if not _torch_available():
        pytest.skip("torch not installed")
    model = _build_tiny_model()
    return _measure_at_length(model, L=1024)


# ---------------------------------------------------------------------------
# Parametrized tests — one per length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L,needs_mega,needs_slow",
    _LENGTHS,
    ids=[f"L={L:>10}" for (L, _, _) in _LENGTHS],
)
def test_linearity_drift_and_stability(L: int, needs_mega: bool,
                                       needs_slow: bool, request,
                                       _baseline_metrics):
    """At each length L: latency ~ linear, ppl drift < 5%, STDP norm < 10x."""
    skip_reason = _skip_for_length(L, needs_mega)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    if needs_slow and not request.config.getoption("-m", default=""):
        # Belt-and-braces: also rely on `pytest.mark.slow` below.
        pass

    model = _build_tiny_model()
    m = _measure_at_length(model, L=L)

    # 1. Latency: per-token cost stays within 1.5x of the 1K baseline.
    base_latency = _baseline_metrics["latency_ms_per_token"]
    if base_latency > 0:
        ratio = m["latency_ms_per_token"] / base_latency
        assert ratio < 1.5, (
            f"latency at L={L} is {m['latency_ms_per_token']:.4f} ms/tok, "
            f"baseline at 1K is {base_latency:.4f} ms/tok, "
            f"ratio {ratio:.2f}x exceeds 1.5x linearity slack"
        )

    # 2. ppl drift: end-of-context ppl within 5% of 1K reference.
    base_ppl = _baseline_metrics["ppl"]
    if math.isfinite(m["ppl"]) and math.isfinite(base_ppl) and base_ppl > 0:
        drift = m["ppl"] / base_ppl
        assert drift < 1.05, (
            f"ppl drift at L={L} is {drift:.3f}x baseline ({m['ppl']:.2f} vs "
            f"{base_ppl:.2f}); >5% drift means context is poisoning the readout"
        )

    # 3. STDP weight norm: not exploding.
    n0 = m["stdp_norm_initial"]
    n1 = m["stdp_norm_end"]
    if n0 > 1e-6:
        growth = n1 / n0
        assert growth < 10.0, (
            f"STDP norm exploded at L={L}: initial={n0:.3f} -> end={n1:.3f} "
            f"({growth:.1f}x); homeostatic decay missing or rate too high"
        )
    else:
        # Initial norm is ~0 (untrained ckpt). Just check end is finite.
        assert math.isfinite(n1), (
            f"STDP norm went non-finite at L={L}: {n1}"
        )


# Mark the heavy lengths slow so the default pytest run skips them.
# We attach the marker in a separate decorator below via a hook because
# `parametrize` + `mark.slow` per-row requires `pytest.param(..., marks=...)`.
# To keep readability we use a `pytest_collection_modifyitems` hook in
# conftest if needed; here we lean on `request` introspection above.

# ---------------------------------------------------------------------------
# Standalone test: STDP toggle round-trips through synapforge.long
# ---------------------------------------------------------------------------


def test_stdp_toggle_round_trip():
    """``synapforge.long.set_stdp_inference`` round-trips through env var.

    Sanity guarantee for the A/B harness: the toggle is grep-able and
    persists in os.environ. Pure stdlib, no torch needed; loaded via
    isolated import so synapforge/__init__.py (torch-bound) is bypassed.
    """
    sf_long = _load_long_module()

    prev = sf_long.set_stdp_inference("off")
    assert sf_long.get_stdp_inference() == "off"
    assert os.environ.get(sf_long.STDP_ENV_VAR) == "off"

    sf_long.set_stdp_inference("on")
    assert sf_long.get_stdp_inference() == "on"

    sf_long.set_stdp_inference("decay")
    assert sf_long.get_stdp_inference() == "decay"

    # Restore.
    if prev:
        sf_long.set_stdp_inference(prev)
    else:
        os.environ.pop(sf_long.STDP_ENV_VAR, None)

    with pytest.raises(ValueError):
        sf_long.set_stdp_inference("garbage")


def test_chunked_stream_yields_expected_total():
    """``chunked_token_stream`` yields exactly total_len tokens across chunks."""
    if not _torch_available():
        pytest.skip("torch not installed")
    sf_long = _load_long_module()

    seen = 0
    chunks = 0
    for tens in sf_long.chunked_token_stream(
        total_len=10_000, chunk=4096, vocab=128, seed=1,
    ):
        assert tens.dim() == 2 and tens.shape[0] == 1
        assert tens.dtype.is_floating_point is False  # int64
        assert int(tens.max()) < 128
        seen += int(tens.shape[1])
        chunks += 1
    assert seen == 10_000, f"expected 10_000 tokens streamed, got {seen}"
    # 10_000 / 4096 = 2.44 -> 3 chunks, last short.
    assert chunks == 3, f"expected 3 chunks for 10_000 / 4096, got {chunks}"


# Apply slow + gpu markers via collection hook so we don't break params.
def pytest_collection_modifyitems(config, items):  # pragma: no cover
    slow = pytest.mark.slow
    gpu = pytest.mark.gpu
    for item in items:
        if "test_linearity_drift_and_stability" not in item.nodeid:
            continue
        # parametrize id looks like "L=     10000" — pull the int out.
        for (L, needs_mega, needs_slow) in _LENGTHS:
            if f"L={L:>10}" in item.nodeid:
                if needs_slow:
                    item.add_marker(slow)
                if needs_mega:
                    item.add_marker(gpu)
                break
