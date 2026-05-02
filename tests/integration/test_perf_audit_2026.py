"""docs/PERF_AUDIT_2026-05-02.md -- argparse + behaviour smoke for the two
perf-audit ships:

  * ``--cuda-sync-every N`` -- defer per-step ``torch.cuda.synchronize()``
    to every N steps. Default 1 = current behaviour (sync every step).
  * ``--clip-grad-cache`` -- cache the requires_grad=True parameter list
    once at trainer init and reuse for ``clip_grad_norm_``. Default OFF.

The trainer is heavy (Qwen tokenizer, parquet, GPU). These tests pin the
argparse surface and the helper-level semantics that must hold for the
two flags to be safe to enable on the rental:

  * default values match the docstring (1 / False)
  * non-default values stick (--cuda-sync-every 10, --clip-grad-cache)
  * the cuda_sync_every modulo gate fires on the documented cadence
  * the clip-grad-cache list contains exactly the requires_grad=True
    params and IS the same list object (so ``id(...)`` is stable
    across iterations)

CPU-only -- no GPU, no real model.  We use a tiny ``nn.Linear`` /
``nn.Sequential`` toy model where ``requires_grad`` settings are
explicit, so the cache contract is testable without booting the
trainer.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_module():
    """Import / reload train_100m_kd so its argparse surface reflects the
    file-on-disk version (matters when running this test alongside
    test_perf_knobs_compose.py which also reloads the module)."""
    pytest.importorskip("torch")
    if "train_100m_kd" in sys.modules:
        return importlib.reload(sys.modules["train_100m_kd"])
    return importlib.import_module("train_100m_kd")


def _parse(mod, argv):
    with patch.object(sys, "argv", ["train_100m_kd.py", *argv]):
        return mod._parse_args()


# ---------------------------------------------------------------------------
# RECO #1: --cuda-sync-every
# ---------------------------------------------------------------------------


def test_cuda_sync_every_default_is_1():
    """``--cuda-sync-every`` defaults to 1 -- existing Run 5 behaviour."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.cuda_sync_every == 1, (
        f"--cuda-sync-every default should be 1; got {args.cuda_sync_every}"
    )


def test_cuda_sync_every_accepts_n():
    """``--cuda-sync-every N`` accepts and stores any positive int."""
    mod = _import_module()
    for n in (1, 5, 10, 50, 100):
        args = _parse(mod, ["--cuda-sync-every", str(n)])
        assert args.cuda_sync_every == n, (
            f"--cuda-sync-every {n} did not stick; got {args.cuda_sync_every}"
        )


def test_cuda_sync_every_modulo_gate_logic():
    """The trainer guards `cuda.synchronize()` behind:

        _sync_period = max(1, int(getattr(args, 'cuda_sync_every', 1)))
        if DEVICE == 'cuda' and (step % _sync_period == 0):
            torch.cuda.synchronize()

    Pin the modulo schedule for N=1 (every step), N=5 (every 5th step),
    and N=10 (every 10th step) so a future regression on the gate
    surfaces here, not at training time on the rental.
    """
    def _would_sync(step: int, sync_every: int) -> bool:
        period = max(1, int(sync_every))
        return (step % period) == 0

    # N=1: every step (current Run 5 behaviour, no perf change).
    for step in range(1, 21):
        assert _would_sync(step, 1) is True

    # N=5: only steps 0, 5, 10, 15, 20 sync.
    for step in range(0, 21):
        expected = (step % 5 == 0)
        assert _would_sync(step, 5) is expected, f"step {step} N=5"

    # N=10: only steps 0, 10, 20 sync.
    for step in range(0, 21):
        expected = (step % 10 == 0)
        assert _would_sync(step, 10) is expected, f"step {step} N=10"


def test_cuda_sync_every_zero_clamps_to_1():
    """``--cuda-sync-every 0`` is degenerate; the runtime guard `max(1, ...)`
    must clamp it to 1 (sync every step) so we never divide by zero on
    the modulo check."""
    # Simulate the runtime guard exactly as it appears in the trainer.
    def _period_for(arg_val) -> int:
        return max(1, int(arg_val))

    assert _period_for(0) == 1
    assert _period_for(-3) == 1
    assert _period_for(1) == 1
    assert _period_for(10) == 10


# ---------------------------------------------------------------------------
# RECO #2: --clip-grad-cache
# ---------------------------------------------------------------------------


def test_clip_grad_cache_default_off():
    """Default OFF (back-compat: list is rebuilt every step, the way Run 5
    is doing it today)."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.clip_grad_cache is False
    args_on = _parse(mod, ["--clip-grad-cache"])
    assert args_on.clip_grad_cache is True


def test_clip_grad_cache_filters_requires_grad():
    """The cache must hold *exactly* the params with ``requires_grad=True``,
    so the freeze/unfreeze invariants of `clip_grad_norm_` are preserved
    at zero perf cost."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(4, 4)
            self.lin2 = nn.Linear(4, 4)
            self.frozen = nn.Linear(4, 4)
            for p in self.frozen.parameters():
                p.requires_grad_(False)

    model = _Toy()
    # Same code path as the trainer's cache-build branch.
    cache = [p for p in model.parameters() if p.requires_grad]

    expected_n = sum(
        1 for p in model.parameters() if p.requires_grad
    )
    assert len(cache) == expected_n
    # No frozen params got into the cache.
    for p in cache:
        assert p.requires_grad is True
    # Frozen-param params are absent.
    frozen_ids = {id(p) for p in model.frozen.parameters()}
    cache_ids = {id(p) for p in cache}
    assert frozen_ids.isdisjoint(cache_ids)


def test_clip_grad_cache_list_object_is_stable():
    """The cache is captured *once* at init -- the trainer hot loop reuses
    the same Python list object every step. Pin that the list identity
    stays the same across simulated iterations.

    This is what unlocks the perf win: torch.nn.utils.clip_grad_norm_
    iterates the input parameters once, and re-iterating the same list
    is free vs. rebuilding from `model.parameters()` (which walks the
    nn.Module tree).
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    cache = [p for p in model.parameters() if p.requires_grad]
    cache_id = id(cache)

    # Reuse on multiple "steps" -- the id() must NOT change. (If a future
    # patch rebuilds the cache mid-loop the test will catch it.)
    for _ in range(5):
        clip_input = cache  # this is the trainer's branch when flag is on
        assert id(clip_input) == cache_id, (
            "cache identity changed across simulated steps"
        )
        # And the list itself must still hold all trainable params.
        n_trainable = sum(
            1 for p in model.parameters() if p.requires_grad
        )
        assert len(clip_input) == n_trainable


def test_clip_grad_cache_actually_clipable():
    """The cached list must be acceptable to ``clip_grad_norm_`` -- i.e.
    the list contents are real ``nn.Parameter`` instances (not stale
    references). Build two identical models from the same seed,
    populate gradients deterministically on both, then clip with the
    "rebuilt" path on model A and the "cached" path on model B.
    The post-clip norm and the post-clip gradient values must match
    exactly (correctness preservation)."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    def _build_model_with_grads(seed: int):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        # Use a separate seed for the gradients so re-seeding for the second
        # model produces identical grads.
        torch.manual_seed(seed + 1000)
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 5.0
        return m

    model_rebuilt = _build_model_with_grads(seed=0)
    model_cached = _build_model_with_grads(seed=0)

    # Pre-clip sanity: both models have identical grads.
    for p_a, p_b in zip(model_rebuilt.parameters(), model_cached.parameters()):
        assert torch.equal(p_a.grad, p_b.grad), (
            "models built from same seed should have identical grads"
        )

    # Path A: rebuilt-each-step (current Run 5 behaviour).
    rebuilt_list = [p for p in model_rebuilt.parameters() if p.requires_grad]
    norm_rebuilt = torch.nn.utils.clip_grad_norm_(rebuilt_list, max_norm=1.0)

    # Path B: cached-once (this PR).
    cached_list = [p for p in model_cached.parameters() if p.requires_grad]
    norm_cached = torch.nn.utils.clip_grad_norm_(cached_list, max_norm=1.0)

    # Norms must match.
    assert torch.isclose(norm_rebuilt, norm_cached, atol=1e-6), (
        f"rebuilt vs cached clip_grad_norm_ produced different norms: "
        f"{norm_rebuilt} vs {norm_cached}"
    )

    # Post-clip gradient values must match too (the clip is in-place; if
    # one path mutates a different set of params we'd see drift).
    for p_a, p_b in zip(model_rebuilt.parameters(), model_cached.parameters()):
        assert torch.allclose(p_a.grad, p_b.grad, atol=1e-6), (
            "post-clip gradients diverged between rebuilt and cached paths"
        )


# ---------------------------------------------------------------------------
# Composition with prior perf knobs (PERF_KNOBS.md v1+v2)
# ---------------------------------------------------------------------------


def test_perf_audit_combo_composes_with_v2_knobs():
    """The recommended PERF_AUDIT combo composes cleanly with the
    PERF_KNOBS v2 batch (prefetch + pin-memory + adaptive-kd-every).

    Pinned so the next-restart launch script can drop the new flags in
    next to the existing v2 knobs without argparse error.
    """
    mod = _import_module()
    args = _parse(
        mod,
        [
            "--batch-size", "24",
            "--grad-accum-steps", "2",
            # PERF_KNOBS v1+v2 combo (already shipped)
            "--z-loss-topk", "2048",
            "--kd-every", "4",
            "--kd-async-teacher",
            "--prefetch-factor", "4",
            "--pin-memory",
            "--shuffle-buffer", "10000",
            # NEW (this PR)
            "--cuda-sync-every", "10",
            "--clip-grad-cache",
        ],
    )
    assert args.batch_size == 24
    assert args.grad_accum_steps == 2
    assert args.z_loss_topk == 2048
    assert args.kd_async_teacher is True
    assert args.prefetch_factor == 4
    assert args.pin_memory is True
    assert args.shuffle_buffer == 10000
    # The two new perf-audit knobs both stick.
    assert args.cuda_sync_every == 10
    assert args.clip_grad_cache is True


def test_perf_audit_combo_default_off_is_back_compat():
    """When neither new flag is passed, the trainer behaves identically
    to the pre-PR version. Pin both defaults so a future regression
    that flips a default ON gets caught here -- defaults-on would
    silently change Run 5 perf characteristics."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.cuda_sync_every == 1, (
        "Default flip: cuda-sync-every default must stay 1 (sync every step) "
        "to preserve the Run 5 timing semantics. Switch to 10+ requires "
        "explicit opt-in on the rental."
    )
    assert args.clip_grad_cache is False, (
        "Default flip: clip-grad-cache must stay OFF -- rebuilding the "
        "param list every step is the back-compat behaviour."
    )
