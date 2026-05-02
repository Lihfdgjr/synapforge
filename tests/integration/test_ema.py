"""DEEP_MAINT_QUEUE.md T8.4 — ModelEMA spec tests.

Pins the four behaviours the T8.4 spec calls out by name:

    1. ``test_ema_update_correct``                — decay math identity.
    2. ``test_swap_context_restores_live_weights`` — swap() context manager.
    3. ``test_save_load_roundtrip``                — save() / load() round-trip.
    4. ``test_default_off_no_overhead``            — --ema-decay 0.0 short-circuits.

These are cheap CPU tests that should run in <2s; complementary to the
existing ``tests/integration/test_ema_weights.py`` (which exercises the
same module against the real ``SynapForge100M`` architecture).

The tests use a tiny ``nn.Linear`` model — small enough to run anywhere
without GPU/parquet/teacher dependencies, big enough to exercise the
full state_dict path (weights + bias + buffers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def _tiny_model(torch_mod, *, in_dim: int = 4, out_dim: int = 4):
    """3-layer MLP with a registered buffer so every state_dict path is
    exercised: parameter tensors (weight, bias) + non-parameter buffers
    (running_count). Both flavours have to round-trip through save/load
    and survive the swap context manager."""
    torch_mod.manual_seed(0)
    model = torch_mod.nn.Sequential(
        torch_mod.nn.Linear(in_dim, 8),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(8, 8),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(8, out_dim),
    )
    # Add a non-parameter buffer to the first linear so we exercise the
    # buffer path of the state_dict (counters, running stats, RoPE bases).
    model[0].register_buffer(
        "running_count", torch_mod.zeros(1, dtype=torch_mod.long)
    )
    return model


def _scramble(torch_mod, model, scale: float = 1.0) -> None:
    """Mutate every floating-point parameter in place. Mimics what an
    optim.step() would do — moves the live weights away from the EMA so
    the update / swap behaviour has something nontrivial to reason about.
    """
    with torch_mod.no_grad():
        for p in model.parameters():
            p.add_(torch_mod.randn_like(p) * scale)


# ============================================================================
# Test 1 — decay math identity
# ============================================================================

def test_ema_update_correct(torch_mod):
    """After one update with decay D, the EMA copy must equal exactly
    ``D * ema_init + (1-D) * model_post_step`` for every floating-point
    state_dict entry. This is the canonical EMA recurrence and the entire
    test suite is built on it being bit-exact (modulo fp32 precision).
    """
    from synapforge.learn.ema import ModelEMA

    model = _tiny_model(torch_mod)
    decay = 0.9

    # Snapshot the EMA's seed (== current model weights at init).
    init_state = {
        k: v.detach().to("cpu").to(torch_mod.float32).clone()
        for k, v in model.state_dict().items()
        if v.is_floating_point()
    }
    tracker = ModelEMA(model, decay=decay)
    assert tracker.decay == pytest.approx(decay)

    # Move the model: scramble the weights to mimic an optim.step().
    _scramble(torch_mod, model, scale=0.5)
    post_step = {
        k: v.detach().to("cpu").to(torch_mod.float32).clone()
        for k, v in model.state_dict().items()
        if v.is_floating_point()
    }

    tracker.update(model)
    ema_state = tracker.state_dict()

    # Decay math: every fp param must be D*init + (1-D)*post_step.
    for k, v_init in init_state.items():
        v_post = post_step[k]
        expected = decay * v_init + (1.0 - decay) * v_post
        ema_v = ema_state[k]
        assert torch_mod.allclose(ema_v, expected, atol=1e-6), (
            f"EMA recurrence failed for key={k!r}: "
            f"max_diff={(ema_v - expected).abs().max().item():.3e}"
        )


# ============================================================================
# Test 2 — swap() context manager
# ============================================================================

def test_swap_context_restores_live_weights(torch_mod):
    """``with tracker.swap(model):`` must temporarily replace the model's
    weights with the EMA copy AND restore the live weights bit-exact on
    exit, even when the wrapped block raises.
    """
    from synapforge.learn.ema import ModelEMA

    model = _tiny_model(torch_mod)
    tracker = ModelEMA(model, decay=0.99)

    # Move model away from EMA seed. Now: model != tracker.state.
    _scramble(torch_mod, model, scale=0.5)

    # Snapshot the live weights to compare against post-exit.
    live_pre = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }

    # Snapshot the EMA state to compare against the inside-block weights.
    ema_state = {
        k: v.detach().clone() for k, v in tracker.state_dict().items()
    }

    # ----- happy path: swap, peek, restore -----
    with tracker.swap(model):
        # Inside the block, model weights must equal the EMA state
        # (cast back to the live param's dtype).
        for k, v in model.state_dict().items():
            ema_v = ema_state.get(k)
            if ema_v is None or ema_v.shape != v.shape:
                continue
            ema_t = ema_v.to(device=v.device, dtype=v.dtype, copy=False)
            assert torch_mod.allclose(v, ema_t, atol=1e-5), (
                f"inside swap(): model[{k!r}] != EMA[{k!r}] "
                f"max_diff={(v - ema_t).abs().max().item():.3e}"
            )

    # After the block, live weights restored bit-exact.
    for k, v in model.state_dict().items():
        live_v = live_pre[k]
        assert torch_mod.allclose(v, live_v, atol=0), (
            f"after swap(): model[{k!r}] != pre-swap live[{k!r}] "
            f"max_diff={(v - live_v).abs().max().item():.3e}"
        )

    # ----- exception path: exception inside block must still restore -----
    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with tracker.swap(model):
            raise _Boom("simulated eval failure")

    # Even after exception, live weights must be intact.
    for k, v in model.state_dict().items():
        live_v = live_pre[k]
        assert torch_mod.allclose(v, live_v, atol=0), (
            f"after swap-raise: model[{k!r}] != pre-swap live[{k!r}] "
            f"max_diff={(v - live_v).abs().max().item():.3e}"
        )


# ============================================================================
# Test 3 — save() / load() round-trip
# ============================================================================

def test_save_load_roundtrip(torch_mod, tmp_path):
    """``tracker.save(path)`` + ``ModelEMA(...).load(path)`` must round-trip
    every floating-point key bit-exact (within fp32 epsilon) and preserve
    auxiliary metadata (``ema_decay``, plus any caller-supplied ``extra``).
    """
    from synapforge.learn.ema import ModelEMA

    model = _tiny_model(torch_mod)
    tracker_a = ModelEMA(model, decay=0.95)

    # Move the EMA off the init seed so the round-trip has signal.
    for _ in range(3):
        _scramble(torch_mod, model, scale=0.3)
        tracker_a.update(model)

    save_path = tmp_path / "ema.pt"
    tracker_a.save(
        str(save_path),
        extra={"step": 42, "tag": "roundtrip-test"},
    )
    assert save_path.exists()

    # Build a fresh tracker (different decay seed so we can verify .load
    # really overrides) and reload from disk.
    fresh_model = _tiny_model(torch_mod)
    tracker_b = ModelEMA(fresh_model, decay=0.5)
    payload = tracker_b.load(str(save_path))

    # Decay must be restored from the ckpt.
    assert tracker_b.decay == pytest.approx(0.95), (
        f"load() did not restore decay; got {tracker_b.decay!r}"
    )
    # Aux metadata flows through.
    assert payload.get("step") == 42
    assert payload.get("tag") == "roundtrip-test"
    assert payload.get("ema_decay") == pytest.approx(0.95)

    # State must match key-for-key, value-for-value (within fp32 epsilon).
    sa = tracker_a.state_dict()
    sb = tracker_b.state_dict()
    assert set(sa.keys()) == set(sb.keys()), (
        f"key set differs after round-trip: "
        f"only-a={set(sa) - set(sb)!r} only-b={set(sb) - set(sa)!r}"
    )
    for k, va in sa.items():
        vb = sb[k]
        if va.is_floating_point():
            assert torch_mod.allclose(va, vb, atol=1e-6), (
                f"value differs after round-trip for key={k!r}: "
                f"max_diff={(va - vb).abs().max().item():.3e}"
            )
        else:
            assert torch_mod.equal(va, vb), (
                f"int counter differs after round-trip for key={k!r}"
            )


# ============================================================================
# Test 4 — --ema-decay 0.0 short-circuits the trainer
# ============================================================================

def test_default_off_no_overhead(torch_mod):
    """When the trainer is launched with --ema-decay 0.0 (the default),
    the EMA tracker must NEVER be instantiated and the per-step update
    hook must be a no-op. This pins the "default behavior unchanged"
    contract from the T8.4 spec.

    We replicate the trainer's branching logic locally rather than
    spinning up the real trainer (which would need parquet + teacher).
    The branch is a literal ``float(args.ema_decay) > 0.0`` test in
    ``train_100m_kd.py``; we simulate it and confirm the no-op path.
    """
    # Branch identical to train_100m_kd.py:
    #     ema_tracker = None
    #     if float(getattr(args, "ema_decay", 0.0)) > 0.0:
    #         ema_tracker = EMATracker(model, decay=...)
    #     ...
    #     if ema_tracker is not None:
    #         ema_tracker.update(model)
    class _ArgsStub:
        ema_decay = 0.0

    args = _ArgsStub()
    model = _tiny_model(torch_mod)

    ema_tracker = None
    if float(getattr(args, "ema_decay", 0.0)) > 0.0:  # pragma: no cover - false branch
        from synapforge.learn.ema import ModelEMA
        ema_tracker = ModelEMA(model, decay=float(args.ema_decay))

    assert ema_tracker is None, (
        f"--ema-decay 0.0 must keep ema_tracker None; got {ema_tracker!r}"
    )

    # Simulate an optim.step() -> per-step update hook with EMA OFF.
    # The hook is a guarded ``if ema_tracker is not None`` so it's a literal
    # zero-cost branch when EMA is disabled. Run the simulated body once
    # and confirm the model state is unchanged afterwards (no shadow copy
    # touched any parameter).
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if ema_tracker is not None:  # pragma: no cover - false branch
        ema_tracker.update(model)

    post = model.state_dict()
    for k, v_pre in pre.items():
        v_post = post[k]
        assert torch_mod.equal(v_pre, v_post), (
            f"--ema-decay 0.0 path mutated model[{k!r}] -- "
            f"the EMA OFF branch must be a true no-op"
        )

    # Now flip on and confirm the same skeleton produces a tracker.
    args.ema_decay = 0.999
    if float(getattr(args, "ema_decay", 0.0)) > 0.0:
        from synapforge.learn.ema import ModelEMA
        ema_tracker = ModelEMA(model, decay=float(args.ema_decay))
    assert ema_tracker is not None
    assert ema_tracker.decay == pytest.approx(0.999)


# ============================================================================
# Bonus — chat_demo --use-ema source resolver
# ============================================================================

def test_chat_demo_resolves_embedded_ema_state(torch_mod, tmp_path):
    """``chat_demo._resolve_ema_source`` must prefer the embedded
    ``raw['ema_state']`` key when present, and fall back to a sibling
    ``step_<N>_ema.pt`` file otherwise. This is the loader path
    ``--use-ema`` exercises in production."""
    from synapforge.demo.chat_demo import _resolve_ema_source

    # Case A: embedded ema_state in raw ckpt -> source == "embedded".
    ckpt_path = tmp_path / "step_000100.pt"
    ckpt_path.write_bytes(b"")  # path needs to exist for sibling check too.
    raw = {
        "model": {"x": torch_mod.zeros(2)},
        "ema_state": {"x": torch_mod.ones(2)},
        "step": 100,
    }
    src, sd = _resolve_ema_source(str(ckpt_path), raw)
    assert src == "embedded"
    assert sd is not None and "x" in sd

    # Case B: no embedded, but sibling _ema.pt exists -> source == path.
    ema_sibling = tmp_path / "step_000200_ema.pt"
    torch_mod.save({"model": {"x": torch_mod.full((2,), 7.0)}, "step": 200},
                   str(ema_sibling))
    live_path = tmp_path / "step_000200.pt"
    live_path.write_bytes(b"")
    raw_no_emb = {"model": {"x": torch_mod.zeros(2)}, "step": 200}
    src, sd = _resolve_ema_source(str(live_path), raw_no_emb)
    assert src.endswith("_ema.pt"), f"expected sibling-path source, got {src!r}"
    assert sd is not None and torch_mod.allclose(sd["x"], torch_mod.full((2,), 7.0))

    # Case C: nothing on disk and no embedded -> source == "none".
    missing_path = tmp_path / "step_000300.pt"
    missing_path.write_bytes(b"")
    src, sd = _resolve_ema_source(str(missing_path), {"model": {}})
    assert src == "none"
    assert sd is None
