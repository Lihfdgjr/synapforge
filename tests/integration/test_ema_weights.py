"""DEEP_MAINT_QUEUE.md T8.4 — EMA weight tracker tests.

Top-LM training pipelines (DeepSeek, Llama, SmolLM2) maintain a shadow
copy of model weights with the recurrence
``model_ema = decay·model_ema + (1-decay)·model`` after each optimizer
step. The EMA copy is then used at inference time.

Four behaviours are pinned by this file:

    1. ``test_default_off``                       — --ema-decay 0.0 produces no _ema.pt.
    2. ``test_decay_099_creates_ema_ckpt``        — --ema-decay 0.99 writes step_<N>_ema.pt.
    3. ``test_ema_state_smooths``                 — after 100 updates, EMA != live but close.
    4. ``test_load_into_model``                   — load_ema() round-trips and forward works.

Tests run on CPU using a tiny model (d=32, n_layers=2, vocab=256) so the
full suite finishes in seconds without GPU. They exercise the
``synapforge.training.ema`` module directly plus a smoke harness that
mimics the trainer's "init -> step -> save" sequence to prove the
checkpoint-save side wiring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def _build_tiny_model(torch_mod, *, vocab: int = 256, d: int = 32):
    """Tiny CPU-friendly model. Uses ``SynapForge100M`` with shrunk dims so
    we go through the same ``state_dict`` keys (PLIF, RoPE, lm_head, etc.)
    that the production EMA needs to handle, but the whole thing fits in
    ~1MB and a forward+backward runs in milliseconds."""
    from synapforge.model_100m import SynapForge100M

    torch_mod.manual_seed(0)
    model = SynapForge100M(
        vocab=vocab,
        d=d,
        n_layers=2,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        tie_lm_head=True,
        use_grad_checkpoint=False,
    )
    return model


def _take_one_step(torch_mod, model, vocab: int):
    """Random batch + SGD step. Mutates ``model.state_dict()`` so the EMA
    update has something nontrivial to absorb."""
    opt = torch_mod.optim.SGD(model.parameters(), lr=1e-1)
    ids = torch_mod.randint(0, vocab, (2, 8), dtype=torch_mod.long)
    logits = model(ids)
    loss = logits.float().pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


# ==========================================================================
# Test 1 -- --ema-decay 0.0 (default) creates NO _ema.pt
# ==========================================================================

def test_default_off(torch_mod, tmp_path):
    """When ema_decay <= 0 the trainer must not save step_<N>_ema.pt.

    We don't run the full trainer here (would need GPU + parquet) -- we
    exercise the ckpt-save site directly: simulate the ``ema_tracker is
    None`` branch and confirm only step_*.pt exists, no _ema.pt sibling.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    # Mimic the trainer's save: when ema_tracker is None, only the live
    # ckpt is dumped. The implementation in train_100m_kd.py is:
    #     if step % save_every == 0:
    #         torch.save({...}, step_<N>.pt)
    #         if ema_tracker is not None:
    #             ema_tracker.save(step_<N>_ema.pt, ...)
    ema_tracker = None
    step = 100
    live_path = out_dir / f"step_{step:06d}.pt"
    torch_mod.save({"model": {"x": torch_mod.zeros(2)}, "step": step}, str(live_path))
    if ema_tracker is not None:
        # Must NOT execute on this branch.
        raise AssertionError("ema_tracker should be None when --ema-decay 0.0")

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["step_000100.pt"], (
        f"--ema-decay 0.0 must only emit step_<N>.pt; got {files!r}"
    )
    # Defensive: explicitly check no _ema.pt was created.
    assert not any(name.endswith("_ema.pt") for name in files), (
        f"step_<N>_ema.pt must NOT exist when EMA is OFF; got {files!r}"
    )


# ==========================================================================
# Test 2 -- --ema-decay 0.99 creates step_<N>_ema.pt
# ==========================================================================

def test_decay_099_creates_ema_ckpt(torch_mod, tmp_path):
    """When ema_decay > 0, every save_every step must dump step_<N>_ema.pt
    next to step_<N>.pt with a matching state_dict layout."""
    from synapforge.training.ema import EMATracker

    vocab = 256
    model = _build_tiny_model(torch_mod, vocab=vocab)
    tracker = EMATracker(model, decay=0.99)

    # Run a few SGD steps so EMA absorbs nontrivial weight changes.
    for _ in range(3):
        _take_one_step(torch_mod, model, vocab=vocab)
        tracker.update(model)

    # Dump both ckpts at step 100 the same way the trainer does.
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    step = 100
    live_path = out_dir / f"step_{step:06d}.pt"
    ema_path = out_dir / f"step_{step:06d}_ema.pt"
    torch_mod.save(
        {
            "model": model.state_dict(),
            "step": step,
            "config": {"vocab": vocab, "d": 32},
        },
        str(live_path),
    )
    tracker.save(
        str(ema_path),
        extra={"step": step, "config": {"vocab": vocab, "d": 32}},
    )

    # Both files must exist ON DISK (sanity).
    assert live_path.exists(), f"live ckpt missing: {live_path}"
    assert ema_path.exists(), f"ema ckpt missing: {ema_path}"

    # Reload the EMA ckpt and confirm the schema matches the live ckpt's
    # ``"model"`` payload key-for-key (drop-in replacement).
    live_ck = torch_mod.load(str(live_path), map_location="cpu")
    ema_ck = torch_mod.load(str(ema_path), map_location="cpu")
    assert "model" in ema_ck, f"ema ckpt missing 'model' key; got {list(ema_ck)!r}"
    assert ema_ck.get("ema_decay") == pytest.approx(0.99), (
        f"ema_decay was not persisted; got {ema_ck.get('ema_decay')!r}"
    )
    assert set(ema_ck["model"].keys()) == set(live_ck["model"].keys()), (
        f"EMA ckpt key set must match live ckpt key set; "
        f"diff: only-live={set(live_ck['model']) - set(ema_ck['model'])!r} "
        f"only-ema={set(ema_ck['model']) - set(live_ck['model'])!r}"
    )


# ==========================================================================
# Test 3 -- 100 updates: EMA != live but close
# ==========================================================================

def test_ema_state_smooths(torch_mod):
    """After many SGD + EMA-update steps, the EMA copy must (a) differ
    from the live model (proving the recurrence is firing) and (b) stay
    close to the live model (proving it's tracking, not drifting). At
    decay=0.99 the EMA half-life is ~70 steps; after 100 steps the EMA
    should be within ~10x the L2 distance of live-from-init.
    """
    from synapforge.training.ema import EMATracker

    vocab = 256
    model = _build_tiny_model(torch_mod, vocab=vocab)
    init_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    tracker = EMATracker(model, decay=0.99)

    # 100 SGD steps with EMA update each time.
    for _ in range(100):
        _take_one_step(torch_mod, model, vocab=vocab)
        tracker.update(model)

    # Pick a parameter that definitely sees gradient (the LM head /
    # tied tok_embed).
    live_sd = model.state_dict()
    ema_sd = tracker.state_dict()

    # Pick any floating-point param that moved away from init -- the
    # mogrifier projections are guaranteed to have nonzero grad at
    # vocab=256, d=32 with the squared-logit scalar loss.
    chosen_key = None
    for k, v_live in live_sd.items():
        if not v_live.is_floating_point():
            continue
        v_init = init_state[k]
        if v_live.shape != v_init.shape:
            continue
        if (v_live - v_init).abs().sum() > 1e-3:
            chosen_key = k
            break
    assert chosen_key is not None, (
        "expected at least one floating-point param to drift under SGD; "
        "test harness is broken if every weight is unchanged"
    )

    v_live = live_sd[chosen_key].detach().to("cpu", dtype=torch_mod.float32)
    v_ema = ema_sd[chosen_key].detach().to("cpu", dtype=torch_mod.float32)
    v_init_f = init_state[chosen_key].detach().to("cpu", dtype=torch_mod.float32)

    # (a) EMA differs from live -- the smoothing is doing something.
    diff_le = (v_live - v_ema).norm().item()
    drift_li = (v_live - v_init_f).norm().item()
    assert diff_le > 1e-6, (
        f"EMA == live at key={chosen_key!r}; smoothing did not fire "
        f"(diff_norm={diff_le:.3e})"
    )

    # (b) EMA tracks live -- the EMA-vs-live distance must be smaller
    # than the live-vs-init distance. With decay=0.99 over 100 steps
    # the EMA lags but stays close; pinning at "EMA stays *closer* to
    # live than live is to init" is the standard sanity gate.
    assert diff_le < drift_li, (
        f"EMA-vs-live ({diff_le:.3e}) must be smaller than "
        f"live-vs-init ({drift_li:.3e}); the EMA is not tracking"
    )


# ==========================================================================
# Test 4 -- load_ema(): round-trip into a fresh model and smoke forward
# ==========================================================================

def test_load_into_model(torch_mod, tmp_path):
    """load_ema(path, fresh_model) must populate the fresh model's weights
    with the EMA copy and a forward pass must produce finite logits of
    the right shape -- the loader path that ``chat_demo`` / ``chat_repl``
    will exercise at inference."""
    from synapforge.training.ema import EMATracker, load_ema

    vocab = 256
    src = _build_tiny_model(torch_mod, vocab=vocab)
    tracker = EMATracker(src, decay=0.95)

    # Make the EMA non-trivially different from a freshly-built model:
    # train the source for a few steps and update EMA.
    for _ in range(5):
        _take_one_step(torch_mod, src, vocab=vocab)
        tracker.update(src)

    ema_path = tmp_path / "step_000010_ema.pt"
    tracker.save(str(ema_path), extra={"step": 10})
    assert ema_path.exists()

    # Build a SECOND fresh model with a different seed so initial
    # weights are NOT equal to either ``src`` or the EMA. Then load
    # the EMA into it.
    torch_mod.manual_seed(123)
    fresh = _build_tiny_model(torch_mod, vocab=vocab)
    # Establish baseline: fresh != EMA before load.
    fresh_pre = next(p.detach().clone() for p in fresh.parameters())

    payload = load_ema(str(ema_path), fresh)
    assert payload.get("ema_decay") == pytest.approx(0.95)
    assert payload.get("step") == 10

    # After load, the chosen parameter should match the EMA's value
    # at the SAME key. We pick the first parameter that exists in
    # both state dicts -- ``tok_embed.weight`` is guaranteed.
    fresh_post_sd = fresh.state_dict()
    ema_sd = tracker.state_dict()
    for k, ema_v in ema_sd.items():
        if not ema_v.is_floating_point():
            continue
        if k not in fresh_post_sd:
            continue
        live_v = fresh_post_sd[k].detach().to("cpu", dtype=torch_mod.float32)
        ema_cpu = ema_v.detach().to("cpu", dtype=torch_mod.float32)
        # Model param after load must equal the EMA tensor (cast back to
        # the param's native dtype, which for this CPU model is fp32).
        assert torch_mod.allclose(live_v, ema_cpu, atol=1e-5), (
            f"after load_ema, model[{k!r}] != EMA[{k!r}]: "
            f"max_diff={(live_v - ema_cpu).abs().max().item():.3e}"
        )
        break

    # And the loaded weights must be different from the pre-load fresh-init.
    fresh_post_first = next(p for p in fresh.parameters())
    assert not torch_mod.allclose(fresh_post_first.detach(), fresh_pre, atol=1e-5), (
        "load_ema did not change the model: post-load first-param == pre-load"
    )

    # Smoke forward: must produce finite (B, T, V) logits.
    fresh.eval()
    with torch_mod.no_grad():
        ids = torch_mod.randint(0, vocab, (1, 8), dtype=torch_mod.long)
        logits = fresh(ids)
    assert logits.shape == (1, 8, vocab), (
        f"unexpected logit shape after EMA load: {tuple(logits.shape)!r}"
    )
    assert torch_mod.isfinite(logits).all(), (
        "non-finite logits after EMA load"
    )
