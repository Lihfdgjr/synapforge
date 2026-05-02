"""Coconut latent thinking at inference time — smoke tests.

Verifies:

1. ``coconut_step`` advances the per-block state by K passes (smoke).
2. ``generate_with_coconut`` is a strict no-op when no ``<bot>`` token
   appears (must NOT degrade non-Coconut output).
3. When a ``<bot>`` token is seeded into the prompt, the latent loop
   actually fires and changes the post-decode state.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from synapforge.inference import (  # noqa: E402
    InferenceState,
    coconut_step,
    generate_rfold,
    generate_with_coconut,
    prefill_state,
)
from synapforge.model_100m import build_synapforge_100m  # noqa: E402


_VOCAB = 32
_D = 64
_N_LAYERS = 2
_LOOP_DEPTH = 1
_MAX_SEQ = 64

# Reserve a synthetic BOT id at the top of the vocab.
_BOT_ID = _VOCAB - 1


def _build(latent_k: int = 0):
    torch.manual_seed(20260502)
    return build_synapforge_100m(
        vocab=_VOCAB,
        d=_D,
        n_layers=_N_LAYERS,
        loop_depth=_LOOP_DEPTH,
        max_seq=_MAX_SEQ,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        use_grad_checkpoint=False,
        freeze_vocab_tail=False,
        live_vocab=_VOCAB,
        lm_head_spectral_norm=False,
        weight_quant_cfc="none",
        latent_k=latent_k,
    ).eval()


def _ids(seed: int = 0, T: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Avoid sampling BOT_ID in the random prompt.
    return torch.randint(0, _VOCAB - 1, (1, T), generator=g, dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. coconut_step requires a seeded state
# ---------------------------------------------------------------------------


def test_coconut_step_requires_seeded_state():
    model = _build()
    with pytest.raises(ValueError):
        coconut_step(model, InferenceState(), k_steps=8)


# ---------------------------------------------------------------------------
# 2. coconut_step k=0 is identity
# ---------------------------------------------------------------------------


def test_coconut_step_k0_identity():
    model = _build()
    prompt = _ids(seed=1, T=4)
    _, state = prefill_state(model, prompt)
    h_before = [h.clone() for h in state.h_cfc]
    v_before = [v.clone() for v in state.v_plif]
    state2 = coconut_step(model, state, k_steps=0)
    for ha, hb in zip(state2.h_cfc, h_before):
        assert torch.allclose(ha, hb, atol=1e-7)
    for va, vb in zip(state2.v_plif, v_before):
        assert torch.allclose(va, vb, atol=1e-7)
    assert state2.position == state.position


# ---------------------------------------------------------------------------
# 3. coconut_step k>0 actually changes the state
# ---------------------------------------------------------------------------


def test_coconut_step_k8_advances_state():
    """K=8 latent passes must perturb the per-block hidden state.

    Identity init on LatentThinker projections means the FIRST pass is
    near-no-op, but K cumulative passes through the HybridBlock stack
    refine the carrier non-trivially.
    """
    model = _build(latent_k=8)
    prompt = _ids(seed=2, T=4)
    _, state = prefill_state(model, prompt)
    h_before = [h.clone() for h in state.h_cfc]
    state2 = coconut_step(model, state, k_steps=8)
    # At least one block must show a non-trivial change.
    max_diff = max(
        (h2 - hb).abs().max().item()
        for h2, hb in zip(state2.h_cfc, h_before)
    )
    assert max_diff > 0.0, (
        "coconut_step k=8 left state untouched; latent loop is bypassed"
    )


# ---------------------------------------------------------------------------
# 4. generate_with_coconut without <bot> == generate_rfold
# ---------------------------------------------------------------------------


def test_no_bot_token_matches_rfold_decode():
    """When no ``<bot>`` token appears, Coconut decode must be bit-identical to plain R-fold.

    This protects the user's quality-must-not-regress 铁律: enabling the
    Coconut hook with bot_id=None (or never sampling it) is strictly a
    no-op vs the unwrapped R-fold path.
    """
    model = _build()
    prompt = _ids(seed=3, T=4)
    max_new = 12

    out_rfold, _ = generate_rfold(
        model, prompt, max_new=max_new, temperature=0.0
    )
    out_coconut, _ = generate_with_coconut(
        model, prompt, max_new=max_new, temperature=0.0,
        coconut_k=8, bot_id=None,
    )
    assert torch.equal(out_rfold, out_coconut), (
        "Coconut decode without bot_id diverged from rfold decode"
    )


def test_bot_id_set_but_never_sampled_matches_rfold():
    """Setting bot_id but never SEEing it must still be a no-op.

    We pick a vocab id that we are certain the model won't emit by:
      * generating reference tokens with rfold first;
      * forcing bot_id to a token that never appears.
    """
    model = _build()
    prompt = _ids(seed=4, T=4)
    max_new = 12

    out_ref, _ = generate_rfold(
        model, prompt, max_new=max_new, temperature=0.0
    )
    emitted = set(out_ref[0, prompt.shape[1]:].tolist())
    bot_id_unseen = next(i for i in range(_VOCAB) if i not in emitted)

    out_coconut, _ = generate_with_coconut(
        model, prompt, max_new=max_new, temperature=0.0,
        coconut_k=8, bot_id=bot_id_unseen,
    )
    assert torch.equal(out_ref, out_coconut)


# ---------------------------------------------------------------------------
# 5. <bot> at end of prompt fires the latent loop
# ---------------------------------------------------------------------------


def test_bot_at_prompt_end_fires_latent_loop():
    """Prompt ending in ``<bot>`` triggers k latent passes BEFORE first emit.

    We compare against an otherwise identical decode where bot_id=None
    (no latent loop). At least one of the next-K emitted tokens must
    differ.
    """
    model = _build()
    base = _ids(seed=5, T=3)
    prompt_with_bot = torch.cat(
        [base, torch.tensor([[_BOT_ID]], dtype=torch.long)], dim=1
    )
    max_new = 6

    out_no_loop, _ = generate_with_coconut(
        model, prompt_with_bot, max_new=max_new, temperature=0.0,
        coconut_k=8, bot_id=None,
    )
    out_with_loop, _ = generate_with_coconut(
        model, prompt_with_bot, max_new=max_new, temperature=0.0,
        coconut_k=8, bot_id=_BOT_ID,
    )
    # The latent loop must perturb at least one emitted token.
    assert not torch.equal(out_no_loop, out_with_loop), (
        "Latent loop at <bot> failed to perturb the next-K emitted tokens"
    )


# ---------------------------------------------------------------------------
# 6. coconut_step works without a LatentThinker (latent_k=0 model)
# ---------------------------------------------------------------------------


def test_coconut_step_works_without_latent_thinker():
    """Inference-side Coconut must not require a Coconut-trained model.

    When ``latent_k=0`` (training-time Coconut disabled), the model has
    no LatentThinker submodule, so we fall back to identity projections.
    The latent loop still runs through the block stack and refines the
    state; quality at the early steps may be worse than a Coconut-trained
    model but the path must not crash.
    """
    model = _build(latent_k=0)
    assert model.latent_thinker is None
    prompt = _ids(seed=6, T=4)
    _, state = prefill_state(model, prompt)
    state2 = coconut_step(model, state, k_steps=4)
    # Fp32 — exact zero diff would be a sign the loop bypassed entirely.
    diffs = [
        (h2 - h1).abs().max().item()
        for h1, h2 in zip(state.h_cfc, state2.h_cfc)
    ]
    assert any(d > 0.0 for d in diffs)
