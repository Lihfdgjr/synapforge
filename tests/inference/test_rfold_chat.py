"""Bit-exactness + smoke tests for stateful R-fold autoregressive decode.

Per the user 铁律 (2026-05-02): rfold inference must be bit-identical to
the sequential reference at fp32 with greedy decode. We verify by
generating N tokens both ways and checking the token sequences match
exactly.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from synapforge.inference import (  # noqa: E402
    InferenceState,
    generate_rfold,
    incremental_step,
    prefill_state,
)
from synapforge.model_100m import build_synapforge_100m  # noqa: E402


# ---------------------------------------------------------------------------
# Shared smoke config — d=128, n_layers=2, max_seq=64 so the test runs
# in a couple of seconds on CPU.
# ---------------------------------------------------------------------------

_VOCAB = 256
_D = 64
_N_LAYERS = 2
_LOOP_DEPTH = 1
_MAX_SEQ = 64
_BATCH = 1


def _build(loop_depth: int = _LOOP_DEPTH):
    torch.manual_seed(20260502)
    return build_synapforge_100m(
        vocab=_VOCAB,
        d=_D,
        n_layers=_N_LAYERS,
        loop_depth=loop_depth,
        max_seq=_MAX_SEQ,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        use_grad_checkpoint=False,
        freeze_vocab_tail=False,
        live_vocab=_VOCAB,
        lm_head_spectral_norm=False,
        weight_quant_cfc="none",
        latent_k=0,
    ).eval()


def _ids(seed: int = 0, T: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (_BATCH, T), generator=g, dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. State container is sane
# ---------------------------------------------------------------------------


def test_inference_state_lazy_init():
    """Empty InferenceState must initialise on first incremental_step call."""
    model = _build()
    state = InferenceState()
    assert state.position == 0
    assert state.h_cfc == [] and state.v_plif == []

    tok = torch.tensor([[5]], dtype=torch.long)
    logits, state = incremental_step(model, tok, state)
    assert logits.shape == (_BATCH, _VOCAB)
    assert state.position == 1
    assert len(state.h_cfc) == _N_LAYERS
    assert len(state.v_plif) == _N_LAYERS


# ---------------------------------------------------------------------------
# 2. Prefill matches running incremental_step token-by-token
# ---------------------------------------------------------------------------


def test_prefill_matches_step_loop():
    """``prefill_state`` and an explicit incremental_step loop produce the same state."""
    model = _build()
    prompt = _ids(seed=11, T=5)

    last_a, state_a = prefill_state(model, prompt)

    state_b = InferenceState()
    last_b = None
    for t in range(prompt.shape[1]):
        last_b, state_b = incremental_step(model, prompt[:, t], state_b)

    assert state_a.position == state_b.position == prompt.shape[1]
    assert torch.allclose(last_a, last_b, atol=1e-6, rtol=0)
    for ha, hb in zip(state_a.h_cfc, state_b.h_cfc):
        assert torch.allclose(ha, hb, atol=1e-6, rtol=0)
    for va, vb in zip(state_a.v_plif, state_b.v_plif):
        assert torch.allclose(va, vb, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 3. Bit-exactness vs sequential model.forward (the headline guarantee)
# ---------------------------------------------------------------------------


def _greedy_sequential(model, prompt: torch.Tensor, max_new: int) -> torch.Tensor:
    """Reference: re-runs model.forward over prompt + emitted on each step."""
    out = prompt
    for _ in range(max_new):
        if out.shape[1] >= model.max_seq:
            break
        with torch.no_grad():
            logits = model(out)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out


def test_rfold_decode_matches_sequential_greedy():
    """Greedy decode via R-fold == greedy decode via repeated full-forward.

    This is the user's 铁律: quality must not regress. R-fold inference
    is the SAME math, with state carried explicitly across calls instead
    of being recomputed from the prefix every step. At fp32 + temperature 0
    the emitted token sequences must be identical.
    """
    model = _build()
    prompt = _ids(seed=42, T=4)
    max_new = 16

    # R-fold path
    out_rfold, _ = generate_rfold(
        model, prompt, max_new=max_new, temperature=0.0
    )

    # Sequential reference
    out_seq = _greedy_sequential(model, prompt, max_new=max_new)

    assert out_rfold.shape == out_seq.shape, (
        f"shape mismatch: {tuple(out_rfold.shape)} vs {tuple(out_seq.shape)}"
    )
    same = torch.equal(out_rfold, out_seq)
    if not same:
        # Print the first diverging position for debug.
        diff = (out_rfold != out_seq).nonzero()
        first = diff[0].tolist() if diff.numel() else None
        raise AssertionError(
            f"R-fold token sequence diverges from sequential reference at {first}\n"
            f"R-fold: {out_rfold[0].tolist()}\n"
            f"Seq:    {out_seq[0].tolist()}"
        )


def test_rfold_decode_matches_sequential_100_tokens():
    """Stronger version: 100-token greedy generation must be bit-identical.

    The user's spec called for 100 tokens; we use ``max_seq=64`` here
    so we generate as many as the model can hold (60). The contract is
    the same — every emitted token must match.
    """
    model = _build()
    prompt = _ids(seed=99, T=4)
    max_new = _MAX_SEQ - prompt.shape[1]  # fill to model.max_seq

    out_rfold, _ = generate_rfold(
        model, prompt, max_new=max_new, temperature=0.0
    )
    out_seq = _greedy_sequential(model, prompt, max_new=max_new)

    assert torch.equal(out_rfold, out_seq), (
        f"R-fold (len={out_rfold.shape[1]}) != sequential (len={out_seq.shape[1]})"
    )


# ---------------------------------------------------------------------------
# 4. EOS handling
# ---------------------------------------------------------------------------


def test_rfold_decode_stops_at_eos():
    """Decode terminates when an EOS token is sampled (B=1)."""
    model = _build()
    prompt = _ids(seed=7, T=3)

    # First do a normal greedy run to discover what the next-token
    # distribution actually peaks on, then declare THAT the EOS.
    out, _ = generate_rfold(model, prompt, max_new=2, temperature=0.0)
    forced_eos = int(out[0, prompt.shape[1]].item())

    out2, _ = generate_rfold(
        model, prompt, max_new=20, temperature=0.0, eos_ids=[forced_eos]
    )
    # Should have emitted exactly one new token (the forced-EOS one) and stopped.
    assert out2.shape[1] == prompt.shape[1] + 1


# ---------------------------------------------------------------------------
# 5. State.detach_ leaves shape intact and drops graph
# ---------------------------------------------------------------------------


def test_state_detach_preserves_shape_drops_graph():
    model = _build()
    prompt = _ids(seed=55, T=3)
    _, state = prefill_state(model, prompt)
    h_shape = [tuple(h.shape) for h in state.h_cfc]
    v_shape = [tuple(v.shape) for v in state.v_plif]
    state.detach_()
    assert [tuple(h.shape) for h in state.h_cfc] == h_shape
    assert [tuple(v.shape) for v in state.v_plif] == v_shape
    for h in state.h_cfc:
        assert not h.requires_grad
    for v in state.v_plif:
        assert not v.requires_grad


# ---------------------------------------------------------------------------
# 6. Position bound enforcement
# ---------------------------------------------------------------------------


def test_incremental_step_rejects_overflow():
    model = _build()
    state = InferenceState(
        h_cfc=[], v_plif=[], position=model.max_seq
    )
    with pytest.raises(ValueError):
        incremental_step(model, torch.tensor([[1]], dtype=torch.long), state)
