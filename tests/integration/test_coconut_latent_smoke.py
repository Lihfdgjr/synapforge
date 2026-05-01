"""T2.9 — Coconut latent thinking forward-smoke tests.

Reference: arXiv:2412.06769 ("Training Large Language Models to Reason in
a Continuous Latent Space"). The Coconut recipe inserts <bot>...<eot>
tokens around a reasoning span; inside that span, the model's hidden
state is fed back as the next-step input WITHOUT going through the LM
head (no token sampling). In our adaptation, ``SynapForge100M`` exposes a
``latent_k`` budget: when > 0, after the normal block stack runs over
the input ids, the last-token hidden is refined by ``latent_k`` extra
continuous-thought passes, then spliced back in before ``ln_f``.

This test suite verifies:

1. ``latent_k=0`` is a strict no-op (default behaviour, zero overhead,
   no extra modules in state_dict).
2. ``latent_k=8`` produces non-NaN output of the correct shape on a
   small (d=128, n_layers=2) factory build.
3. ``latent_k=8`` actually changes the logits compared to ``latent_k=0``
   (sanity that the latent loop is wired, not an accidental bypass).
4. Forward at ``d=128, n_layers=2, latent_k=8`` completes in under 5 s on
   CPU (smoke perf budget).

All tests use the real ``build_synapforge_100m`` factory — no mock model.
"""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.thinking.coconut import LatentThinker  # noqa: E402


# ---------------------------------------------------------------------------
# Shared smoke config -- d=128 n_layers=2 keeps each factory build under ~1 s
# on CPU and lets the latent_k=8 forward complete in well under 5 s.
# ---------------------------------------------------------------------------

_VOCAB = 256
_D = 128
_N_LAYERS = 2
_LOOP_DEPTH = 1
_MAX_SEQ = 16
_BATCH = 2
_SEQ = 8


def _build(latent_k: int):
    return build_synapforge_100m(
        vocab=_VOCAB,
        d=_D,
        n_layers=_N_LAYERS,
        loop_depth=_LOOP_DEPTH,
        max_seq=_MAX_SEQ,
        ffn_ratio=4.0,
        sparsity=0.5,
        dropout=0.0,
        use_grad_checkpoint=False,
        freeze_vocab_tail=False,
        live_vocab=_VOCAB,
        lm_head_spectral_norm=False,
        weight_quant_cfc="none",
        latent_k=latent_k,
    )


def _ids(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=g, dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. default: latent_k=0 is a no-op
# ---------------------------------------------------------------------------


def test_default_no_latent():
    """``latent_k=0`` (default) must NOT instantiate LatentThinker.

    This guarantees zero parameter / state_dict overhead for the Coconut
    feature when disabled — preserving exact bit-equivalence with the
    pre-T2.9 baseline checkpoint footprint.
    """
    model = _build(latent_k=0)
    assert model.latent_k == 0
    assert model.latent_thinker is None, (
        "latent_thinker must be None when latent_k=0 (no extra params)"
    )
    # No LatentThinker instance should appear anywhere in the module tree.
    for m in model.modules():
        assert not isinstance(m, LatentThinker), (
            "found a LatentThinker submodule despite latent_k=0; this would "
            "leak unused params into state_dict and break warmstart parity"
        )


# ---------------------------------------------------------------------------
# 2. latent_k=8 forward produces non-NaN output of correct shape
# ---------------------------------------------------------------------------


def test_latent_k_8_no_nan():
    """Forward at latent_k=8 must complete and produce finite logits."""
    torch.manual_seed(1234)
    model = _build(latent_k=8).eval()
    assert isinstance(model.latent_thinker, LatentThinker)
    assert model.latent_k == 8

    ids = _ids(seed=1)
    with torch.no_grad():
        logits = model(ids)

    assert torch.isfinite(logits).all(), (
        "Coconut latent k=8 forward produced non-finite logits "
        "(NaN/Inf) — latent loop diverged"
    )


def test_latent_k_8_shape():
    """Output shape (B, T, V) must be preserved by the latent splice."""
    torch.manual_seed(1234)
    model = _build(latent_k=8).eval()
    ids = _ids(seed=2)
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (_BATCH, _SEQ, _VOCAB), (
        f"shape mismatch after latent splice: got {tuple(logits.shape)}, "
        f"expected ({_BATCH}, {_SEQ}, {_VOCAB})"
    )


# ---------------------------------------------------------------------------
# 3. latent_k=8 actually changes the logits vs latent_k=0
# ---------------------------------------------------------------------------


def test_latent_k_changes_logits():
    """Sanity: latent_k=8 must produce different logits than latent_k=0.

    Both models share the SAME init seed for backbone weights, so any
    difference must come from the latent loop (the LatentThinker projections
    are init'd to identity, so the first latent step is a near-no-op, but
    K=8 cumulative passes through the HybridBlock stack do refine the
    last-token hidden non-trivially).
    """
    torch.manual_seed(20260502)
    m_off = _build(latent_k=0).eval()
    # Re-seed and rebuild so backbone weights are identical between runs.
    torch.manual_seed(20260502)
    m_on = _build(latent_k=8).eval()
    # Copy backbone weights from m_off into m_on so ONLY the latent path
    # differs — this isolates the effect of the K=8 thinking passes.
    sd_off = m_off.state_dict()
    sd_on = m_on.state_dict()
    for key in sd_off:
        if key in sd_on:
            sd_on[key] = sd_off[key].clone()
    m_on.load_state_dict(sd_on, strict=False)

    ids = _ids(seed=3)
    with torch.no_grad():
        logits_off = m_off(ids)
        logits_on = m_on(ids)

    # Last-token logits must differ (that is where the latent splice lands).
    diff_last = (logits_on[:, -1] - logits_off[:, -1]).abs().max().item()
    assert diff_last > 0.0, (
        "latent_k=8 produced identical last-token logits to latent_k=0 — "
        "the latent loop is silently bypassed"
    )


# ---------------------------------------------------------------------------
# 4. perf budget: latent_k=8 forward < 5 s on CPU
# ---------------------------------------------------------------------------


def test_latent_k_8_perf_budget_5s():
    """Forward must finish in <5 s on CPU at d=128, n_layers=2, k=8."""
    torch.manual_seed(7)
    model = _build(latent_k=8).eval()
    ids = _ids(seed=4)
    # Warm: torch.compile / lazy-init paths for the HybridBlock primitives.
    with torch.no_grad():
        _ = model(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(ids)
    dt = time.perf_counter() - t0
    assert dt < 5.0, (
        f"latent_k=8 forward took {dt:.2f}s on CPU; expected <5s for "
        f"d={_D} n_layers={_N_LAYERS} loop_depth={_LOOP_DEPTH}"
    )


# ---------------------------------------------------------------------------
# 5. Defensive: latent_k must be non-negative
# ---------------------------------------------------------------------------


def test_latent_k_negative_rejected():
    with pytest.raises(ValueError):
        _build(latent_k=-1)
