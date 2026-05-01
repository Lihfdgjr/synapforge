"""docs/PERF_KNOBS.md / DEEP_MAINT_QUEUE.md T2.6 — LM head spectral norm.

Validates that ``SynapForge100M(lm_head_spectral_norm=True)`` correctly
reparameterises the LM projection via ``torch.nn.utils.spectral_norm``,
bounding the operator-2-norm of the LM head matrix and therefore the
partition function (logsumexp over vocab).

Why this matters (P28 z-loss linear drift):
    The trainer uses a PaLM/Gemma-style ``z = log Z`` regularizer and the
    observation in PROGRESS.md shows ``z`` growing roughly linearly with
    step count. That happens because nothing bounds the LM head weight
    norm — Adam pushes it up, ``log Z`` follows, and the regularizer
    fights but never wins. Spectral_norm reparameterises ``W`` as
    ``W_orig / sigma_top(W_orig)``, capping the operator norm at 1
    (modulo the per-step power-iteration drift). With sigma capped, the
    logits are bounded ``||logits|| <= ||x||``, which means ``log Z``
    is bounded too — so z-loss can no longer drift linearly.

Tests:
    1. tied path: build with default ``tie_lm_head=True`` and
       ``lm_head_spectral_norm=True``, assert ``tok_embed`` carries
       the spectral_norm reparametrisation artifacts (``weight_orig``,
       ``weight_u``, ``weight_v``) and a forward pass produces a
       finite logits tensor of the right shape.
    2. untied path: same but ``tie_lm_head=False``, artifacts on
       ``lm_head`` instead.
    3. bf16 path: cast the tied model to bf16 and run forward; the
       spectral_norm power-iteration buffers stay fp32 internally
       under the standard PyTorch implementation, but the public
       ``weight`` projection still works under bf16 input.
    4. flag-off default: ``lm_head_spectral_norm=False`` (default)
       does NOT add the artifacts, so existing checkpoints stay
       backwards-compatible and the trainer can opt-in via the new
       ``--lm-head-spectral-norm`` CLI flag without touching any
       model that was saved before this commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_torch_and_model():
    """Lazy-import torch + the 100M model; skip cleanly if torch absent."""
    pytest.importorskip("torch")
    import torch  # noqa: F401  (used by callers)

    from synapforge.model_100m import SynapForge100M
    return torch, SynapForge100M


def _tiny_kwargs() -> dict:
    """Tiny model dims so CPU CI can run end-to-end in a few seconds.

    vocab=128 keeps the (B, T, V) softmax intermediate negligible.
    d=32, n_layers=1, loop_depth=1 means one HybridBlock unrolled once
    per forward — enough to exercise tok_embed -> blocks -> ln_f ->
    F.linear(x, tok_embed.weight) without paying the 99M-param tax.
    """
    return dict(
        vocab=128,
        d=32,
        n_layers=1,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.1,  # SparseSynapse requires (0, 1]
        dropout=0.0,
        freeze_vocab_tail=False,  # vocab=128 < QWEN25_LIVE_VOCAB
    )


def test_tied_spectral_norm_artifacts_and_forward():
    """Tied path: spectral_norm wraps tok_embed; forward returns finite."""
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(
        tie_lm_head=True, lm_head_spectral_norm=True, **_tiny_kwargs()
    )

    # spectral_norm replaces ``weight`` with a property; the underlying
    # parameter is ``weight_orig`` and the power-iteration vectors are
    # ``weight_u`` / ``weight_v``. (PyTorch >= 1.0 contract.)
    assert hasattr(model.tok_embed, "weight_orig"), (
        "tied + spectral_norm: tok_embed should expose weight_orig"
    )
    assert hasattr(model.tok_embed, "weight_u"), (
        "tied + spectral_norm: tok_embed should expose weight_u "
        "(power-iter buffer)"
    )
    assert hasattr(model.tok_embed, "weight_v"), (
        "tied + spectral_norm: tok_embed should expose weight_v "
        "(power-iter buffer)"
    )
    # The recomputed ``weight`` is still accessible — and that's the one
    # forward() / F.linear() use.
    assert model.tok_embed.weight.shape == (128, 32)

    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all(), "fp32 forward produced non-finite"


def test_untied_spectral_norm_artifacts():
    """Untied path: spectral_norm wraps lm_head, NOT tok_embed."""
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(
        tie_lm_head=False, lm_head_spectral_norm=True, **_tiny_kwargs()
    )

    assert model.lm_head is not None
    assert hasattr(model.lm_head, "weight_orig"), (
        "untied + spectral_norm: lm_head should expose weight_orig"
    )
    assert hasattr(model.lm_head, "weight_u"), (
        "untied + spectral_norm: lm_head should expose weight_u"
    )
    # tok_embed stays vanilla in the untied path.
    assert not hasattr(model.tok_embed, "weight_orig"), (
        "untied + spectral_norm: tok_embed must NOT be wrapped"
    )

    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()


def test_flag_off_default_no_artifacts():
    """Default lm_head_spectral_norm=False keeps ckpts backwards-compatible.

    Existing best_*.pt checkpoints don't carry weight_u/weight_v/weight_orig
    keys; loading them into a model built with the flag OFF must not
    require any name mapping. So we assert the artifacts are absent in
    the default path.
    """
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(tie_lm_head=True, **_tiny_kwargs())
    assert not hasattr(model.tok_embed, "weight_orig"), (
        "flag OFF default must not add spectral_norm artifacts"
    )
    # Sanity: forward still works in the no-spectral-norm baseline.
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)


def test_bf16_forward_under_spectral_norm():
    """bf16 path: cast model + input to bf16 and verify forward is finite.

    The standard PyTorch spectral_norm hook keeps its internal
    ``weight_u``/``weight_v`` power-iter vectors in whatever dtype
    the parameter has. After ``.bfloat16()`` they will be bf16 too —
    which is fine for the forward path (sigma is computed once as a
    scalar, the divide-by-sigma broadcast stays in bf16). The known
    quirk is gradient-direction drift over many steps; that's a
    training-time concern, not a forward-time one. Here we just
    assert no NaN / no shape error on a single forward.
    """
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(
        tie_lm_head=True, lm_head_spectral_norm=True, **_tiny_kwargs()
    ).to(dtype=torch.bfloat16)

    # Sanity: the spectral_norm artifacts survived the dtype cast.
    assert hasattr(model.tok_embed, "weight_orig")
    assert model.tok_embed.weight_orig.dtype == torch.bfloat16

    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all(), (
        "bf16 forward produced non-finite logits under spectral_norm"
    )
