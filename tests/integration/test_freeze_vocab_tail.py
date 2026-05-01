"""DEEP_MAINT_QUEUE.md T2.4 — frozen vocab tail backward hook.

Asserts that rows ``[QWEN25_LIVE_VOCAB, vocab)`` of ``tok_embed.weight``
(and ``lm_head.weight`` when untied) do NOT change after a few SGD-driven
optimizer steps when ``freeze_vocab_tail=True``, and DO change when the
flag is off.

Why: Qwen 2.5 emits IDs in ``[0, 151643)``; rows beyond that are random-init
padding that never see real gradient through the forward path. They WOULD
still drift under Adam noise + weight decay, polluting the LM head distance
distribution. The backward hook in ``SynapForge100M.__init__`` zeros their
gradient slice so the optimizer step leaves them at their init values.

CPU-only — uses a tiny model (d=32, n_layers=2, vocab=151936) so the test
runs in seconds and doesn't need a GPU. We make the live_vocab boundary
small (e.g. 64) so a random-id batch reliably hits both halves of the
embedding within 5 steps without us having to draw 151K-id minibatches.
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


def _build_model(torch_mod, *, freeze: bool, tie: bool, vocab: int = 256,
                 live_vocab: int = 64):
    from synapforge.model_100m import SynapForge100M

    torch_mod.manual_seed(0)
    model = SynapForge100M(
        vocab=vocab,
        d=32,
        n_layers=2,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        tie_lm_head=tie,
        use_grad_checkpoint=False,
        freeze_vocab_tail=freeze,
        live_vocab=live_vocab,
    )
    return model


def _run_steps(torch_mod, model, *, steps: int, vocab: int, lr: float = 1e-1):
    """5 random-batch SGD steps. lr=1e-1 makes any unfrozen drift highly
    visible; SGD (not Adam) so we don't need to model momentum decay,
    keeping the freeze test purely about the gradient hook."""
    opt = torch_mod.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        ids = torch_mod.randint(0, vocab, (4, 8), dtype=torch_mod.long)
        logits = model(ids)
        # arbitrary scalar loss that touches all logits, so all live-vocab
        # rows DO get a real gradient (proving the test isn't trivially
        # passing because nothing has any gradient).
        loss = logits.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def test_freeze_tail_keeps_tied_lm_head_rows_unchanged(torch_mod):
    """freeze_vocab_tail=True + tied head: tail rows of tok_embed are frozen."""
    vocab = 256
    live = 64
    model = _build_model(torch_mod, freeze=True, tie=True,
                         vocab=vocab, live_vocab=live)
    before_tail = model.tok_embed.weight[live:].detach().clone()
    before_live = model.tok_embed.weight[:live].detach().clone()

    _run_steps(torch_mod, model, steps=5, vocab=vocab)

    after_tail = model.tok_embed.weight[live:].detach()
    after_live = model.tok_embed.weight[:live].detach()

    assert torch_mod.allclose(before_tail, after_tail, atol=0.0), (
        "tail rows changed under freeze_vocab_tail=True; backward hook "
        "did not zero the tail gradient"
    )
    # Sanity: live rows DID change. If they didn't, the experiment is
    # vacuous (no gradient, no drift).
    diff_live = (after_live - before_live).abs().max().item()
    assert diff_live > 1e-6, (
        f"live rows didn't change ({diff_live=}); test is vacuous. "
        "Either lr too small or no gradient reached the embedding."
    )


def test_freeze_tail_disabled_lets_tail_rows_drift(torch_mod):
    """freeze_vocab_tail=False: tail rows DO change under SGD noise.

    Acts as the negative control for the main test — proves the hook
    is the thing keeping the rows frozen, not some accident of the
    forward graph never touching them. We feed random ids that include
    tail-ids on purpose, so a real gradient reaches every row.
    """
    vocab = 256
    live = 64
    model = _build_model(torch_mod, freeze=False, tie=True,
                         vocab=vocab, live_vocab=live)
    before_tail = model.tok_embed.weight[live:].detach().clone()

    _run_steps(torch_mod, model, steps=5, vocab=vocab)

    after_tail = model.tok_embed.weight[live:].detach()
    diff_tail = (after_tail - before_tail).abs().max().item()
    assert diff_tail > 1e-6, (
        f"tail rows didn't change with freeze_vocab_tail=False ({diff_tail=}); "
        "negative control failed — hook may be applied unconditionally"
    )


def test_freeze_tail_works_for_untied_lm_head(torch_mod):
    """freeze_vocab_tail=True + untied head: lm_head.weight tail also frozen.

    Two backward hooks are registered — one on tok_embed.weight and one on
    lm_head.weight. Both tail slices must stay pinned to their init values.
    """
    vocab = 256
    live = 64
    model = _build_model(torch_mod, freeze=True, tie=False,
                         vocab=vocab, live_vocab=live)
    before_emb = model.tok_embed.weight[live:].detach().clone()
    before_head = model.lm_head.weight[live:].detach().clone()

    _run_steps(torch_mod, model, steps=5, vocab=vocab)

    after_emb = model.tok_embed.weight[live:].detach()
    after_head = model.lm_head.weight[live:].detach()
    assert torch_mod.allclose(before_emb, after_emb, atol=0.0), (
        "tok_embed tail rows changed under freeze_vocab_tail=True (untied)"
    )
    assert torch_mod.allclose(before_head, after_head, atol=0.0), (
        "lm_head tail rows changed under freeze_vocab_tail=True (untied); "
        "the second backward hook on lm_head.weight is not firing"
    )


def test_qwen25_live_vocab_constant(torch_mod):
    """Sanity check: the canonical Qwen2.5 boundary is 151643.

    Locks the constant so refactoring doesn't silently shift the freeze
    boundary on the real 151936-vocab production model.
    """
    from synapforge.model_100m import QWEN25_LIVE_VOCAB
    assert QWEN25_LIVE_VOCAB == 151643, (
        f"QWEN25_LIVE_VOCAB drifted to {QWEN25_LIVE_VOCAB}; production "
        "trainer freezes rows >= 151643 only. If Qwen tokenizer changes, "
        "update this constant + re-verify the freeze boundary on disk."
    )
