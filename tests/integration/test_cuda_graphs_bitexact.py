"""GraphedHybridBlock — bit-exact contract tests.

Pins the contract that ``GraphedHybridBlock`` produces the same loss
value (within fp32 round-off, ``rel_err < 1e-6``) as the eager path
for the SAME ``(model, x, y)``. The captured graph runs the same
kernels in the same order; the only difference is dispatch overhead.

CUDA-only: the module's static-shape pinning + warmup + capture path
requires a real CUDA device. Tests are skipped on no-CUDA builds via
``pytest.importorskip("torch.cuda")`` + ``torch.cuda.is_available()``.

We additionally check the **structural** contract on no-CUDA builds:
the wrapper's ``capture_active`` must be False with a populated
``skip_reason`` (no crashes, no surprise exceptions).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from synapforge.training.cuda_graphs import (  # noqa: E402
    GraphedBlockCfg,
    GraphedHybridBlock,
    cross_entropy_loss,
    _cuda_graphs_available,
)


class _TinyLM(torch.nn.Module):
    """Smallest LM-shaped module that exercises the same kernel
    sequence as the production HybridBlock stack: embed → 1 linear →
    head. Cheap enough to capture+replay on any GPU we'd test on.
    """

    def __init__(self, vocab: int = 64, d: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.linear = torch.nn.Linear(d, d, bias=False)
        self.head = torch.nn.Linear(d, vocab, bias=False)
        # Deterministic init so different test runs see identical
        # outputs.
        torch.manual_seed(42)
        torch.nn.init.normal_(self.embed.weight, std=0.02)
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = torch.nn.functional.silu(self.linear(h))
        return self.head(h)


def test_graphed_no_cuda_is_safe_skip() -> None:
    """On a no-CUDA build, ``GraphedHybridBlock`` must construct
    without raising and report ``capture_active=False`` with a
    populated ``skip_reason``. The trainer relies on this so the
    flag is safe to leave on in CI / dev boxes.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA available — exercise the real path elsewhere")
    model = _TinyLM()
    cfg = GraphedBlockCfg(
        batch_size=4, seq_len=8,
        device=torch.device("cpu"),  # type: ignore[arg-type]
        dtype=None,
        n_warmup_iters=2,
    )
    g = GraphedHybridBlock(model, cross_entropy_loss, cfg)
    assert g.capture_active is False
    assert g.skip_reason  # non-empty


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graphs require a real CUDA device",
)
def test_graphed_loss_matches_eager_fp32() -> None:
    """Headline bit-exact test: the captured graph's loss must match
    the eager-path loss to within fp32 round-off.

    We run with ``dtype=None`` (no autocast) so both paths run pure
    fp32 — autocast bf16 has non-deterministic atomic adds in the
    backward that drive rel_err up to ~1e-3, which is fine for
    training but not the right gauge for a bit-exact regression
    check.
    """
    if not _cuda_graphs_available():
        pytest.skip("torch.cuda.CUDAGraph not available")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _TinyLM(vocab=64, d=16).to(device)
    cfg = GraphedBlockCfg(
        batch_size=4, seq_len=8,
        device=device, dtype=None,
        n_warmup_iters=5,
        accumulate_grad=False,
    )
    graphed = GraphedHybridBlock(model, cross_entropy_loss, cfg)
    if not graphed.capture_active:
        pytest.skip(
            f"capture failed on this GPU: {graphed.skip_reason}"
        )

    # Snapshot the post-warmup post-capture model state. We compare
    # against a fresh "eager" reference model that starts from the
    # SAME state, so capture-side warmup + the in-graph backward are
    # both already baked in.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    rng = torch.Generator(device=device).manual_seed(123)
    x = torch.randint(0, 64, (4, 8), generator=rng, device=device)
    y = torch.randint(0, 64, (4, 8), generator=rng, device=device)

    # Run graph step.
    model.zero_grad(set_to_none=True)
    loss_g = graphed.step(x, y).item()
    grads_g = {
        n: p.grad.detach().clone() for n, p in model.named_parameters()
        if p.grad is not None
    }

    # Reference eager path on a fresh model (same init).
    model_ref = _TinyLM(vocab=64, d=16).to(device)
    model_ref.load_state_dict(state)
    model_ref.zero_grad(set_to_none=True)
    logits = model_ref(x)
    loss_e = cross_entropy_loss(logits, y)
    loss_e.backward()
    loss_e_val = float(loss_e.detach().item())
    grads_e = {
        n: p.grad.detach().clone() for n, p in model_ref.named_parameters()
        if p.grad is not None
    }

    # The captured graph runs the SAME kernels — loss should match to
    # fp32 round-off.
    rel_err = abs(loss_g - loss_e_val) / max(abs(loss_e_val), 1e-12)
    assert rel_err < 1e-6, (
        f"loss rel_err {rel_err:.3e} exceeds 1e-6 (graph={loss_g}, "
        f"eager={loss_e_val})"
    )
    # Gradients should also match (the graph captured the backward).
    for name in grads_e:
        gg = grads_g[name]
        ge = grads_e[name]
        max_abs = (gg - ge).abs().max().item()
        ge_norm = ge.abs().max().item()
        rel = max_abs / max(ge_norm, 1e-12)
        assert rel < 1e-5, (
            f"grad {name} rel_err {rel:.3e} exceeds 1e-5 "
            f"(max_abs={max_abs:.3e}, ge_norm={ge_norm:.3e})"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graphs require a real CUDA device",
)
def test_graphed_replay_is_repeatable() -> None:
    """Calling ``.step()`` twice with the SAME inputs must give the
    same loss (modulo accumulating gradients into ``param.grad``).
    """
    if not _cuda_graphs_available():
        pytest.skip("torch.cuda.CUDAGraph not available")

    device = torch.device("cuda")
    torch.manual_seed(7)
    model = _TinyLM(vocab=64, d=16).to(device)
    cfg = GraphedBlockCfg(
        batch_size=4, seq_len=8,
        device=device, dtype=None,
        n_warmup_iters=5,
        accumulate_grad=False,  # graph zeros grads inside
    )
    graphed = GraphedHybridBlock(model, cross_entropy_loss, cfg)
    if not graphed.capture_active:
        pytest.skip(
            f"capture failed on this GPU: {graphed.skip_reason}"
        )
    rng = torch.Generator(device=device).manual_seed(11)
    x = torch.randint(0, 64, (4, 8), generator=rng, device=device)
    y = torch.randint(0, 64, (4, 8), generator=rng, device=device)
    losses = []
    for _ in range(3):
        loss = graphed.step(x, y).item()
        losses.append(loss)
    # All three replays should give the same loss; no model param
    # update happens inside the graph (we don't capture optimizer.step).
    for i in (1, 2):
        rel_err = abs(losses[i] - losses[0]) / max(abs(losses[0]), 1e-12)
        assert rel_err < 1e-6, (
            f"replay {i} drifted: {losses[i]} vs {losses[0]} "
            f"rel_err={rel_err:.3e}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graphs require a real CUDA device",
)
def test_graphed_partial_batch_falls_back_to_eager() -> None:
    """A shape-mismatched call must transparently fall back to the
    eager step (no exception, just slower).
    """
    if not _cuda_graphs_available():
        pytest.skip("torch.cuda.CUDAGraph not available")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _TinyLM(vocab=64, d=16).to(device)
    cfg = GraphedBlockCfg(
        batch_size=4, seq_len=8,
        device=device, dtype=None, n_warmup_iters=5,
    )
    graphed = GraphedHybridBlock(model, cross_entropy_loss, cfg)
    if not graphed.capture_active:
        pytest.skip(
            f"capture failed on this GPU: {graphed.skip_reason}"
        )
    # B=3 instead of B=4 — shape mismatch.
    rng = torch.Generator(device=device).manual_seed(13)
    x = torch.randint(0, 64, (3, 8), generator=rng, device=device)
    y = torch.randint(0, 64, (3, 8), generator=rng, device=device)
    model.zero_grad(set_to_none=True)
    loss = graphed.step(x, y)
    # Should produce a finite scalar — eager fell back.
    assert torch.isfinite(loss).all(), "eager fallback produced non-finite loss"


def test_graphed_repr_safe() -> None:
    """``repr()`` runs without crashing on no-CUDA builds."""
    model = _TinyLM()
    if torch.cuda.is_available():
        # On CUDA, capture might succeed or fall back, both fine.
        cfg = GraphedBlockCfg(
            batch_size=2, seq_len=4,
            device=torch.device("cuda"), dtype=None, n_warmup_iters=2,
        )
        model = model.to("cuda")
    else:
        cfg = GraphedBlockCfg(
            batch_size=2, seq_len=4,
            device=torch.device("cpu"),  # type: ignore[arg-type]
            dtype=None, n_warmup_iters=2,
        )
    g = GraphedHybridBlock(model, cross_entropy_loss, cfg)
    s = repr(g)
    assert "GraphedHybridBlock" in s
    assert "B=2" in s
    assert "T=4" in s
