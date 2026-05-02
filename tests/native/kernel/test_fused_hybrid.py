"""Bit-exact tests for ``FusedHybridBlock`` vs reference ``HybridBlock``.

These tests run on CPU (PyTorch reference path inside FusedHybridBlock)
so they're CI-portable. The Triton path is exercised by the rental
GPU test job.

Test plan
---------
1. ``test_smoke_tiny`` -- d=128, n_layers=1, B=2, T=8, fp32. Just
   checks the module instantiates and does forward + backward.
2. ``test_bitexact_fp32`` -- compare ``FusedHybridBlock(orig)(x)`` to
   ``orig(x)`` for x in fp32. Tolerance 1e-4. Same generator -> same
   weights, same ATan surrogate, same RMSNorm reduction order.
3. ``test_bitexact_bf16`` -- same as above but bf16, tolerance 5e-3.
4. ``test_grad_finite_difference`` -- numerical gradient check on a
   single scalar parameter for the simplest config (no SEW).
5. ``test_can_fuse_rejects_kwta`` -- capability probe gates kwta_k>0.
6. ``test_can_fuse_rejects_hp_residual`` -- capability probe gates
   high-pass conv.
"""
from __future__ import annotations

import os
import sys

# Allow `pytest tests/native/kernel/test_fused_hybrid.py` from the repo root.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import math
import torch
import pytest

from synapforge.model_100m import HybridBlock
from synapforge.native.kernel import (
    FusedHybridBlock,
    fused_hybrid_block_apply,
    can_fuse_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_block(d: int = 128, dtype: torch.dtype = torch.float32, **kw) -> HybridBlock:
    torch.manual_seed(0)
    defaults = dict(
        d=d,
        ffn_ratio=2.0,                      # smaller FFN for speed
        sparsity=0.5,                        # denser mask -> tighter test
        dropout=0.0,                         # fused doesn't fuse dropout
        sew_shortcut=False,                  # reference path
    )
    defaults.update(kw)
    block = HybridBlock(**defaults)
    block.eval()
    block = block.to(dtype)
    return block


def _input(B: int, T: int, D: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, T, D, dtype=dtype) * 0.5


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_smoke_tiny():
    """Most basic: builds, forward, backward all run without crash."""
    block = _make_block(d=64)
    fused = FusedHybridBlock.from_hybrid_block(block)
    x = _input(B=2, T=4, D=64)
    x.requires_grad_(True)
    y = fused(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_bitexact_fp32():
    """Compare fused output to reference HybridBlock at fp32, tol 1e-4.

    Both paths share the SAME weight tensors (FusedHybridBlock just
    holds references). The Triton fused kernel and the PyTorch
    reference inside FusedHybridBlock both compute the chain in fp32
    promotion -- they should agree to fp32 round-off.
    """
    d = 128
    block = _make_block(d=d, dtype=torch.float32)
    fused = FusedHybridBlock.from_hybrid_block(block)

    x = _input(B=2, T=8, D=d, dtype=torch.float32)
    with torch.no_grad():
        y_ref = block(x)
        y_fused = fused(x)
    rel_err = ((y_ref - y_fused).abs() / (y_ref.abs() + 1e-6)).max().item()
    abs_err = (y_ref - y_fused).abs().max().item()
    assert abs_err < 1e-4, (
        f"fp32 fused vs reference mismatch: abs={abs_err:.2e}, rel={rel_err:.2e}"
    )


def test_bitexact_with_sew():
    """Same as bitexact_fp32 but with sew_shortcut=True."""
    d = 128
    block = _make_block(d=d, dtype=torch.float32, sew_shortcut=True)
    fused = FusedHybridBlock.from_hybrid_block(block)

    x = _input(B=2, T=8, D=d, dtype=torch.float32)
    with torch.no_grad():
        y_ref = block(x)
        y_fused = fused(x)
    abs_err = (y_ref - y_fused).abs().max().item()
    assert abs_err < 1e-4, f"SEW fused vs reference mismatch: abs={abs_err:.2e}"


def test_bitexact_bf16():
    """bf16 output check, MEAN-abs tolerance 5e-3.

    bf16 has 7-bit mantissa, so per-element errors compound through
    the chain of exp + tanh + sigmoid + matmul ops. Worst-case max-abs
    can hit 5% on values near zero, but the AVERAGE error should stay
    in the 0.1-0.5% range. We test:

    * mean abs error < 5e-3  (matches the rest of the repo's bf16 bar)
    * abs error < 1e-1 element-wise   (no NaN / catastrophic cancel)
    * cosine similarity > 0.999       (direction agrees)
    """
    d = 128
    block = _make_block(d=d, dtype=torch.bfloat16)
    fused = FusedHybridBlock.from_hybrid_block(block)

    x = _input(B=2, T=8, D=d, dtype=torch.bfloat16)
    with torch.no_grad():
        y_ref = block(x)
        y_fused = fused(x)
    diff = (y_ref.float() - y_fused.float())
    mean_abs = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten().unsqueeze(0),
        y_fused.float().flatten().unsqueeze(0),
    ).item()
    assert mean_abs < 5e-3, f"bf16 mean_abs={mean_abs:.2e} too large"
    assert max_abs < 1e-1, f"bf16 max_abs={max_abs:.2e} catastrophic"
    assert cos > 0.999, f"bf16 cos sim {cos:.5f} too low"


def test_backward_grad_matches_autograd_fp32():
    """Compare the closed-form fused backward to torch autograd.

    Both paths use the SAME parameter tensors; the difference is the
    BWD compute graph. autograd unrolls the per-step Python loop,
    fused uses our closed-form. They must agree to fp32 round-off.
    """
    d = 64
    B, T = 2, 6

    block = _make_block(d=d, dtype=torch.float32)
    block.train()
    block_ref = HybridBlock(d=d, ffn_ratio=2.0, sparsity=0.5, dropout=0.0)
    block_ref.load_state_dict(block.state_dict())
    # Copy mask buffer too
    block_ref.synapse.mask.copy_(block.synapse.mask)
    block_ref.train()

    x = _input(B=B, T=T, D=d, dtype=torch.float32)

    # ---- reference path: vanilla HybridBlock.forward -> autograd ----
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = block_ref(x_ref)
    y_ref.sum().backward()
    g_x_ref = x_ref.grad.clone()
    grads_ref = {n: p.grad.clone() for n, p in block_ref.named_parameters() if p.grad is not None}

    # ---- fused path ----
    fused = FusedHybridBlock.from_hybrid_block(block)
    # zero block grads (the from_hybrid_block constructor SHARES weight refs;
    # clear them so grads accumulate cleanly).
    for p in block.parameters():
        if p.grad is not None:
            p.grad.zero_()

    x_fused = x.clone().detach().requires_grad_(True)
    y_fused = fused(x_fused)
    y_fused.sum().backward()
    g_x_fused = x_fused.grad.clone()
    grads_fused = {n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None}

    # Forward outputs match
    abs_fwd = (y_ref - y_fused).abs().max().item()
    assert abs_fwd < 1e-4, f"forward mismatch: {abs_fwd:.2e}"

    # Input gradient bit-exact (fp32 round-off only)
    abs_gx = (g_x_ref - g_x_fused).abs().max().item()
    assert abs_gx < 1e-5, f"grad_x abs={abs_gx:.2e} not within fp32 round-off"

    # Every parameter gradient matches the reference autograd to fp32
    # round-off.  Tolerance 1e-5 absolute (closed-form vs unrolled
    # autograd both done in fp32 -- only difference is op ORDERING).
    for n in grads_ref:
        if n in grads_fused:
            gr = grads_ref[n]
            gf = grads_fused[n]
            d_abs = (gr - gf).abs().max().item()
            ref_max = gr.abs().max().item()
            rel = d_abs / (ref_max + 1e-12)
            assert (d_abs < 1e-5 or rel < 1e-5), (
                f"param '{n}' grad mismatch: abs={d_abs:.2e} rel={rel:.2e} ref_max={ref_max:.2e}"
            )


def test_can_fuse_rejects_kwta():
    block = HybridBlock(d=64, ffn_ratio=2.0, sparsity=0.5, kwta_k=8)
    ok, reason = can_fuse_block(block)
    assert not ok
    assert "kwta" in reason


def test_can_fuse_rejects_hp_residual():
    block = HybridBlock(
        d=64, ffn_ratio=2.0, sparsity=0.5,
        high_pass_residual_weight=0.1,
    )
    ok, reason = can_fuse_block(block)
    assert not ok
    assert "high-pass" in reason


def test_can_fuse_accepts_baseline():
    block = HybridBlock(d=64, ffn_ratio=2.0, sparsity=0.5)
    ok, reason = can_fuse_block(block)
    assert ok, f"baseline block should fuse: {reason}"


def test_can_fuse_accepts_sew():
    block = HybridBlock(d=64, ffn_ratio=2.0, sparsity=0.5, sew_shortcut=True)
    ok, reason = can_fuse_block(block)
    assert ok


def test_capability_falls_back_kwta():
    """When kwta_k>0 the FusedHybridBlock CTOR must refuse."""
    block = HybridBlock(d=64, ffn_ratio=2.0, sparsity=0.5, kwta_k=8)
    with pytest.raises(ValueError, match="kwta"):
        FusedHybridBlock.from_hybrid_block(block)


def test_grad_finite_difference_scalar_param():
    """Numerical gradient check on a SMOOTH-ONLY parameter entry.

    Validates the closed-form backward against finite differences.

    Constraint: FD on parameters whose gradient flows ONLY through the
    discrete spike Heaviside (e.g. plif.threshold for fp32 inputs that
    are not within `eps` of a spike boundary) gives ZERO numerically
    because no spike state changes within the perturbation. The
    ATan surrogate gradient that the analytical bwd reports is the
    backward-only object that has no direct FD analog.

    To get a meaningful FD check we target an FFN weight (W_down) --
    its gradient flows entirely through smooth operations, so FD and
    the closed-form must agree to fp64 round-off.
    """
    d = 32
    B, T = 2, 4
    block = _make_block(d=d, dtype=torch.float64)
    fused = FusedHybridBlock.from_hybrid_block(block)
    x = _input(B=B, T=T, D=d, dtype=torch.float64)

    # Pick a single W_down entry.
    p = block.ffn.w_down.weight
    idx = (3, 7)  # arbitrary
    eps = 1e-5

    # Analytical grad through fused.
    for q in block.parameters():
        if q.grad is not None:
            q.grad.zero_()
    x_in = x.clone().detach().requires_grad_(True)
    y = fused(x_in)
    loss = y.sum()
    loss.backward()
    analytical = p.grad[idx[0], idx[1]].item()

    # Finite difference (central).
    with torch.no_grad():
        p[idx[0], idx[1]] += eps
        y_plus = fused(x).sum().item()
        p[idx[0], idx[1]] -= 2 * eps
        y_minus = fused(x).sum().item()
        p[idx[0], idx[1]] += eps  # restore

    numerical = (y_plus - y_minus) / (2 * eps)

    rel = abs(analytical - numerical) / (abs(numerical) + 1e-12)
    abs_err = abs(analytical - numerical)
    assert (rel < 1e-3) or (abs_err < 1e-6), (
        f"FD mismatch on w_down[{idx}]: analytical={analytical:.6e} "
        f"numerical={numerical:.6e} rel={rel:.2e} abs={abs_err:.2e}"
    )


if __name__ == "__main__":  # pragma: no cover
    test_smoke_tiny()
    test_bitexact_fp32()
    test_bitexact_with_sew()
    test_bitexact_bf16()
    test_backward_grad_matches_autograd_fp32()
    test_can_fuse_rejects_kwta()
    test_can_fuse_rejects_hp_residual()
    test_can_fuse_accepts_baseline()
    test_can_fuse_accepts_sew()
    test_capability_falls_back_kwta()
    test_grad_finite_difference_scalar_param()
    print("All fused HybridBlock tests passed.")
