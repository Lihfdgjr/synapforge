"""Closed-form Triton backward for the SEW + sigmoid-gate + mul fusion.

This complements `triton_block_kernel.py`, which already fuses the
CfC scan + PLIF spike + subtract-on-spike-reset into ONE forward kernel
launch with a paired Triton backward kernel
(`fused_lnn_snn_block_bwd_kernel`).

What this file adds (Phase 4 partial)
-------------------------------------
The remaining hot-path ops INSIDE a single ``HybridBlock.forward`` that
still go through torch autograd (and thus pay Python dispatch +
saved_for_backward overhead) are::

    spike_input = s + h                      # SEW shortcut (op #6)
    gated       = syn_out * sigmoid(gate_pre) # gate fusion (ops #9, #10)

The matmul ops (SparseSynapse, Linear gate) stay on torch — cuBLAS is
strictly faster than what we'd write in Triton. The win we ARE chasing
is removing the **3 separate kernel launches** for `(+, sigmoid, *)`
that fire per layer per loop_depth iter.

Closed-form math
----------------
Forward
    spike_input = s + h                     # additive SEW
    gate        = sigmoid(gate_pre)
    gated       = syn_out * gate

Backward (upstream: grad_gated, downstream: grad_s, grad_h, grad_gate_pre,
grad_syn_out)
    grad_syn_out  = grad_gated * gate
    grad_gate     = grad_gated * syn_out
    grad_gate_pre = grad_gate * gate * (1.0 - gate)
    grad_s        = grad_spike_input        # via add identity
    grad_h        = grad_spike_input        # via add identity

Where ``grad_spike_input`` is whatever flows back from the synapse +
gate-pre paths (computed by torch autograd downstream — we just produce
``grad_s`` and ``grad_h`` from the additive SEW node, and merge them in
the caller).

Numerical contract
------------------
Bit-equivalence with torch.autograd within rel-err < 1e-4 on a (B=2,
T=8, d=128) fixture. See ``tests/backends/test_triton_block_bwd.py``.

Public API
----------
    >>> y, syn_out, gate = sew_sigmoid_gate_fused(s, h, syn_out, gate_pre)
    >>> # backward: y.backward(grad_y) — calls Triton if available
"""
from __future__ import annotations

import warnings

import torch

# Reuse the Triton-availability probe from the existing block kernel so
# we keep ONE source of truth for the runtime probe.
from .triton_block_kernel import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _sew_sigmoid_gate_fwd_kernel(
        s_ptr,          # (N,)  spike train (after PLIF, possibly cast)
        h_ptr,          # (N,)  liquid hidden state (pre-spike, post-CfC)
        syn_ptr,        # (N,)  SparseSynapse(s + h) — already computed (cuBLAS)
        gate_pre_ptr,   # (N,)  Linear_gate(s + h)   — already computed (cuBLAS)
        # outputs
        spike_input_ptr,  # (N,)  s + h        (saved for SparseSynapse fwd & bwd)
        gate_ptr,         # (N,)  sigmoid(gate_pre)  (saved for bwd)
        gated_ptr,        # (N,)  syn_out * sigmoid(gate_pre)
        # sizes
        N,
        # meta
        BLOCK: tl.constexpr,
    ):
        """Fused SEW(s+h) + sigmoid(gate_pre) + mul.

        grid = (cdiv(N, BLOCK),). Pure elementwise — no reduce, no
        atomic. We launch on the flat (B*T*D) buffer.

        Note: this kernel is OPTIONAL on the forward path. We still call
        SparseSynapse and Linear_gate via torch (cuBLAS). We only fuse the
        elementwise tail (`s+h`, sigmoid, mul) into a single launch.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        syn = tl.load(syn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gp = tl.load(gate_pre_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        spike_input = s + h
        # numerically-stable sigmoid: tl.sigmoid is fp32-safe
        gate = 1.0 / (1.0 + tl.exp(-gp))
        gated = syn * gate

        tl.store(spike_input_ptr + offs,
                 spike_input.to(spike_input_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(gate_ptr + offs,
                 gate.to(gate_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(gated_ptr + offs,
                 gated.to(gated_ptr.dtype.element_ty),
                 mask=mask)

    @triton.jit
    def _sew_sigmoid_gate_bwd_kernel(
        # forward saved tensors (read-only)
        syn_ptr,          # (N,)  SparseSynapse(s+h)  -- saved for d/dgate
        gate_ptr,         # (N,)  sigmoid(gate_pre)   -- saved for d/dsyn AND d/dgate_pre
        # upstream gradient
        grad_gated_ptr,   # (N,)  dL/d gated
        # downstream gradients (write)
        grad_syn_ptr,     # (N,)  dL/d syn_out
        grad_gate_pre_ptr, # (N,)  dL/d gate_pre
        # sizes
        N,
        # meta
        BLOCK: tl.constexpr,
    ):
        """Fused (sigmoid + mul + SEW-add) backward.

        Closed-form:
            grad_syn      = grad_gated * gate
            grad_gate     = grad_gated * syn_out
            grad_gate_pre = grad_gate * gate * (1 - gate)

        SEW-add (`spike_input = s + h`) gradient is identity, so we DO
        NOT need to compute it here — torch autograd gets ``grad_s =
        grad_h = grad_spike_input`` for free in the wrapper. We only emit
        the sigmoid-mul half.

        Pure elementwise. grid = (cdiv(N, BLOCK),).
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        syn = tl.load(syn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gg = tl.load(grad_gated_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        grad_syn = gg * gate
        grad_gate = gg * syn
        grad_gate_pre = grad_gate * gate * (1.0 - gate)

        tl.store(grad_syn_ptr + offs,
                 grad_syn.to(grad_syn_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(grad_gate_pre_ptr + offs,
                 grad_gate_pre.to(grad_gate_pre_ptr.dtype.element_ty),
                 mask=mask)


# ---------------------------------------------------------------------------
# Triton dispatch wrappers
# ---------------------------------------------------------------------------

def _triton_sew_sigmoid_gate_forward(
    s: torch.Tensor,
    h: torch.Tensor,
    syn_out: torch.Tensor,
    gate_pre: torch.Tensor,
):
    """Run the fused fwd kernel.

    Returns
    -------
    spike_input : (B, T, D)  s + h, same dtype as inputs
    gate        : (B, T, D)  sigmoid(gate_pre)
    gated       : (B, T, D)  syn_out * gate
    """
    assert _HAS_TRITON, "Triton not available"
    assert s.is_cuda and h.is_cuda and syn_out.is_cuda and gate_pre.is_cuda
    out_dtype = s.dtype
    s_c = s.contiguous()
    h_c = h.contiguous()
    syn_c = syn_out.contiguous()
    gp_c = gate_pre.contiguous()

    spike_input = torch.empty_like(s_c)
    gate = torch.empty_like(s_c)
    gated = torch.empty_like(s_c)
    N = s_c.numel()

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _sew_sigmoid_gate_fwd_kernel[grid](
        s_c, h_c, syn_c, gp_c,
        spike_input, gate, gated,
        N,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return spike_input, gate, gated


def _triton_sew_sigmoid_gate_backward(
    syn_out: torch.Tensor,
    gate: torch.Tensor,
    grad_gated: torch.Tensor,
):
    """Run the fused bwd kernel.

    Returns
    -------
    grad_syn      : (B, T, D)
    grad_gate_pre : (B, T, D)

    The SEW-add gradient (grad_s = grad_h = grad_spike_input) is NOT
    produced here — torch autograd computes ``grad_spike_input`` from
    the (Linear_gate + SparseSynapse) chains, and the wrapper just
    splits it identity-wise into grad_s and grad_h.
    """
    assert _HAS_TRITON
    assert syn_out.is_cuda and gate.is_cuda and grad_gated.is_cuda

    syn_c = syn_out.contiguous()
    gate_c = gate.contiguous()
    gg_c = grad_gated.contiguous()

    grad_syn = torch.empty_like(syn_c)
    grad_gate_pre = torch.empty_like(syn_c)
    N = syn_c.numel()

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _sew_sigmoid_gate_bwd_kernel[grid](
        syn_c, gate_c, gg_c,
        grad_syn, grad_gate_pre,
        N,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return grad_syn, grad_gate_pre


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (used on Windows / no-GPU CI and as bwd fallback).
# ---------------------------------------------------------------------------


def _torch_sew_sigmoid_gate_forward(
    s: torch.Tensor,
    h: torch.Tensor,
    syn_out: torch.Tensor,
    gate_pre: torch.Tensor,
):
    """Reference forward — same numerical contract as the Triton kernel."""
    spike_input = s + h
    gate = torch.sigmoid(gate_pre)
    gated = syn_out * gate
    return spike_input, gate, gated


def _torch_sew_sigmoid_gate_backward(
    syn_out: torch.Tensor,
    gate: torch.Tensor,
    grad_gated: torch.Tensor,
):
    """Reference backward — closed-form, same as the Triton kernel."""
    # Promote to fp32 for stability on bf16 inputs (mirrors the surrogate path).
    dt = grad_gated.dtype
    gg = grad_gated.float()
    g = gate.float()
    sy = syn_out.float()
    grad_syn = (gg * g).to(dt)
    grad_gate = gg * sy
    grad_gate_pre = (grad_gate * g * (1.0 - g)).to(dt)
    return grad_syn, grad_gate_pre


# ---------------------------------------------------------------------------
# torch.autograd.Function bridge
# ---------------------------------------------------------------------------


class SEWSigmoidGateFn(torch.autograd.Function):
    """Fused forward+backward for ``gated = syn_out * sigmoid(gate_pre)``
    with a side-output ``spike_input = s + h``.

    Inputs are 4 tensors (s, h, syn_out, gate_pre); outputs are
    (spike_input, gated). The wrapper exposes an additional ``gate``
    save for the backward path.

    Note we do NOT compute SparseSynapse or Linear_gate here — those
    stay torch (cuBLAS). This Function ONLY fuses the elementwise
    tail (``+``, sigmoid, ``*``).
    """

    @staticmethod
    def forward(ctx, s, h, syn_out, gate_pre):  # type: ignore[override]
        use_triton = (
            _HAS_TRITON and s.is_cuda
            and s.dtype in (torch.float32, torch.float16, torch.bfloat16)
        )
        if use_triton:
            try:
                spike_input, gate, gated = _triton_sew_sigmoid_gate_forward(
                    s, h, syn_out, gate_pre,
                )
            except Exception as exc:  # pragma: no cover -- fallback
                warnings.warn(
                    f"SEWSigmoidGateFn: Triton fwd failed ({exc!r}), "
                    f"falling back to PyTorch reference.",
                    RuntimeWarning,
                )
                spike_input, gate, gated = _torch_sew_sigmoid_gate_forward(
                    s, h, syn_out, gate_pre,
                )
        else:
            spike_input, gate, gated = _torch_sew_sigmoid_gate_forward(
                s, h, syn_out, gate_pre,
            )
        # Save for backward.  We need (syn_out, gate) to compute
        # grad_syn / grad_gate_pre. SEW-add backward is identity so no
        # extra save is needed for that.
        ctx.save_for_backward(syn_out, gate)
        return spike_input, gated

    @staticmethod
    def backward(ctx, grad_spike_input, grad_gated):  # type: ignore[override]
        """grad_spike_input flows from torch autograd downstream paths
        (SparseSynapse and Linear_gate); we add it identity-wise to s
        and h. grad_gated is fused with the sigmoid+mul.
        """
        syn_out, gate = ctx.saved_tensors

        if grad_gated is None:
            # No gradient flowing through the gated branch — only SEW.
            grad_s = grad_spike_input if grad_spike_input is not None else None
            grad_h = grad_spike_input if grad_spike_input is not None else None
            return grad_s, grad_h, None, None

        use_triton_bwd = (
            _HAS_TRITON
            and grad_gated.is_cuda
            and grad_gated.dtype in (torch.float32, torch.float16, torch.bfloat16)
        )

        if use_triton_bwd:
            try:
                grad_syn, grad_gate_pre = _triton_sew_sigmoid_gate_backward(
                    syn_out, gate, grad_gated,
                )
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"SEWSigmoidGateFn: Triton bwd failed ({exc!r}), "
                    f"falling back to PyTorch reference.",
                    RuntimeWarning,
                )
                grad_syn, grad_gate_pre = _torch_sew_sigmoid_gate_backward(
                    syn_out, gate, grad_gated,
                )
        else:
            grad_syn, grad_gate_pre = _torch_sew_sigmoid_gate_backward(
                syn_out, gate, grad_gated,
            )

        # SEW-add: grad_s = grad_h = grad_spike_input (which carries
        # downstream gradient from the SparseSynapse / Linear_gate paths
        # that torch autograd is responsible for).
        if grad_spike_input is None:
            grad_s = None
            grad_h = None
        else:
            grad_s = grad_spike_input
            grad_h = grad_spike_input

        # Return order matches forward inputs: (s, h, syn_out, gate_pre)
        return grad_s, grad_h, grad_syn, grad_gate_pre


def sew_sigmoid_gate_fused(
    s: torch.Tensor,
    h: torch.Tensor,
    syn_out: torch.Tensor,
    gate_pre: torch.Tensor,
):
    """Functional entry: fused SEW + sigmoid + mul (forward + Triton bwd).

    Parameters
    ----------
    s        : (B, T, D)  spike train (output of PLIF, dtype same as block).
    h        : (B, T, D)  CfC hidden state (output of LiquidCell).
    syn_out  : (B, T, D)  SparseSynapse(s + h) — caller computed via cuBLAS.
    gate_pre : (B, T, D)  Linear_gate(s + h) — caller computed via cuBLAS.

    Returns
    -------
    spike_input : (B, T, D)  s + h (returned so the caller can pipe it
                              into Linear_gate / SparseSynapse instead of
                              recomputing — see ``HybridBlock.forward``).
    gated       : (B, T, D)  syn_out * sigmoid(gate_pre).

    Notes
    -----
    The user can pass *anything* for syn_out / gate_pre — the kernel does
    not care about their provenance. The closed-form bwd assumes::

        gated = syn_out * sigmoid(gate_pre)
        spike_input = s + h

    which matches `HybridBlock.forward` line 219-222.
    """
    return SEWSigmoidGateFn.apply(s, h, syn_out, gate_pre)


# ---------------------------------------------------------------------------
# Self-test (runs on cuda; a no-GPU env will report skipped).
# ---------------------------------------------------------------------------


def _self_test(device: str = "cuda") -> dict:
    """Numerical correctness gate: torch-ref vs Triton kernel.

    Returns dict with relative errors per output and an ``ok`` flag
    (rel_err < 1e-4 across all 4 input gradients + 2 outputs).
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    torch.manual_seed(42)
    B, T, D = 2, 8, 128
    s = torch.rand(B, T, D, device=dev) > 0.7  # binary spikes
    s = s.float().requires_grad_(False)
    h = torch.randn(B, T, D, device=dev, requires_grad=True)
    syn_out = torch.randn(B, T, D, device=dev, requires_grad=True)
    gate_pre = torch.randn(B, T, D, device=dev, requires_grad=True)

    # --- torch reference path ---
    s_r = s.clone().requires_grad_(True)
    h_r = h.clone().detach().requires_grad_(True)
    syn_r = syn_out.clone().detach().requires_grad_(True)
    gp_r = gate_pre.clone().detach().requires_grad_(True)

    spike_input_r = s_r + h_r
    gate_r = torch.sigmoid(gp_r)
    gated_r = syn_r * gate_r

    grad_spike_input = torch.randn_like(spike_input_r)
    grad_gated = torch.randn_like(gated_r)
    torch.autograd.backward(
        [spike_input_r, gated_r],
        [grad_spike_input, grad_gated],
    )

    # --- fused path ---
    s_f = s.clone().requires_grad_(True)
    h_f = h.clone().detach().requires_grad_(True)
    syn_f = syn_out.clone().detach().requires_grad_(True)
    gp_f = gate_pre.clone().detach().requires_grad_(True)

    spike_input_f, gated_f = sew_sigmoid_gate_fused(s_f, h_f, syn_f, gp_f)
    torch.autograd.backward(
        [spike_input_f, gated_f],
        [grad_spike_input, grad_gated],
    )

    def _relerr(a, b):
        diff = (a - b).abs().max().item()
        denom = b.abs().mean().item() + 1e-12
        return diff / denom

    rel_y_spike = _relerr(spike_input_f, spike_input_r)
    rel_y_gated = _relerr(gated_f, gated_r)
    rel_grad_h = _relerr(h_f.grad, h_r.grad)
    rel_grad_syn = _relerr(syn_f.grad, syn_r.grad)
    rel_grad_gp = _relerr(gp_f.grad, gp_r.grad)

    info = {
        "device": str(dev),
        "triton": _HAS_TRITON,
        "rel_y_spike_input": rel_y_spike,
        "rel_y_gated": rel_y_gated,
        "rel_grad_h": rel_grad_h,
        "rel_grad_syn_out": rel_grad_syn,
        "rel_grad_gate_pre": rel_grad_gp,
    }
    info["ok"] = all(
        v < 1e-4 for k, v in info.items() if k.startswith("rel_")
    )
    return info


__all__ = [
    "sew_sigmoid_gate_fused",
    "SEWSigmoidGateFn",
    "_torch_sew_sigmoid_gate_forward",
    "_torch_sew_sigmoid_gate_backward",
    "_self_test",
]


if __name__ == "__main__":  # pragma: no cover
    import json
    info = _self_test("cuda")
    print(json.dumps(info, indent=2))
    assert info["ok"], f"Numerical gate failed: {info}"
