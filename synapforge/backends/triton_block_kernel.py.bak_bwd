"""Triton fused kernel for the ENTIRE LNN+SNN HybridBlock.

This goes one layer above `triton_cfc_scan.py` (which only fused the CfC scan)
by fusing CfC scan + PLIF spike + subtract-on-spike reset + STDP eligibility
update into a SINGLE kernel launch per layer per loop_depth iteration.

Per token t, per channel d (independent across (B, D) for the recurrence; the
STDP outer-product is gathered via atomics):

    delta_t = softplus(W_delta x_t)                     # done outside  (matmul)
    A_t   = exp(-delta_t * A)                           # done outside  (elementwise)
    b_t   = delta_t * (W_B x_t)                         # done outside  (matmul)
    h_t   = A_t * h_{t-1} + b_t                         # FUSED          (CfC scan)
    s_t   = Heaviside(h_t - threshold_d)                # FUSED          (PLIF spike)
    h_t   = h_t * (1 - s_t)                             # FUSED          (subtract reset)
    E[i,j] += s_pre_t[i] * s_post_t[j] * decay          # FUSED, atomic  (STDP elig.)
                  (with s_pre = s_{t-1}, s_post = s_t -- Hebbian co-firing)

Backward: ATanSurrogate (`grad = a/(2(1+(pi/2 a (h-thr))^2))`) over the spike,
plus exact BPTT through the linear scan via cached pre-reset h. Done in
PyTorch -- same forward-only-Triton pattern as `triton_cfc_scan.py`. STDP
eligibility is a non-differentiated buffer (Hebbian, not gradient-based) so
it has no backward.

Public API
----------
    block = TritonHybridBlock(d_in=256, d_hidden=256).cuda()
    h_out, spikes = block(x, h0=None, stdp_eligibility=E_buffer)
    # h_out:  (B, T, D)  -- post-spike-reset hidden state, post out_norm
    # spikes: (B, T, D)  -- binary spike train (0/1), same dtype as h_out
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Triton availability probe (lazy: file importable on Windows / no-GPU CI).
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    _TRITON_VERSION = tuple(int(p) for p in triton.__version__.split(".")[:2])
except Exception:  # pragma: no cover -- Windows / no GPU
    triton = None
    tl = None
    _TRITON_VERSION = (0, 0)


if _HAS_TRITON:

    @triton.jit
    def fused_lnn_snn_block_kernel(
        # forward pointers
        a_ptr,              # (B, T, D)  decay multiplier  A_t
        b_ptr,              # (B, T, D)  input             b_t
        thr_ptr,            # (D,)       per-channel learnable threshold
        h0_ptr,             # (B, D)     initial state
        # scratch / output pointers
        h_pre_ptr,          # (B, T, D)  pre-reset hidden state (for backward)
        s_ptr,              # (B, T, D)  spike train (output + backward)
        m_ptr,              # (B, T, D)  membrane (h_pre - thr) for surrogate bw
        h_post_ptr,         # (B, T, D)  post-reset hidden state (output)
        # STDP buffer (intra-tile only in v1)
        elig_ptr,           # (D, D)     STDP eligibility, atomic accumulator
        elig_decay,         # f32 scalar trace decay (kept tile-local)
        # strides
        a_b_str, a_t_str, a_d_str,
        b_b_str, b_t_str, b_d_str,
        thr_d_str,
        h0_b_str, h0_d_str,
        out_b_str, out_t_str, out_d_str,
        elig_pre_str, elig_post_str,
        # sizes
        B, T, D,
        # meta
        BLOCK_D: tl.constexpr,
        ENABLE_STDP: tl.constexpr,
    ):
        """grid = (cdiv(D, BLOCK_D), B). One program owns BLOCK_D channels."""
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)

        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        # Load h_0 (always materialized as a real tensor in the wrapper).
        h_ptrs = h0_ptr + pid_b * h0_b_str + d_offsets * h0_d_str
        h = tl.load(h_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load per-channel threshold once (constant across T).
        thr = tl.load(thr_ptr + d_offsets * thr_d_str, mask=d_mask, other=0.0).to(tl.float32)

        # Carry the previous-step spike for STDP co-firing inside this tile.
        # At t=0, s_prev = 0 -> no STDP update on the first step.
        s_prev = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Sequential T-loop, fully fused (no Python overhead, no kernel relaunch).
        for t in range(0, T):
            # ---- Step 1: CfC scan ----
            a_ptrs = a_ptr + pid_b * a_b_str + t * a_t_str + d_offsets * a_d_str
            b_ptrs = b_ptr + pid_b * b_b_str + t * b_t_str + d_offsets * b_d_str
            a_t = tl.load(a_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            b_t = tl.load(b_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            h = a_t * h + b_t                                # h_pre at time t

            # Save pre-reset h.
            h_pre_p = h_pre_ptr + pid_b * out_b_str + t * out_t_str + d_offsets * out_d_str
            tl.store(h_pre_p, h.to(h_pre_ptr.dtype.element_ty), mask=d_mask)

            # ---- Step 2-3: PLIF membrane + Heaviside spike ----
            m = h - thr
            s = tl.where(m > 0.0, 1.0, 0.0)

            m_p = m_ptr + pid_b * out_b_str + t * out_t_str + d_offsets * out_d_str
            tl.store(m_p, m.to(m_ptr.dtype.element_ty), mask=d_mask)

            s_p = s_ptr + pid_b * out_b_str + t * out_t_str + d_offsets * out_d_str
            tl.store(s_p, s.to(s_ptr.dtype.element_ty), mask=d_mask)

            # ---- Step 4: subtract-on-spike reset ----
            h = h * (1.0 - s)

            h_post_p = h_post_ptr + pid_b * out_b_str + t * out_t_str + d_offsets * out_d_str
            tl.store(h_post_p, h.to(h_post_ptr.dtype.element_ty), mask=d_mask)

            # ---- Step 5: STDP co-firing (intra-tile outer product) ----
            if ENABLE_STDP:
                op = s[:, None] * s_prev[None, :] * elig_decay  # (BLOCK_D, BLOCK_D), fp32
                row_idx = d_offsets[:, None]                   # (BLOCK_D, 1)
                col_idx = d_offsets[None, :]                   # (1, BLOCK_D)
                row_mask = d_mask[:, None]
                col_mask = d_mask[None, :]
                tile_mask = row_mask & col_mask
                elig_offs = row_idx * elig_pre_str + col_idx * elig_post_str
                tl.atomic_add(elig_ptr + elig_offs, op, mask=tile_mask)

            # roll s into s_prev for next step
            s_prev = s


def _triton_block_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: torch.Tensor,
    h0,
    elig_buffer,
    elig_decay: float,
    enable_stdp: bool,
):
    """Run the fused Triton kernel.

    Returns:
        h_pre:  (B, T, D)
        s:      (B, T, D)
        m:      (B, T, D)
        h_post: (B, T, D)
    """
    assert _HAS_TRITON
    assert a.is_cuda and b.is_cuda and threshold.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    threshold = threshold.contiguous()
    B, T, D = a.shape
    if h0 is None:
        h0_t = torch.zeros(B, D, device=a.device, dtype=a.dtype)
    else:
        h0_t = h0.contiguous()

    h_pre = torch.empty_like(a)
    s = torch.empty_like(a)
    m = torch.empty_like(a)
    h_post = torch.empty_like(a)

    if enable_stdp and elig_buffer is None:
        elig_buffer = torch.zeros(D, D, device=a.device, dtype=torch.float32)
    if not enable_stdp:
        elig_buffer = torch.zeros(1, 1, device=a.device, dtype=torch.float32)

    BLOCK_D = 64 if D >= 64 else D
    grid = ((D + BLOCK_D - 1) // BLOCK_D, B)

    fused_lnn_snn_block_kernel[grid](
        a, b, threshold, h0_t,
        h_pre, s, m, h_post,
        elig_buffer, float(elig_decay),
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        threshold.stride(0),
        h0_t.stride(0), h0_t.stride(1),
        h_pre.stride(0), h_pre.stride(1), h_pre.stride(2),
        elig_buffer.stride(0), elig_buffer.stride(1),
        B, T, D,
        BLOCK_D=BLOCK_D,
        ENABLE_STDP=bool(enable_stdp),
        num_warps=4,
    )
    return h_pre, s, m, h_post


# ---------------------------------------------------------------------------
# Pure-PyTorch reference / fallback (used on Windows + as backward).
# ---------------------------------------------------------------------------


def _pytorch_block_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: torch.Tensor,
    h0,
    elig_buffer,
    elig_decay: float,
    enable_stdp: bool,
):
    """Reference: per-step Python loop. Returns (h_pre, s, m, h_post)."""
    B, T, D = a.shape
    h = a.new_zeros(B, D) if h0 is None else h0.clone()
    h_pre_buf = a.new_empty(B, T, D)
    s_buf = a.new_empty(B, T, D)
    m_buf = a.new_empty(B, T, D)
    h_post_buf = a.new_empty(B, T, D)

    if enable_stdp and elig_buffer is None:
        elig_buffer = torch.zeros(D, D, device=a.device, dtype=torch.float32)

    s_prev = a.new_zeros(B, D)
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        h_pre_buf[:, t] = h
        m = h - threshold
        s = (m > 0).to(a.dtype)
        m_buf[:, t] = m
        s_buf[:, t] = s
        h = h * (1.0 - s)
        h_post_buf[:, t] = h
        if enable_stdp:
            elig_buffer.add_(
                (s.float().t() @ s_prev.float()) * float(elig_decay)
            )
        s_prev = s
    return h_pre_buf, s_buf, m_buf, h_post_buf


class TritonHybridBlockFn(torch.autograd.Function):
    """Forward via Triton fused kernel; backward via cached states."""

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        threshold: torch.Tensor,
        h0,
        elig_buffer,
        elig_decay: float,
        enable_stdp: bool,
        alpha: float,
    ):
        use_triton = (
            _HAS_TRITON and a.is_cuda
            and a.dtype in (torch.float32, torch.float16, torch.bfloat16)
        )
        if use_triton:
            try:
                h_pre, s, m, h_post = _triton_block_forward(
                    a, b, threshold, h0, elig_buffer, elig_decay, enable_stdp,
                )
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"TritonHybridBlock: Triton path failed ({exc!r}), "
                    f"falling back to PyTorch reference."
                )
                h_pre, s, m, h_post = _pytorch_block_forward(
                    a, b, threshold, h0, elig_buffer, elig_decay, enable_stdp,
                )
                use_triton = False
        else:
            h_pre, s, m, h_post = _pytorch_block_forward(
                a, b, threshold, h0, elig_buffer, elig_decay, enable_stdp,
            )

        # Cross-tile STDP correction (cheap PyTorch post-pass)
        if enable_stdp and use_triton and elig_buffer is not None:
            s_post = s.float()
            s_pre = torch.cat(
                [s.new_zeros(s.shape[0], 1, s.shape[2]), s[:, :-1]], dim=1
            ).float()
            full_elig = (
                s_post.reshape(-1, s.shape[2]).t() @ s_pre.reshape(-1, s.shape[2])
            ) * float(elig_decay)
            D = s.shape[2]
            BLOCK_D = 64 if D >= 64 else D
            tile_idx = torch.arange(D, device=s.device) // BLOCK_D
            cross_mask = (tile_idx[:, None] != tile_idx[None, :]).float()
            elig_buffer.add_(full_elig * cross_mask)

        ctx.save_for_backward(
            a, b, threshold, h_pre, s, m,
            h0 if h0 is not None else a.new_zeros(a.shape[0], a.shape[2]),
        )
        ctx.had_h0 = h0 is not None
        ctx.alpha = float(alpha)
        return h_post, s

    @staticmethod
    def backward(ctx, grad_h_post: torch.Tensor, grad_s: torch.Tensor):
        a, b, threshold, h_pre, s, m, h0 = ctx.saved_tensors
        B, T, D = a.shape
        alpha = ctx.alpha

        grad_a = torch.zeros_like(a)
        grad_b = torch.zeros_like(b)
        grad_thr = torch.zeros_like(threshold)
        grad_h_post_prev = a.new_zeros(B, D)

        m_f32 = m.float()
        x = alpha * m_f32
        ds_dm = (alpha / (2.0 * (1.0 + (math.pi / 2.0 * x).pow(2)))).to(a.dtype)

        for t in range(T - 1, -1, -1):
            grad_h_post_t = grad_h_post[:, t] + grad_h_post_prev

            ds_dm_t = ds_dm[:, t]
            grad_s_t = grad_s[:, t] if grad_s is not None else None
            one_minus_s = (1.0 - s[:, t])
            dh_post_dh_pre = one_minus_s - h_pre[:, t] * ds_dm_t

            grad_h_pre_t = grad_h_post_t * dh_post_dh_pre
            if grad_s_t is not None:
                grad_h_pre_t = grad_h_pre_t + grad_s_t * ds_dm_t

            if t > 0:
                h_post_prev = h_pre[:, t - 1] * (1.0 - s[:, t - 1])
            else:
                h_post_prev = h0
            grad_a[:, t] = grad_h_pre_t * h_post_prev
            grad_b[:, t] = grad_h_pre_t

            grad_thr_t = grad_h_post_t * h_pre[:, t] * ds_dm_t
            if grad_s_t is not None:
                grad_thr_t = grad_thr_t - grad_s_t * ds_dm_t
            grad_thr = grad_thr + grad_thr_t.sum(dim=0)

            grad_h_post_prev = grad_h_pre_t * a[:, t]

        grad_h0 = grad_h_post_prev if ctx.had_h0 else None
        return grad_a, grad_b, grad_thr, grad_h0, None, None, None, None


def hybrid_block_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: torch.Tensor,
    h0=None,
    elig_buffer=None,
    elig_decay: float = 0.95,
    enable_stdp: bool = True,
    alpha: float = 2.0,
):
    """Functional entry: fused CfC + PLIF + STDP block."""
    return TritonHybridBlockFn.apply(
        a, b, threshold, h0, elig_buffer, elig_decay, enable_stdp, alpha,
    )


class TritonHybridBlock(nn.Module):
    """Drop-in fused replacement for the standard mscfc HybridBlock.

        delta_t = softplus(W_delta x_t)
        A     = exp(A_log)
        A_t   = exp(-delta_t * A)
        b_t   = delta_t * (W_B x_t)
        h_t   = A_t h_{t-1} + b_t           |
        s_t   = ATan(h_t - thr_d)           |  fused 1 kernel
        h_t   = h_t * (1 - s_t)             |
        E += s_t * s_{t-1} * decay          /
        y_t   = LayerNorm(h_t)

    Returns (y, spikes).
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        elig_decay: float = 0.95,
        alpha: float = 2.0,
        enable_stdp: bool = True,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden_size = int(d_hidden)
        self.delta_proj = nn.Linear(d_in, d_hidden)
        self.b_proj = nn.Linear(d_in, d_hidden)
        # Same A_log init as mscfc HybridBlock (Mamba convention).
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, d_hidden)))
        # Per-channel learnable PLIF threshold (init around 1.0).
        self.threshold = nn.Parameter(torch.ones(d_hidden))
        nn.init.constant_(self.delta_proj.bias, 0.0)
        self.out_norm = nn.LayerNorm(d_hidden)

        self.register_buffer("elig", torch.zeros(d_hidden, d_hidden))
        self.elig_decay = float(elig_decay)
        self.alpha = float(alpha)
        self.enable_stdp = bool(enable_stdp)

    def get_decay_rate(self) -> torch.Tensor:
        return torch.exp(self.A_log)

    def reset_eligibility(self) -> None:
        self.elig.zero_()

    def forward(self, x: torch.Tensor, h0=None, R: int = 1):
        """Same API as the existing HybridBlock; returns (y, spikes)."""
        B, T, _ = x.shape
        delta = F.softplus(self.delta_proj(x))
        b_step = delta * self.b_proj(x)
        A = self.get_decay_rate()
        A_step = torch.exp(-delta * A)

        if R > 1:
            log_A = torch.log(A_step.clamp_min(1e-30))
            A_t = torch.exp(R * log_A)
            geom = (1.0 - A_t) / (1.0 - torch.exp(log_A)).clamp_min(1e-9)
            b_t = geom * b_step
        else:
            A_t = A_step
            b_t = b_step

        h_post, spikes = hybrid_block_fused(
            A_t, b_t, self.threshold, h0,
            elig_buffer=self.elig if self.enable_stdp else None,
            elig_decay=self.elig_decay,
            enable_stdp=self.enable_stdp,
            alpha=self.alpha,
        )
        return self.out_norm(h_post), spikes


class _PyHybridBlockRef(nn.Module):
    """Bit-exact reference for TritonHybridBlock -- pure-PyTorch per-step loop."""

    def __init__(self, d_in: int, d_hidden: int, alpha: float = 2.0):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden_size = int(d_hidden)
        self.delta_proj = nn.Linear(d_in, d_hidden)
        self.b_proj = nn.Linear(d_in, d_hidden)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, d_hidden)))
        self.threshold = nn.Parameter(torch.ones(d_hidden))
        nn.init.constant_(self.delta_proj.bias, 0.0)
        self.out_norm = nn.LayerNorm(d_hidden)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, h0=None):
        B, T, _ = x.shape
        D = self.hidden_size
        delta = F.softplus(self.delta_proj(x))
        b_step = delta * self.b_proj(x)
        A = torch.exp(self.A_log)
        A_step = torch.exp(-delta * A)

        h = x.new_zeros(B, D) if h0 is None else h0.clone()
        h_post = x.new_empty(B, T, D)
        spikes = x.new_empty(B, T, D)
        for t in range(T):
            h = A_step[:, t] * h + b_step[:, t]
            m = h - self.threshold
            s = (m > 0).to(x.dtype)
            h = h * (1.0 - s)
            h_post[:, t] = h
            spikes[:, t] = s
        return self.out_norm(h_post), spikes


def _self_test(device: str = "cuda") -> dict:
    """Bit-correctness gate. Returns dict with rel_err and ok flag."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    torch.manual_seed(0)
    B, T, D = 4, 64, 32

    block = TritonHybridBlock(d_in=D, d_hidden=D, enable_stdp=False).to(dev)
    block.eval()
    ref = _PyHybridBlockRef(d_in=D, d_hidden=D).to(dev)
    with torch.no_grad():
        ref.delta_proj.weight.copy_(block.delta_proj.weight)
        ref.delta_proj.bias.copy_(block.delta_proj.bias)
        ref.b_proj.weight.copy_(block.b_proj.weight)
        ref.b_proj.bias.copy_(block.b_proj.bias)
        ref.A_log.copy_(block.A_log)
        ref.threshold.copy_(block.threshold)
        ref.out_norm.weight.copy_(block.out_norm.weight)
        ref.out_norm.bias.copy_(block.out_norm.bias)

    x = torch.randn(B, T, D, device=dev) * 0.5
    with torch.no_grad():
        y_t, s_t = block(x)
        y_r, s_r = ref(x)
    err = (y_t - y_r).abs().max().item()
    mag = y_r.abs().mean().item() + 1e-12
    rel = err / mag
    spike_match = (s_t == s_r).float().mean().item()

    return {
        "device": str(dev),
        "triton": _HAS_TRITON,
        "triton_version": ".".join(str(p) for p in _TRITON_VERSION) if _HAS_TRITON else None,
        "rel_err_y": rel,
        "abs_err_y": err,
        "spike_match_rate": spike_match,
        "ok": rel < 1e-3 and spike_match > 0.99,
    }


__all__ = [
    "TritonHybridBlock",
    "TritonHybridBlockFn",
    "hybrid_block_fused",
    "_PyHybridBlockRef",
    "_pytorch_block_forward",
    "_self_test",
    "_HAS_TRITON",
]


if __name__ == "__main__":
    import json
    info = _self_test("cuda")
    print(json.dumps(info, indent=2))
    assert info["ok"], f"Numerical gate failed: {info}"
