"""PyTorch glue for the packed-spike matmul kernels.

This file owns:

* ``pack_spikes_torch`` / ``unpack_spikes_torch`` -- torch-tensor pack
  and unpack utilities (mirroring the numpy primitives in ``pack.py``
  but operating on ``torch.Tensor`` so they integrate with autograd
  and CUDA).
* ``PackedSpikeMatmul`` -- ``torch.autograd.Function`` that runs the
  Triton fwd / bwd kernels from ``packed_matmul.py`` with the right
  tensor lifetimes and tile sizes.
* ``packed_spike_linear`` -- ergonomic wrapper that takes ``s, h,
  linear`` (matching the existing ``sparse_spike_linear`` API in
  ``synapforge.kernels.sparse_spike_matmul``) and dispatches to
  packed / dense based on the ``--packed-spikes`` flag and the
  measured density.

Auto-fallback policy
--------------------
The bit-pack saving only matters when the spike density is low.
At density > 30%, dense GEMM (cuBLAS) wins on cache locality and
predictable access.  The dispatcher computes density (or accepts a
caller-provided estimate -- ``--log-spike-per-layer`` already tracks
it for free) and falls back to ``F.linear(s.float() + h, W, bias)``
above the threshold.

Why this is safe
----------------
* Below threshold: packed-spike kernel dominates.
* At threshold: dense GEMM is faster, fallback is a no-op for
  correctness.
* Dead PLIF (density = 0): we still take the packed path; cost is
  one MMA per K-block where every spike bit is zero -- equivalent to
  multiplying the weight by zero, which is silently optimal.

CLI integration
---------------
``train_100m_kd.py --packed-spikes`` toggles this on the synapse
matmul; default OFF since current Run 7 PLIF is dead.  The flag is
**dormant until PLIF revives** -- see docs/NATIVE_SPIKE_PACKING.md.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from synapforge.native.spike import packed_matmul as _km

# Default density threshold (matches sparse_spike_synapse).
PACKED_SPIKE_DEFAULT_THRESHOLD = 0.30

# Default block sizes: Triton needs >= 16 along all MMA axes; choose tile
# that divides typical d=1280 (80 packed words) and bs*T=12288.
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K_PACKED = 4   # -> BLOCK_K = 64


# ---------------------------------------------------------------------------
# Torch-tensor pack / unpack (CUDA + autograd-friendly).
# ---------------------------------------------------------------------------
def pack_spikes_torch(spikes: torch.Tensor) -> torch.Tensor:
    """Pack a binary tensor along its last dim into uint16 slots.

    Mirrors ``synapforge.native.spike.pack.pack_spikes`` but on
    torch tensors so it can run inside a CUDA graph / autograd path.

    Returns a tensor of shape ``(..., (d + 15) // 16)``.  PyTorch lacks
    native ``uint16`` (pre-2.4); we emit ``int32`` which carries the
    same 0..65535 range bit-exactly and compresses 8x relative to
    fp16.  The Triton kernel reads the int32 buffer and casts to
    uint32 internally before the bit decode.
    """
    if spikes.numel() == 0:
        return torch.empty(
            spikes.shape[:-1] + (0,), dtype=torch.int32,
            device=spikes.device,
        )

    d = spikes.shape[-1]
    pad = (16 - d % 16) % 16
    if pad > 0:
        spikes = torch.nn.functional.pad(spikes, (0, pad))

    bits = (spikes != 0).to(torch.int32)
    bits = bits.view(*bits.shape[:-1], -1, 16)
    shifts = torch.arange(16, device=bits.device, dtype=torch.int32)
    packed = (bits << shifts).sum(dim=-1).to(torch.int32)
    return packed


def unpack_spikes_torch(packed: torch.Tensor, d: int,
                        *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack int32 packed-spike tensor back to dense.

    Counterpart of :func:`pack_spikes_torch`.  Default dtype is
    ``torch.float32`` for downstream GEMM; pass ``torch.bool`` for the
    memory-efficient consumer path.
    """
    if packed.numel() == 0 or d == 0:
        return torch.zeros(
            packed.shape[:-1] + (d,), dtype=dtype, device=packed.device,
        )

    n_slots = packed.shape[-1]
    expected_slots = (d + 15) // 16
    if n_slots != expected_slots:
        raise ValueError(
            f"unpack_spikes_torch: packed has {n_slots} slots; d={d} "
            f"expects {expected_slots} slots."
        )

    shifts = torch.arange(16, device=packed.device, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(*packed.shape[:-1], n_slots * 16)
    bits = bits[..., :d]
    return bits.to(dtype)


# ---------------------------------------------------------------------------
# autograd.Function over the packed-spike matmul.
# ---------------------------------------------------------------------------
class PackedSpikeMatmul(torch.autograd.Function):
    """Compute ``y = s.float() @ W`` on packed binary spikes.

    Forward
    -------
    1. Pack ``s`` (boolean / 0-1) along the last dim into uint16.
    2. Triton kernel ``packed_spike_matmul_fwd_kernel`` reads packed
       words and contributes to ``y`` without unpacking to HBM.
    3. Cache the packed tensor (NOT ``s``) for backward -- saves
       another 16x in activation memory.

    Backward
    --------
    1. ``grad_W = s.T @ grad_y`` -- packed kernel
       ``packed_spike_matmul_bwd_dW_kernel``.
    2. ``grad_s = grad_y @ W.T`` -- dense (grad_s is fp surrogate
       gradient, not binary).

    Limitations
    -----------
    * GPU + Triton only.  CPU path raises and the caller's
      ``packed_spike_linear`` falls back to dense.
    * The Triton bit-expansion trick fixes ``BLOCK_K = 16 *
      BLOCK_K_PACKED``; we set ``BLOCK_K_PACKED = 4`` so ``BLOCK_K =
      64``.  ``d_in`` should be divisible by 64 for best perf;
      otherwise the kernel handles the tail with masking.
    """

    @staticmethod
    def forward(ctx, s: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not _km._HAS_TRITON:
            raise RuntimeError(
                "PackedSpikeMatmul: Triton not available. "
                "Use synapforge.native.spike.torch_glue.packed_spike_linear "
                "which auto-falls-back to dense F.linear on no-Triton boxes."
            )
        if not s.is_cuda or not weight.is_cuda:
            raise RuntimeError(
                "PackedSpikeMatmul: tensors must be CUDA. "
                "Got s.device={s.device}, W.device={weight.device}."
            )
        # Flatten leading dims to (M, d_in).
        leading = s.shape[:-1]
        d_in = s.shape[-1]
        d_out = weight.shape[1] if weight.shape[0] == d_in else weight.shape[0]
        # Standardise weight to (d_in, d_out).  nn.Linear stores (out, in).
        if weight.shape[0] == d_in:
            W_in_out = weight                          # (d_in, d_out)
        elif weight.shape[1] == d_in:
            W_in_out = weight.t().contiguous()         # transpose nn.Linear shape
        else:
            raise ValueError(
                f"PackedSpikeMatmul: weight shape {tuple(weight.shape)} "
                f"does not match d_in={d_in}."
            )

        s_flat = s.reshape(-1, d_in).contiguous()
        M = s_flat.shape[0]
        d_out = W_in_out.shape[1]

        # Pack along last dim.  We promise int32 storage; the kernel reads
        # via .data_ptr() and treats the buffer as uint32.
        packed = pack_spikes_torch(s_flat)             # (M, K_packed) int32
        K_packed = packed.shape[-1]

        # Allocate output.
        y = torch.empty((M, d_out), dtype=torch.float32, device=s.device)

        # Launch fwd kernel.
        BLOCK_M, BLOCK_N, BLOCK_K_PACKED = (
            DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K_PACKED,
        )
        grid = (
            (M + BLOCK_M - 1) // BLOCK_M,
            (d_out + BLOCK_N - 1) // BLOCK_N,
        )
        _km.packed_spike_matmul_fwd_kernel[grid](
            packed,                  # uint16 packed (M, K_packed)
            W_in_out,                # weight (d_in, d_out)
            y,                       # output (M, d_out)
            M, d_out, d_in, K_packed,
            packed.stride(0), packed.stride(1),
            W_in_out.stride(0), W_in_out.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
        )

        if bias is not None:
            y = y + bias

        # Cache packed (16x smaller than caching s).  Save d_in in saved-state.
        ctx.save_for_backward(packed, W_in_out)
        ctx.d_in = d_in
        ctx.d_out = d_out
        ctx.M = M
        ctx.K_packed = K_packed
        ctx.leading = leading
        ctx.has_bias = bias is not None

        return y.reshape(leading + (d_out,))

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        packed, W_in_out = ctx.saved_tensors
        d_in, d_out, M, K_packed = ctx.d_in, ctx.d_out, ctx.M, ctx.K_packed

        grad_y_flat = grad_y.reshape(M, d_out).contiguous().to(torch.float32)

        # ---- grad_W = s.T @ grad_y ---- (packed kernel) -----------------
        BLOCK_M, BLOCK_N, BLOCK_K_PACKED = (
            DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K_PACKED,
        )
        grad_W = torch.zeros((d_in, d_out), dtype=torch.float32,
                             device=W_in_out.device)
        grid_w = (
            (K_packed + BLOCK_K_PACKED - 1) // BLOCK_K_PACKED,
            (d_out + BLOCK_N - 1) // BLOCK_N,
        )
        _km.packed_spike_matmul_bwd_dW_kernel[grid_w](
            packed,
            grad_y_flat,
            grad_W,
            M, d_out, d_in, K_packed,
            packed.stride(0), packed.stride(1),
            grad_y_flat.stride(0), grad_y_flat.stride(1),
            grad_W.stride(0), grad_W.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
        )

        # ---- grad_s = grad_y @ W.T ---- (dense kernel) -------------------
        # grad_s is the surrogate-gradient signal feeding atan(); fp dense.
        # Use the existing dense path (the bit-pack saving doesn't apply
        # to a fp output anyway).
        grad_s_flat = grad_y_flat @ W_in_out.t()                   # (M, d_in)
        grad_s = grad_s_flat.reshape(ctx.leading + (d_in,))

        # Bias grad: sum over leading dims if bias was passed.
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.reshape(-1, d_out).sum(dim=0)

        return grad_s.to(W_in_out.dtype), grad_W.to(W_in_out.dtype), grad_bias


# ---------------------------------------------------------------------------
# High-level API: packed_spike_linear / packed_spike_matmul.
# ---------------------------------------------------------------------------
def _estimate_density(s: torch.Tensor) -> float:
    """Return the fraction of non-zero entries in ``s`` as a float in [0, 1]."""
    if s.numel() == 0:
        return 0.0
    return float(s.detach().float().mean().item())


def packed_spike_matmul(
    s: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    density_threshold: float = PACKED_SPIKE_DEFAULT_THRESHOLD,
    density_estimate: Optional[float] = None,
) -> torch.Tensor:
    """Compute ``(s + h) @ weight.T + bias`` exploiting bit-packed sparse spikes.

    Mirrors the API of ``synapforge.kernels.sparse_spike_matmul`` so the
    drop-in is transparent at the model layer.

    Strategy
    --------
    * Density >= threshold: dense path ``F.linear(s.float() + h, W, bias)``.
    * Density <  threshold and Triton + CUDA available: packed kernel.
    * Otherwise: dense path (identical numerics).

    The dense ``h @ W.T`` part is split out and run separately when the
    packed kernel is used: ``y = h @ W.T + packed_path(s, W) [+ bias]``.
    This keeps the dense GEMM (already cuBLAS-optimal) and only routes
    the sparse-spike contribution through the bit-packed path.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"packed_spike_matmul: weight must be 2D, got {tuple(weight.shape)}"
        )
    if h.shape[-1] != s.shape[-1]:
        raise ValueError(
            f"packed_spike_matmul: h.shape[-1]={h.shape[-1]} must equal "
            f"s.shape[-1]={s.shape[-1]}"
        )

    # Decide path based on density.
    if density_estimate is None:
        density_estimate = _estimate_density(s)

    use_packed = (
        density_estimate < density_threshold
        and _km._HAS_TRITON
        and s.is_cuda
        and weight.is_cuda
    )
    if not use_packed:
        # Dense fallback -- bit-equivalent.
        s_typed = s.to(h.dtype) if s.dtype != h.dtype else s
        return torch.nn.functional.linear(s_typed + h, weight, bias)

    # Split: y = h @ W.T + packed(s, W) [+ bias]
    # We pass bias only once -- on the dense h-path -- so the packed
    # kernel handles the spike contribution alone.
    h_out = torch.nn.functional.linear(h, weight, bias=None)
    spike_out = PackedSpikeMatmul.apply(s, weight, None)
    out = h_out + spike_out
    if bias is not None:
        out = out + bias
    return out


def packed_spike_linear(
    s: torch.Tensor,
    h: torch.Tensor,
    linear: nn.Module,
    *,
    density_threshold: float = PACKED_SPIKE_DEFAULT_THRESHOLD,
    density_estimate: Optional[float] = None,
) -> torch.Tensor:
    """Convenience wrapper accepting an ``nn.Linear`` (or ``SparseSynapse``).

    For ``SparseSynapse`` the structural mask is composed with the
    weight so the packed-spike path stays bit-equivalent to the masked
    dense reference (same pattern as
    ``synapforge.kernels.sparse_spike_matmul.sparse_spike_linear``).
    """
    weight = linear.weight
    bias = getattr(linear, "bias", None)

    mask = getattr(linear, "mask", None)
    if mask is not None:
        weight = weight * mask.to(weight.dtype)

    return packed_spike_matmul(
        s, h, weight, bias,
        density_threshold=density_threshold,
        density_estimate=density_estimate,
    )


__all__ = [
    "PACKED_SPIKE_DEFAULT_THRESHOLD",
    "PackedSpikeMatmul",
    "pack_spikes_torch",
    "unpack_spikes_torch",
    "packed_spike_linear",
    "packed_spike_matmul",
]
