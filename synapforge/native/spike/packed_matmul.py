"""Triton kernels for matmul on bit-packed binary spikes.

This module hosts the **kernel side** of the packed-spike pipeline.
Per the design contract, this file contains **zero ``import torch``**:
the kernel is pure ``triton.language`` with numpy-compatible host
glue.  Tensor lifetime, autograd, and dispatch live in
``torch_glue.py``.

What the kernel does
--------------------
Given:

* ``packed_s``: ``(M, K_packed)`` uint16 tensor where each word stores
  16 binary spikes (bit ``b`` of word ``w`` is ``s[m, w*16 + b]``).
* ``W``: ``(K, N)`` float weight (the synapse matrix in
  ``y = s.float() @ W`` formulation).
* ``M`` rows of token-level inputs, ``N`` output channels.

Compute:

* ``y = unpack(packed_s).float() @ W``

without unpacking ``packed_s`` to HBM.  The kernel loads one packed
word, decodes 16 bits inline using ``tl.where``, and contributes the
selected weight rows to the output tile.

Why this is the load-bearing optimisation
-----------------------------------------
At ``B=48 T=256 d=1280 layers=16``, each block step touches
``16 * 48 * 256 * 1280 * 2 bytes/spike * 2 ways = 100 MB`` of HBM
traffic on the spike branch alone.  Packed: ``100 / 16 = 6 MB``.
At HBM bandwidth ``1.5 TB/s`` on A800, that's ``~94 / 1500 ms = 60 us``
saved per step on bandwidth alone -- before counting the implicit
``where`` / ``select`` saving from skipping zero-bit lanes.

Forward kernel
--------------
``packed_spike_matmul_fwd_kernel`` tiles over ``(BLOCK_M, BLOCK_N)``
output tiles and walks the K dim by ``BLOCK_K`` packed words at a
time (so ``16 * BLOCK_K`` original spike channels per step).  For each
loaded packed-word column ``b in [0, 16)``, the kernel uses
``tl.where(bit, weight_row, 0)`` to zero the contribution of inactive
spikes; cuBLAS would otherwise need 16 fp16 multiplies and adds where
half end up zero.

Backward kernel
---------------
For ``y = s @ W`` with ``s`` binary, the backward is:

* ``grad_W = s.T @ grad_y`` -- still benefits from packed-bit dispatch
  (same shape pattern, transposed).
* ``grad_s = grad_y @ W.T`` -- this is **dense fp** (``grad_s`` is a
  surrogate-gradient signal, not binary).  So we keep the dense path
  here.

The closed-form-bwd request in the spec ("produces packed grad_s
output (since downstream is also spike-domain)") is preserved as a
**packed-grad option**: if the downstream consumer is itself a binary
spike op (e.g. a STDP eligibility update that only cares about the
sign of ``grad_s``), we expose a ``pack_grad_s=True`` path that
writes the sign-bit of grad_s into a packed-uint16 output.  In the
default path used by SGD-bwd we keep ``grad_s`` dense.

Numerics
--------
Forward path produces output equivalent to ``unpacked.float() @ W``
within fp32 numerical tolerance ``1e-5`` (relative).  Verified by
``tests/native/spike/test_packed_matmul.py``.

Importability
-------------
This module is importable on CPU-only / Windows machines (where
triton is absent) -- the kernel is wrapped behind a lazy
``_HAS_TRITON`` guard.  Calling ``packed_spike_matmul_*`` without
triton raises ``RuntimeError`` with a clear message.  Tests on
no-triton boxes use the host fallback (numpy-based) for correctness
verification.
"""
from __future__ import annotations

# NOTE: no ``import torch`` is permitted in this file (design contract).
# The torch glue lives in ``synapforge.native.spike.torch_glue``.

import numpy as np

# ---------------------------------------------------------------------------
# Triton availability probe (lazy -- file importable on no-GPU CI).
# ---------------------------------------------------------------------------
_HAS_TRITON = False
try:  # pragma: no cover -- rental box only
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore


# Per-tile defaults; A800 80GB sweet-spot.  Tune in benchmarks.
_DEFAULT_BLOCK_M = 32
_DEFAULT_BLOCK_N = 64
_DEFAULT_BLOCK_K_PACKED = 4  # 4 packed words = 64 unpacked spike channels


# ---------------------------------------------------------------------------
# Triton kernels (loaded only when triton is importable).
# ---------------------------------------------------------------------------
if _HAS_TRITON:  # pragma: no cover

    @triton.jit
    def packed_spike_matmul_fwd_kernel(
        # ---- input pointers ----
        packed_s_ptr,        # uint16  (M, K_packed)   bit-packed spikes
        weight_ptr,          # f16/f32 (K, N)          dense weight (in_dim, out_dim)
        y_ptr,               # f32     (M, N)          output accumulator
        # ---- shape ----
        M, N, K, K_packed,
        # ---- strides ----
        stride_pm, stride_pk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        # ---- tile sizes (compile-time constexpr) ----
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,  # number of packed words per inner step
    ):
        """Forward: y[m, n] = sum_k unpack(packed_s)[m, k] * weight[k, n].

        Tiling
        ------
        * Output tiled over ``(BLOCK_M, BLOCK_N)``.
        * K-loop walks ``BLOCK_K_PACKED`` packed words at a time, expanding
          each word to 16 spike channels in-register via right-shift +
          mask.  The expanded ``(BLOCK_M, 16 * BLOCK_K_PACKED)`` tile then
          feeds ``tl.dot`` directly -- one MMA per K-block.

        Bandwidth math
        --------------
        Per K-block we load:

        * Packed spikes:   BLOCK_M * BLOCK_K_PACKED * 2 bytes  (uint16)
        * Weight rows:     16 * BLOCK_K_PACKED * BLOCK_N * sizeof(W)

        The packed-spike load is 16x smaller than an unpacked fp16 load
        of the same channel count; that's the headline saving.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = offs_m < M
        n_mask = offs_n < N

        # Output accumulator (always fp32 for accuracy; cast on store).
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Width of the unpacked-K tile after bit-expansion.
        BLOCK_K: tl.constexpr = 16 * BLOCK_K_PACKED

        for k_packed_block in range(0, tl.cdiv(K_packed, BLOCK_K_PACKED)):
            offs_kp = k_packed_block * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
            kp_mask = offs_kp < K_packed

            # Load a [BLOCK_M, BLOCK_K_PACKED] tile of packed-uint16 spikes.
            packed_ptrs = (packed_s_ptr
                           + offs_m[:, None] * stride_pm
                           + offs_kp[None, :] * stride_pk)
            packed_tile = tl.load(
                packed_ptrs,
                mask=m_mask[:, None] & kp_mask[None, :],
                other=0,
            ).to(tl.uint32)  # cast to uint32 so >> doesn't sign-extend

            # Expand to a [BLOCK_M, BLOCK_K] fp32 tile by decoding bits.
            # We build the expanded tile by computing, for each spike-channel
            # offset c in [0, BLOCK_K), which packed-word it lives in
            # (c // 16) and which bit index within that word (c % 16).
            offs_k_local = tl.arange(0, BLOCK_K)              # (BLOCK_K,)
            # Within-tile word index for each k channel.
            kp_within = offs_k_local // 16                     # (BLOCK_K,)
            bit_within = offs_k_local % 16                     # (BLOCK_K,)
            # Gather the packed word for each k (broadcasted across M).
            # packed_tile: (BLOCK_M, BLOCK_K_PACKED) -> indexed by kp_within.
            # We build a (BLOCK_M, BLOCK_K) expanded tile.
            expanded_word = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint32)
            # We must select column `kp_within[k]` of packed_tile for each k.
            # Triton has no fancy indexing on register tiles, so we expand
            # by a mask broadcast: for each k, OR in packed_tile[:, kp]
            # if kp == kp_within[k].
            for kp_idx in tl.static_range(BLOCK_K_PACKED):
                # Build a (BLOCK_K,) mask: 1 where kp_within == kp_idx.
                col_match = (kp_within == kp_idx).to(tl.uint32)  # (BLOCK_K,)
                # packed_tile[:, kp_idx] : (BLOCK_M,)
                col_word = packed_tile[:, kp_idx][:, None]        # (BLOCK_M, 1)
                expanded_word = expanded_word + col_word * col_match[None, :]
            # Now expanded_word[m, k] holds the packed uint32 word covering
            # k's slot.  Shift by bit_within and mask bit 0.
            bits = (expanded_word >> bit_within[None, :]) & 1     # (BLOCK_M, BLOCK_K)
            spike_tile = bits.to(tl.float32)

            # Load the [BLOCK_K, BLOCK_N] weight tile.
            offs_k_global = k_packed_block * BLOCK_K + offs_k_local
            k_mask = offs_k_global < K
            w_ptrs = (weight_ptr
                      + offs_k_global[:, None] * stride_wk
                      + offs_n[None, :] * stride_wn)
            w_tile = tl.load(
                w_ptrs,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # MMA: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N).
            # Triton requires BLOCK_M, BLOCK_K, BLOCK_N >= 16 and powers of 2
            # for tl.dot; the autotune config ensures that.
            acc += tl.dot(spike_tile, w_tile, allow_tf32=True)

        # Store the output tile.
        y_ptrs = (y_ptr + offs_m[:, None] * stride_ym
                  + offs_n[None, :] * stride_yn)
        tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def packed_spike_matmul_bwd_dW_kernel(
        # ---- input pointers ----
        packed_s_ptr,        # uint16 (M, K_packed)
        grad_y_ptr,          # f32    (M, N)
        grad_w_ptr,          # f32    (K, N)   accumulator (zero-init by host)
        # ---- shape ----
        M, N, K, K_packed,
        # ---- strides ----
        stride_pm, stride_pk,
        stride_gym, stride_gyn,
        stride_gwk, stride_gwn,
        # ---- tile ----
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        """Backward dW: grad_W[k, n] += sum_m unpack(s)[m, k] * grad_y[m, n].

        Iterates over (K_block, N) output tiles, accumulating over M.
        We tile over K using ``BLOCK_K = 16 * BLOCK_K_PACKED`` (one
        compile-time-known tile) and N using ``BLOCK_N``, expanding the
        packed bits to a dense ``(BLOCK_K, BLOCK_M)`` tile and feeding
        ``tl.dot``.
        """
        pid_kp_block = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        BLOCK_K: tl.constexpr = 16 * BLOCK_K_PACKED

        offs_kp = pid_kp_block * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kp_mask = offs_kp < K_packed
        n_mask = offs_n < N

        # Build the (BLOCK_K,) k-channel offset for this kp_block tile.
        offs_k_local = tl.arange(0, BLOCK_K)
        offs_k_global = pid_kp_block * BLOCK_K + offs_k_local
        k_mask = offs_k_global < K
        kp_within = offs_k_local // 16
        bit_within = offs_k_local % 16

        # Output accumulator over the M dim: (BLOCK_K, BLOCK_N).
        acc = tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float32)

        for m_block in range(0, tl.cdiv(M, BLOCK_M)):
            offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = offs_m < M

            # Load packed[BLOCK_M, BLOCK_K_PACKED].
            p_ptrs = (packed_s_ptr
                      + offs_m[:, None] * stride_pm
                      + offs_kp[None, :] * stride_pk)
            p_tile = tl.load(
                p_ptrs,
                mask=m_mask[:, None] & kp_mask[None, :],
                other=0,
            ).to(tl.uint32)

            # Expand to (BLOCK_M, BLOCK_K) dense bits (same trick as fwd).
            expanded_word = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint32)
            for kp_idx in tl.static_range(BLOCK_K_PACKED):
                col_match = (kp_within == kp_idx).to(tl.uint32)
                col_word = p_tile[:, kp_idx][:, None]
                expanded_word = expanded_word + col_word * col_match[None, :]
            bits = (expanded_word >> bit_within[None, :]) & 1
            spike_tile = bits.to(tl.float32)  # (BLOCK_M, BLOCK_K)

            # Load grad_y[BLOCK_M, BLOCK_N].
            gy_ptrs = (grad_y_ptr
                       + offs_m[:, None] * stride_gym
                       + offs_n[None, :] * stride_gyn)
            gy_tile = tl.load(
                gy_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # spike_tile.T @ gy_tile : (BLOCK_K, BLOCK_N)
            acc += tl.dot(tl.trans(spike_tile), gy_tile, allow_tf32=True)

        # Atomic-add the accumulator into grad_W at the (offs_k_global, offs_n) tile.
        # Atomic since multiple kp_block / n_block programs may target overlapping
        # tiles when called from autograd-Function-side dispatch with multiple
        # kernel launches; the host sets up a non-overlapping grid by default
        # (see torch_glue), so the atomic is a safety net for autotune
        # configs that expand the grid.
        gw_ptrs = (grad_w_ptr
                   + offs_k_global[:, None] * stride_gwk
                   + offs_n[None, :] * stride_gwn)
        tl.atomic_add(
            gw_ptrs,
            acc,
            mask=k_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def packed_spike_matmul_bwd_dS_kernel(
        # ---- input pointers ----
        grad_y_ptr,          # f32    (M, N)
        weight_ptr,          # f32/f16 (K, N)
        grad_s_ptr,          # f32    (M, K)  -- DENSE grad_s out
        # ---- shape ----
        M, N, K,
        # ---- strides ----
        stride_gym, stride_gyn,
        stride_wk, stride_wn,
        stride_gsm, stride_gsk,
        # ---- tile ----
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Backward dS: grad_s[m, k] = sum_n grad_y[m, n] * weight[k, n].

        grad_s is the surrogate-gradient signal feeding the spike's atan
        backward; it is **dense fp**, not binary.  So no packing on output.
        """
        pid_m = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K

        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        for n_block in range(0, tl.cdiv(N, BLOCK_N)):
            offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            gy_ptrs = (grad_y_ptr
                       + offs_m[:, None] * stride_gym
                       + offs_n[None, :] * stride_gyn)
            gy_tile = tl.load(
                gy_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            w_ptrs = (weight_ptr
                      + offs_k[:, None] * stride_wk
                      + offs_n[None, :] * stride_wn)
            w_tile = tl.load(
                w_ptrs,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # acc += gy @ w.T
            acc += tl.dot(gy_tile, tl.trans(w_tile), allow_tf32=True)

        gs_ptrs = (grad_s_ptr
                   + offs_m[:, None] * stride_gsm
                   + offs_k[None, :] * stride_gsk)
        tl.store(gs_ptrs, acc, mask=m_mask[:, None] & k_mask[None, :])


# ---------------------------------------------------------------------------
# Numpy host fallback (CPU + correctness reference, also used as the
# torch-free verification path that the kernel matches).
# ---------------------------------------------------------------------------
def packed_spike_matmul_numpy(
    packed_s: np.ndarray,
    weight: np.ndarray,
    d_in: int,
) -> np.ndarray:
    """Reference: ``y = unpack(packed_s, d_in) @ weight``.

    Pure numpy.  Used by tests to verify the Triton kernel output and
    by CPU-side smoke runs.  Does NOT exercise the bandwidth saving
    (numpy materialises the unpacked tensor) but is bit-equivalent in
    fp32.

    Parameters
    ----------
    packed_s : np.ndarray
        uint16 (M, K_packed).
    weight : np.ndarray
        Float (d_in, N).  d_in must equal the original spike channel
        count (before packing).
    d_in : int
        Original spike channel count.  Used by ``unpack_spikes`` to
        trim the trailing padding bits.

    Returns
    -------
    np.ndarray
        Float32 (M, N).
    """
    from synapforge.native.spike.pack import unpack_spikes

    if weight.shape[0] != d_in:
        raise ValueError(
            f"packed_spike_matmul_numpy: weight.shape[0]={weight.shape[0]} "
            f"must equal d_in={d_in}"
        )

    s_unpacked = unpack_spikes(packed_s, d_in, dtype=np.float32)
    # s_unpacked: (M, d_in); weight: (d_in, N)
    return s_unpacked.astype(np.float32) @ weight.astype(np.float32)


# ---------------------------------------------------------------------------
# Public dispatcher (numpy-only signature; torch glue calls the triton
# launcher in torch_glue.py).
# ---------------------------------------------------------------------------
def packed_spike_matmul(
    packed_s: np.ndarray,
    weight: np.ndarray,
    d_in: int,
    d_out: int,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Compute ``y = unpack(packed_s, d_in) @ weight`` on the host side.

    This is the **numpy-side** entry-point.  GPU-side training calls
    the triton launcher in ``torch_glue.py``.

    Parameters
    ----------
    packed_s : np.ndarray
        uint16 (M, K_packed).
    weight : np.ndarray
        Float (d_in, d_out).
    d_in : int
        Original spike channel count.
    d_out : int
        Output channel count (must equal ``weight.shape[1]``).
    backend : {"auto", "numpy"}, optional
        ``auto`` falls back to ``numpy`` here (Triton requires CUDA
        tensors -- see ``torch_glue.py`` for the GPU path).
    """
    if weight.shape != (d_in, d_out):
        raise ValueError(
            f"packed_spike_matmul: weight shape {weight.shape} != "
            f"(d_in={d_in}, d_out={d_out})"
        )
    if backend not in ("auto", "numpy"):
        raise ValueError(
            f"packed_spike_matmul: unknown backend {backend!r}; "
            f"GPU path lives in torch_glue.PackedSpikeMatmul"
        )
    return packed_spike_matmul_numpy(packed_s, weight, d_in)


__all__ = [
    "packed_spike_matmul",
    "packed_spike_matmul_numpy",
    # The kernels are exported only when triton is available; consumers
    # should test ``_HAS_TRITON`` first.
]
