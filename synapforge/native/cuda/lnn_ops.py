"""lnn_ops -- LNN+SNN-specific ops on CudaTensor (no torch).

These are the building blocks the SynapForge HybridBlock needs and that
are NOT in the standard cuBLAS / cupy library. They are correct on
both the cupy GPU path and the numpy CPU fallback.

Coverage
--------
* ``cfc_step_dense``   -- single-step dense Liquid CfC update
* ``cfc_scan``         -- O(T) sequential scan (used when fused
                          Triton kernel is unavailable)
* ``plif_spike``       -- Heaviside spike with optional ATan surrogate
                          gradient (forward + a separately-callable
                          surrogate derivative for the VJP package)
* ``rope_apply``       -- Rotary position embeddings (LLaMA layout)
* ``swiglu``           -- x * silu(g) -- the SwiGLU FFN intermediate
* ``rfold_cumprod``    -- numerically-stable cumulative product, used
                          in the R-fold inference path

NO torch import. Pure numpy/cupy.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as _np

from .tensor import CudaTensor

try:  # pragma: no cover -- optional GPU path
    import cupy as _cp

    try:
        _cp.zeros(1, dtype=_cp.float32) + 0
        _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
except Exception:  # pragma: no cover
    _cp = None
    _HAS_CUPY = False


__all__ = [
    "cfc_step_dense",
    "cfc_scan",
    "plif_spike",
    "atan_surrogate_grad",
    "rope_apply",
    "swiglu",
    "rfold_cumprod",
]


def _xp():
    return _cp if _HAS_CUPY else _np


def _unwrap(t: Any) -> Any:
    return t.data if isinstance(t, CudaTensor) else t


# ---------------------------------------------------------------------------
# Liquid CfC -- closed-form Hasani 2022 Eq 5
# ---------------------------------------------------------------------------


def cfc_step_dense(
    h_prev: CudaTensor,
    a: CudaTensor,
    b: CudaTensor,
) -> CudaTensor:
    """Single CfC step: ``h_new = a * h_prev + b``.

    All three inputs are shape (B, D). Used as the inner loop of
    ``cfc_scan`` and as the per-step update inside the Triton kernel
    (handled by ``triton_glue``).
    """
    res = _unwrap(a) * _unwrap(h_prev) + _unwrap(b)
    return CudaTensor(res, device=h_prev.device)


def cfc_scan(
    a: CudaTensor,
    b: CudaTensor,
    h0: Optional[CudaTensor] = None,
) -> CudaTensor:
    """O(T) sequential CfC scan.

    Parameters
    ----------
    a, b : CudaTensor
        Shape (B, T, D). ``a`` is the decay multiplier, ``b`` is the input.
    h0 : CudaTensor, optional
        Initial state (B, D). Zeros if None.

    Returns
    -------
    h : CudaTensor of shape (B, T, D)
        h[:, t, :] = a[:, t, :] * h[:, t-1, :] + b[:, t, :]
    """
    xp = _xp()
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    B, T, D = a_arr.shape
    out = xp.empty_like(a_arr)
    if h0 is None:
        h = xp.zeros((B, D), dtype=a_arr.dtype)
    else:
        h = _unwrap(h0).astype(a_arr.dtype, copy=True)
    for t in range(T):
        h = a_arr[:, t, :] * h + b_arr[:, t, :]
        out[:, t, :] = h
    return CudaTensor(out, device=a.device)


# ---------------------------------------------------------------------------
# PLIF spike + ATan surrogate
# ---------------------------------------------------------------------------


def plif_spike(membrane: CudaTensor, threshold: CudaTensor) -> Tuple[CudaTensor, CudaTensor]:
    """Heaviside spike: ``s = 1[membrane > threshold]``.

    Parameters
    ----------
    membrane : CudaTensor of shape (..., D)
    threshold : CudaTensor of shape (D,) -- broadcast over leading dims

    Returns
    -------
    spike : CudaTensor (binary 0/1, same dtype as membrane)
    pre_threshold_membrane : CudaTensor (membrane - threshold), kept
        for the surrogate-gradient VJP downstream.
    """
    xp = _xp()
    m = _unwrap(membrane)
    thr = _unwrap(threshold)
    delta = m - thr
    s = (delta > 0.0).astype(m.dtype)
    return (
        CudaTensor(s, device=membrane.device),
        CudaTensor(delta, device=membrane.device),
    )


def atan_surrogate_grad(
    pre_threshold_membrane: CudaTensor,
    alpha: float = 2.0,
) -> CudaTensor:
    """ATan surrogate gradient: ``ds/dm = alpha / (2 * (1 + (pi/2 * alpha * m)^2))``.

    Used by the manual VJP of ``plif_spike`` in the synapforge native
    backward pass. Reference: Fang 2021 "Incorporating Learnable Membrane
    Time Constant ...".
    """
    xp = _xp()
    m = _unwrap(pre_threshold_membrane)
    pi_over_2 = math.pi / 2.0
    x = alpha * m
    denom = 1.0 + (pi_over_2 * x) * (pi_over_2 * x)
    grad = m.dtype.type(alpha / 2.0) / denom
    return CudaTensor(grad, device=pre_threshold_membrane.device)


# ---------------------------------------------------------------------------
# RoPE (rotary position embeddings, LLaMA layout)
# ---------------------------------------------------------------------------


def rope_apply(
    x: CudaTensor,
    cos_cache: CudaTensor,
    sin_cache: CudaTensor,
    *,
    pos_offset: int = 0,
) -> CudaTensor:
    """Apply rotary position embedding to ``x``.

    Parameters
    ----------
    x : (B, T, n_head, head_dim)
        Q or K tensor. ``head_dim`` must be even.
    cos_cache, sin_cache : (max_T, head_dim)
        Pre-computed RoPE rotation tables.
    pos_offset : int
        For incremental decoding: the timestep offset of x[:, 0, ...].

    Returns
    -------
    rotated : same shape as x

    Implementation
    --------------
    Splits ``x`` into even/odd halves along ``head_dim`` and applies
    the 2D rotation matrix derived from cos/sin. This is the LLaMA
    GPT-NeoX layout (``[x_even, x_odd]`` interleaved).
    """
    xp = _xp()
    arr = _unwrap(x)
    cos_arr = _unwrap(cos_cache)
    sin_arr = _unwrap(sin_cache)
    B, T, H, D = arr.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    cos = cos_arr[pos_offset : pos_offset + T, :]   # (T, D)
    sin = sin_arr[pos_offset : pos_offset + T, :]

    # Reshape so the last dim splits into pairs: (..., D/2, 2)
    arr2 = arr.reshape(B, T, H, D // 2, 2)
    cos2 = cos.reshape(T, D // 2, 2)
    sin2 = sin.reshape(T, D // 2, 2)

    # Pull apart the pair so we can rotate.
    x_even = arr2[..., 0]   # (B, T, H, D/2)
    x_odd = arr2[..., 1]
    # broadcast cos/sin over (B, H)
    cos_e = cos2[None, :, None, :, 0]   # (1, T, 1, D/2)
    sin_e = sin2[None, :, None, :, 0]

    rot_even = x_even * cos_e - x_odd * sin_e
    rot_odd = x_even * sin_e + x_odd * cos_e

    out = xp.stack([rot_even, rot_odd], axis=-1).reshape(B, T, H, D)
    return CudaTensor(out, device=x.device)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def swiglu(x: CudaTensor, gate: CudaTensor) -> CudaTensor:
    """SwiGLU FFN intermediate: ``x * silu(gate)``.

    Used as ``swiglu(W_v(x), W_g(x))`` in our SwiGLU FFN. The two
    matmuls are caller's responsibility -- this is the elementwise
    fusion only.
    """
    xp = _xp()
    g = _unwrap(gate)
    silu_g = g * (1.0 / (1.0 + xp.exp(-g)))
    res = _unwrap(x) * silu_g
    return CudaTensor(res, device=x.device)


# ---------------------------------------------------------------------------
# rfold cumulative product (numerically stable in log space)
# ---------------------------------------------------------------------------


def rfold_cumprod(x: CudaTensor, axis: int = -1, eps: float = 1e-8) -> CudaTensor:
    """Numerically stable cumulative product.

    Computes ``cumprod(x)`` along ``axis`` via ``exp(cumsum(log(|x| + eps)))
    * sign(x)`` to avoid the underflow that plain ``cumprod`` hits when
    the per-step gating values shrink the running product to fp32-zero.

    Used in the R-fold inference path (Coconut latent thinking).
    """
    xp = _xp()
    arr = _unwrap(x)
    # Magnitude via log space; sign tracked by a parallel cumprod over signs.
    abs_arr = xp.abs(arr) + arr.dtype.type(eps)
    log_abs = xp.log(abs_arr)
    log_cum = xp.cumsum(log_abs, axis=axis)
    abs_cum = xp.exp(log_cum)
    sign = xp.sign(arr)
    sign_cum = xp.cumprod(sign, axis=axis)
    out = abs_cum * sign_cum
    return CudaTensor(out, device=x.device)
