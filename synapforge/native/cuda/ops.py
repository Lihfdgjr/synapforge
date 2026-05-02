"""ops -- cuBLAS / cupy elementwise + reductions for CudaTensor.

All ops:
* Accept and return ``CudaTensor`` (numpy fallback when cupy missing).
* Default to fp32. Pass ``out=`` to recycle a pre-allocated CudaTensor.
* Have an explicit forward signature -- no autograd, no graph. Backward
  for these lives in ``synapforge/native/vjp/`` (sister package).

Coverage (matches request):
    matmul, bmm, gemm
    silu, gelu, sigmoid, tanh
    sum, mean, var, max, argmax, topk
    add, mul, addcmul, addcdiv
    layernorm, rmsnorm
    softmax (numerically stable)
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

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
    "matmul", "bmm", "gemm",
    "silu", "gelu", "sigmoid", "tanh",
    "sum", "mean", "var", "max", "argmax", "topk",
    "add", "mul", "addcmul", "addcdiv",
    "layernorm", "rmsnorm",
    "softmax",
]


def _xp():
    return _cp if _HAS_CUPY else _np


def _wrap(arr: Any, device: Optional[str] = None) -> CudaTensor:
    return CudaTensor(arr, device=device)


def _unwrap(t: Union[CudaTensor, Any]) -> Any:
    if isinstance(t, CudaTensor):
        return t.data
    return t


# ---------------------------------------------------------------------------
# Linear algebra: matmul / bmm / gemm
# ---------------------------------------------------------------------------


def matmul(a: CudaTensor, b: CudaTensor, *, out: Optional[CudaTensor] = None) -> CudaTensor:
    """C = A @ B. Routes through cuBLAS via cupy.matmul.

    Shapes: ``A`` is (..., M, K), ``B`` is (..., K, N) -> (..., M, N).
    Broadcasts on leading dims.
    """
    xp = _xp()
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    res = xp.matmul(a_arr, b_arr)
    if out is not None:
        out.data[...] = res  # type: ignore[index]
        return out
    dev = "cuda:0" if (_HAS_CUPY and _cp is not None and isinstance(res, _cp.ndarray)) else "cpu"
    return _wrap(res, device=dev)


def bmm(a: CudaTensor, b: CudaTensor, *, out: Optional[CudaTensor] = None) -> CudaTensor:
    """Batched matmul. ``A`` (B, M, K) x ``B`` (B, K, N) -> (B, M, N).

    Forwards to ``matmul`` since cupy/numpy already support batched.
    """
    return matmul(a, b, out=out)


def gemm(
    a: CudaTensor,
    b: CudaTensor,
    *,
    transa: bool = False,
    transb: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: Optional[CudaTensor] = None,
) -> CudaTensor:
    """C = alpha * op(A) * op(B) + beta * C, the cuBLAS GEMM signature.

    ``op(X) = X.T`` if the corresponding ``transX`` is True.

    cupy exposes ``cupy.cublas.gemm`` for the raw cuBLAS call but the
    signature is awkward (in-place, fp32 only via dgemm/sgemm). We
    transpose-then-matmul which lets cupy choose between cuBLAS sgemm
    / hgemm / bf16-emulation depending on dtype.
    """
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    if transa:
        a_arr = a_arr.swapaxes(-1, -2)
    if transb:
        b_arr = b_arr.swapaxes(-1, -2)
    xp = _xp()
    res = xp.matmul(a_arr, b_arr)
    if alpha != 1.0:
        res = res * a_arr.dtype.type(alpha)  # keep dtype
    if c is not None and beta != 0.0:
        res = res + b_arr.dtype.type(beta) * _unwrap(c)
    elif c is not None and beta == 0.0:
        c.data[...] = res  # type: ignore[index]
        return c
    dev = "cuda:0" if (_HAS_CUPY and _cp is not None and isinstance(res, _cp.ndarray)) else "cpu"
    return _wrap(res, device=dev)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def silu(x: CudaTensor) -> CudaTensor:
    """SiLU / Swish: x * sigmoid(x)."""
    xp = _xp()
    arr = _unwrap(x)
    res = arr * (1.0 / (1.0 + xp.exp(-arr)))
    return _wrap(res, device=x.device)


def gelu(x: CudaTensor, approximate: str = "tanh") -> CudaTensor:
    """GELU activation. ``approximate='tanh'`` matches GPT-style GELU."""
    xp = _xp()
    arr = _unwrap(x)
    if approximate == "tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c = math.sqrt(2.0 / math.pi)
        inner = c * (arr + 0.044715 * arr * arr * arr)
        res = 0.5 * arr * (1.0 + xp.tanh(inner))
    else:
        # exact via erf
        c = 1.0 / math.sqrt(2.0)
        # Some xp impls expose erf via xp.special; fall back to tanh form.
        try:
            res = 0.5 * arr * (1.0 + xp.special.erf(arr * c))  # type: ignore[attr-defined]
        except AttributeError:
            c2 = math.sqrt(2.0 / math.pi)
            inner = c2 * (arr + 0.044715 * arr * arr * arr)
            res = 0.5 * arr * (1.0 + xp.tanh(inner))
    return _wrap(res, device=x.device)


def sigmoid(x: CudaTensor) -> CudaTensor:
    xp = _xp()
    arr = _unwrap(x)
    res = 1.0 / (1.0 + xp.exp(-arr))
    return _wrap(res, device=x.device)


def tanh(x: CudaTensor) -> CudaTensor:
    xp = _xp()
    arr = _unwrap(x)
    return _wrap(xp.tanh(arr), device=x.device)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _reduce(fn_name: str, x: CudaTensor, axis: Any = None, keepdims: bool = False) -> CudaTensor:
    xp = _xp()
    arr = _unwrap(x)
    res = getattr(xp, fn_name)(arr, axis=axis, keepdims=keepdims)
    if not isinstance(res, (xp.ndarray,)):
        # scalar -> wrap into 0-d
        res = xp.asarray(res)
    return _wrap(res, device=x.device)


def sum(x: CudaTensor, axis: Any = None, keepdims: bool = False) -> CudaTensor:  # noqa: A001
    return _reduce("sum", x, axis, keepdims)


def mean(x: CudaTensor, axis: Any = None, keepdims: bool = False) -> CudaTensor:
    return _reduce("mean", x, axis, keepdims)


def var(x: CudaTensor, axis: Any = None, keepdims: bool = False, ddof: int = 0) -> CudaTensor:
    xp = _xp()
    arr = _unwrap(x)
    res = xp.var(arr, axis=axis, keepdims=keepdims, ddof=ddof)
    if not isinstance(res, (xp.ndarray,)):
        res = xp.asarray(res)
    return _wrap(res, device=x.device)


def max(x: CudaTensor, axis: Any = None, keepdims: bool = False) -> CudaTensor:  # noqa: A001
    return _reduce("max", x, axis, keepdims)


def argmax(x: CudaTensor, axis: Any = None) -> CudaTensor:
    xp = _xp()
    arr = _unwrap(x)
    res = xp.argmax(arr, axis=axis)
    if not isinstance(res, (xp.ndarray,)):
        res = xp.asarray(res)
    return _wrap(res, device=x.device)


def topk(x: CudaTensor, k: int, axis: int = -1) -> Tuple[CudaTensor, CudaTensor]:
    """Return (values, indices) of the top-k along ``axis``.

    Both numpy and cupy expose ``argpartition``, which we use to avoid
    a full sort. Final order within the top-k is descending by value.
    """
    xp = _xp()
    arr = _unwrap(x)
    if axis < 0:
        axis += arr.ndim
    if k >= arr.shape[axis]:
        sort_idx = xp.argsort(-arr, axis=axis)
        sort_idx = xp.take(sort_idx, xp.arange(k), axis=axis)
    else:
        part_idx = xp.argpartition(-arr, k - 1, axis=axis)
        topk_idx = xp.take(part_idx, xp.arange(k), axis=axis)
        # Sort the top-k by descending value for deterministic order.
        topk_vals = xp.take_along_axis(arr, topk_idx, axis=axis)
        order = xp.argsort(-topk_vals, axis=axis)
        topk_idx = xp.take_along_axis(topk_idx, order, axis=axis)
        sort_idx = topk_idx
    topk_vals = xp.take_along_axis(arr, sort_idx, axis=axis)
    return _wrap(topk_vals, device=x.device), _wrap(sort_idx, device=x.device)


# ---------------------------------------------------------------------------
# Elementwise binary
# ---------------------------------------------------------------------------


def add(a: CudaTensor, b: CudaTensor, *, alpha: float = 1.0) -> CudaTensor:
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    res = a_arr + (alpha * b_arr if alpha != 1.0 else b_arr)
    return _wrap(res, device=a.device)


def mul(a: CudaTensor, b: CudaTensor) -> CudaTensor:
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    return _wrap(a_arr * b_arr, device=a.device)


def addcmul(a: CudaTensor, b: CudaTensor, c: CudaTensor, *, value: float = 1.0) -> CudaTensor:
    """Out = A + value * B * C. Matches torch.addcmul semantics."""
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    c_arr = _unwrap(c)
    res = a_arr + b_arr.dtype.type(value) * (b_arr * c_arr) if value != 1.0 else a_arr + b_arr * c_arr
    return _wrap(res, device=a.device)


def addcdiv(a: CudaTensor, b: CudaTensor, c: CudaTensor, *, value: float = 1.0, eps: float = 0.0) -> CudaTensor:
    """Out = A + value * B / (C + eps). Used in AdamW."""
    a_arr = _unwrap(a)
    b_arr = _unwrap(b)
    c_arr = _unwrap(c)
    if eps != 0.0:
        denom = c_arr + b_arr.dtype.type(eps)
    else:
        denom = c_arr
    res = a_arr + b_arr.dtype.type(value) * (b_arr / denom) if value != 1.0 else a_arr + b_arr / denom
    return _wrap(res, device=a.device)


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------


def layernorm(
    x: CudaTensor,
    gamma: Optional[CudaTensor] = None,
    beta: Optional[CudaTensor] = None,
    eps: float = 1e-5,
    axis: int = -1,
) -> CudaTensor:
    """Layernorm along ``axis``. ``gamma``, ``beta`` broadcast over the norm axis.

    cupy ships fast ``ElementwiseKernel`` we could compile for this; the
    pure-xp version below is correct on both backends and avoids
    bringing in an extra compile step. For training-time hot path,
    consider replacing with a fused Triton kernel (see triton_glue.py).
    """
    xp = _xp()
    arr = _unwrap(x)
    mu = xp.mean(arr, axis=axis, keepdims=True)
    sigma2 = xp.var(arr, axis=axis, keepdims=True)
    inv_sigma = 1.0 / xp.sqrt(sigma2 + arr.dtype.type(eps))
    out = (arr - mu) * inv_sigma
    if gamma is not None:
        out = out * _unwrap(gamma)
    if beta is not None:
        out = out + _unwrap(beta)
    return _wrap(out, device=x.device)


def rmsnorm(
    x: CudaTensor,
    gamma: Optional[CudaTensor] = None,
    eps: float = 1e-6,
    axis: int = -1,
) -> CudaTensor:
    """RMSNorm: x * gamma / sqrt(mean(x^2) + eps). LLaMA / synapforge style."""
    xp = _xp()
    arr = _unwrap(x)
    sq_mean = xp.mean(arr * arr, axis=axis, keepdims=True)
    inv_rms = 1.0 / xp.sqrt(sq_mean + arr.dtype.type(eps))
    out = arr * inv_rms
    if gamma is not None:
        out = out * _unwrap(gamma)
    return _wrap(out, device=x.device)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def softmax(x: CudaTensor, axis: int = -1) -> CudaTensor:
    """Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))."""
    xp = _xp()
    arr = _unwrap(x)
    m = xp.max(arr, axis=axis, keepdims=True)
    e = xp.exp(arr - m)
    s = xp.sum(e, axis=axis, keepdims=True)
    return _wrap(e / s, device=x.device)
