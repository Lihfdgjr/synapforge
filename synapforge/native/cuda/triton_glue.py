"""triton_glue -- bridge so existing Triton kernels accept CudaTensor I/O.

Background
----------
``synapforge/backends/triton_block_kernel.py`` defines the production
fused LNN+SNN forward kernel as a ``@triton.jit`` function. Triton
takes raw cuda pointers via ``T_ptr`` arguments, but the high-level
wrappers in that file accept ``torch.Tensor`` because that is what
Triton's JIT prologue expects (it pulls ``.data_ptr()`` and ``.stride()``
off torch tensors).

This module provides a thin shim:

    1.  Take ``CudaTensor`` inputs.
    2.  Materialize a ``torch.Tensor`` view that *aliases* the same GPU
        memory (no copy) via ``torch.utils.dlpack.from_dlpack``.
    3.  Call the existing torch-API Triton wrapper.
    4.  Wrap the torch.Tensor outputs back as ``CudaTensor`` (again no
        copy, via dlpack).

This is the **only** place in ``synapforge/native/cuda/`` that touches
torch -- and it is gated behind ``HAS_TORCH``. When torch (or Triton or
cupy) is missing, ``run_fused_lnn_snn_block`` raises a clear error and
the test suite skips.

DLPack reference: https://dmlc.github.io/dlpack/latest/python_spec.html
cupy DLPack:     https://docs.cupy.dev/en/stable/user_guide/interoperability.html#dlpack-data-exchange-protocol

NO ``import torch`` at module top. Torch is imported *inside* function
bodies so the wrapper module remains importable on hosts without torch.
"""
from __future__ import annotations

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
    "HAS_TORCH",
    "HAS_TRITON",
    "cuda_tensor_to_torch",
    "torch_to_cuda_tensor",
    "run_fused_lnn_snn_block",
]


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401  -- intentional probe
        return True
    except Exception:
        return False


def _triton_present() -> bool:
    try:
        import triton  # noqa: F401  -- intentional probe
        return True
    except Exception:
        return False


HAS_TORCH = _torch_present()
HAS_TRITON = _triton_present()


# ---------------------------------------------------------------------------
# DLPack-based zero-copy bridge
# ---------------------------------------------------------------------------


def cuda_tensor_to_torch(t: CudaTensor) -> Any:
    """Return a torch.Tensor aliasing the same GPU memory as ``t``.

    Zero-copy on the cupy path (DLPack); a synchronous copy on the
    numpy fallback path. Caller must keep ``t`` alive for as long as
    the returned tensor is used (the underlying cupy array owns the
    memory; torch only borrows the buffer).
    """
    if not HAS_TORCH:
        raise RuntimeError(
            "cuda_tensor_to_torch requires torch; install torch in the "
            "training environment (Triton itself depends on torch)."
        )

    import torch  # local import; never appears at module top

    # Fast path: cupy.ndarray <-> torch.Tensor via DLPack.
    if t.is_cuda and _HAS_CUPY and _cp is not None and isinstance(t.data, _cp.ndarray):
        # cupy.ndarray.toDlpack() / cupy.ndarray.__dlpack__() are both
        # available; prefer the new dlpack v2 protocol when supported.
        try:
            return torch.utils.dlpack.from_dlpack(t.data)
        except Exception:
            # Older cupy / torch -- fall through to capsule API.
            cap = t.data.toDlpack()
            return torch.utils.dlpack.from_dlpack(cap)

    # numpy fallback: synchronous H2D copy. Useful for smoke tests on
    # GPU machines too (e.g. when callers force CPU for a debug pass).
    host = t.to_cpu() if t.is_cuda else _np.asarray(t.data)
    if torch.cuda.is_available():
        return torch.from_numpy(host).cuda()
    return torch.from_numpy(host)


def torch_to_cuda_tensor(tt: Any) -> CudaTensor:
    """Inverse of :func:`cuda_tensor_to_torch`.

    Wraps a ``torch.Tensor`` as a ``CudaTensor`` without copying when
    cupy is available and the tensor is on CUDA; otherwise drops the
    data through a numpy host buffer.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch_to_cuda_tensor requires torch.")
    import torch

    if isinstance(tt, torch.Tensor) and tt.is_cuda and _HAS_CUPY and _cp is not None:
        try:
            arr = _cp.from_dlpack(tt)
        except Exception:
            arr = _cp.fromDlpack(tt.to_dlpack())
        return CudaTensor(arr, device=f"cuda:{tt.device.index or 0}")

    # CPU path
    host = tt.detach().cpu().numpy()
    return CudaTensor.from_cpu(host)


# ---------------------------------------------------------------------------
# Public wrapper around the existing fused LNN+SNN Triton kernel
# ---------------------------------------------------------------------------


def run_fused_lnn_snn_block(
    a: CudaTensor,
    b: CudaTensor,
    threshold: CudaTensor,
    h0: Optional[CudaTensor] = None,
    elig_buffer: Optional[CudaTensor] = None,
    elig_decay: float = 0.99,
    enable_stdp: bool = False,
) -> Tuple[CudaTensor, CudaTensor, CudaTensor, CudaTensor]:
    """Run the fused LNN+SNN forward kernel with CudaTensor I/O.

    Routes to ``synapforge.backends.triton_block_kernel._triton_block_forward``
    for the actual kernel launch. We just wrap the I/O.

    Parameters
    ----------
    a, b
        CfC scan inputs of shape (B, T, D), fp32 (cast inside the kernel).
    threshold
        Per-channel learnable threshold of shape (D,).
    h0
        Initial hidden state of shape (B, D), or None for zeros.
    elig_buffer
        STDP eligibility accumulator (D, D) when ``enable_stdp`` is set.
    elig_decay
        Trace decay used by the STDP atomic update.
    enable_stdp
        Whether to write into ``elig_buffer``.

    Returns
    -------
    h_pre, s, m, h_post
        All shape (B, T, D), wrapped as CudaTensor. ``h_post`` is the
        post-spike-reset hidden state (the actual block output);
        ``h_pre``, ``s``, ``m`` are kept for the backward pass.

    Raises
    ------
    RuntimeError
        If torch / triton / cupy are not all available, or if the
        upstream kernel module cannot be imported.
    """
    if not HAS_TORCH:
        raise RuntimeError(
            "run_fused_lnn_snn_block requires torch (Triton's runtime "
            "is built on top of it). Install torch in the training env."
        )
    if not HAS_TRITON:
        raise RuntimeError(
            "run_fused_lnn_snn_block requires Triton. CPU-only smoke "
            "tests should construct the wrapper but skip this call."
        )

    # Lazy imports so module stays importable without these deps.
    import torch  # noqa: F401
    from synapforge.backends.triton_block_kernel import _triton_block_forward

    # Convert each input to torch.Tensor (zero-copy through DLPack on cupy).
    a_t = cuda_tensor_to_torch(a)
    b_t = cuda_tensor_to_torch(b)
    thr_t = cuda_tensor_to_torch(threshold)
    h0_t = cuda_tensor_to_torch(h0) if h0 is not None else None
    elig_t = cuda_tensor_to_torch(elig_buffer) if elig_buffer is not None else None

    h_pre, s, m, h_post = _triton_block_forward(
        a_t, b_t, thr_t, h0_t, elig_t, float(elig_decay), bool(enable_stdp),
    )

    return (
        torch_to_cuda_tensor(h_pre),
        torch_to_cuda_tensor(s),
        torch_to_cuda_tensor(m),
        torch_to_cuda_tensor(h_post),
    )
