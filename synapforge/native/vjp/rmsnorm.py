"""RMSNorm -- Root-Mean-Square LayerNorm closed-form VJP.

Forward
-------
For x of shape (..., d) and gamma of shape (d,):

    rms(x) = sqrt( mean(x^2) + eps )            scalar per row
    rstd   = 1 / rms                            saved for backward
    y      = x * rstd * gamma

Backward
--------
Let s = rstd, g = gamma. Per row (writing N = d):

    dy/dx_i = s * g_i  -  (g . y)/N * x_i * s^2 / s
            = s * g_i  -  (1/N) * (s^3) * x_i * sum_j g_j * x_j

Which simplifies (using y_j = s * x_j * g_j) to:

    grad_x_i = s * (g_i * grad_y_i  -  (x_i / N) * sum_j (grad_y_j * g_j * x_j) * s^2)
             = s * g_i * grad_y_i  -  s * x_i / N * sum_j (grad_y_j * g_j * x_j) * s^2

Equivalent vectorised form (this is what we implement, mirroring the
canonical RMSNorm reference in Zhang & Sennrich 2019 and rms_norm
implementations in PyTorch / Apex):

    norm   = x * rstd
    grad_norm = grad_y * gamma
    grad_x = rstd * (grad_norm
                     - norm * mean(norm * grad_norm, axis=-1, keepdims=True))
    grad_gamma = sum_{leading dims} (grad_y * x * rstd)

Numerical pitfall
-----------------
``rstd`` blows up when ``mean(x^2) -> 0``. The eps inside the sqrt
controls this: rstd <= 1 / sqrt(eps). For default eps=1e-6 this caps
rstd at 1e3, well-bounded for fp32 / bf16. Setting eps too small
(<1e-12) can wash out the stabilising effect.

References
----------
* Zhang & Sennrich, "Root Mean Square Layer Normalization" (NeurIPS 2019).
* Apex/Triton ``rms_norm`` kernels: same closed form.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def rmsnorm_fwd(
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """RMSNorm forward.

    Parameters
    ----------
    x : (..., d) input.
    gamma : (d,) per-channel gain.
    eps : float, regulariser inside the sqrt.

    Returns
    -------
    y : (..., d) normalised + scaled output.
    rstd : (..., 1) saved 1/sqrt(mean(x^2) + eps); stored as fp32 even if
        ``x`` is bf16 to keep the bwd numerically faithful.
    """
    if gamma.shape != (x.shape[-1],):
        raise ValueError(
            f"rmsnorm_fwd: gamma shape {gamma.shape} != ({x.shape[-1]},)"
        )
    # Promote to fp32 internally so the squared-mean step doesn't underflow
    # in bf16 (bf16 mantissa is 7 bits, mean(x^2) routinely needs more).
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    rstd = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    y = (x32 * rstd) * gamma.astype(np.float32, copy=False)
    return y.astype(x.dtype, copy=False), rstd


def rmsnorm_bwd(
    grad_y: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    rstd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """RMSNorm backward.

    Parameters
    ----------
    grad_y : (..., d) upstream gradient.
    x : (..., d) saved forward input.
    gamma : (d,) saved gain.
    rstd : (..., 1) saved 1/sqrt(mean(x^2)+eps).

    Returns
    -------
    (grad_x, grad_gamma)
        grad_x : (..., d) -- same shape as x.
        grad_gamma : (d,) -- summed over all leading dims.
    """
    if grad_y.shape != x.shape:
        raise ValueError(
            f"rmsnorm_bwd: grad_y shape {grad_y.shape} != x {x.shape}"
        )
    out_dtype = grad_y.dtype
    # All math in fp32 for stability.
    x32 = x.astype(np.float32, copy=False)
    g32 = gamma.astype(np.float32, copy=False)
    gy32 = grad_y.astype(np.float32, copy=False)
    rstd32 = rstd.astype(np.float32, copy=False)

    d = x.shape[-1]
    norm = x32 * rstd32                               # (..., d)
    grad_norm = gy32 * g32                            # (..., d)
    # Standard RMSNorm bwd: grad_x = rstd * (grad_norm
    #     - norm * mean(norm * grad_norm, axis=-1, keepdims=True))
    inner = np.mean(norm * grad_norm, axis=-1, keepdims=True)
    grad_x = rstd32 * (grad_norm - norm * inner)

    # grad_gamma = sum_{leading} (grad_y * x * rstd) = sum_{leading} (grad_y * norm)
    flat_gy = gy32.reshape(-1, d)
    flat_norm = norm.reshape(-1, d)
    grad_gamma = (flat_gy * flat_norm).sum(axis=0)

    return grad_x.astype(out_dtype, copy=False), grad_gamma.astype(g32.dtype)
