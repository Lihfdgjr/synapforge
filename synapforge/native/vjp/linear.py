"""Affine layer y = x @ W.T + b -- closed-form VJP.

Forward
-------
    x : (..., in_dim)
    W : (out_dim, in_dim)
    b : (out_dim,) or None
    y : (..., out_dim) = x @ W.T + b

Backward
--------
Let f = sum_{*} grad_y * y. Then for the affine map:

    grad_x = grad_y @ W
    grad_W = grad_y.T @ x       (sum over leading dims)
    grad_b = grad_y.sum over leading dims

The summing-over-leading-dims is handled by reshaping (..., d) -> (N, d).

References
----------
* Goodfellow et al, "Deep Learning" (2016), section 6.5 backprop algebra.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def linear_fwd(
    x: np.ndarray,
    W: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Affine layer forward, y = x @ W.T + b.

    Parameters
    ----------
    x : (..., in_dim)
    W : (out_dim, in_dim)
    b : optional (out_dim,)

    Returns
    -------
    y : (..., out_dim)
    """
    if W.ndim != 2:
        raise ValueError(f"linear_fwd: W must be 2D, got {W.shape}")
    out_dim, in_dim = W.shape
    if x.shape[-1] != in_dim:
        raise ValueError(
            f"linear_fwd: x last dim {x.shape[-1]} != W in_dim {in_dim}"
        )
    y = x @ W.T
    if b is not None:
        if b.shape != (out_dim,):
            raise ValueError(
                f"linear_fwd: b shape {b.shape} != ({out_dim},)"
            )
        y = y + b
    return y


def linear_bwd(
    grad_y: np.ndarray,
    x: np.ndarray,
    W: np.ndarray,
    has_bias: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Affine layer backward.

    Parameters
    ----------
    grad_y : (..., out_dim) upstream gradient.
    x : (..., in_dim) saved forward activation.
    W : (out_dim, in_dim) weight (only its shape & values used for grad_x).
    has_bias : if True returns ``grad_b``, else returns ``None`` for grad_b.

    Returns
    -------
    (grad_x, grad_W, grad_b) where
        grad_x : (..., in_dim)
        grad_W : (out_dim, in_dim)
        grad_b : (out_dim,) or None
    """
    if grad_y.shape[-1] != W.shape[0]:
        raise ValueError(
            f"linear_bwd: grad_y last dim {grad_y.shape[-1]} != "
            f"W out_dim {W.shape[0]}"
        )
    if x.shape[-1] != W.shape[1]:
        raise ValueError(
            f"linear_bwd: x last dim {x.shape[-1]} != W in_dim {W.shape[1]}"
        )
    # grad_x = grad_y @ W  (broadcasts naturally over leading dims)
    grad_x = grad_y @ W
    # grad_W = grad_y.T @ x  (sum over leading dims)
    flat_grad = grad_y.reshape(-1, grad_y.shape[-1])  # (N, out)
    flat_x = x.reshape(-1, x.shape[-1])               # (N, in)
    grad_W = flat_grad.T @ flat_x                     # (out, in)
    if has_bias:
        grad_b = flat_grad.sum(axis=0)
    else:
        grad_b = None
    return grad_x, grad_W, grad_b
