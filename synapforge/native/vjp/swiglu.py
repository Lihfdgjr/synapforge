"""SwiGLU FFN -- closed-form VJP.

Forward
-------
SwiGLU (Shazeer 2020, "GLU Variants Improve Transformer"):

    g = silu(x @ W_gate.T)            (..., h)
    u = x @ W_up.T                    (..., h)
    a = g * u                         (..., h)
    y = a @ W_down.T                  (..., d)

Backward
--------
By chain rule (writing the silu derivative as silu' for compactness),
with silu(z) = z * sigmoid(z) and silu'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z)):

    grad_a       = grad_y @ W_down                        # (..., h)
    grad_W_down  = grad_y.T @ a                           # (d, h)  -> sum over leading
    grad_g       = grad_a * u                             # (..., h)
    grad_u       = grad_a * g                             # (..., h)
    grad_z_gate  = grad_g * silu'(z_gate)                 # (..., h)
    grad_W_up    = grad_u.T @ x                           # (h, d_in)
    grad_W_gate  = grad_z_gate.T @ x                      # (h, d_in)
    grad_x       = grad_z_gate @ W_gate + grad_u @ W_up   # (..., d_in)

We reuse ``linear.linear_bwd`` for the three linear pieces -- this keeps
the implementation DRY and lets the linear module's correctness tests
cover the projection grads. The only SwiGLU-specific piece is the SiLU
+ multiplicative gate.

References
----------
* Shazeer, "GLU Variants Improve Transformer" (2020), arXiv:2002.05202.
* Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016) -- SiLU
  is also called Swish; ``silu(x) = x * sigmoid(x)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from synapforge.native.vjp.linear import linear_bwd, linear_fwd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    pos_mask = x >= 0
    out = np.empty_like(x)
    # for x >= 0: 1 / (1 + exp(-x))
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    # for x < 0: exp(x) / (1 + exp(x))
    neg = x[~pos_mask]
    e = np.exp(neg)
    out[~pos_mask] = e / (1.0 + e)
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _silu_grad(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    # d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    return s + x * s * (1.0 - s)


def swiglu_fwd(
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    W_down: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """SwiGLU FFN forward.

    Parameters
    ----------
    x : (..., d_in)
    W_gate : (h, d_in)
    W_up : (h, d_in)
    W_down : (d_in, h)

    Returns
    -------
    y : (..., d_in)
    saved : dict with keys ``z_gate, z_up, g, u, a`` for the bwd pass.
    """
    if W_gate.shape != W_up.shape:
        raise ValueError(
            f"swiglu_fwd: W_gate {W_gate.shape} != W_up {W_up.shape}"
        )
    if W_down.shape[1] != W_gate.shape[0]:
        raise ValueError(
            f"swiglu_fwd: W_down second dim {W_down.shape[1]} != "
            f"W_gate first dim {W_gate.shape[0]}"
        )

    z_gate = linear_fwd(x, W_gate, b=None)   # (..., h)
    z_up = linear_fwd(x, W_up, b=None)       # (..., h)
    g = _silu(z_gate)                        # (..., h)
    u = z_up                                 # (..., h) -- u and z_up are same
    a = g * u                                # (..., h)
    y = linear_fwd(a, W_down, b=None)        # (..., d_in)

    saved = dict(z_gate=z_gate, z_up=z_up, g=g, u=u, a=a, x=x,
                 W_gate=W_gate, W_up=W_up, W_down=W_down)
    return y, saved


def swiglu_bwd(
    grad_y: np.ndarray,
    saved: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SwiGLU FFN backward.

    Parameters
    ----------
    grad_y : (..., d_in) upstream gradient.
    saved : dict from ``swiglu_fwd``.

    Returns
    -------
    (grad_x, grad_W_gate, grad_W_up, grad_W_down)
    """
    x = saved["x"]
    z_gate = saved["z_gate"]
    g = saved["g"]
    u = saved["u"]
    a = saved["a"]
    W_gate = saved["W_gate"]
    W_up = saved["W_up"]
    W_down = saved["W_down"]

    # y = a @ W_down.T  ->  grad_a, grad_W_down
    grad_a, grad_W_down, _ = linear_bwd(grad_y, a, W_down, has_bias=False)

    # a = g * u  ->  grad_g = grad_a * u  ; grad_u = grad_a * g
    grad_g = grad_a * u
    grad_u = grad_a * g

    # g = silu(z_gate)
    grad_z_gate = grad_g * _silu_grad(z_gate)

    # u = linear(x, W_up) ; z_gate = linear(x, W_gate)
    grad_x_up, grad_W_up, _ = linear_bwd(grad_u, x, W_up, has_bias=False)
    grad_x_gate, grad_W_gate, _ = linear_bwd(grad_z_gate, x, W_gate, has_bias=False)
    grad_x = grad_x_up + grad_x_gate

    return grad_x, grad_W_gate, grad_W_up, grad_W_down
