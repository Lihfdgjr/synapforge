"""Liquid CfC continuous-time recurrence -- closed-form VJP.

This is the Hasani et al. 2022 closed-form CfC recurrence, in the
specific selective-dt form used by ``synapforge.cells.liquid.LiquidCell``
(matching ``mscfc.liquid_s4.LiquidS4Cell`` byte-for-byte in the forward).

Forward (single step)
---------------------
For input x_t of shape (B, in_dim) and prior state h_{t-1} of shape (B, d):

    delta_in = x_t @ W_in.T                              # (B, d)
    delta_t  = softplus(delta_in)                        # (B, d), >0
    A_t      = exp(-delta_t * exp(A_log))                # (B, d), in (0, 1]
    b_in     = x_t @ W_h.T                               # (B, d)  ("h" for "hidden" input proj)
    B_t      = delta_t * b_in   (+ b broadcast if given) # (B, d)
    h_t      = A_t * h_{t-1} + B_t                       # (B, d)  pre-tanh state
    out_t    = tanh(h_t)                                 # (B, d)  bound output

Forward returns ``(h_t, out_t, cache)`` where ``h_t`` is the pre-tanh
state used as ``h_{t-1}`` for the next step (this matches Hasani's
closed-form scan convention -- the tanh is a *bound*, not part of the
recurrence).

Multi-step ``cfc_seq_fwd`` simply scans over T using the pre-tanh
state. We do NOT use the cumsum-based parallel scan here because the
backward for that scan involves cumsum subtractions that are
numerically delicate -- the sequential walk gives byte-tight VJPs.

Backward (single step)
----------------------
Given upstream grads (grad_out_t, grad_h_t_from_next), where
``grad_h_t_from_next`` is the gradient flowing back from step t+1's
recurrence (zero at t = T-1):

    1. tanh chain:     grad_h_t = grad_out_t * (1 - out_t^2)
                                + grad_h_t_from_next
    2. h_t = A * h_{t-1} + B:
                       grad_A = grad_h_t * h_{t-1}
                       grad_h_prev = grad_h_t * A
                       grad_B = grad_h_t
    3. B = delta_t * b_in:
                       grad_delta_from_B = grad_B * b_in
                       grad_b_in = grad_B * delta_t
    4. A = exp(-delta_t * exp(A_log)):
                       grad_delta_from_A = grad_A * (-exp(A_log)) * A
                       grad_A_log         = sum_{batch} grad_A * (-delta_t) * exp(A_log) * A
    5. delta = softplus(delta_in):
                       grad_delta_in = (grad_delta_from_B + grad_delta_from_A)
                                       * sigmoid(delta_in)
    6. delta_in = x_t @ W_in.T  ;  b_in = x_t @ W_h.T:
                       grad_x_t = grad_delta_in @ W_in + grad_b_in @ W_h
                       grad_W_in = grad_delta_in.T @ x_t   (sum over batch)
                       grad_W_h  = grad_b_in.T @ x_t       (sum over batch)
                       grad_b   = grad_delta_in.sum_{batch} (if b given to fwd via B)

References
----------
* Hasani et al., "Closed-form Continuous-Time Neural Networks" (Nature
  Machine Intelligence 2022), §3.2.
* Heinsen, "Efficient Parallelization of an Ubiquitous Sequential
  Computation" (arXiv:2311.06281, 2023) -- parallel-scan equivalent
  forward (we do sequential bwd here for VJP simplicity).
* synapforge.cells.liquid.LiquidCell -- matching forward in the codebase.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    # softplus(x) = log1p(exp(x)) -- numerically stable
    # for large positive x: softplus(x) ~ x (avoid overflow in exp)
    # for large negative x: softplus(x) ~ exp(x) (no overflow)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # d softplus / dx = sigmoid(x)
    pos_mask = x >= 0
    out = np.empty_like(x)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    e = np.exp(x[~pos_mask])
    out[~pos_mask] = e / (1.0 + e)
    return out


def cfc_step_fwd(
    h_prev: np.ndarray,
    x: np.ndarray,
    W_in: np.ndarray,
    W_h: np.ndarray,
    A_log: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """One step of Liquid-S4 CfC recurrence (closed-form Hasani 2022).

    Parameters
    ----------
    h_prev : (B, d) prior pre-tanh state.
    x : (B, in_dim) current input.
    W_in : (d, in_dim) selective-dt projection (called W_delta in mscfc).
    W_h : (d, in_dim) input-to-hidden projection (called W_b in mscfc).
    A_log : (d,) log of per-channel decay rate.
    b : optional (d,) bias added to B_t.

    Returns
    -------
    h_t : (B, d) new pre-tanh state.
    out_t : (B, d) tanh-bounded output.
    cache : dict of saved tensors for the backward.
    """
    if W_in.shape != W_h.shape:
        raise ValueError(
            f"cfc_step_fwd: W_in {W_in.shape} != W_h {W_h.shape}"
        )
    d, in_dim = W_in.shape
    if x.shape[-1] != in_dim:
        raise ValueError(
            f"cfc_step_fwd: x last dim {x.shape[-1]} != in_dim {in_dim}"
        )
    if h_prev.shape[-1] != d:
        raise ValueError(
            f"cfc_step_fwd: h_prev last dim {h_prev.shape[-1]} != d {d}"
        )
    if A_log.shape != (d,):
        raise ValueError(
            f"cfc_step_fwd: A_log shape {A_log.shape} != ({d},)"
        )

    delta_in = x @ W_in.T                  # (B, d)
    delta_t = _softplus(delta_in)          # (B, d)
    expA = np.exp(A_log)                   # (d,)
    A_t = np.exp(-delta_t * expA)          # (B, d)
    b_in = x @ W_h.T                       # (B, d)
    B_t = delta_t * b_in                   # (B, d)
    if b is not None:
        if b.shape != (d,):
            raise ValueError(
                f"cfc_step_fwd: b shape {b.shape} != ({d},)"
            )
        B_t = B_t + b
    h_t = A_t * h_prev + B_t               # (B, d)
    out_t = np.tanh(h_t)                   # (B, d)

    cache = dict(
        x=x, h_prev=h_prev, W_in=W_in, W_h=W_h, A_log=A_log, b=b,
        delta_in=delta_in, delta_t=delta_t, expA=expA,
        A_t=A_t, b_in=b_in, B_t=B_t, h_t=h_t, out_t=out_t,
        has_bias=(b is not None),
    )
    return h_t, out_t, cache


def cfc_step_bwd(
    grad_out_t: np.ndarray,
    grad_h_t_from_next: np.ndarray,
    cache: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """One-step backward through the CfC recurrence.

    Parameters
    ----------
    grad_out_t : (B, d) gradient on tanh output.
    grad_h_t_from_next : (B, d) gradient on pre-tanh state from next step
        (zero on the final timestep).
    cache : dict from ``cfc_step_fwd``.

    Returns
    -------
    (grad_x, grad_h_prev, grad_W_in, grad_W_h, grad_A_log, grad_b)
        grad_b is None when forward did not receive a bias.
    """
    x = cache["x"]
    h_prev = cache["h_prev"]
    W_in = cache["W_in"]
    W_h = cache["W_h"]
    delta_in = cache["delta_in"]
    delta_t = cache["delta_t"]
    expA = cache["expA"]
    A_t = cache["A_t"]
    b_in = cache["b_in"]
    out_t = cache["out_t"]
    has_bias = cache["has_bias"]

    # Step 1: tanh derivative + add recurrence path
    grad_h_t = grad_out_t * (1.0 - out_t * out_t) + grad_h_t_from_next  # (B, d)

    # Step 2: h_t = A_t * h_prev + B_t
    grad_A_t = grad_h_t * h_prev
    grad_h_prev = grad_h_t * A_t
    grad_B_t = grad_h_t  # alias is fine; we don't mutate

    # Step 3: B_t = delta_t * b_in [+ bias broadcast]
    grad_delta_from_B = grad_B_t * b_in
    grad_b_in = grad_B_t * delta_t
    if has_bias:
        # b broadcast across batch -> sum over leading (B,)
        grad_b = grad_B_t.reshape(-1, grad_B_t.shape[-1]).sum(axis=0)
    else:
        grad_b = None

    # Step 4: A_t = exp(-delta_t * expA)
    # dA/d(delta_t) = -expA * A     ; dA/d(A_log) = -delta_t * expA * A
    grad_delta_from_A = grad_A_t * (-expA) * A_t
    grad_A_log = (grad_A_t * (-delta_t) * expA * A_t).reshape(
        -1, A_t.shape[-1]
    ).sum(axis=0)

    # Step 5: delta_t = softplus(delta_in) -> sigmoid
    grad_delta_total = grad_delta_from_B + grad_delta_from_A
    grad_delta_in = grad_delta_total * _sigmoid(delta_in)

    # Step 6: linear projections
    in_dim = W_in.shape[1]
    out_dim = W_in.shape[0]
    grad_x_from_delta = grad_delta_in @ W_in   # (B, in_dim)
    grad_x_from_b = grad_b_in @ W_h            # (B, in_dim)
    grad_x = grad_x_from_delta + grad_x_from_b

    flat_delta = grad_delta_in.reshape(-1, out_dim)
    flat_b_in = grad_b_in.reshape(-1, out_dim)
    flat_x = x.reshape(-1, in_dim)
    grad_W_in = flat_delta.T @ flat_x          # (d, in_dim)
    grad_W_h = flat_b_in.T @ flat_x            # (d, in_dim)

    return grad_x, grad_h_prev, grad_W_in, grad_W_h, grad_A_log, grad_b


def cfc_seq_fwd(
    x: np.ndarray,
    h0: np.ndarray,
    W_in: np.ndarray,
    W_h: np.ndarray,
    A_log: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Sequence forward over T timesteps.

    Parameters
    ----------
    x : (B, T, in_dim).
    h0 : (B, d) initial pre-tanh state.
    W_in, W_h, A_log, b : same as ``cfc_step_fwd``.

    Returns
    -------
    h_seq : (B, T, d) pre-tanh states at each timestep (for residual paths).
    out_seq : (B, T, d) tanh-bounded outputs.
    caches : list of length T with per-step cache dicts.
    """
    if x.ndim != 3:
        raise ValueError(f"cfc_seq_fwd: x must be 3D, got {x.shape}")
    B, T, _in_dim = x.shape
    d = W_in.shape[0]

    h_seq = np.empty((B, T, d), dtype=x.dtype)
    out_seq = np.empty((B, T, d), dtype=x.dtype)
    caches: list = []
    h_prev = h0
    for t in range(T):
        h_t, out_t, cache = cfc_step_fwd(
            h_prev, x[:, t, :], W_in, W_h, A_log, b
        )
        h_seq[:, t, :] = h_t
        out_seq[:, t, :] = out_t
        caches.append(cache)
        h_prev = h_t
    return h_seq, out_seq, caches


def cfc_seq_bwd(
    grad_out_seq: np.ndarray,
    grad_h_T: Optional[np.ndarray],
    caches: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sequence backward over T timesteps (BPTT, closed-form).

    Parameters
    ----------
    grad_out_seq : (B, T, d) upstream gradient on tanh outputs.
    grad_h_T : (B, d) optional gradient on the final pre-tanh state
        (used if h_T is read by a separate consumer); zeros if not.
    caches : list of T caches from ``cfc_seq_fwd``.

    Returns
    -------
    (grad_x, grad_h0, grad_W_in, grad_W_h, grad_A_log, grad_b)
    """
    if grad_out_seq.ndim != 3:
        raise ValueError(
            f"cfc_seq_bwd: grad_out_seq must be 3D, got {grad_out_seq.shape}"
        )
    B, T, d = grad_out_seq.shape
    in_dim = caches[0]["W_in"].shape[1]

    grad_x = np.zeros((B, T, in_dim), dtype=grad_out_seq.dtype)
    grad_W_in = np.zeros_like(caches[0]["W_in"])
    grad_W_h = np.zeros_like(caches[0]["W_h"])
    grad_A_log = np.zeros_like(caches[0]["A_log"])
    has_bias = caches[0]["has_bias"]
    if has_bias:
        grad_b = np.zeros_like(caches[0]["b"])
    else:
        grad_b = None

    grad_h_t_from_next = (
        grad_h_T if grad_h_T is not None else np.zeros((B, d), dtype=grad_out_seq.dtype)
    )

    for t in reversed(range(T)):
        gx, gh_prev, gWin, gWh, gAlog, gb = cfc_step_bwd(
            grad_out_seq[:, t, :], grad_h_t_from_next, caches[t]
        )
        grad_x[:, t, :] = gx
        grad_W_in += gWin
        grad_W_h += gWh
        grad_A_log += gAlog
        if has_bias:
            grad_b += gb
        grad_h_t_from_next = gh_prev

    grad_h0 = grad_h_t_from_next   # final accumulated grad on h0
    return grad_x, grad_h0, grad_W_in, grad_W_h, grad_A_log, grad_b
