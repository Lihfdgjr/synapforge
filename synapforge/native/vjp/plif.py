"""PLIF (Parametric Leaky-Integrate-Fire) -- closed-form VJP.

Forward
-------
For input current ``x`` of shape (B, d) and prior membrane ``v_prev`` of
shape (B, d), with per-channel ``tau`` (we accept ``tau_log``) and
threshold ``thr`` (per-channel learnable):

    tau    = exp(tau_log)                            (d,)
    decay  = exp(-dt / tau)                          (d,)
    v_pre  = v_prev * decay + x                      (B, d)  pre-spike membrane
    spike  = indicator(v_pre >= thr)                 (B, d)  hard threshold
    v_new  = v_pre - spike * thr                     (B, d)  reset-by-subtract

Backward
--------
The hard indicator ``v_pre >= thr`` is non-differentiable. The standard
SNN trick is the **ATan surrogate** (Fang et al., ICCV 2021), which
keeps the forward indicator but replaces the gradient with:

    d spike / d v_pre = alpha / (2 * (1 + (pi/2 * alpha * (v_pre - thr))^2))

The two upstream gradients are ``grad_spike`` (downstream consumers of
the spike) and ``grad_v_new`` (next-timestep recurrence). Walking
backward through the four equations:

    1. v_new = v_pre - spike * thr
       grad_v_pre += grad_v_new
       grad_spike  += grad_v_new * (-thr)
       grad_thr    += sum_{batch} grad_v_new * (-spike)

    2. spike = surrogate(v_pre - thr)  via ATan
       Let s' = surrogate'(v_pre - thr) = alpha / (2 * (1 + (pi/2 * alpha * (v_pre - thr))^2))
       grad_v_pre += grad_spike_total * s'
       grad_thr   += sum_{batch} grad_spike_total * (-s')

    3. v_pre = v_prev * decay + x
       grad_v_prev = grad_v_pre * decay
       grad_x      = grad_v_pre

    4. decay = exp(-dt / tau) ; tau = exp(tau_log)
       d decay / d tau_log = decay * dt / tau
       grad_tau_log = sum_{batch} (grad_v_pre * v_prev) * (decay * dt / tau)

Note the surrogate constant matches Fang et al. and is identical to
synapforge.cells.plif._ATanSurrogate.backward (codebase ground truth).

Numerical notes
---------------
* When |v_pre - thr| is large, the ATan surrogate decays as 1/x^2,
  giving vanishing-grad behaviour exactly as desired (saturates outside
  the firing band).
* tau_log = exp(tau_log) is unbounded above; the codebase clamps to
  [1e-2, 1e3]. We do not clamp here to keep the closed form pure;
  callers that need clamping should do so on the input.

References
----------
* Fang et al., "Incorporating Learnable Membrane Time Constant to
  Enhance Learning of Spiking Neural Networks" (ICCV 2021).
* synapforge.cells.plif._ATanSurrogate -- reference implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Default ATan slope (matches synapforge.cells.plif default).
DEFAULT_ALPHA: float = 2.0


def plif_fwd(
    v_prev: np.ndarray,
    x: np.ndarray,
    tau_log: np.ndarray,
    thr: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """PLIF forward (one timestep).

    Parameters
    ----------
    v_prev : (B, d) prior membrane potential.
    x : (B, d) input current.
    tau_log : (d,) log of per-channel time constant.
    thr : (d,) per-channel firing threshold.
    alpha : ATan surrogate slope; saved for backward.
    dt : numeric integration step.

    Returns
    -------
    spike : (B, d) hard indicator (0 or 1) -- same dtype as ``x``.
    v_new : (B, d) post-reset membrane.
    saved : dict for the backward.
    """
    if v_prev.shape != x.shape:
        raise ValueError(
            f"plif_fwd: v_prev {v_prev.shape} != x {x.shape}"
        )
    if tau_log.shape != (x.shape[-1],):
        raise ValueError(
            f"plif_fwd: tau_log shape {tau_log.shape} != ({x.shape[-1]},)"
        )
    if thr.shape != (x.shape[-1],):
        raise ValueError(
            f"plif_fwd: thr shape {thr.shape} != ({x.shape[-1]},)"
        )

    tau = np.exp(tau_log)                                 # (d,)
    decay = np.exp(-dt / tau)                             # (d,)
    v_pre = v_prev * decay + x                            # (B, d)
    spike = (v_pre >= thr).astype(x.dtype)                # (B, d)
    v_new = v_pre - spike * thr                           # (B, d)

    saved = dict(
        v_prev=v_prev, x=x, tau_log=tau_log, thr=thr,
        tau=tau, decay=decay, v_pre=v_pre, spike=spike,
        alpha=float(alpha), dt=float(dt),
    )
    return spike, v_new, saved


def plif_bwd(
    grad_spike: np.ndarray,
    grad_v_new: np.ndarray,
    saved: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PLIF backward (one timestep) -- closed-form ATan surrogate.

    Parameters
    ----------
    grad_spike : (B, d) gradient on the spike output.
    grad_v_new : (B, d) gradient on the post-reset membrane (recurrence).
    saved : dict from ``plif_fwd``.

    Returns
    -------
    (grad_x, grad_v_prev, grad_tau_log, grad_thr)
    """
    v_prev = saved["v_prev"]
    tau_log = saved["tau_log"]
    thr = saved["thr"]
    tau = saved["tau"]
    decay = saved["decay"]
    v_pre = saved["v_pre"]
    spike = saved["spike"]
    alpha = saved["alpha"]
    dt = saved["dt"]

    # -- Step 1: v_new = v_pre - spike * thr --
    grad_v_pre = grad_v_new.copy()
    grad_spike_from_reset = grad_v_new * (-thr)            # (B, d)
    # threshold accumulator: sum over batch (broadcast in fwd)
    grad_thr_from_reset = (grad_v_new * (-spike)).reshape(
        -1, spike.shape[-1]
    ).sum(axis=0)

    # -- Step 2: ATan surrogate for spike --
    # x_hat = alpha * (v_pre - thr)
    x_hat = alpha * (v_pre - thr)
    surrogate = alpha / (2.0 * (1.0 + (np.pi / 2.0 * x_hat) ** 2))
    grad_spike_total = grad_spike + grad_spike_from_reset
    grad_v_pre = grad_v_pre + grad_spike_total * surrogate
    grad_thr = grad_thr_from_reset + (
        grad_spike_total * (-surrogate)
    ).reshape(-1, surrogate.shape[-1]).sum(axis=0)

    # -- Step 3: v_pre = v_prev * decay + x --
    grad_v_prev = grad_v_pre * decay
    grad_x = grad_v_pre.copy()

    # -- Step 4: decay = exp(-dt/tau) ; tau = exp(tau_log) --
    # d decay / d tau_log = decay * dt / tau
    d_decay_d_tau_log = decay * dt / tau
    grad_decay = (grad_v_pre * v_prev).reshape(
        -1, decay.shape[-1]
    ).sum(axis=0)
    grad_tau_log = grad_decay * d_decay_d_tau_log

    return grad_x, grad_v_prev, grad_tau_log, grad_thr
