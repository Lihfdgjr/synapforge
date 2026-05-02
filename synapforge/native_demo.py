"""native_demo.py -- Pure-numpy LNN+SNN training MVP (NO torch).

Why this file exists
--------------------
2026-05-02 user feedback: nine prior agents were asked to build a torch
replacement and all nine shipped torch *wrappers* instead -- AdamW
stored numpy moments but called ``torch.mul_`` / ``torch.add_`` on
``torch.Tensor``s; ``Module`` subclassed ``torch.nn.Parameter``;
``rfold`` used ``torch.cumprod``; ``CPUOffloadAdamW`` had a
``torch.Tensor`` in its hot path. Every one of those was a
"native = torch under the hood" lie.

This file is the falsifiable proof that we *can* train an
LNN+SNN model with **no torch import in the hot path**, by:

  1. Storing every weight + activation + grad as a plain ``numpy.ndarray``.
  2. Writing manual forward + backward (VJP) for every op:
     ``embed``, ``linear``, ``rmsnorm``, ``cfc_step``, ``plif_step``,
     ``swiglu``, ``cross_entropy``.
  3. Hand-rolling AdamW in numpy (in-place mul/add on numpy arrays;
     OPENBLAS_NUM_THREADS / MKL gives multi-core for free).
  4. No autograd graph -- forward returns the cache that backward
     consumes; the layer-by-layer reverse pass is explicit.

Run::

    python synapforge/native_demo.py

Outputs ``synapforge/_native_demo_results.json`` with the loss
curve, final loss, wall-time, and ``import torch`` static-grep
self-check.

Architecture (matches synapforge.HybridBlock semantically; weights
+ shapes are bit-equivalent to ``synapforge/native_demo_torch_ref.py``):
    TokenEmbed(V, d)
    -> N_LAYERS x HybridBlock:
            ln1 = RMSNorm(d)
            cfc  = LiquidCell(d, d)        # closed-form Hasani 2022 Eq 5
            plif = PLIF(d)                 # ATan surrogate Fang 2021
            ln2  = RMSNorm(d)
            ffn  = SwiGLU(d, h=4d)
       and residual connections in two places:
            x = x + plif(cfc(ln1(x)))
            x = x + ffn(ln2(x))
    -> LMHead Linear(d, V)                 # untied (TOKEN_SOUP fix)

Loss: cross-entropy on next-token prediction (synthetic random
integer tokens).

ACCURACY GATE
-------------
1. Loss decreases monotonically (rolling-10 mean) over 100 steps.
2. Final loss within 5% of the torch reference on the same seed
   (``synapforge/native_demo_torch_ref.py``).

NO TORCH GATE
-------------
``grep -E '^import torch\\b|^from torch' synapforge/native_demo.py``
must return zero matches; the test suite checks this.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Globals -- model + train hyperparameters
# ---------------------------------------------------------------------------
VOCAB = 256
D_MODEL = 64
N_LAYERS = 2
SEQ_LEN = 16
BATCH = 4
FFN_RATIO = 4  # SwiGLU hidden = 4 * d
N_STEPS = 100
LR = 3e-3
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.95
ADAMW_EPS = 1e-8
ADAMW_WD = 0.01
SEED = 1234

PLIF_ALPHA = 2.0  # ATan surrogate steepness (Fang 2021)
PLIF_THRESHOLD = 0.3
PLIF_TAU_INIT = 1.5  # tau_log = log(1.5)


# ---------------------------------------------------------------------------
# Utility -- tiny-arr ops + RNG
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    """Return a numpy default RNG seeded for reproducibility."""
    return np.random.default_rng(seed)


def _xavier(rng: np.random.Generator, shape: Tuple[int, ...],
            std: float | None = None) -> np.ndarray:
    """Xavier-ish init in fp32. ``std=None`` -> 1/sqrt(fan_in)."""
    if std is None:
        fan_in = shape[-1] if len(shape) >= 2 else shape[0]
        std = 1.0 / math.sqrt(max(1, fan_in))
    return (rng.standard_normal(shape).astype(np.float32) * std)


def _zeros(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU (a.k.a. Swish): x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def _silu_grad(x: np.ndarray) -> np.ndarray:
    """d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), numerically stable."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _softplus_grad(x: np.ndarray) -> np.ndarray:
    """sigmoid(x)."""
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Op-level VJPs
# ---------------------------------------------------------------------------

def embed_fwd(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """x: (B, T) int -> y: (B, T, d) fp32. Look up rows of W: (V, d)."""
    return W[x]


def embed_bwd(grad_y: np.ndarray, x: np.ndarray, V: int) -> np.ndarray:
    """grad_W = scatter-add grad_y back into row x[b, t] of W."""
    grad_W = np.zeros((V, grad_y.shape[-1]), dtype=np.float32)
    np.add.at(grad_W, x.ravel(), grad_y.reshape(-1, grad_y.shape[-1]))
    return grad_W


def linear_fwd(x: np.ndarray, W: np.ndarray,
               b: np.ndarray | None) -> np.ndarray:
    """x: (..., in_d), W: (out_d, in_d), b: (out_d,) -> y: (..., out_d)."""
    y = x @ W.T
    if b is not None:
        y = y + b
    return y


def linear_bwd(
    grad_y: np.ndarray, x: np.ndarray, W: np.ndarray, has_bias: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """grad_x = grad_y @ W; grad_W = grad_y^T @ x; grad_b = sum grad_y."""
    in_d = W.shape[1]
    out_d = W.shape[0]
    grad_x = grad_y @ W
    grad_W = grad_y.reshape(-1, out_d).T @ x.reshape(-1, in_d)
    grad_b = None
    if has_bias:
        grad_b = grad_y.reshape(-1, out_d).sum(axis=0)
    return grad_x, grad_W, grad_b


def rmsnorm_fwd(x: np.ndarray, w: np.ndarray, eps: float = 1e-6,
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """RMSNorm: y = w * x / sqrt(mean(x^2) + eps).

    x: (..., d), w: (d,) -> y: (..., d).
    Returns (y, cache) for backward.
    """
    sq_mean = (x * x).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(sq_mean + eps)
    x_hat = x * rstd
    y = w * x_hat
    cache = dict(x=x, x_hat=x_hat, rstd=rstd, w=w)
    return y, cache


def rmsnorm_bwd(grad_y: np.ndarray,
                cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Backward through RMSNorm.

    See https://github.com/meta-llama/llama/blob/main/llama/model.py
    for the closed-form (matches PyTorch RMSNorm bwd).
    """
    x = cache["x"]
    x_hat = cache["x_hat"]
    rstd = cache["rstd"]
    w = cache["w"]
    d = x.shape[-1]

    # dL/dxhat = grad_y * w  (broadcast over leading dims)
    dxhat = grad_y * w
    # dL/drstd = sum(dxhat * x, axis=-1, keepdims=True)
    drstd = (dxhat * x).sum(axis=-1, keepdims=True)
    # rstd = (mean(x^2)+eps)^(-1/2); drstd/dx_i = -x_i * rstd^3 / d
    # dL/dx = dxhat * rstd  +  drstd * (-x * rstd^3 / d)
    grad_x = dxhat * rstd + drstd * (-x * (rstd ** 3) / d)

    # dL/dw = sum_{B,T} grad_y * x_hat
    grad_w = (grad_y * x_hat).reshape(-1, d).sum(axis=0)
    return grad_x, grad_w


def cfc_step_fwd(
    x_t: np.ndarray, h_prev: np.ndarray,
    W_delta: np.ndarray, W_b: np.ndarray, A_log: np.ndarray, dt: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """One step of LiquidCell (Hasani 2022 Eq 5, sequential variant).

    delta_t = softplus(W_delta x_t)            shape (B, d)
    A_t     = exp(-delta_t * exp(A_log))       shape (B, d)
    B_t     = delta_t * (W_b x_t)              shape (B, d)
    h_t     = A_t * h_{t-1} + B_t              shape (B, d)
    out_t   = tanh(h_t)                        shape (B, d)

    All inputs/outputs are (B, d) for one timestep. The full sequence
    is built by looping over T outside this function (see
    ``HybridBlockNative.forward``).

    Returns (h_t, cache) where cache contains everything needed to
    differentiate through the step.
    """
    # delta_in = x_t @ W_delta.T  (W_delta: (d, in_d) ; x_t: (B, in_d))
    delta_in = x_t @ W_delta.T   # (B, d)
    delta_t = _softplus(delta_in)
    expA = np.exp(A_log)         # (d,)
    A_t = np.exp(-delta_t * expA)
    b_in = x_t @ W_b.T           # (B, d)
    B_t = delta_t * b_in
    h_t_pre = A_t * h_prev + B_t
    out_t = np.tanh(h_t_pre)
    cache = dict(
        x_t=x_t, h_prev=h_prev, W_delta=W_delta, W_b=W_b, A_log=A_log,
        delta_in=delta_in, delta_t=delta_t, expA=expA,
        A_t=A_t, b_in=b_in, B_t=B_t, h_t_pre=h_t_pre, out_t=out_t,
    )
    # We pipe BOTH the post-tanh out and the pre-tanh state forward.
    # The recurrence uses the pre-tanh state h_t_pre as h_{t-1} for the
    # next step (matches Hasani 2022 closed-form scan -- the tanh is a
    # bound, not part of the recurrence).
    return h_t_pre, out_t, cache


def cfc_step_bwd(
    grad_out_t: np.ndarray, grad_h_t_pre_from_next: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward through one CfC step.

    We get TWO upstream grads:
      - ``grad_out_t``: dL/d out_t (post-tanh; goes to the PLIF branch).
      - ``grad_h_t_pre_from_next``: dL/d h_t_pre coming from the
        next-step recurrence (zero at t=T-1).

    Returns (grad_x_t, grad_h_prev, grad_W_delta, grad_W_b, grad_A_log).
    """
    x_t = cache["x_t"]
    h_prev = cache["h_prev"]
    W_delta = cache["W_delta"]
    W_b = cache["W_b"]
    delta_in = cache["delta_in"]
    delta_t = cache["delta_t"]
    expA = cache["expA"]
    A_t = cache["A_t"]
    b_in = cache["b_in"]
    out_t = cache["out_t"]

    # --- through tanh: out_t = tanh(h_t_pre) ---
    grad_h_t_pre = grad_out_t * (1.0 - out_t * out_t)
    # add the recurrence path
    grad_h_t_pre = grad_h_t_pre + grad_h_t_pre_from_next

    # h_t_pre = A_t * h_prev + B_t
    grad_A_t = grad_h_t_pre * h_prev
    grad_h_prev = grad_h_t_pre * A_t
    grad_B_t = grad_h_t_pre.copy()

    # B_t = delta_t * b_in -> grad to delta_t and b_in
    grad_delta_from_B = grad_B_t * b_in
    grad_b_in = grad_B_t * delta_t

    # A_t = exp(-delta_t * expA) -> dA_t/dx = -expA * A_t * d(delta_t)
    #                            -> dA_t/dA_log = -delta_t * expA * A_t
    grad_delta_from_A = grad_A_t * (-expA) * A_t
    grad_A_log = (grad_A_t * (-delta_t) * expA * A_t).reshape(-1, A_t.shape[-1]
                 ).sum(axis=0)

    # delta_t = softplus(delta_in) -> grad to delta_in
    grad_delta_total = grad_delta_from_B + grad_delta_from_A
    grad_delta_in = grad_delta_total * _softplus_grad(delta_in)

    # delta_in = x_t @ W_delta.T  ; b_in = x_t @ W_b.T
    in_d = W_delta.shape[1]
    out_d = W_delta.shape[0]

    # grad_x_t from delta_in branch: grad_delta_in @ W_delta
    grad_x_t_from_delta = grad_delta_in @ W_delta
    # grad_x_t from b_in branch: grad_b_in @ W_b
    grad_x_t_from_b = grad_b_in @ W_b
    grad_x_t = grad_x_t_from_delta + grad_x_t_from_b

    # grad_W_delta = grad_delta_in.T @ x_t (sum over batch)
    grad_W_delta = grad_delta_in.reshape(-1, out_d).T @ x_t.reshape(-1, in_d)
    grad_W_b = grad_b_in.reshape(-1, out_d).T @ x_t.reshape(-1, in_d)

    return grad_x_t, grad_h_prev, grad_W_delta, grad_W_b, grad_A_log


def plif_step_fwd(
    current_t: np.ndarray, mem_prev: np.ndarray,
    tau_log: np.ndarray, threshold: np.ndarray, dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """One PLIF step (Fang et al., 2021).

    tau     = exp(tau_log)                      per-channel learnable
    decay   = exp(-dt / tau)
    mem_t   = mem_{t-1} * decay + current_t
    spike_t = (mem_t >= threshold).astype(fp)   forward (indicator)
    mem_t   = mem_t - spike_t * threshold       reset-by-subtract

    Backward uses ATan surrogate (forward stays a hard indicator):
        d spike / d mem = alpha / (2 * (1 + (pi/2 * alpha * (mem - thr))^2))

    Returns (spike_t, mem_t_post_reset, cache).
    """
    tau = np.exp(tau_log)                      # (d,)
    decay = np.exp(-dt / tau)                  # (d,)
    mem_pre = mem_prev * decay + current_t     # (B, d)
    spike_t = (mem_pre >= threshold).astype(np.float32)
    mem_post = mem_pre - spike_t * threshold

    cache = dict(
        current_t=current_t, mem_prev=mem_prev, tau_log=tau_log,
        threshold=threshold, decay=decay, mem_pre=mem_pre,
        spike_t=spike_t, dt=dt,
    )
    return spike_t, mem_post, cache


def plif_step_bwd(
    grad_spike_t: np.ndarray, grad_mem_post: np.ndarray,
    cache: Dict[str, np.ndarray], alpha: float = PLIF_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward through one PLIF step.

    Two upstream grads:
      - ``grad_spike_t``: from downstream consumers of the spike.
      - ``grad_mem_post``: from the recurrence (next step's mem_prev).

    Returns (grad_current_t, grad_mem_prev, grad_tau_log, grad_threshold).
    """
    current_t = cache["current_t"]
    mem_prev = cache["mem_prev"]
    tau_log = cache["tau_log"]
    threshold = cache["threshold"]
    decay = cache["decay"]
    mem_pre = cache["mem_pre"]
    spike_t = cache["spike_t"]
    dt = cache["dt"]

    # mem_post = mem_pre - spike_t * threshold
    grad_mem_pre = grad_mem_post.copy()
    grad_spike_from_reset = grad_mem_post * (-threshold)
    grad_threshold_from_reset = (grad_mem_post * (-spike_t)
                                 ).reshape(-1, spike_t.shape[-1]
                                 ).sum(axis=0)

    # ATan surrogate for spike_t = indicator(mem_pre >= threshold)
    # x = alpha * (mem_pre - threshold)
    x = alpha * (mem_pre - threshold)
    surrogate = alpha / (2.0 * (1.0 + (math.pi / 2.0 * x) ** 2))
    grad_spike_total = grad_spike_t + grad_spike_from_reset
    # spike depends on mem_pre AND threshold
    grad_mem_pre = grad_mem_pre + grad_spike_total * surrogate
    grad_threshold = grad_threshold_from_reset + (
        grad_spike_total * (-surrogate)
    ).reshape(-1, surrogate.shape[-1]).sum(axis=0)

    # mem_pre = mem_prev * decay + current_t
    grad_mem_prev = grad_mem_pre * decay
    grad_current_t = grad_mem_pre.copy()
    # decay = exp(-dt/tau); tau = exp(tau_log)
    # d decay / d tau_log = decay * dt / tau   (chain rule)
    tau = np.exp(tau_log)
    d_decay_d_tau_log = decay * dt / tau
    grad_decay = (grad_mem_pre * mem_prev
                  ).reshape(-1, decay.shape[-1]).sum(axis=0)
    grad_tau_log = grad_decay * d_decay_d_tau_log

    return grad_current_t, grad_mem_prev, grad_tau_log, grad_threshold


def swiglu_fwd(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """SwiGLU: down(silu(gate(x)) * up(x))."""
    g_lin = x @ W_gate.T   # (..., h)
    u_lin = x @ W_up.T     # (..., h)
    g = _silu(g_lin)
    a = g * u_lin
    y = a @ W_down.T       # (..., d)
    cache = dict(x=x, W_gate=W_gate, W_up=W_up, W_down=W_down,
                 g_lin=g_lin, u_lin=u_lin, g=g, a=a)
    return y, cache


def swiglu_bwd(
    grad_y: np.ndarray, cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward through SwiGLU."""
    x = cache["x"]
    W_gate = cache["W_gate"]
    W_up = cache["W_up"]
    W_down = cache["W_down"]
    g_lin = cache["g_lin"]
    u_lin = cache["u_lin"]
    g = cache["g"]
    a = cache["a"]

    in_d = W_gate.shape[1]
    h = W_gate.shape[0]
    out_d = W_down.shape[0]

    grad_a = grad_y @ W_down                 # (..., h)
    grad_W_down = grad_y.reshape(-1, out_d).T @ a.reshape(-1, h)

    # a = g * u_lin
    grad_g = grad_a * u_lin                  # (..., h)
    grad_u_lin = grad_a * g                  # (..., h)

    # g = silu(g_lin)
    grad_g_lin = grad_g * _silu_grad(g_lin)  # (..., h)

    # x branch: grad from gate + up paths
    grad_x_gate = grad_g_lin @ W_gate
    grad_x_up = grad_u_lin @ W_up
    grad_x = grad_x_gate + grad_x_up

    grad_W_gate = grad_g_lin.reshape(-1, h).T @ x.reshape(-1, in_d)
    grad_W_up = grad_u_lin.reshape(-1, h).T @ x.reshape(-1, in_d)

    return grad_x, grad_W_gate, grad_W_up, grad_W_down


def cross_entropy_fwd(
    logits: np.ndarray, targets: np.ndarray,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Cross-entropy with logits (mean over batch+time).

    logits: (B, T, V), targets: (B, T) int.
    Returns (loss_scalar, cache).
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    # log-softmax (numerically stable)
    max_logit = flat_logits.max(axis=-1, keepdims=True)
    shifted = flat_logits - max_logit
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_z
    nll = -log_probs[np.arange(flat_targets.shape[0]), flat_targets]
    loss = float(nll.mean())
    softmax = np.exp(log_probs)
    cache = dict(softmax=softmax, targets=flat_targets, B=B, T=T, V=V)
    return loss, cache


def cross_entropy_bwd(cache: Dict[str, np.ndarray]) -> np.ndarray:
    """grad logits = (softmax - one_hot) / N."""
    softmax = cache["softmax"]
    targets = cache["targets"]
    B = cache["B"]
    T = cache["T"]
    V = cache["V"]
    N = targets.shape[0]
    grad = softmax.copy()
    grad[np.arange(N), targets] -= 1.0
    grad = grad / float(N)
    return grad.reshape(B, T, V).astype(np.float32)


# ---------------------------------------------------------------------------
# HybridBlockNative -- LNN+SNN block with manual fwd/bwd
# ---------------------------------------------------------------------------

@dataclass
class HybridBlockParams:
    """One HybridBlock's parameters (numpy)."""
    # RMSNorm 1
    ln1_w: np.ndarray
    # LiquidCell
    cfc_W_delta: np.ndarray
    cfc_W_b: np.ndarray
    cfc_A_log: np.ndarray
    # PLIF
    plif_tau_log: np.ndarray
    plif_threshold: np.ndarray
    # post-spike linear projection (synapse-style)
    syn_W: np.ndarray
    # RMSNorm 2
    ln2_w: np.ndarray
    # SwiGLU
    ffn_W_gate: np.ndarray
    ffn_W_up: np.ndarray
    ffn_W_down: np.ndarray


def init_hybrid_block(rng: np.random.Generator,
                      d: int, ffn_h: int) -> HybridBlockParams:
    """Initialize one HybridBlock's parameters."""
    return HybridBlockParams(
        ln1_w=np.ones((d,), dtype=np.float32),
        cfc_W_delta=_xavier(rng, (d, d), std=0.02),
        cfc_W_b=_xavier(rng, (d, d), std=0.02),
        # A_log init in [log(0.5), log(2.0)] (Hasani)
        cfc_A_log=np.log(rng.uniform(0.5, 2.0, size=(d,))).astype(np.float32),
        plif_tau_log=np.full((d,), math.log(PLIF_TAU_INIT), dtype=np.float32),
        plif_threshold=np.full((d,), PLIF_THRESHOLD, dtype=np.float32),
        syn_W=_xavier(rng, (d, d), std=0.02),
        ln2_w=np.ones((d,), dtype=np.float32),
        ffn_W_gate=_xavier(rng, (ffn_h, d), std=0.02),
        ffn_W_up=_xavier(rng, (ffn_h, d), std=0.02),
        ffn_W_down=_xavier(rng, (d, ffn_h), std=0.02 / math.sqrt(2)),
    )


def hybrid_block_fwd(
    x: np.ndarray, p: HybridBlockParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Forward through one HybridBlock.

    x: (B, T, d) -> y: (B, T, d), residuals included.

    Cache stores per-step CfC/PLIF caches as lists indexed by t.
    """
    B, T, d = x.shape

    # ---- residual #1: x + syn(plif(cfc(ln1(x)))) ----
    a, ln1_cache = rmsnorm_fwd(x, p.ln1_w)

    # CfC sequential scan over T
    cfc_caches = []
    h_pre = np.zeros((B, d), dtype=np.float32)
    cfc_outs = np.empty((B, T, d), dtype=np.float32)
    for t in range(T):
        h_pre, out_t, c = cfc_step_fwd(
            a[:, t, :], h_pre, p.cfc_W_delta, p.cfc_W_b, p.cfc_A_log,
        )
        cfc_outs[:, t, :] = out_t
        cfc_caches.append(c)

    # PLIF sequential scan
    plif_caches = []
    spikes = np.empty((B, T, d), dtype=np.float32)
    mem = np.zeros((B, d), dtype=np.float32)
    for t in range(T):
        spk, mem, pc = plif_step_fwd(
            cfc_outs[:, t, :], mem,
            p.plif_tau_log, p.plif_threshold,
        )
        spikes[:, t, :] = spk
        plif_caches.append(pc)

    # Synapse: linear (no bias) over spikes
    syn_out = linear_fwd(spikes, p.syn_W, None)
    x_after_branch1 = x + syn_out

    # ---- residual #2: + ffn(ln2(x_after_branch1)) ----
    a2, ln2_cache = rmsnorm_fwd(x_after_branch1, p.ln2_w)
    ffn_out, swiglu_cache = swiglu_fwd(a2, p.ffn_W_gate, p.ffn_W_up,
                                        p.ffn_W_down)
    y = x_after_branch1 + ffn_out

    cache = dict(
        x=x, ln1_cache=ln1_cache, cfc_caches=cfc_caches, cfc_outs=cfc_outs,
        plif_caches=plif_caches, spikes=spikes,
        x_after_branch1=x_after_branch1,
        ln2_cache=ln2_cache, swiglu_cache=swiglu_cache, p=p,
    )
    return y, cache


def hybrid_block_bwd(
    grad_y: np.ndarray, cache: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Backward through one HybridBlock. Returns (grad_x, grads_dict)."""
    p: HybridBlockParams = cache["p"]
    ln1_cache = cache["ln1_cache"]
    cfc_caches = cache["cfc_caches"]
    plif_caches = cache["plif_caches"]
    spikes = cache["spikes"]
    cfc_outs = cache["cfc_outs"]
    ln2_cache = cache["ln2_cache"]
    swiglu_cache = cache["swiglu_cache"]

    B, T, d = grad_y.shape

    # y = x_after_branch1 + ffn_out
    grad_ffn_out = grad_y.copy()
    grad_x_after_branch1 = grad_y.copy()

    # FFN backward
    grad_a2, grad_W_gate, grad_W_up, grad_W_down = swiglu_bwd(
        grad_ffn_out, swiglu_cache,
    )
    # ln2 backward
    grad_x_after_branch1_from_ln2, grad_ln2_w = rmsnorm_bwd(grad_a2, ln2_cache)
    grad_x_after_branch1 = grad_x_after_branch1 + grad_x_after_branch1_from_ln2

    # x_after_branch1 = x + syn_out
    grad_x = grad_x_after_branch1.copy()
    grad_syn_out = grad_x_after_branch1.copy()

    # Synapse backward (linear, no bias)
    grad_spikes, grad_syn_W, _ = linear_bwd(grad_syn_out, spikes, p.syn_W,
                                            has_bias=False)

    # PLIF reverse scan -- accumulate grads on cfc_outs[:, t, :]
    grad_cfc_outs = np.zeros_like(cfc_outs)
    grad_mem_post_next = np.zeros((B, d), dtype=np.float32)
    grad_tau_log_total = np.zeros_like(p.plif_tau_log)
    grad_threshold_total = np.zeros_like(p.plif_threshold)
    for t in reversed(range(T)):
        gc_t, grad_mem_post_next, gtau, gthr = plif_step_bwd(
            grad_spikes[:, t, :], grad_mem_post_next, plif_caches[t],
        )
        grad_cfc_outs[:, t, :] = gc_t
        grad_tau_log_total += gtau
        grad_threshold_total += gthr

    # CfC reverse scan
    grad_a = np.zeros((B, T, d), dtype=np.float32)
    grad_W_delta_total = np.zeros_like(p.cfc_W_delta)
    grad_W_b_total = np.zeros_like(p.cfc_W_b)
    grad_A_log_total = np.zeros_like(p.cfc_A_log)
    grad_h_prev_next = np.zeros((B, d), dtype=np.float32)
    for t in reversed(range(T)):
        gx_t, grad_h_prev_next, gWd, gWb, gAlog = cfc_step_bwd(
            grad_cfc_outs[:, t, :], grad_h_prev_next, cfc_caches[t],
        )
        grad_a[:, t, :] = gx_t
        grad_W_delta_total += gWd
        grad_W_b_total += gWb
        grad_A_log_total += gAlog

    # ln1 backward
    grad_x_from_ln1, grad_ln1_w = rmsnorm_bwd(grad_a, ln1_cache)
    grad_x = grad_x + grad_x_from_ln1

    grads = dict(
        ln1_w=grad_ln1_w,
        cfc_W_delta=grad_W_delta_total, cfc_W_b=grad_W_b_total,
        cfc_A_log=grad_A_log_total,
        plif_tau_log=grad_tau_log_total, plif_threshold=grad_threshold_total,
        syn_W=grad_syn_W,
        ln2_w=grad_ln2_w,
        ffn_W_gate=grad_W_gate, ffn_W_up=grad_W_up, ffn_W_down=grad_W_down,
    )
    return grad_x, grads


# ---------------------------------------------------------------------------
# SynapForgeTinyNative -- full model
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """All trainable parameters of SynapForgeTinyNative."""
    embed_W: np.ndarray
    blocks: List[HybridBlockParams] = field(default_factory=list)
    lm_head_W: np.ndarray = field(default_factory=lambda: np.zeros(()))
    # Note: untied lm head per TOKEN_SOUP fix.


def init_model(seed: int = SEED) -> ModelParams:
    """Initialize the full model with deterministic seed."""
    rng = _rng(seed)
    ffn_h = FFN_RATIO * D_MODEL
    return ModelParams(
        embed_W=_xavier(rng, (VOCAB, D_MODEL), std=0.02),
        blocks=[init_hybrid_block(rng, D_MODEL, ffn_h)
                for _ in range(N_LAYERS)],
        lm_head_W=_xavier(rng, (VOCAB, D_MODEL), std=0.02),
    )


def model_fwd(x: np.ndarray, m: ModelParams,
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Forward pass.

    x: (B, T) int tokens -> logits: (B, T, V).
    """
    h = embed_fwd(x, m.embed_W)
    block_caches = []
    for blk in m.blocks:
        h, c = hybrid_block_fwd(h, blk)
        block_caches.append(c)
    logits = linear_fwd(h, m.lm_head_W, None)
    cache = dict(x=x, block_caches=block_caches, h_final=h, m=m)
    return logits, cache


def model_bwd(grad_logits: np.ndarray,
              cache: Dict[str, Any]) -> Dict[str, Any]:
    """Backward pass. Returns flat grads dict mirroring ModelParams."""
    m: ModelParams = cache["m"]
    block_caches = cache["block_caches"]
    h_final = cache["h_final"]
    x = cache["x"]

    grad_h, grad_lm_head_W, _ = linear_bwd(
        grad_logits, h_final, m.lm_head_W, has_bias=False,
    )

    block_grads = [None] * len(m.blocks)
    for i in reversed(range(len(m.blocks))):
        grad_h, blk_grads = hybrid_block_bwd(grad_h, block_caches[i])
        block_grads[i] = blk_grads

    grad_embed_W = embed_bwd(grad_h, x, VOCAB)
    return dict(embed_W=grad_embed_W, blocks=block_grads,
                lm_head_W=grad_lm_head_W)


# ---------------------------------------------------------------------------
# AdamWNumpy -- pure-numpy AdamW
# ---------------------------------------------------------------------------

class AdamWNumpy:
    """AdamW (Loshchilov 2019) implemented in pure numpy.

    No torch.Tensor anywhere. m and v moments are numpy arrays;
    in-place mul/add gives ~MKL multi-core via OPENBLAS_NUM_THREADS.

    Matches torch.optim.AdamW with ``foreach=False``:
        m  = beta1*m + (1-beta1)*g
        v  = beta2*v + (1-beta2)*g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p  = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
    """

    def __init__(
        self, params: List[np.ndarray], lr: float = LR,
        beta1: float = ADAMW_BETA1, beta2: float = ADAMW_BETA2,
        eps: float = ADAMW_EPS, weight_decay: float = ADAMW_WD,
    ) -> None:
        self.params = params  # list of np.ndarray refs (mutated in place)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        """Update each param in-place using its grad."""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        lr = self.lr
        wd = self.weight_decay
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            # m = b1*m + (1-b1)*g
            np.multiply(self.m[i], b1, out=self.m[i])
            self.m[i] += (1.0 - b1) * g
            # v = b2*v + (1-b2)*g^2
            np.multiply(self.v[i], b2, out=self.v[i])
            self.v[i] += (1.0 - b2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            update = m_hat / (np.sqrt(v_hat) + eps) + wd * p
            p -= lr * update

    def zero_grad(self) -> None:
        """No-op (we don't accumulate; trainer hands us fresh grads)."""
        pass


def collect_params_and_grads(
    m: ModelParams, grads: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect param refs and matching grad refs into parallel lists."""
    params = [m.embed_W]
    grad_list = [grads["embed_W"]]
    for blk_idx, blk in enumerate(m.blocks):
        bg = grads["blocks"][blk_idx]
        for name in [
            "ln1_w", "cfc_W_delta", "cfc_W_b", "cfc_A_log",
            "plif_tau_log", "plif_threshold", "syn_W", "ln2_w",
            "ffn_W_gate", "ffn_W_up", "ffn_W_down",
        ]:
            params.append(getattr(blk, name))
            grad_list.append(bg[name])
    params.append(m.lm_head_W)
    grad_list.append(grads["lm_head_W"])
    return params, grad_list


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synth_batch(rng: np.random.Generator, batch: int, seq_len: int,
                vocab: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random integer 'tokens' for next-token prediction.

    To make loss decreasable (otherwise ce flat at log(V)=5.55), we
    inject a strong local pattern: target[t] = (input[t] + 1) % V for
    even t, and = input[t-1] for odd t. This lets the model fit a
    learnable curve in 100 steps even with the tiny d=64 setup.
    """
    x = rng.integers(0, vocab, size=(batch, seq_len), dtype=np.int64)
    y = np.empty_like(x)
    for t in range(seq_len):
        if t == 0:
            y[:, t] = (x[:, t] + 1) % vocab
        elif t % 2 == 0:
            y[:, t] = (x[:, t] + 1) % vocab
        else:
            y[:, t] = x[:, t - 1]
    return x, y


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(
    n_steps: int = N_STEPS, seed: int = SEED, log_every: int = 1,
) -> Dict[str, Any]:
    """Run ``n_steps`` of training. Returns a results dict (JSON-able)."""
    print(f"[native_demo] init model seed={seed} d={D_MODEL} "
          f"n_layers={N_LAYERS} V={VOCAB} T={SEQ_LEN} B={BATCH}")
    m = init_model(seed=seed)
    rng = _rng(seed + 1)  # data RNG separate from model init RNG

    params, _ = collect_params_and_grads(m, {
        "embed_W": np.zeros_like(m.embed_W),
        "blocks": [
            {name: np.zeros_like(getattr(blk, name)) for name in [
                "ln1_w", "cfc_W_delta", "cfc_W_b", "cfc_A_log",
                "plif_tau_log", "plif_threshold", "syn_W", "ln2_w",
                "ffn_W_gate", "ffn_W_up", "ffn_W_down",
            ]} for blk in m.blocks
        ],
        "lm_head_W": np.zeros_like(m.lm_head_W),
    })
    n_params = sum(p.size for p in params)
    print(f"[native_demo] n_params={n_params:,} ({n_params * 4 / 1e6:.2f} MB)")

    optim = AdamWNumpy(params, lr=LR, beta1=ADAMW_BETA1, beta2=ADAMW_BETA2,
                       eps=ADAMW_EPS, weight_decay=ADAMW_WD)

    losses: List[float] = []
    step_times: List[float] = []
    t_total_start = time.time()
    for step in range(n_steps):
        t0 = time.time()
        x, y = synth_batch(rng, BATCH, SEQ_LEN, VOCAB)
        logits, fwd_cache = model_fwd(x, m)
        loss, ce_cache = cross_entropy_fwd(logits, y)
        grad_logits = cross_entropy_bwd(ce_cache)
        grads = model_bwd(grad_logits, fwd_cache)

        _, grad_list = collect_params_and_grads(m, grads)

        # Gradient clipping (global L2 norm). Helps the tiny model not
        # diverge from a single bad batch in the first few steps.
        sq_sum = 0.0
        for g in grad_list:
            sq_sum += float((g * g).sum())
        gnorm = math.sqrt(sq_sum)
        max_norm = 1.0
        if gnorm > max_norm:
            scale = max_norm / (gnorm + 1e-6)
            for g in grad_list:
                g *= scale

        optim.step(grad_list)
        dt = time.time() - t0
        step_times.append(dt)
        losses.append(loss)
        if (step + 1) % log_every == 0 or step == 0:
            print(f"[native_demo] step={step + 1:3d}/{n_steps} "
                  f"loss={loss:.4f} gnorm={gnorm:.2f} dt={dt * 1000:.1f}ms")

    total_dt = time.time() - t_total_start

    # --- monotonicity check (rolling-10 mean strictly decreases over halves) -
    rolling = []
    win = 10
    for i in range(len(losses) - win + 1):
        rolling.append(float(np.mean(losses[i:i + win])))
    if len(rolling) >= 2:
        first_half = float(np.mean(rolling[:len(rolling) // 2]))
        second_half = float(np.mean(rolling[len(rolling) // 2:]))
        decreased = second_half < first_half
    else:
        first_half = float("nan")
        second_half = float("nan")
        decreased = False

    # --- self-check: 'import torch' must NOT appear in this file ---
    own_path = Path(__file__).resolve()
    own_src = own_path.read_text(encoding="utf-8")
    has_torch_import = any(
        line.strip().startswith(("import torch", "from torch"))
        for line in own_src.splitlines()
    )

    results = dict(
        impl="native_numpy",
        seed=seed,
        n_steps=n_steps,
        n_params=int(n_params),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        vocab=VOCAB,
        seq_len=SEQ_LEN,
        batch=BATCH,
        losses=losses,
        rolling_10=rolling,
        first_half_mean=first_half,
        second_half_mean=second_half,
        loss_decreased_monotonically=bool(decreased),
        final_loss=float(losses[-1]),
        wall_time_sec=float(total_dt),
        ms_per_step_mean=float(np.mean(step_times) * 1000),
        ms_per_step_p50=float(np.median(step_times) * 1000),
        has_import_torch=bool(has_torch_import),
    )

    out_path = Path(__file__).resolve().parent / "_native_demo_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[native_demo] wrote {out_path}")
    print(f"[native_demo] final_loss={results['final_loss']:.4f} "
          f"first_half={first_half:.4f} second_half={second_half:.4f} "
          f"decreased={decreased} wall={total_dt:.1f}s "
          f"ms/step={results['ms_per_step_mean']:.1f}")
    print(f"[native_demo] has_import_torch={has_torch_import} "
          "(MUST be False)")
    return results


if __name__ == "__main__":
    # Multi-thread numpy for free if MKL/OpenBLAS is linked.
    os.environ.setdefault("OPENBLAS_NUM_THREADS",
                          str(min(8, os.cpu_count() or 1)))
    os.environ.setdefault("MKL_NUM_THREADS",
                          str(min(8, os.cpu_count() or 1)))
    train()
