"""Cross-entropy loss (with built-in softmax) -- closed-form VJP.

Forward
-------
For logits ``z`` of shape (N, V) and targets ``t`` of shape (N,) with
class indices in [0, V):

    softmax(z)_v = exp(z_v - max_v(z)) / sum_v exp(z_v - max_v(z))   # log-sum-exp stable
    loss_n       = -log(softmax(z_n)[t_n])
    loss         = mean_n(loss_n)   if reduction == "mean"
                 = sum_n(loss_n)    if reduction == "sum"
                 = loss_n           if reduction == "none"

Optional ``ignore_index`` masks targets equal to that value (used by
HF / pytorch convention; common values are -100 or pad-id).

Backward
--------
The famous closed form: when loss = -log(softmax_t), the gradient of
loss with respect to logits is simply

    grad_z = (softmax - one_hot(target)) / scale

where ``scale`` is the count of *non-ignored* tokens for ``mean``,
``1`` for ``sum``, and ``1`` element-wise for ``none``. This bypasses
the chain through softmax + log entirely -- one of the cleanest VJPs
in the catalogue.

References
----------
* Goodfellow et al., "Deep Learning" (2016), section 4.3.2.
* PyTorch ``F.cross_entropy`` -- functionally identical.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _softmax_logsumexp(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce_fwd(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: Optional[int] = -100,
    reduction: str = "mean",
) -> Tuple[np.ndarray, dict]:
    """Cross-entropy forward with built-in softmax.

    Parameters
    ----------
    logits : (..., V) raw class logits. We flatten leading dims internally.
    targets : (...,) integer class indices.
    ignore_index : int or None. Targets equal to ``ignore_index`` contribute
        0 to the loss and 0 to the gradient. Default -100 matches PyTorch.
    reduction : "mean" | "sum" | "none".

    Returns
    -------
    loss : scalar (mean / sum) or array (none).
    saved : dict containing softmax, targets, mask, scale -- for the bwd.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"ce_fwd: reduction must be mean|sum|none, got {reduction}")
    if logits.ndim < 1:
        raise ValueError("ce_fwd: logits must have >= 1 dim")
    V = logits.shape[-1]
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"ce_fwd: targets shape {targets.shape} != logits leading dims {logits.shape[:-1]}"
        )

    flat_logits = logits.reshape(-1, V)              # (N, V)
    flat_targets = targets.reshape(-1).astype(np.int64)  # (N,)
    N = flat_logits.shape[0]

    if ignore_index is not None:
        mask = (flat_targets != ignore_index).astype(np.float32)  # (N,)
    else:
        mask = np.ones(N, dtype=np.float32)

    # Substitute a safe class index (0) for ignored entries so gather doesn't
    # index out-of-bounds. The mask zeros their loss anyway.
    safe_targets = np.where(mask > 0, flat_targets, 0)

    softmax = _softmax_logsumexp(flat_logits.astype(np.float32))   # (N, V)
    # gather softmax[n, target[n]]
    log_prob = np.log(np.clip(softmax[np.arange(N), safe_targets], 1e-30, 1.0))
    per_token = -log_prob * mask                                   # (N,)

    if reduction == "mean":
        denom = max(mask.sum(), 1.0)
        loss = per_token.sum() / denom
    elif reduction == "sum":
        denom = 1.0
        loss = per_token.sum()
    else:  # "none"
        denom = None  # type: ignore[assignment]
        loss = per_token.reshape(targets.shape)

    saved = dict(
        softmax=softmax,
        targets=flat_targets,
        safe_targets=safe_targets,
        mask=mask,
        denom=denom,
        reduction=reduction,
        logits_shape=logits.shape,
    )
    return loss, saved


def ce_bwd(
    grad_loss: np.ndarray,
    saved: dict,
) -> np.ndarray:
    """Cross-entropy backward (closed: softmax - one_hot, scaled).

    Parameters
    ----------
    grad_loss : scalar for "mean"/"sum", array of shape ``targets`` for "none".
    saved : dict from ``ce_fwd``.

    Returns
    -------
    grad_logits : same shape as the original logits passed to ``ce_fwd``.
    """
    softmax = saved["softmax"]
    safe_targets = saved["safe_targets"]
    mask = saved["mask"]
    denom = saved["denom"]
    reduction = saved["reduction"]
    logits_shape = saved["logits_shape"]
    N, V = softmax.shape

    # diff = softmax - one_hot(target)
    diff = softmax.copy()
    diff[np.arange(N), safe_targets] -= 1.0
    # zero out ignored rows (they contributed nothing to loss)
    diff = diff * mask[:, None]

    if reduction == "mean":
        scale = float(grad_loss) / float(denom)
        grad_flat = diff * scale
    elif reduction == "sum":
        scale = float(grad_loss)
        grad_flat = diff * scale
    else:  # "none"
        # grad_loss has shape of targets. Broadcast to (N, V) then mul.
        gl_flat = grad_loss.reshape(-1)               # (N,)
        grad_flat = diff * gl_flat[:, None]

    return grad_flat.reshape(logits_shape)
