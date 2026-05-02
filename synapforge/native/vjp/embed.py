"""Token embedding lookup -- closed-form VJP.

Forward
-------
    y[b, t, :] = W[input_ids[b, t], :]
    shape:      W is (V, d), input_ids is (B, T) int, y is (B, T, d)

Backward
--------
For each (b, t), grad_y[b, t, :] flows into the row W[input_ids[b, t], :].
Multiple tokens with the same id accumulate (sparse scatter-add):

    grad_W[v, :] = sum_{b,t : input_ids[b,t] == v} grad_y[b, t, :]

Note: We keep the full (V, d) gradient matrix for simplicity. A truly
sparse representation (id -> rows) is a future optimisation; for fp32
training-loop scale (V <= 200k, d <= 4096) the dense scatter is fine.

References
----------
* Mikolov 2013, "Efficient Estimation of Word Representations in Vector
  Space", section 4.1 (lookup as one-hot * W).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def embed_fwd(
    input_ids: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Token embedding forward.

    Parameters
    ----------
    input_ids : (B, T) int -- token indices in [0, V).
    W : (V, d) float -- embedding table. ``W[v]`` is the row for token id v.

    Returns
    -------
    y : (B, T, d) float -- gathered embedding rows. dtype matches ``W``.
    """
    if input_ids.dtype.kind not in ("i", "u"):
        raise TypeError(
            f"embed_fwd: input_ids must be integer, got dtype {input_ids.dtype}"
        )
    if W.ndim != 2:
        raise ValueError(f"embed_fwd: W must be 2D (V, d), got {W.shape}")
    V = W.shape[0]
    if input_ids.size:
        max_id = int(input_ids.max())
        min_id = int(input_ids.min())
        if min_id < 0 or max_id >= V:
            raise IndexError(
                f"embed_fwd: token id range [{min_id}, {max_id}] out of bounds for V={V}"
            )
    return W[input_ids]


def embed_bwd(
    grad_y: np.ndarray,
    input_ids: np.ndarray,
    V: int,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Token embedding backward (sparse scatter-add over rows).

    Parameters
    ----------
    grad_y : (B, T, d) -- upstream gradient.
    input_ids : (B, T) int -- same indices used in forward.
    V : int -- vocabulary size (rows of W).
    dtype : optional numpy dtype for the output grad_W (defaults to grad_y.dtype).

    Returns
    -------
    grad_W : (V, d) -- gradient on the embedding table. Rows untouched
        by ``input_ids`` are zero.
    """
    if grad_y.ndim != 3:
        raise ValueError(
            f"embed_bwd: grad_y must be 3D (B, T, d), got {grad_y.shape}"
        )
    if input_ids.shape != grad_y.shape[:2]:
        raise ValueError(
            f"embed_bwd: input_ids shape {input_ids.shape} mismatch "
            f"grad_y batch dims {grad_y.shape[:2]}"
        )
    d = grad_y.shape[-1]
    out_dtype = dtype if dtype is not None else grad_y.dtype
    grad_W = np.zeros((V, d), dtype=out_dtype)
    flat_ids = input_ids.reshape(-1)
    flat_grad = grad_y.reshape(-1, d).astype(out_dtype, copy=False)
    # np.add.at performs unbuffered (correct) scatter-add for repeated indices.
    np.add.at(grad_W, flat_ids, flat_grad)
    return grad_W
