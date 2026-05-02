"""SEW (Spike-Element-Wise) shortcut residual -- closed-form VJP.

Forward
-------
Per Fang et al. (NeurIPS 2021), the additive SEW residual is:

    y = spike + h_dense

where ``spike`` is the (binary 0/1) spike output of a PLIF cell and
``h_dense`` is the parallel dense (CfC) path output. SEW lets the
gradient flow through the dense path even when spikes are sparse,
which fixes the bootstrap "PLIF-dead" pathology in the codebase.

Backward
--------
Both branches are identity in the upstream gradient:

    grad_spike   = grad_y
    grad_h_dense = grad_y

This op is trivial but lives in its own module so the catalogue is
complete and tests cover the residual path explicitly.

References
----------
* Fang et al., "Deep Residual Learning in Spiking Neural Networks"
  (NeurIPS 2021), §3 "SEW residual block".
* synapforge.cells.plif -- the bootstrap discussion in the docstring
  references SEW for the dead-PLIF rescue.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sew_fwd(
    spike: np.ndarray,
    h_dense: np.ndarray,
) -> np.ndarray:
    """SEW additive residual forward.

    Parameters
    ----------
    spike : (..., d)
    h_dense : (..., d)

    Returns
    -------
    y : (..., d) -- broadcastable sum.
    """
    if spike.shape != h_dense.shape:
        raise ValueError(
            f"sew_fwd: spike {spike.shape} != h_dense {h_dense.shape}"
        )
    return spike + h_dense


def sew_bwd(
    grad_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """SEW additive residual backward.

    Parameters
    ----------
    grad_y : upstream gradient.

    Returns
    -------
    (grad_spike, grad_h_dense) -- both equal grad_y (identity split).
    Returned as separate arrays to avoid aliasing surprises in callers
    that mutate one of the gradients in place.
    """
    return grad_y.copy(), grad_y.copy()
