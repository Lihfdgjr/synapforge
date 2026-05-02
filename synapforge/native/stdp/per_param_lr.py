"""per_param_lr — match plasticity rate to layer's CfC tau.

CfC neurons in early layers have *slow* tau (long memory, integrate
many inputs); late layers have *fast* tau (rapid response). Plasticity
should match: fast layers should adapt fast, slow layers slow. We
scale the base STDP learning rate per-parameter:

    alpha_layer = alpha_base * (tau_layer / tau_avg)

This is the same heuristic used by the multiband-tau LIF tuner in
``synapforge.bio.multiband``. The wrapper here is intentionally
storage-agnostic: it produces a single Python float per parameter id,
not a tensor.

Hard constraint: NO ``import torch``. We accept and return raw types
(int IDs, float taus, dict mappings).
"""
from __future__ import annotations

from collections.abc import Mapping


def per_param_alpha(
    base_alpha: float,
    tau_per_param: Mapping[int, float],
    *,
    floor: float = 1e-6,
    ceil: float = 1.0,
) -> dict[int, float]:
    """Compute per-param STDP learning rate from per-layer tau.

    Parameters
    ----------
    base_alpha : float
        The single global STDP rate (eg. 0.02). All layer-specific
        scalars are computed as ``base_alpha * (tau_layer / tau_avg)``.
    tau_per_param : Mapping[int, float]
        ``{id(param): tau}`` — the CfC tau (in steps) of the layer
        owning ``param``. We compute the geometric average across all
        param IDs to set ``tau_avg``.
    floor, ceil : float
        Clip the resulting per-param alpha into ``[floor, ceil]`` to
        protect against pathological tau distributions.

    Returns
    -------
    dict[int, float]
        Mapping ``id(param) -> alpha_param``.

    Notes
    -----
    Geometric mean is preferred over arithmetic because tau spans an
    order of magnitude (4 step → 32 step is typical for the
    multiband-tau setup). With arithmetic mean the slowest layer
    pulls everyone toward zero.
    """
    if base_alpha <= 0.0:
        raise ValueError(f"base_alpha must be > 0; got {base_alpha}")
    if not tau_per_param:
        return {}
    if floor >= ceil:
        raise ValueError(f"floor {floor} must be < ceil {ceil}")
    # Geometric mean — equivalent to exp(mean(log(tau))). Robust to
    # tau ranging over an order of magnitude.
    log_sum = 0.0
    n = 0
    for tau in tau_per_param.values():
        if tau <= 0.0:
            raise ValueError(
                f"tau must be > 0 for all params; got {tau}"
            )
        # math.log; avoid the dependency-free path through ln series
        # (we are already using float arithmetic, so import math is fine).
        import math
        log_sum += math.log(tau)
        n += 1
    import math
    tau_avg = math.exp(log_sum / n) if n > 0 else 1.0
    out: dict[int, float] = {}
    for pid, tau in tau_per_param.items():
        scaled = base_alpha * (tau / tau_avg)
        if scaled < floor:
            scaled = floor
        elif scaled > ceil:
            scaled = ceil
        out[pid] = float(scaled)
    return out


def alpha_from_uniform_tau(base_alpha: float, n_params: int) -> dict[int, float]:
    """Return ``{0..n-1: base_alpha}`` — used when no tau metadata exists.

    This is the noop case: every parameter gets ``base_alpha``. Useful
    when the model does not expose per-layer tau (e.g. a pure SNN with
    LIF only, no CfC). Keeps the optimizer's per-param-LR codepath
    uniform.
    """
    if base_alpha <= 0.0:
        raise ValueError(f"base_alpha must be > 0; got {base_alpha}")
    if n_params <= 0:
        return {}
    return {i: float(base_alpha) for i in range(n_params)}


__all__ = ["per_param_alpha", "alpha_from_uniform_tau"]
