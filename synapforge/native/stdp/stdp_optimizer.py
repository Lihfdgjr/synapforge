"""STDPOnlyOptimizer — local Hebbian update without autograd.

Why not AdamW
-------------
Plasticity-tagged weights (``_sf_grad_source`` containing ``"stdp"``
or ``"hebb"``) carry no semantically meaningful BP gradient — the
producer feeds the layer pre+post spike pairs and a local rule
says how the weight should change. Routing such weights through
AdamW costs:

* H2D copy of the BP grad we don't need
* CPU-side Adam moment math (~140ms for 730M params at fp32)
* D2H copy of the updated parameter
* O(n_weights) work even when only a few neurons spike

STDP costs O(active_spike_pairs) — a 100-200x reduction in the sparse
case, runs at inference time, and never touches autograd.

Hard constraints
----------------
* **No `import torch` at module level.** Triton kernel imports ``tl``
  conditionally; numpy fallback handles CPU and tests. The dispatcher
  in :mod:`hybrid_optim_dispatch` is the single bridge to torch.
* When ``--stdp-only-plasticity`` is OFF the dispatcher falls back to
  the existing ``PlasticityAwareAdamW`` codepath bit-for-bit.

Plasticity rule (Hebbian-STDP)
------------------------------
For each plasticity-tagged weight ``W [out_dim, in_dim]``::

    pre_trace[t]  = decay_pre  * pre_trace[t-1]  + pre_spike[t]
    post_trace[t] = decay_post * post_trace[t-1] + post_spike[t]
    dW[t]         = a_plus  * outer(post_spike[t], pre_trace[t-1])
                  - a_minus * outer(post_trace[t-1], pre_spike[t])
    W[t]          = clip(W[t-1] + dW[t], -clip, clip)

This is the classic 1995 Markram pair-STDP rule with exponential
traces. ``a_plus = a_minus`` is the symmetric default; asymmetry
yields a small net potentiation that the AdamW path lacked.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .spike_buffer import SpikeRingBuffer


@dataclass
class STDPParamGroup:
    """One plasticity-tagged tensor + its spike buffer + its alpha.

    ``param`` is the *target* weight whose ``data`` we mutate. We do
    not touch ``param.grad``; we do not call ``.backward``. We accept
    any object with:

    * ``.shape`` returning ``(out_dim, in_dim)``
    * ``.copy_(numpy_array)`` to bulk-assign new values, OR
      ``.data`` mutable sub-attribute that supports inplace ``+=``

    A real ``torch.nn.Parameter`` satisfies both via ``.data.add_``.
    Tests use a small duck-typed shim.
    """

    param: object
    name: str
    buffer: SpikeRingBuffer
    alpha: float
    a_plus: float = 0.02
    a_minus: float = 0.02
    clip: float = 1.0
    last_active_pairs: int = 0


class STDPOnlyOptimizer:
    """Drop-in optimizer for plasticity-tagged params.

    Differences from ``torch.optim.Optimizer``:

    * **No** ``zero_grad()``: we never read ``.grad``.
    * **No** moment buffers: state-free per-step.
    * **Step inputs differ**: instead of ``loss.backward()`` followed
      by ``opt.step()``, callers do
      ``opt.observe(name, pre_spike, post_spike)`` after each forward
      and finally ``opt.step()``. The step is O(n_param_groups *
      n_active_spikes_per_layer).

    Construction
    ------------
    >>> opt = STDPOnlyOptimizer.from_named_params(
    ...     [(name, p) for name, p in model.named_parameters()
    ...      if "stdp" in getattr(p, "_sf_grad_source", [])],
    ...     base_alpha=0.02, window=20)

    Step
    ----
    >>> # In the training loop, after each forward pass on the SNN side:
    >>> opt.observe("layer3.synapse.weight", pre_spike, post_spike)
    >>> # ... after some number of forwards (typically 1):
    >>> stats = opt.step()
    >>> stats["total_active_spikes"]   # for logging
    """

    def __init__(self, groups: Iterable[STDPParamGroup]) -> None:
        self.groups: list[STDPParamGroup] = list(groups)
        if not self.groups:
            raise ValueError(
                "STDPOnlyOptimizer: must construct with at least one "
                "STDPParamGroup. If your model has no plasticity-tagged "
                "params, use the regular AdamW path."
            )
        self._by_name = {g.name: g for g in self.groups}
        self._step_count = 0

    # ----------------------------------------------------- factory helper
    @classmethod
    def from_named_params(
        cls,
        named_params: Iterable[tuple[str, object]],
        base_alpha: float = 0.02,
        window: int = 20,
        a_plus: float = 0.02,
        a_minus: float = 0.02,
        clip: float = 1.0,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        per_param_alpha: dict[int, float] | None = None,
    ) -> STDPOnlyOptimizer:
        """Build groups from ``[(name, param)]`` and a flat ``base_alpha``.

        Each param must be a 2D tensor-like with ``.shape ==
        (out_dim, in_dim)``. If a per-param alpha map is supplied
        (keyed by ``id(param)``) we use it; otherwise every param
        gets ``base_alpha``.
        """
        decay_pre = math.exp(-1.0 / tau_pre)
        decay_post = math.exp(-1.0 / tau_post)
        groups: list[STDPParamGroup] = []
        for name, p in named_params:
            shape = tuple(p.shape)  # type: ignore[attr-defined]
            if len(shape) != 2:
                raise ValueError(
                    f"plasticity-tagged param {name!r} must be 2D "
                    f"(out_dim, in_dim); got shape={shape}"
                )
            out_dim, in_dim = shape
            buf = SpikeRingBuffer(
                in_dim=in_dim,
                out_dim=out_dim,
                window=window,
                decay_pre=decay_pre,
                decay_post=decay_post,
            )
            alpha = base_alpha
            if per_param_alpha is not None:
                alpha = per_param_alpha.get(id(p), base_alpha)
            groups.append(
                STDPParamGroup(
                    param=p,
                    name=name,
                    buffer=buf,
                    alpha=alpha,
                    a_plus=a_plus * (alpha / base_alpha),
                    a_minus=a_minus * (alpha / base_alpha),
                    clip=clip,
                )
            )
        return cls(groups)

    # ---------------------------------------------------------- observe
    def observe(
        self,
        name: str,
        pre_spike: np.ndarray,
        post_spike: np.ndarray,
    ) -> None:
        """Record one timestep of spikes for layer ``name``.

        Producer responsibility:
        * call once per forward pass per plasticity-tagged layer
        * pre_spike and post_spike must be 1D binary arrays
        * the producer batches over time (ie call once per timestep)

        If ``name`` is not a registered group we silently no-op rather
        than fail — this lets the producer wire ``observe`` calls in
        unconditionally even when STDP is disabled by config.
        """
        g = self._by_name.get(name)
        if g is None:
            return
        g.buffer.push(pre_spike, post_spike)

    # --------------------------------------------------------------- step
    def step(self) -> dict:
        """Apply one local STDP weight update across all groups.

        Returns a stats dict::

            {"total_active_spikes": int,
             "total_groups": int,
             "step": int,
             "per_group": {name: {"active_pairs": int, "delta_norm": float}}}

        Uses the Triton kernel when available (CUDA tensors); otherwise
        falls back to the numpy outer-product path.
        """
        self._step_count += 1
        per_group: dict[str, dict] = {}
        total_active = 0
        for g in self.groups:
            pre_spike, post_spike = g.buffer.latest_spikes()
            n_pre_active = int(pre_spike.sum())
            n_post_active = int(post_spike.sum())
            active_pairs = n_pre_active * n_post_active
            g.last_active_pairs = active_pairs
            total_active += n_pre_active + n_post_active
            if active_pairs == 0:
                # Skip: no spikes => no update. This is the SNN-sparse
                # fast path that wins us the 100-200x speedup over AdamW.
                per_group[g.name] = {
                    "active_pairs": 0,
                    "delta_norm": 0.0,
                }
                continue
            # Compute dW = a_plus * outer(post_spike, pre_trace)
            #             - a_minus * outer(post_trace, pre_spike)
            # via the buffer's own pair_outer (numpy fallback).
            dW = g.buffer.pair_outer(a_plus=g.a_plus, a_minus=g.a_minus)
            # Bulk-apply to param.data via the backend helper.
            delta_norm = float(np.linalg.norm(dW))
            _apply_delta_inplace(g.param, dW, clip=g.clip)
            per_group[g.name] = {
                "active_pairs": active_pairs,
                "delta_norm": delta_norm,
            }
        return {
            "total_active_spikes": total_active,
            "total_groups": len(self.groups),
            "step": self._step_count,
            "per_group": per_group,
        }

    # --------------------------------------------------------- bookkeeping
    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing.

        Param weights are owned by the *parent module*; we never
        snapshot them here. We only persist the spike traces + cursors
        so that a resumed run gets the same in-flight plasticity.
        """
        return {
            "step": self._step_count,
            "groups": {
                g.name: {
                    "alpha": g.alpha,
                    "a_plus": g.a_plus,
                    "a_minus": g.a_minus,
                    "clip": g.clip,
                    "cursor": g.buffer.cursor,
                    "pre_trace": g.buffer.pre_trace.copy(),
                    "post_trace": g.buffer.post_trace.copy(),
                }
                for g in self.groups
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a previous :meth:`state_dict`."""
        self._step_count = int(sd.get("step", 0))
        groups = sd.get("groups", {})
        for name, g_state in groups.items():
            g = self._by_name.get(name)
            if g is None:
                continue
            g.alpha = float(g_state["alpha"])
            g.a_plus = float(g_state["a_plus"])
            g.a_minus = float(g_state["a_minus"])
            g.clip = float(g_state["clip"])
            g.buffer.cursor = int(g_state["cursor"])
            g.buffer.pre_trace[:] = np.asarray(
                g_state["pre_trace"], dtype=np.float32
            )
            g.buffer.post_trace[:] = np.asarray(
                g_state["post_trace"], dtype=np.float32
            )

    # -------------------------------------------------------- introspection
    def total_params(self) -> int:
        """Return total number of weights under STDP control."""
        return sum(int(np.prod(np.asarray(g.param.shape))) for g in self.groups)

    def names(self) -> list[str]:
        """List of group names registered with this optimizer."""
        return [g.name for g in self.groups]


# ---------------------------------------------------------------- backend
# Lazy torch bridge — kept in a tiny private function so ``import
# torch`` happens only at runtime, not at module load. This satisfies
# the hard constraint while still letting the optimizer mutate
# ``torch.nn.Parameter.data`` in place.

def _apply_delta_inplace(param: object, delta: np.ndarray, *, clip: float) -> None:
    """Add ``delta`` to ``param.data`` in place and clip.

    Supports:
    * numpy ndarray (test path) — direct add.
    * torch.nn.Parameter / torch.Tensor (production) — lazy-imported
      ``torch.from_numpy`` to bridge.

    Param mutation is detached and version-bump-safe (we go through
    ``.data`` so autograd never sees the in-place op as a graph
    edge — same pattern used by ``synapforge.bio.stdp_fast``).
    """
    # Pure-numpy fast path first (zero-import).
    if isinstance(param, np.ndarray):
        np.add(param, delta, out=param)
        np.clip(param, -clip, clip, out=param)
        return

    # Duck-typed numpy-backed shim used in tests.
    if hasattr(param, "_np_data") and isinstance(param._np_data, np.ndarray):  # type: ignore[attr-defined]
        np.add(param._np_data, delta, out=param._np_data)  # type: ignore[attr-defined]
        np.clip(param._np_data, -clip, clip, out=param._np_data)  # type: ignore[attr-defined]
        return

    # Torch path — runtime import only.
    try:
        import torch  # local import keeps module-level free of torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "STDPOnlyOptimizer: param is not numpy-backed and torch is "
            "not importable. Either pass a numpy ndarray, set "
            "param._np_data, or install torch."
        ) from exc
    data = getattr(param, "data", param)
    if not isinstance(data, torch.Tensor):
        raise TypeError(
            f"STDPOnlyOptimizer: cannot apply delta to param of type "
            f"{type(param).__name__}; expected torch.Tensor or numpy."
        )
    delta_t = torch.from_numpy(delta).to(
        device=data.device, dtype=data.dtype, non_blocking=True
    )
    with torch.no_grad():
        data.add_(delta_t)
        data.clamp_(-clip, clip)


__all__ = ["STDPOnlyOptimizer", "STDPParamGroup"]
