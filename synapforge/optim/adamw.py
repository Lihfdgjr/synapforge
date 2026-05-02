"""synapforge.optim.adamw — pure-python AdamW (Phase 1 of torch removal).

Why this exists
---------------
``train_100m_kd.py`` currently builds either ``PlasticityAwareAdamW``
(``synapforge.optim.PlasticityAwareAdamW``) or — when ``--fused-adamw``
is set and no plasticity sources are wired — ``torch.optim.AdamW``.

Both inherit from ``torch.optim.Optimizer``. As Phase 1 of the
torch-replacement roadmap (see ``docs/TORCH_REPLACEMENT_PLAN.md``), we
ship a pure-python ``AdamW`` here that:

1. Operates on iterables of ``torch.Tensor`` (so it remains drop-in
   while the rest of the trainer still uses ``torch.Tensor`` storage).
2. Implements the standard AdamW update bit-by-bit on top of plain
   tensor ops — no ``torch.optim`` inheritance, no ``torch._C`` /
   ``torch._foreach_*`` fused kernels.
3. Matches ``torch.optim.AdamW(fused=True)`` numerics to within 1e-5
   (the unit test in ``tests/optim/test_synapforge_adamw.py`` enforces
   this; see also Phase 1 entry in the roadmap).
4. Provides ``state_dict`` / ``load_state_dict`` with the canonical
   ``exp_avg`` / ``exp_avg_sq`` / ``step`` keys so warmstart from a
   ``--fused-adamw`` ckpt round-trips losslessly.

Why pure-python and not Triton/CUDA
-----------------------------------
The Adam moment update is bandwidth-bound, not compute-bound, so a
fused kernel (``torch.optim.AdamW(fused=True)``) wins ~2-3 % step time
on a 100M model — meaningful but not dominant. Phase 1 trades that
~2-3 % for getting out of ``torch.optim`` entirely; Phase 4 of the
roadmap will replace the underlying ``torch.Tensor`` storage with
``synapforge.tensor`` and at that point this same loop will dispatch
to a Triton ``adamw_step`` kernel without changing the public API.

Public API
----------
    >>> from synapforge.optim import AdamW
    >>> opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    >>> for batch in dl:
    ...     opt.zero_grad()
    ...     loss = step(model, batch)
    ...     loss.backward()
    ...     opt.step()

Limitations (intentional, will close in Phase 2-4)
--------------------------------------------------
* Single param-group only (no per-layer LR schedules). The trainer
  doesn't currently use multi-group LR; if it ever does, fork
  ``param_groups`` support from ``PlasticityAwareAdamW``.
* No closure support (the trainer doesn't pass one).
* No fused kernel — see "Why pure-python" above.
* No plasticity merge — that lives in
  ``synapforge.optim.PlasticityAwareAdamW``. ``AdamW`` is the
  plasticity-free path that mirrors the ``--fused-adamw`` semantics.
* Still imports ``torch`` for tensor storage. Phase 4 of the roadmap
  removes this import; until then, the only ``torch.*`` surface this
  module touches is: ``torch.Tensor`` (type), ``torch.no_grad`` (ctx),
  ``torch.zeros_like`` (state init), and arithmetic methods on
  ``Tensor`` (``mul_``, ``add_``, ``addcmul_``, ``sqrt``, ``addcdiv_``).
  No ``torch.optim`` import. No ``torch._C``. No fused kernels.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

__all__ = ["AdamW"]


class AdamW:
    """Pure-python AdamW optimizer.

    Numerically equivalent to ``torch.optim.AdamW`` (with or without
    ``fused=True``) when operating on ``torch.Tensor`` parameters in
    the standard regime (no NaN/Inf grads, finite weight_decay).

    Update rule (per-step, per-param):

    .. code-block:: text

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p   <- p - lr * weight_decay * p          # decoupled (AdamW)
        p   <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Implementation notes:
      * The bias-correction division is fused into ``step_size`` and
        ``denom`` so the per-element math matches the reference
        implementation in PyTorch's ``adamw.py`` line-by-line. See the
        cross-check test in ``tests/optim/test_synapforge_adamw.py``.
      * ``step()`` is decorated with ``@torch.no_grad()`` to mirror the
        reference; we do not need to build a graph through the moment
        EMA. (Phase 4 will remove this when we move off ``torch.autograd``.)
      * ``state_dict`` uses keys ``step`` / ``exp_avg`` / ``exp_avg_sq``,
        identical to ``torch.optim.AdamW``, so ckpts round-trip.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in [0,1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in [0,1), got {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        params_list = [p for p in params if p is not None]
        if not params_list:
            raise ValueError("AdamW: empty params iterable")
        # Reject anything that doesn't quack like a tensor with .data + .grad
        # to surface misuse early (e.g., passing a model instead of
        # model.parameters()).
        for i, p in enumerate(params_list):
            if not hasattr(p, "data") or not hasattr(p, "grad"):
                raise TypeError(
                    f"AdamW: params[{i}] is not a torch.Tensor-like "
                    f"(missing .data or .grad); got {type(p).__name__!r}. "
                    "Did you pass `model` instead of `model.parameters()`?"
                )

        self.defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        # Single param-group for now (see "Limitations" in module
        # docstring). Mirroring torch.optim's API shape so trainers can
        # iterate ``opt.param_groups[0]["lr"]`` for LR scheduling.
        self.param_groups: list[dict[str, Any]] = [
            {
                "params": params_list,
                **self.defaults,
            }
        ]
        # state[id(p)] = {"step": int, "exp_avg": Tensor, "exp_avg_sq": Tensor}
        # Keyed by id() rather than the tensor itself so we avoid
        # holding a hash dependency on the tensor (which would prevent
        # GC of stale params).
        self.state: dict[int, dict[str, Any]] = {}

    # --------------------------------------------------------------- API

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on all wrapped parameters.

        Mirrors ``torch.optim.Optimizer.zero_grad`` semantics:
        ``set_to_none=True`` (default in PyTorch >= 2.0) sets
        ``p.grad = None`` rather than zeroing in-place. This is faster
        and avoids a memory write per param.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Run one AdamW update on every wrapped parameter.

        Skips params with ``p.grad is None`` (matches ``torch.optim``).
        Skips params whose grad contains NaN/Inf (poisoning Adam's
        moments with NaN propagates and is harder to recover from than
        a single skipped step).
        """
        if closure is not None:
            raise NotImplementedError(
                "synapforge.optim.AdamW does not support closures; "
                "use the standard zero_grad/backward/step idiom."
            )
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                # Skip NaN/Inf grads (would poison the moments).
                # Use a single all-reduce over isfinite to keep this
                # cheap on GPU (one CUDA dispatch per param).
                if not torch.isfinite(grad).all():
                    continue

                pid = id(p)
                state = self.state.get(pid)
                if state is None:
                    state = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format,
                        ),
                        "exp_avg_sq": torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format,
                        ),
                    }
                    self.state[pid] = state

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ----- moment EMAs (in-place, decoupled from autograd graph) -----
                # m_t = beta1*m_{t-1} + (1-beta1)*g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v_t = beta2*v_{t-1} + (1-beta2)*g_t*g_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # ----- bias correction -----
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t
                # ``step_size`` and ``denom`` together produce the bias-
                # corrected update without materializing m_hat/v_hat.
                # This matches the reference impl in PyTorch
                # (torch/optim/adamw.py: _single_tensor_adamw).
                step_size = lr / bias_c1
                # denom = sqrt(v_hat) + eps = sqrt(v) / sqrt(bias_c2) + eps
                denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)

                # ----- decoupled weight decay (AdamW, not Adam+L2) -----
                # p <- p * (1 - lr*wd) BEFORE the gradient step. Matches
                # PyTorch's reference; the order matters: the WD shrink
                # uses the un-updated p, then the Adam step further
                # adjusts. (Adam+L2 would instead fold wd into the
                # gradient before the EMAs — DO NOT do that here.)
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # ----- the actual Adam step -----
                # p <- p - step_size * m / denom
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

    # ----------------------------------------------------- ckpt round-trip

    def state_dict(self) -> dict[str, Any]:
        """Return a state_dict compatible with ``torch.optim.AdamW``.

        Layout mirrors PyTorch's: top-level ``state`` dict keyed by
        param index (NOT id, so it's portable across processes), plus
        ``param_groups`` carrying hyperparams.

        The chief reason for the index-based key (rather than id-based)
        is portability: ``id(p)`` is not stable across runs, so we map
        each param to its position in the param-group's ``params``
        list at serialization time.
        """
        # Build a stable param-index → state mapping.
        packed_state: dict[int, dict[str, Any]] = {}
        global_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(id(p))
                if st is not None:
                    packed_state[global_idx] = {
                        "step": int(st["step"]),
                        "exp_avg": st["exp_avg"].detach().clone(),
                        "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
                    }
                global_idx += 1

        # Don't include the tensor list in the serialized groups —
        # PyTorch encodes them as integer index lists. We do the same.
        packed_groups: list[dict[str, Any]] = []
        global_idx = 0
        for group in self.param_groups:
            n = len(group["params"])
            packed_groups.append(
                {
                    "lr": group["lr"],
                    "betas": group["betas"],
                    "eps": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "params": list(range(global_idx, global_idx + n)),
                }
            )
            global_idx += n
        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state.

        Tolerates the ``torch.optim.AdamW(fused=True)`` ckpt layout (same
        ``exp_avg`` / ``exp_avg_sq`` / ``step`` keys) so warmstart from a
        ``--fused-adamw`` run round-trips losslessly.
        """
        if not isinstance(state_dict, dict) or "state" not in state_dict:
            raise ValueError(
                "AdamW.load_state_dict: expected a dict with 'state' and "
                f"'param_groups' keys, got {type(state_dict).__name__}"
            )
        groups = state_dict.get("param_groups", [])
        if groups:
            group_in = groups[0]
            self.param_groups[0]["lr"] = float(group_in.get(
                "lr", self.param_groups[0]["lr"]))
            self.param_groups[0]["betas"] = tuple(group_in.get(
                "betas", self.param_groups[0]["betas"]))
            self.param_groups[0]["eps"] = float(group_in.get(
                "eps", self.param_groups[0]["eps"]))
            self.param_groups[0]["weight_decay"] = float(group_in.get(
                "weight_decay", self.param_groups[0]["weight_decay"]))

        # Map index-based state back to id(p)-keyed state.
        params_flat: list[torch.Tensor] = []
        for group in self.param_groups:
            params_flat.extend(group["params"])
        new_state: dict[int, dict[str, Any]] = {}
        for idx, st in state_dict.get("state", {}).items():
            idx_i = int(idx)
            if idx_i >= len(params_flat):
                # Stale ckpt entry; silently skip (matches torch.optim).
                continue
            p = params_flat[idx_i]
            new_state[id(p)] = {
                "step": int(st["step"]),
                "exp_avg": st["exp_avg"].detach().clone().to(
                    device=p.device, dtype=p.dtype),
                "exp_avg_sq": st["exp_avg_sq"].detach().clone().to(
                    device=p.device, dtype=p.dtype),
            }
        self.state = new_state

    # ----------------------------------------------------- compat helpers

    def __repr__(self) -> str:  # pragma: no cover
        n_params = sum(len(g["params"]) for g in self.param_groups)
        return (
            f"synapforge.optim.AdamW(n_params={n_params}, "
            f"lr={self.defaults['lr']}, betas={self.defaults['betas']}, "
            f"eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']})"
        )
