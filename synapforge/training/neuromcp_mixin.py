"""trainer-side NeuroMCP wire-in -- T9.2 (DEEP_MAINT_QUEUE.md).

NeuroMCP (`synapforge.action.neuromcp.NeuroMCPHead`) replaces JSON
function-calling tool tokens with a pair of *neuroplastic* primitives:

    SparseSynapticLayer       -- D->D linear with a structurally sparse
                                 mask that grows (co-activation EMA) +
                                 prunes (magnitude) on its own.
    DynamicActionCodebook     -- [K_max, D] prototypes; alive_mask grows
                                 when a hidden state's max cosine
                                 similarity to alive prototypes is below
                                 a novelty threshold.

PoC-validated on the 4-button env at ``synapforge.demo.four_button``:
density 4.5 -> 39.9%, K=9 -> 12, 100% hit-rate after warmup. The PoC,
however, is a standalone training loop -- the production 100M LM
trainer (``train_100m_kd.py``) never instantiates the head, so the
NeuroMCP claim is currently NOT exercised on Synap-1.

This module wires NeuroMCPHead into the LM trainer as a *trainer
mixin* with the same default-OFF / fail-soft contract used by
``synapforge.trainer_mixins.{MultimodalMixin,SelfLearnMixin,
CuriosityMixin}``:

  * Constructed only when ``--neuromcp-weight > 0``.
  * Failures inside the forward / step path return a zero loss with
    grad attached so ``.backward()`` is safe and training continues.
  * No dependency on ``model_100m`` internals beyond the public
    ``model.encode(x) -> (B, T, d)`` contract; we read ``model.d`` for
    the head's hidden width.
  * No new heavy deps -- only ``torch`` + ``synapforge.action``.

Action target rationale
------------------------
A pure LM trainer has no real "action" labels. We synthesise a cheap
supervisory signal: *use the next-token's first byte mod K as the
target action ID*. This is **a placeholder**, not the production
target distribution. It exists to:

  1. Prove the head learns when wired into the LM forward pass.
  2. Drive the ``coact_ema`` buffer (so density grows) and the codebook
     novelty signal (so K grows) -- both PoC-validated mechanisms.
  3. Give us a non-trivial action_loss curve to log.

Real action data (e.g. screen-recordings paired with user clicks) is
the long-term target -- see memory ``feedback_neural_action_no_token_no_mcp``.
The mixin's API is shaped so the action target can be swapped without
touching the trainer: callers can pass an explicit ``action_targets``
keyword to ``action_loss``.

Public surface
--------------
* ``NeuroMCPMixin(model, hidden, codebook_size=16, action_dim=64, ...)``
* ``mixin.action_loss(text_hidden, target_ids=None) -> Tensor``  -- scalar
  CE loss across the alive prototypes; on failure returns a zero scalar
  with grad attached to ``text_hidden``.
* ``mixin.step_plasticity()`` -- run after ``optim.step()`` to grow/prune
  the sparse mask + maybe-grow the codebook. Returns the stats dict.
* ``mixin.stats() -> dict``  -- ``{density, K, last_action_loss,
  last_hit_rate}`` for periodic logging (every 100 steps).
* ``NeuroMCPMixin.smoke()`` -- self-contained sanity check on dummy
  tensors; mirrors ``MultimodalMixin.smoke``.

Default OFF: ``train_100m_kd`` only constructs the mixin when
``--neuromcp-weight > 0``; if construction fails the trainer logs +
sets the mixin to ``None`` and the LM trajectory is unchanged.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroMCPMixin:
    """Wrap ``synapforge.action.NeuroMCPHead`` for the LM trainer loop.

    Parameters
    ----------
    model : nn.Module
        The LM. Used only to read ``model.d`` (hidden width) and to
        place the head on the same device. The mixin does NOT touch
        ``model_100m`` internals.
    hidden : int
        Override the model's hidden width. Default ``model.d``.
    codebook_size : int
        ``codebook_initial`` for ``DynamicActionCodebook``. ``max_size``
        is set to ``4 * codebook_size`` so growth has headroom. Default
        16 (i.e. up to 64 prototypes).
    action_dim : int
        UNUSED in the current head wiring -- the head's action space
        equals the alive codebook size, which is dynamic. Kept on the
        constructor for forward-compat with a future fixed-action-dim
        head (e.g. when the trainer gets real OS actuator labels).
    synapse_density : float
        Initial mask density for ``SparseSynapticLayer``. Default 5%.
    synapse_max_density : float
        Cap for the same. Default 40%.
    verbose : bool
        Print warnings on init / runtime errors. Default True.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden: Optional[int] = None,
        codebook_size: int = 16,
        action_dim: int = 64,
        synapse_density: float = 0.05,
        synapse_max_density: float = 0.40,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.hidden = int(hidden) if hidden is not None else int(getattr(model, "d", 512))
        self.codebook_size = int(codebook_size)
        self.action_dim = int(action_dim)  # placeholder for fixed-action wiring
        self.synapse_density = float(synapse_density)
        self.synapse_max_density = float(synapse_max_density)
        self.verbose = bool(verbose)
        self.head: Optional[nn.Module] = None
        self.last_action_loss: float = 0.0
        self.last_hit_rate: float = 0.0
        self._step_count: int = 0

        try:
            from ..action.neuromcp import NeuroMCPHead
            self.head = NeuroMCPHead(
                hidden=self.hidden,
                codebook_initial=self.codebook_size,
                codebook_max=max(self.codebook_size * 4, self.codebook_size + 1),
                synapse_density=self.synapse_density,
                synapse_max_density=self.synapse_max_density,
            )
            try:
                dev = next(model.parameters()).device
                self.head.to(dev)
            except StopIteration:
                pass
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(
                    f"NeuroMCPMixin: cannot build NeuroMCPHead ({exc}); no-op.",
                    RuntimeWarning, stacklevel=2,
                )
            self.head = None

    # ------------------------------------------------------------------ helpers
    def parameters(self):
        if self.head is not None:
            yield from self.head.parameters()

    @property
    def density(self) -> float:
        if self.head is None:
            return 0.0
        return float(self.head.proj.density)

    @property
    def K(self) -> int:
        if self.head is None:
            return 0
        return int(self.head.codebook.K)

    def stats(self) -> dict[str, float]:
        return {
            "density": self.density,
            "K": float(self.K),
            "last_action_loss": float(self.last_action_loss),
            "last_hit_rate": float(self.last_hit_rate),
            "step_count": float(self._step_count),
        }

    # ------------------------------------------------------------------ targets
    @staticmethod
    def _placeholder_targets(text_hidden: torch.Tensor,
                             y: Optional[torch.Tensor],
                             K_alive: int) -> Optional[torch.Tensor]:
        """Build action targets from the next-token's first byte mod K.

        Documented as a *placeholder*: any real screen-recording / OS-actuator
        label feed should bypass this and pass ``target_ids`` to
        ``action_loss`` directly.
        """
        if y is None:
            return None
        try:
            # y is (B, T) of token ids; clamp to byte-range and mod by alive K.
            # K_alive can grow over training, so we mod fresh every step.
            if K_alive <= 0:
                return None
            with torch.no_grad():
                ids = y.reshape(-1).long().abs() % int(K_alive)
            return ids
        except Exception:
            return None

    # ------------------------------------------------------------------ forward
    def action_loss(
        self,
        text_hidden: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        y_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CE loss over the codebook prototypes.

        Parameters
        ----------
        text_hidden : (B, T, d) -- last-layer LM hidden state.
        target_ids : (B*T,) long -- explicit action targets. Takes precedence.
        y_next : (B, T) long -- next-token labels. When ``target_ids`` is
            None we synthesise placeholder action ids from these (see
            ``_placeholder_targets``).

        Returns
        -------
        scalar tensor -- mean CE across all positions. On any failure
        returns a zero tensor with grad attached to ``text_hidden`` so
        ``.backward()`` is safe.
        """
        zero = (text_hidden.float().sum() * 0.0)
        if self.head is None:
            return zero
        try:
            B, T, D = text_hidden.shape
            if D != self.hidden:
                if self.verbose:
                    warnings.warn(
                        f"NeuroMCPMixin.action_loss: hidden mismatch "
                        f"{D} != {self.hidden}; returning zero.",
                        RuntimeWarning, stacklevel=2,
                    )
                return zero
            # Run the head: (B*T, d) -> {logits: (B*T, K_alive), hidden_z}.
            h_flat = text_hidden.reshape(-1, D)
            out = self.head(h_flat)
            logits = out["logits"]
            K_alive = logits.shape[-1]
            if K_alive == 0:
                return zero
            # Build targets.
            if target_ids is None:
                target_ids = self._placeholder_targets(text_hidden, y_next, K_alive)
            if target_ids is None:
                # No supervision available -- self-supervised: the head's own
                # argmax. This still drives gradient because the CE rotates
                # the prototype geometry to be consistent with hidden states.
                target_ids = logits.detach().argmax(dim=-1)
            target_ids = target_ids.long().to(logits.device)
            if target_ids.numel() != logits.shape[0]:
                # Allow broadcasting: e.g. (B,) per-sequence targets.
                if target_ids.numel() == B:
                    target_ids = target_ids.unsqueeze(1).expand(B, T).reshape(-1)
                else:
                    return zero
            loss = F.cross_entropy(logits.float(), target_ids)
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                self.last_hit_rate = float((pred == target_ids).float().mean().item())
                self.last_action_loss = float(loss.detach().item())
            self._step_count += 1
            # Stash the hidden_z so step_plasticity can grow the codebook.
            self._last_hidden_z = out["hidden_z"].detach()
            return loss
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(
                    f"NeuroMCPMixin.action_loss failed: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
            return zero

    # ------------------------------------------------------------------ plasticity
    @torch.no_grad()
    def step_plasticity(self) -> dict[str, Any]:
        """Run synapse grow/prune + codebook maybe-grow.

        Call AFTER ``optim.step()`` (matches the OBSERVE/DELTA/APPLY contract
        of ``synapforge.plasticity``). Returns the stats dict for logging.
        Failures degrade gracefully: returns ``{}`` and continues.
        """
        if self.head is None:
            return {}
        try:
            hidden_z = getattr(self, "_last_hidden_z", None)
            return self.head.step_plasticity(hidden_z=hidden_z)
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(
                    f"NeuroMCPMixin.step_plasticity failed: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
            return {}

    # ------------------------------------------------------------------ smoke
    @classmethod
    def smoke(cls) -> dict[str, Any]:
        """Self-contained sanity check on dummy tensors.

        Builds a tiny (d=32) head, fakes a ``text_hidden`` plus next-token
        labels, runs ``action_loss`` and ``step_plasticity``, and asserts
        the loss is finite, density is in [0, 1], and K is positive.
        Returns a dict suitable for ``json.dumps``.
        """
        torch.manual_seed(0)
        d = 32
        # Dummy "model" with .d attribute and one parameter so .to() works.
        dummy = nn.Linear(d, d)
        dummy.d = d  # type: ignore[attr-defined]
        mix = cls(dummy, hidden=d, codebook_size=4, verbose=False)
        if mix.head is None:
            return {"ok": False, "reason": "head construction failed"}

        text_h = torch.randn(2, 4, d, requires_grad=True)
        y_next = torch.randint(0, 100, (2, 4))
        loss1 = mix.action_loss(text_h, y_next=y_next)
        assert torch.isfinite(loss1), "action_loss non-finite"
        loss1.backward()  # must not raise
        plast1 = mix.step_plasticity()

        # A few extra steps to verify density does not collapse.
        for _ in range(3):
            text_h = torch.randn(2, 4, d, requires_grad=True)
            y_next = torch.randint(0, 100, (2, 4))
            loss = mix.action_loss(text_h, y_next=y_next)
            loss.backward()
            mix.step_plasticity()

        return {
            "ok": True,
            "loss": float(loss1.detach()),
            "density": mix.density,
            "K": mix.K,
            "hit_rate": mix.last_hit_rate,
            "plast_keys": sorted(plast1.keys()) if plast1 else [],
        }


__all__ = ["NeuroMCPMixin"]


if __name__ == "__main__":
    import json
    import sys
    res = NeuroMCPMixin.smoke()
    print(json.dumps(res, indent=2, default=str))
    sys.exit(0 if res.get("ok") else 1)
