"""synapforge.neuromcp.action_head -- ActionHead with codebook + ParamHead.

Layer 1 + Layer 2 + Layer 3 of the NeuroMCP closed-loop stack.

Three responsibilities:

    1) ActionHead        :: hidden -> primitive_logits   (Layer 1)
    2) ParamHead         :: hidden -> 8-slot params      (Layer 2)
    3) DynamicCodebook   :: K growing prototypes         (Layer 3, delegated
                                                          to sf.action.neuromcp)

Usage
-----

    from synapforge.neuromcp.action_head import NeuroActionHead

    head = NeuroActionHead(hidden=512, num_primitives=24)
    out = head(text_hidden)          # (B, T, d) -> dict
    # out["primitive_id"]    : (B, T) long
    # out["params"]          : (B, T, 8) float
    # out["confidence"]      : (B, T) float in [0,1]
    # out["primitive_logits"]: (B, T, K_alive)
    # out["raw_params"]      : (B, T, 8) the un-clipped Linear output

Confidence < 0.5 -> the actuator should HALT and ask the user (per the
brief).  We expose ``out["should_halt"]`` as a convenience bool tensor.

torch is **lazily imported** inside the class so a fresh checkout can
``import synapforge.neuromcp.action_head`` for static analysis even if
torch is not installed.  Only when the class is *constructed* do we
actually pull torch in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .primitives import NUM_PARAM_SLOTS, NUM_PRIMITIVES


@dataclass
class ActionHeadConfig:
    """Configuration for the NeuroActionHead.

    Defaults are tuned for the LM trainer's d=512 hidden width and the
    24-primitive vocabulary.
    """

    hidden: int = 512
    num_primitives: int = NUM_PRIMITIVES
    num_param_slots: int = NUM_PARAM_SLOTS
    halt_threshold: float = 0.5
    use_codebook: bool = True
    codebook_initial: int = 16
    codebook_max: int = 64
    synapse_density: float = 0.05
    synapse_max_density: float = 0.40


class NeuroActionHead:
    """Hidden-state -> (primitive_id, params, confidence) head.

    Wraps:
      * ``primitive_proj`` :: Linear(hidden, num_primitives)
      * ``param_proj``     :: Linear(hidden, num_param_slots)
      * Optional ``codebook`` :: synapforge.action.neuromcp.NeuroMCPHead

    The codebook gives us the dynamic-K growth path used by Layer-4
    CompoundGrowth.  When ``use_codebook=False`` the head degenerates to
    a static linear classifier (useful for unit tests on machines
    without ``synapforge.plasticity``).

    The class itself is NOT a torch.nn.Module so this file can be
    imported torch-free.  Once constructed, a torch.nn.Module child is
    held in ``self._module`` -- this child IS a torch.nn.Module so it
    plugs into ``optimizer.add_param_group(self.parameters())``.
    """

    def __init__(self, cfg: Optional[ActionHeadConfig] = None, **kwargs: Any) -> None:
        cfg = cfg or ActionHeadConfig(**kwargs)
        self.cfg = cfg
        # Lazy-import torch.  We do it once, here, at construction time.
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - hard dep at runtime
            raise ImportError(
                "synapforge.neuromcp.action_head requires torch at construction "
                "time.  Module import is OK without torch (lazy)."
            ) from exc
        self._torch = torch
        self._nn = nn

        # Inner torch.nn.Module wraps the actual learnable weights.  This
        # keeps NeuroActionHead torch-import-free at module-scope while
        # still exposing .parameters() / .to(device) / .train()/.eval()
        # to the caller.
        class _Inner(nn.Module):  # noqa: N801 -- private inner module
            def __init__(self_inner: "_Inner") -> None:
                super().__init__()
                self_inner.primitive_proj = nn.Linear(cfg.hidden, cfg.num_primitives)
                self_inner.param_proj = nn.Linear(cfg.hidden, cfg.num_param_slots)
                self_inner.confidence_proj = nn.Linear(cfg.hidden, 1)
                # Zero-init param + confidence head so early training is
                # neutral (params ~ 0.5 after sigmoid, confidence ~ 0.5).
                nn.init.zeros_(self_inner.param_proj.weight)
                nn.init.zeros_(self_inner.param_proj.bias)
                nn.init.zeros_(self_inner.confidence_proj.weight)
                nn.init.zeros_(self_inner.confidence_proj.bias)

        self._module = _Inner()

        # Optional dynamic codebook.  Bridges to the validated head in
        # ``synapforge.action.neuromcp``.  We do not duplicate that file
        # -- we *delegate*.
        self._codebook = None
        if cfg.use_codebook:
            try:
                from ..action.neuromcp import NeuroMCPHead
                self._codebook = NeuroMCPHead(
                    hidden=cfg.hidden,
                    codebook_initial=cfg.codebook_initial,
                    codebook_max=cfg.codebook_max,
                    synapse_density=cfg.synapse_density,
                    synapse_max_density=cfg.synapse_max_density,
                )
            except Exception:  # pragma: no cover - defensive
                self._codebook = None

    # -- torch.nn passthrough -------------------------------------------
    def parameters(self):
        yield from self._module.parameters()
        if self._codebook is not None:
            yield from self._codebook.parameters()

    def to(self, device):
        self._module.to(device)
        if self._codebook is not None:
            self._codebook.to(device)
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        if self._codebook is not None:
            self._codebook.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # -- core forward ----------------------------------------------------
    def forward(self, hidden):
        """Hidden state -> action dict.

        Args
        ----
        hidden : torch.Tensor of shape (B, T, d) or (N, d)

        Returns
        -------
        dict with keys
            primitive_logits : (.., num_primitives)
            primitive_id     : (..,) long
            params           : (.., 8) in [0,1] (sigmoid)
            raw_params       : (.., 8) raw linear output
            confidence       : (..,) float in [0,1]
            should_halt      : (..,) bool  -- confidence < halt_threshold
            codebook_logits  : (.., K_alive) when codebook enabled, else None
            codebook_z       : (.., d)       when codebook enabled, else None
        """
        torch = self._torch
        if hidden.dim() < 2:
            raise ValueError(f"hidden needs >=2 dims, got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.cfg.hidden:
            raise ValueError(
                f"last-dim mismatch: expected {self.cfg.hidden}, got {hidden.shape[-1]}"
            )

        primitive_logits = self._module.primitive_proj(hidden)
        raw_params = self._module.param_proj(hidden)
        params = torch.sigmoid(raw_params)
        # confidence = sigmoid of a single learned scalar projection per
        # token.  Init=0 -> sigmoid(0)=0.5 -> agent is "ambivalent" until
        # taught, which is the desired safety prior.
        confidence_raw = self._module.confidence_proj(hidden).squeeze(-1)
        confidence = torch.sigmoid(confidence_raw)
        primitive_id = primitive_logits.argmax(dim=-1)
        should_halt = confidence < float(self.cfg.halt_threshold)

        # Optional codebook path.  On failure (e.g. wrong shape inside
        # the bridged NeuroMCPHead) we silently leave it None -- the
        # rest of the pipeline is unaffected.
        cb_logits = None
        cb_z = None
        if self._codebook is not None:
            try:
                # NeuroMCPHead expects flattened (N, d) or (B, T, d); it
                # is fine with both.
                cb_out = self._codebook(hidden)
                cb_logits = cb_out["logits"]
                cb_z = cb_out["hidden_z"]
            except Exception:  # pragma: no cover - defensive
                cb_logits = None
                cb_z = None

        return {
            "primitive_logits": primitive_logits,
            "primitive_id": primitive_id,
            "params": params,
            "raw_params": raw_params,
            "confidence": confidence,
            "should_halt": should_halt,
            "codebook_logits": cb_logits,
            "codebook_z": cb_z,
        }

    __call__ = forward

    # -- plasticity ------------------------------------------------------
    def step_plasticity(self, hidden_z=None) -> dict:
        """Tick the bridged codebook's plasticity rule.

        Returns the codebook's stats dict, or {} when the codebook is
        disabled.  Safe to call after ``optim.step()``.
        """
        if self._codebook is None:
            return {}
        try:
            return self._codebook.step_plasticity(hidden_z=hidden_z)
        except Exception:  # pragma: no cover - defensive
            return {}

    # -- introspection ---------------------------------------------------
    @property
    def K_alive(self) -> int:
        """Currently-alive prototype count, or num_primitives when off."""
        if self._codebook is None:
            return int(self.cfg.num_primitives)
        try:
            return int(self._codebook.codebook.K)
        except Exception:  # pragma: no cover - defensive
            return int(self.cfg.num_primitives)

    @property
    def density(self) -> float:
        """Sparse synapse density (0 if codebook disabled)."""
        if self._codebook is None:
            return 0.0
        try:
            return float(self._codebook.proj.density)
        except Exception:  # pragma: no cover - defensive
            return 0.0


__all__ = ["ActionHeadConfig", "NeuroActionHead"]
