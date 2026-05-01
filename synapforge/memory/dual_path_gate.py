"""
Dual-path PLIF gate — Stage 3 of the L3 drift fix recipe.

At long context (>32K tokens), PLIF spike rate often drops below 8% as
the membrane potentials saturate and channels die. When this happens,
the spiking pathway becomes uninformative — most channels gated to 0,
parametric recall starves.

Solution: at position > 32K AND PLIF spike rate < 8%, switch the model
to dense CfC (PLIF observe_only=True) for the next 4K tokens. This lets
parametric channels stay alive when they would otherwise be gated.

Triggers: ~5-10% of long context. Negligible cost (just observe_only flag).

Per agent synthesis 2026-04-30: Stage 3 of 5-day L3 drift fix recipe.
Combined with Stages 0 (BM25 sidecar), 1 (trained PQ codebook), 2
(MultiBandTau-PLIF): L3 50M drift 8-15% → <5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DualPathGateState:
    position: int = 0
    spike_rate_ema: float = 0.10
    in_dense_mode: bool = False
    dense_remaining: int = 0
    n_switches: int = 0
    n_dense_tokens: int = 0
    n_total_tokens: int = 0


class DualPathPLIFGate:
    """Switches PLIF modules between spiking (default) and dense (bypass).

    Tracks position + spike rate EMA across all PLIF modules. When position
    exceeds long_threshold AND spike rate falls below low_threshold, all
    PLIFs flip to observe_only=True for `dense_window` tokens, then revert.

    Usage:
        plif_modules = [m for m in model.modules()
                        if m.__class__.__name__ == "PLIF"]
        gate = DualPathPLIFGate(plif_modules)

        for step, tokens in enumerate(stream):
            gate.step(n_new_tokens=tokens.shape[1])
            # ... model forward (PLIFs auto-bypassed if gate active)
    """

    def __init__(
        self,
        plif_modules: List,
        long_threshold: int = 32_000,
        low_spike_rate: float = 0.08,
        dense_window: int = 4096,
        ema_alpha: float = 0.95,
    ) -> None:
        self.plifs = plif_modules
        self.long_threshold = long_threshold
        self.low_spike_rate = low_spike_rate
        self.dense_window = dense_window
        self.ema_alpha = ema_alpha
        self.state = DualPathGateState()

    def _read_spike_rates(self) -> float:
        rates = []
        for m in self.plifs:
            if hasattr(m, "last_spike_rate"):
                r = float(m.last_spike_rate)
                rates.append(r)
        if not rates:
            return self.state.spike_rate_ema
        return sum(rates) / len(rates)

    def _set_observe_only(self, observe: bool) -> None:
        for m in self.plifs:
            if hasattr(m, "observe_only"):
                m.observe_only = observe

    def step(self, n_new_tokens: int = 1) -> bool:
        """Advance position by n_new_tokens, decide if dense mode toggles.

        Returns True if model is currently in dense mode (PLIF observe_only).
        """
        self.state.position += n_new_tokens
        self.state.n_total_tokens += n_new_tokens

        rate_now = self._read_spike_rates()
        self.state.spike_rate_ema = (
            self.ema_alpha * self.state.spike_rate_ema
            + (1.0 - self.ema_alpha) * rate_now
        )

        if self.state.in_dense_mode:
            self.state.dense_remaining -= n_new_tokens
            self.state.n_dense_tokens += n_new_tokens
            if self.state.dense_remaining <= 0:
                self.state.in_dense_mode = False
                self._set_observe_only(False)

        elif (self.state.position > self.long_threshold
              and self.state.spike_rate_ema < self.low_spike_rate):
            self.state.in_dense_mode = True
            self.state.dense_remaining = self.dense_window
            self.state.n_switches += 1
            self._set_observe_only(True)

        return self.state.in_dense_mode

    def reset(self) -> None:
        """Reset between documents."""
        self._set_observe_only(False)
        self.state = DualPathGateState()

    def stats(self) -> dict:
        denom = max(self.state.n_total_tokens, 1)
        return {
            "position": self.state.position,
            "spike_rate_ema": self.state.spike_rate_ema,
            "in_dense_mode": self.state.in_dense_mode,
            "n_switches": self.state.n_switches,
            "dense_token_fraction": self.state.n_dense_tokens / denom,
        }


def attach_dual_path_gate_to_model(
    model,
    long_threshold: int = 32_000,
    low_spike_rate: float = 0.08,
    dense_window: int = 4096,
) -> DualPathPLIFGate:
    """Auto-discover all PLIF modules in model and wire a single gate."""
    plif_modules = [
        m for m in model.modules()
        if m.__class__.__name__ == "PLIF" and hasattr(m, "observe_only")
    ]
    if not plif_modules:
        raise RuntimeError("no PLIF modules found in model")
    return DualPathPLIFGate(
        plif_modules=plif_modules,
        long_threshold=long_threshold,
        low_spike_rate=low_spike_rate,
        dense_window=dense_window,
    )
