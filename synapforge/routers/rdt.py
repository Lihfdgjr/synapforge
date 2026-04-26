"""sf.routers.RDTLoop — Recurrent-Depth Transformer loop wrapper.

Reference
---------
- Recurrent-Depth Transformer (RDT, OpenMythos kyegomez/OpenMythos 2026-04):
  recurrent shared backbone with per-loop adaptations. The same body block
  runs ``R`` times; per-step features (loop-index embedding, depth LoRA,
  residual gate bias, layer scale) inject "which-step-am-I" signal so the
  shared weights can specialise.

What this module provides
-------------------------
1. ``LoopIndexEmbedding``  — sinusoidal step-index bias added to body input
                             on the first ``hidden//frac`` channels.
2. ``LayerScale``           — per-channel γ (init 1e-4) on the residual delta
                             so the loop is near-identity at init.
3. ``ResidualGateBias``     — σ(linear(x_in) + b) with b init -2 → init gate
                             ≈ 0.12, near-identity, learns to open with depth.
4. ``DepthLoRAAdapter``     — low-rank (rank=8 default) per-loop scale embed
                             additive delta. Shared A/B, per-step scale so
                             cost scales with rank not depth.
5. ``AccelExit``            — inference-only early-exit when the residual
                             "acceleration" stays below tau for 2 steps.
6. ``RDTConfig``            — single dataclass for all flags.
7. ``RDTLoop``              — the wrapper module, runs body for R steps with
                             all enabled features.

bf16-friendly. Both gpu_dense and triton_block backends accept this as
opaque; the inner body is re-compiled normally.

API
---
    >>> body = HybridBlock(d=256)
    >>> rdt  = RDTLoop(body, hidden=256, max_loops_train=4, max_loops_infer=8,
    ...                enable_loop_index_embed=True, enable_layer_scale=True,
    ...                enable_residual_gate_bias=True, enable_depth_lora=True)
    >>> y = rdt(x)                # train: R=4 (cfg.max_loops_train)
    >>> rdt.eval()
    >>> y_deep = rdt(x)           # infer: R=8 (cfg.max_loops_infer)
    >>> y_more = rdt(x, n_loops=16)  # explicit override
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


# ---------------------------------------------------------------------------
# 1. Loop-index sinusoidal embedding
# ---------------------------------------------------------------------------


class LoopIndexEmbedding(Module):
    """Sinusoidal loop-index embedding on the first ``hidden//frac`` channels.

    Zero learnable parameters. Provides "which-step-am-I" signal to the
    shared body block. Channel occupancy < 13% (default frac=8).
    """

    def __init__(
        self,
        hidden: int,
        max_loops: int = 32,
        frac: int = 8,
        theta: float = 10000.0,
    ):
        super().__init__()
        n_ch = hidden // frac
        if n_ch < 2:
            n_ch = 2  # need at least sin+cos pair
        # Pre-compute sinusoidal table (max_loops, n_ch).
        pos = torch.arange(max_loops, dtype=torch.float32).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(0, n_ch, 2, dtype=torch.float32) * (-math.log(theta) / n_ch)
        )
        pe = torch.zeros(max_loops, n_ch)
        pe[:, 0::2] = torch.sin(pos * inv_freq)
        pe[:, 1::2] = torch.cos(pos * inv_freq[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)
        self.n_ch = n_ch
        self.hidden = hidden

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """x: (..., hidden); step: int in [0, max_loops)."""
        step = min(step, self.pe.shape[0] - 1)
        bias = self.pe[step].to(x.dtype)  # (n_ch,)
        # Pad zeros after n_ch up to hidden, then broadcast-add.
        full = F.pad(bias, (0, self.hidden - self.n_ch))
        return x + full


# ---------------------------------------------------------------------------
# 2. LayerScale (per-channel γ)
# ---------------------------------------------------------------------------


class LayerScale(Module):
    """Per-channel γ scalar on the residual delta.

    init 1e-4 → at init each loop step contributes a tiny fraction of the
    body's delta, so the loop is near-identity. Training learns the right
    amplitude.
    """

    def __init__(self, hidden: int, init_scale: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_scale * torch.ones(hidden))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.gamma * delta


# ---------------------------------------------------------------------------
# 3. Residual gate bias (sigmoid gate, init ≈ 0.12)
# ---------------------------------------------------------------------------


class ResidualGateBias(Module):
    """σ(proj(x_in) + b) gate on the residual delta.

    b init -2 → σ(-2) ≈ 0.12 → near-identity at init. The gate can learn to
    open as depth increases.
    """

    def __init__(self, hidden: int, bias_init: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(self, x_in: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.proj(x_in))
        return g * delta


# ---------------------------------------------------------------------------
# 4. Depth LoRA adapter
# ---------------------------------------------------------------------------


class DepthLoRAAdapter(Module):
    """Per-loop low-rank delta with clamp-for-extrapolation.

    Shared A (D -> r) + per-loop scale embedding (max_loops, r) + Shared B
    (r -> D). At loop t, delta = B((A x) * scale[t_clamped]).

    During training t ∈ [0, max_t-1]; at inference t > max_t-1 we clamp
    (NOT wrap), so deeper inference loops keep using the deepest learned
    scale. ~5k params at hidden=256, rank=8.
    """

    def __init__(self, hidden: int, rank: int = 8, max_loops: int = 8):
        super().__init__()
        self.A = nn.Linear(hidden, rank, bias=False)
        self.scale = nn.Embedding(max_loops, rank)
        self.B = nn.Linear(rank, hidden, bias=False)
        self.max_t = int(max_loops)
        self.rank = int(rank)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.B.weight)  # init delta = 0

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        t_idx = min(int(loop_t), self.max_t - 1)
        idx = torch.tensor(t_idx, device=x.device, dtype=torch.long)
        s = self.scale(idx)  # (rank,)
        # Broadcast scale across all leading dims.
        return self.B(self.A(x) * s)


# ---------------------------------------------------------------------------
# 5. AccelExit (inference-only)
# ---------------------------------------------------------------------------


class AccelExit:
    """Acceleration-based early exit. O(d) overhead, no params.

    Tracks Δ_t = x_t - x_{t-1}. When ‖Δ_t - Δ_{t-1}‖ / (‖Δ_t‖+‖Δ_{t-1}‖+ε)
    < tau for ``streak_required`` consecutive steps → exit.
    """

    def __init__(self, tau: float = 1e-3, eps: float = 1e-6, streak_required: int = 2):
        self.tau = float(tau)
        self.eps = float(eps)
        self.streak_required = int(streak_required)
        self.reset()

    def reset(self) -> None:
        self.prev_delta: torch.Tensor | None = None
        self.prev_x: torch.Tensor | None = None
        self.streak: int = 0

    def update_and_should_exit(self, x_t: torch.Tensor) -> bool:
        """Call after each loop step with current hidden. True = should break."""
        if self.prev_x is None:
            self.prev_x = x_t.detach()
            return False
        delta = x_t.detach() - self.prev_x
        if self.prev_delta is None:
            self.prev_delta = delta
            self.prev_x = x_t.detach()
            return False
        num = (delta - self.prev_delta).norm()
        den = delta.norm() + self.prev_delta.norm() + self.eps
        if (num / den).item() < self.tau:
            self.streak += 1
        else:
            self.streak = 0
        self.prev_delta = delta
        self.prev_x = x_t.detach()
        return self.streak >= self.streak_required


# ---------------------------------------------------------------------------
# 6. Config + RDTLoop
# ---------------------------------------------------------------------------


@dataclass
class RDTConfig:
    """All RDT flags consolidated. Pass to ``RDTLoop`` ctor or kwargs."""
    enable_loop_index_embed: bool = True
    enable_layer_scale: bool = True
    layer_scale_init: float = 1e-4
    enable_residual_gate_bias: bool = True
    enable_accel_exit: bool = False
    accel_exit_tau: float = 1e-3
    enable_depth_lora: bool = True
    depth_lora_rank: int = 8
    max_loops_train: int = 4
    max_loops_infer: int = 8


class RDTLoop(Module):
    """Recurrent-Depth Transformer loop wrapper around any sf.Module body.

    The body is called R times (R = max_loops_train in training, max_loops_infer
    at eval, or override via ``forward(..., n_loops=R)``). Per-step features
    inject "which-step-am-I" signal so shared body weights can specialise.

    Body signature (any of)::
        body(x) -> Tensor
        body(x, *aux) -> Tensor
        body(x, *aux) -> (Tensor, *more)

    Aux is forwarded verbatim. Only the first tensor (``x``) gets the
    per-step adaptations.

    If the body is wrapped via ``attach_coe_to_block``, RDTLoop sets
    ``body._coe_step_t = t`` before each call so the CoE selects the
    right per-step router.
    """

    def __init__(
        self,
        body: nn.Module,
        hidden: int,
        max_loops_train: int = 4,
        max_loops_infer: int = 8,
        cfg: RDTConfig | None = None,
        **flags,
    ):
        super().__init__()
        self.body = body
        self.hidden = int(hidden)
        # Build cfg from kwargs unless explicit cfg passed.
        if cfg is None:
            cfg = RDTConfig(
                max_loops_train=max_loops_train,
                max_loops_infer=max_loops_infer,
            )
            for k, v in flags.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise TypeError(f"unknown RDT flag {k!r}")
        elif flags:
            raise TypeError("pass either cfg= or kwargs, not both")
        self.cfg = cfg

        max_loops = max(cfg.max_loops_train, cfg.max_loops_infer)

        self.loop_embed = (
            LoopIndexEmbedding(hidden, max_loops=max_loops)
            if cfg.enable_loop_index_embed else None
        )
        self.layer_scale = (
            LayerScale(hidden, init_scale=cfg.layer_scale_init)
            if cfg.enable_layer_scale else None
        )
        self.residual_gate = (
            ResidualGateBias(hidden) if cfg.enable_residual_gate_bias else None
        )
        self.depth_lora = (
            DepthLoRAAdapter(hidden, rank=cfg.depth_lora_rank, max_loops=max_loops)
            if cfg.enable_depth_lora else None
        )

    @staticmethod
    def _split_out(out):
        """Normalise body output to (Tensor, aux_tuple)."""
        if torch.is_tensor(out):
            return out, ()
        if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            return out[0], tuple(out[1:])
        raise TypeError(
            f"RDTLoop: body must return Tensor or tuple(Tensor, ...); got {type(out).__name__}"
        )

    def forward(
        self,
        x: torch.Tensor,
        *aux_in,
        n_loops: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run body for R loops with per-step adaptations.

        Returns the final feature tensor (matches body's first output).
        Auxiliary outputs from the body's last call are dropped to keep
        the API simple — for HybridBlocks needing to thread state across
        loops, capture them via a closure on the body itself.
        """
        if n_loops is None:
            n_loops = self.cfg.max_loops_train if self.training else self.cfg.max_loops_infer
        n_loops = max(1, int(n_loops))

        accel_exit = (
            AccelExit(tau=self.cfg.accel_exit_tau)
            if (self.cfg.enable_accel_exit and not self.training) else None
        )

        x_in_orig = x  # used by depth LoRA (each step takes original input ref)
        aux = aux_in

        for t in range(n_loops):
            x_before = x
            x_step_in = x

            # 1. Loop-index sinusoidal bias.
            if self.loop_embed is not None:
                x_step_in = self.loop_embed(x_step_in, step=t)

            # 2. CoE hint: tell the body which loop step (no-op if not CoE).
            if hasattr(self.body, "_coe_step_t"):
                self.body._coe_step_t = t

            # 3. Body call.
            out = self.body(x_step_in, *aux, **kwargs) if aux else self.body(x_step_in, **kwargs)
            x_new, aux_new = self._split_out(out)
            aux = aux_new

            # 4. Residual delta.
            delta = x_new - x_before

            # 5. Depth LoRA additive delta (per-loop scale, clamp at max_t).
            if self.depth_lora is not None:
                delta = delta + self.depth_lora(x_in_orig, loop_t=t)

            # 6. LayerScale γ.
            if self.layer_scale is not None:
                delta = self.layer_scale(delta)

            # 7. Residual gate bias.
            if self.residual_gate is not None:
                delta = self.residual_gate(x_before, delta)

            x = x_before + delta

            # 8. Acceleration-based early exit (infer only).
            if accel_exit is not None and accel_exit.update_and_should_exit(x):
                break

        return x


def scaled_residual_init(module: nn.Module, effective_depth: int) -> None:
    """std = 0.02 / sqrt(2 * effective_depth) on all Linear/Embedding.

    effective_depth = num_layers * loop_depth. Loop unroll is the residual
    path length, not just the layer count.
    """
    std = 0.02 / max(1.0, math.sqrt(2.0 * effective_depth))
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)


__all__ = [
    "RDTLoop",
    "RDTConfig",
    "LoopIndexEmbedding",
    "LayerScale",
    "ResidualGateBias",
    "DepthLoRAAdapter",
    "AccelExit",
    "scaled_residual_init",
]
