"""sf.LiquidCell — continuous-time CfC with Heinsen 2023 parallel scan.

Math (identical to mscfc.LiquidS4Cell so the correctness test stays bit-tight)
-----------------------------------------------------------------------------
    delta_t = softplus(W_delta x_t)         per-channel selective dt
    A_t     = exp(-delta_t * exp(A_log))    per-channel input-dep decay (0,1]
    B_t     = delta_t * (W_B x_t)
    h_t     = A_t * h_{t-1} + B_t

Closed-form parallel scan (Heinsen 2023, arXiv 2311.06281):
    S_t   = cumsum(log A_t, dim=T)
    inner = exp(-S) * B
    h_t   = exp(S_t) * cumsum(inner, dim=T) [+ exp(S_t) * h0]
    out   = tanh(h_t)

The fp32 promote inside the scan matches mscfc/liquid_s4.py exactly so
synapforge.LiquidCell vs mscfc.LiquidS4Cell match within fp32 noise.

API
---
    cell = sf.LiquidCell(in_dim, hidden_dim, init="hasani")
    h = cell(x)                # x: (B, T, in), h: (B, T, hidden)
    h = cell(x, h0=h_init)     # provide initial state
    h_step = cell.step(x_t, h) # per-token API for compat with old code

`init="hasani"` matches Hasani et al's weak-init convention (A in [0.5, 2.0])
also used by mscfc; `init="mamba"` uses Mamba's log(1..16) which decays harder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


class LiquidCell(Module):
    """Closed-form-scannable Liquid (CfC-S4) recurrence cell.

    Parameters
    ----------
    in_dim, hidden_dim
        Standard CfC dimensions.
    init
        ``"hasani"`` (default, A in [0.5, 2.0]) or ``"mamba"``
        (A in [1.0, 16.0]).
    bound
        If True, wrap the output in ``tanh``.
    weight_quant
        Quantization scheme for the input-projection weights
        (``delta_proj`` and ``b_proj`` only — the recurrent decay
        ``A_log`` is *never* quantized because per-channel tau is
        the most sensitive parameter in a CfC).

        * ``"none"`` (default): standard fp ``nn.Linear``, matches
          historical behaviour.
        * ``"ternary"``: BitNet b1.58 AbsMean ternary QAT
          (arXiv:2402.17764). Forward emits weights in
          ``{-gamma, 0, +gamma}`` per-tensor; backward uses STE
          (straight-through estimator). Compatible with bf16 autocast
          (``gamma`` stays in fp32). This is the M1 milestone of
          ``docs/MATMUL_FREE.md``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        init: str = "hasani",
        bound: bool = True,
        weight_quant: str = "none",
        rfold: bool = False,
        rfold_chunk: int = 16,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.bound = bool(bound)
        self.weight_quant = str(weight_quant)
        # 2026-05-02 / docs/MASTER_PLAN.md "math simplification" pack:
        # rfold replaces the per-token Python ``for t in range(T)`` loop
        # in ``forward`` with a chunked closed-form parallel scan
        # (synapforge.cells.rfold.liquid_rfold). Numerically equivalent
        # to the sequential reference up to fp32 round-off for chunk<=16
        # (verified by tests/cells/test_rfold_equivalence.py). Default
        # OFF -- opt-in via the ``--rfold`` flag in the trainer to keep
        # Run 6 byte-identical to the historical sequential path.
        self.rfold = bool(rfold)
        if rfold_chunk < 1:
            raise ValueError(
                f"rfold_chunk must be >= 1, got {rfold_chunk}"
            )
        self.rfold_chunk = int(rfold_chunk)
        if self.weight_quant not in ("none", "ternary"):
            raise ValueError(
                f"unknown weight_quant {weight_quant!r}; "
                f"use 'none' or 'ternary'"
            )

        if self.weight_quant == "ternary":
            # BitNet b1.58 AbsMean ternary QAT on the two input
            # projections only. Imported lazily so cells.liquid stays
            # importable even when synapforge.quantize fails to load
            # (e.g. minimal CI without the full package).
            from ..quantize import TernaryLinear
            self.delta_proj = TernaryLinear(in_dim, hidden_dim, bias=True)
            self.b_proj = TernaryLinear(in_dim, hidden_dim, bias=True)
        else:
            self.delta_proj = nn.Linear(in_dim, hidden_dim)
            self.b_proj = nn.Linear(in_dim, hidden_dim)

        if init == "hasani":
            A_init = torch.log(torch.linspace(0.5, 2.0, hidden_dim))
        elif init == "mamba":
            A_init = torch.log(torch.linspace(1.0, 16.0, hidden_dim))
        else:
            raise ValueError(f"unknown init {init!r}; use 'hasani' or 'mamba'")
        self.A_log = nn.Parameter(A_init)

        nn.init.constant_(self.delta_proj.bias, 0.0)
        nn.init.normal_(self.b_proj.weight, std=0.02)
        nn.init.zeros_(self.b_proj.bias)

    def get_decay_rate(self) -> torch.Tensor:
        return torch.exp(self.A_log)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """Sequence forward via Heinsen parallel scan.

        Args
            x:  (B, T, in_dim)
            h0: optional (B, hidden_dim)

        Returns
            h:  (B, T, hidden_dim) — tanh-bounded if self.bound
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, in_dim), got {tuple(x.shape)}")

        delta = F.softplus(self.delta_proj(x))     # (B, T, D), positive
        b_t = delta * self.b_proj(x)               # (B, T, D)
        A = self.get_decay_rate()                  # (D,)
        A_t = torch.exp(-delta * A)                # (B, T, D), in (0, 1]

        # Stable per-step recurrence in fp32. Heinsen scan overflows for
        # T>=128 because cumsum(log A) -> -inf; the cumprod-divide variant
        # has gradient explosion through 1/cumA. Direct h = A*h + b is
        # O(T*B*D) compute — same FLOPs, no divisions, autograd-safe.
        A_f = A_t.float()
        b_f = b_t.float()
        B, T, D = A_f.shape
        h_prev = (h0.float() if h0 is not None else
                  A_f.new_zeros(B, D))
        if self.rfold:
            # Math simplification path: ONE closed-form parallel scan per
            # chunk (default chunk=16) instead of T Python iterations. On
            # GPU this swaps T kernel launches for T/chunk launches, which
            # is the dominant cost at small per-step compute. See
            # synapforge.cells.rfold.liquid_rfold for the math.
            from .rfold import liquid_rfold
            h_full = liquid_rfold(
                A_f, b_f, h_prev, chunk=self.rfold_chunk,
            )
        else:
            h_chunks = []
            for t in range(T):
                h_prev = A_f[:, t] * h_prev + b_f[:, t]
                h_chunks.append(h_prev)
            h_full = torch.stack(h_chunks, dim=1)

        out = h_full.to(x.dtype)
        return torch.tanh(out) if self.bound else out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Per-token forward (calls forward with T=1). Slow path."""
        return self.forward(x_t.unsqueeze(1), h0=h).squeeze(1)


__all__ = ["LiquidCell"]
