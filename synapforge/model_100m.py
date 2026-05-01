"""sf.model_100m — 100M-param synapforge LM.

Architecture
------------
Token embed (50257, d) -> Position embed (256, d) ->
    [HybridBlock] x N
    -> RMSNorm -> tied LM head (vocab, d)

HybridBlock (sf primitives only):
    h_in = x
    for loop_step in range(loop_depth):  # RDT-style depth recurrence
        h = LiquidCell(LayerNorm(h_in))            # CfC parallel scan
        s, _ = PLIFCell.forward_seq(h)             # spike train, learnable tau/thr
        gated = SparseSynapse(s) * sigmoid(...)    # Hebbian-eligible
        h_in = h_in + dropout(gated)               # residual
        # FFN (SwiGLU)
        ff = w_down(silu(w_gate(LayerNorm(h_in))) * w_up(...))
        h_in = h_in + dropout(ff)
    return h_in

The block is applied with WEIGHT-SHARED loop_depth=1 default for speed (or 4 for used 4× per
forward, but only counted 1× — this is the RDT trick). All weights are
typed as sf.Param with grad_source=["bp"] for the synapse and ["bp"] for
the rest; flipping a synapse to ["bp", "hebb"] enables plasticity merge.

Sizing
------
hidden=512, layers=10, ffn_ratio=8 -> ~97M params (within target +/- 10%).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells.liquid import LiquidCell
from .cells.synapse import SparseSynapse
from .module import Module
from .surrogate import PLIFCell


def _swiglu_ffn(d: int, ratio: float) -> nn.Module:
    """SwiGLU FFN: silu(W_gate x) * W_up x -> W_down. 3 matrices."""
    h = int(d * ratio)
    return _SwiGLU(d, h)


class _RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, affine scale only)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _SwiGLU(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class HybridBlock(Module):
    """LiquidCell + PLIFCell + SparseSynapse + SwiGLU FFN, RDT-loopable."""

    def __init__(
        self,
        d: int,
        ffn_ratio: float = 8.0,
        sparsity: float = 0.95,  # density of the synapse mask
        dropout: float = 0.0,
        plif_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.d = int(d)

        self.ln1 = _RMSNorm(d)
        self.liquid = LiquidCell(d, d, init="hasani")
        self.plif = PLIFCell(d, tau_init=1.5, threshold_init=plif_threshold,
                             surrogate="atan", reset="subtract")
        self.synapse = SparseSynapse(d, d, sparsity=sparsity, bias=False)
        # Tag synapse weight for plasticity merge (off by default).
        self.synapse.weight._sf_grad_source = ["bp"]
        # Smooth gate so that even with sparse mask we can learn channel gain.
        self.gate = nn.Linear(d, d, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.01)

        self.ln2 = _RMSNorm(d)
        self.ffn = _swiglu_ffn(d, ffn_ratio)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- liquid + spike + sparse synapse ---- (residual #1)
        a = self.ln1(x)
        h = self.liquid(a)              # (B, T, d)
        s, _ = self.plif.forward_seq(h)  # (B, T, d) spikes in {0,1}
        gated = self.synapse(s) * torch.sigmoid(self.gate(s))
        x = x + self.drop(gated)

        # ---- SwiGLU FFN ---- (residual #2)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class SynapForge100M(Module):
    """100M synapforge LM. Stack of HybridBlocks with optional RDT depth-loop."""

    def __init__(
        self,
        vocab: int = 50257,
        d: int = 512,
        n_layers: int = 10,
        loop_depth: int = 4,
        max_seq: int = 256,
        ffn_ratio: float = 8.0,
        sparsity: float = 0.95,
        dropout: float = 0.0,
        tie_lm_head: bool = True,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.vocab = int(vocab)
        self.n_layers = int(n_layers)
        self.loop_depth = int(loop_depth)
        self.max_seq = int(max_seq)
        self.tie_lm_head = bool(tie_lm_head)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)

        self.tok_embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            HybridBlock(d, ffn_ratio=ffn_ratio, sparsity=sparsity,
                        dropout=dropout)
            for _ in range(n_layers)
        )
        self.ln_f = _RMSNorm(d)
        if tie_lm_head:
            self.lm_head = None  # tied to tok_embed.weight
        else:
            self.lm_head = nn.Linear(d, vocab, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

    @torch.no_grad()
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not only_trainable)
        )

    def _run_blocks(self, x):
        if self.use_grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint as _ckpt
            for blk in self.blocks:
                for _ in range(self.loop_depth):
                    x = _ckpt(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                for _ in range(self.loop_depth):
                    x = blk(x)
        return x

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> (B, T, d) hidden after backbone (post ln_f)."""
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T), got {tuple(ids.shape)}")
        B, T = ids.shape
        if T > self.max_seq:
            raise ValueError(f"seq_len {T} > max_seq {self.max_seq}")
        x = self.tok_embed(ids) + self.pos_embed[:T].unsqueeze(0)
        x = self._run_blocks(x)
        return self.ln_f(x)

    def forward_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Skip tok_embed; feed pre-embedded (B, T, d). Returns hidden post ln_f.

        Used by multi-modal callers (sf.modal.UnifiedEmbed already produced z).
        Position embed truncated to first T positions and added.
        """
        if z.dim() != 3:
            raise ValueError(f"expected (B, T, d), got {tuple(z.shape)}")
        B, T, D = z.shape
        if D != self.d:
            raise ValueError(f"hidden mismatch: z d={D} vs backbone d={self.d}")
        if T > self.max_seq:
            # extend pos_embed dynamically (rare)
            extra = T - self.max_seq
            extra_pos = torch.zeros(extra, D, device=z.device, dtype=z.dtype)
            pos = torch.cat([self.pos_embed.to(z.dtype), extra_pos], dim=0)[:T]
        else:
            pos = self.pos_embed[:T].to(z.dtype)
        x = z + pos.unsqueeze(0)
        x = self._run_blocks(x)
        return self.ln_f(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) int64. Returns logits (B, T, vocab)."""
        x = self.encode(ids)
        if self.tie_lm_head:
            logits = F.linear(x, self.tok_embed.weight)
        else:
            logits = self.lm_head(x)
        return logits


def build_synapforge_100m(
    vocab: int = 50257,
    d: int = 512,
    n_layers: int = 10,
    loop_depth: int = 4,
    max_seq: int = 256,
    ffn_ratio: float = 8.0,
    sparsity: float = 0.95,
    dropout: float = 0.0,
    use_grad_checkpoint: bool = False,
) -> SynapForge100M:
    return SynapForge100M(
        vocab=vocab, d=d, n_layers=n_layers, loop_depth=loop_depth,
        use_grad_checkpoint=use_grad_checkpoint,
        max_seq=max_seq, ffn_ratio=ffn_ratio, sparsity=sparsity,
        dropout=dropout,
    )


__all__ = ["SynapForge100M", "HybridBlock", "build_synapforge_100m"]
