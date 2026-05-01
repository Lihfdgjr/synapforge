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
        plif_threshold: float = 0.05,  # was 0.3 — too high for tanh-bounded
        # CfC inputs gated by (1-decay)~=0.49 in PLIFCell; eff input ceiling
        # ~=0.49*1.0=0.49 but most channels at init have |x|<0.3, so
        # threshold=0.3 -> nearly zero spikes (dead=N/N). Lowered to 0.05
        # to match LearnableThreshold default and let learning bootstrap.
        weight_quant: str = "none",
        # T2.8 / MATMUL_FREE.md M1 — when "ternary", the LiquidCell input
        # projections (delta_proj, b_proj) are BitNet b1.58 AbsMean
        # ternary-QAT layers. ALL OTHER weights (PLIF tau/threshold,
        # SparseSynapse, gate, FFN, RMSNorm) stay fp/bf16. Default
        # "none" matches the historical fp baseline.
    ) -> None:
        super().__init__()
        self.d = int(d)

        self.ln1 = _RMSNorm(d)
        self.liquid = LiquidCell(d, d, init="hasani",
                                 weight_quant=weight_quant)
        # tau_init=2.5 (was 1.5): 1-decay = 0.33 (was 0.49), so per-step
        # input drive is smaller but membrane integrates over more steps,
        # which is the textbook LIF behavior (Fang 2021 uses tau~2-4).
        self.plif = PLIFCell(d, tau_init=2.5, threshold_init=plif_threshold,
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


#: Effective Qwen 2.5 token id range. The tokenizer/teacher emits IDs in
#: ``[0, QWEN25_LIVE_VOCAB)``; rows ``[QWEN25_LIVE_VOCAB, vocab)`` are
#: random-init padding that never see real gradient yet still drift under
#: optimizer noise (Adam moments + weight decay). We freeze them by zeroing
#: their gradients in a backward hook (P26 / DEEP_MAINT_QUEUE.md T2.4).
QWEN25_LIVE_VOCAB = 151643


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
        freeze_vocab_tail: bool = True,
        live_vocab: int = QWEN25_LIVE_VOCAB,
        lm_head_spectral_norm: bool = False,
        weight_quant_cfc: str = "none",
        # T2.8 / MATMUL_FREE.md M1 — when "ternary" the LiquidCell input
        # projections inside every HybridBlock are BitNet b1.58 ternary
        # QAT layers. PLIF / Synapse / FFN / LM head untouched. Default
        # "none" preserves the historical fp baseline.
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.vocab = int(vocab)
        self.n_layers = int(n_layers)
        self.loop_depth = int(loop_depth)
        self.max_seq = int(max_seq)
        self.tie_lm_head = bool(tie_lm_head)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self.freeze_vocab_tail = bool(freeze_vocab_tail)
        self.live_vocab = int(live_vocab)
        self.lm_head_spectral_norm = bool(lm_head_spectral_norm)
        self.weight_quant_cfc = str(weight_quant_cfc)
        if self.weight_quant_cfc not in ("none", "ternary"):
            raise ValueError(
                f"weight_quant_cfc must be 'none' or 'ternary', "
                f"got {weight_quant_cfc!r}"
            )

        self.tok_embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            HybridBlock(d, ffn_ratio=ffn_ratio, sparsity=sparsity,
                        dropout=dropout,
                        weight_quant=self.weight_quant_cfc)
            for _ in range(n_layers)
        )
        self.ln_f = _RMSNorm(d)
        if tie_lm_head:
            self.lm_head = None  # tied to tok_embed.weight
        else:
            self.lm_head = nn.Linear(d, vocab, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

        # T2.4 — freeze the rows [live_vocab, vocab) of tok_embed.weight (and
        # lm_head.weight when untied) by zeroing their gradient slice on the
        # backward pass. Qwen 2.5 emits IDs in [0, 151643); rows beyond that
        # are random-init padding that never see real gradient through the
        # forward path, but Adam + weight decay would still push noise into
        # them. The backward hook drops them to zero so the optimizer step
        # leaves them untouched.
        if self.freeze_vocab_tail and 0 < self.live_vocab < self.vocab:
            tail_start = self.live_vocab

            def _zero_tail_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                # grad is a fresh tensor returned to autograd; mutating it
                # in-place is allowed and avoids an extra alloc.
                grad[tail_start:] = 0
                return grad

            self.tok_embed.weight.register_hook(_zero_tail_grad_hook)
            if self.lm_head is not None:
                self.lm_head.weight.register_hook(_zero_tail_grad_hook)

        # ---- T2.6 — optional spectral norm on LM head (P28 z-loss drift) ----
        # Bound the Lipschitz constant of the LM projection so the partition
        # function (logsumexp over vocab) cannot grow unboundedly during long
        # training. Default OFF: opt-in, since bf16 + spectral_norm has known
        # quirks (the power-iteration buffers ``weight_u``/``weight_v`` are
        # tracked separately and the parametrisation reparameterises ``weight``
        # as ``weight_orig / sigma`` per forward, where ``sigma`` is the top
        # singular value estimated by power iteration).
        #
        # Note on tied embeddings: when ``tie_lm_head=True`` we have no
        # separate ``lm_head`` module — ``forward()`` calls
        # ``F.linear(x, self.tok_embed.weight)``. Applying ``spectral_norm``
        # to ``tok_embed`` reparameterises ``self.tok_embed.weight`` via the
        # same power-iteration recipe, so the LM head logits are bounded
        # automatically (and the ``tok_embed(ids)`` lookup also goes through
        # the bounded weight, which is the desired symmetric behaviour for a
        # tied model). When ``tie_lm_head=False`` we wrap ``lm_head``
        # directly.
        if self.lm_head_spectral_norm:
            if self.tie_lm_head:
                nn.utils.spectral_norm(self.tok_embed, name="weight")
            else:
                nn.utils.spectral_norm(self.lm_head, name="weight")

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
    freeze_vocab_tail: bool = True,
    live_vocab: int = QWEN25_LIVE_VOCAB,
    lm_head_spectral_norm: bool = False,
    weight_quant_cfc: str = "none",
) -> SynapForge100M:
    return SynapForge100M(
        vocab=vocab, d=d, n_layers=n_layers, loop_depth=loop_depth,
        use_grad_checkpoint=use_grad_checkpoint,
        max_seq=max_seq, ffn_ratio=ffn_ratio, sparsity=sparsity,
        dropout=dropout,
        freeze_vocab_tail=freeze_vocab_tail,
        live_vocab=live_vocab,
        lm_head_spectral_norm=lm_head_spectral_norm,
        weight_quant_cfc=weight_quant_cfc,
    )


__all__ = [
    "SynapForge100M",
    "HybridBlock",
    "build_synapforge_100m",
    "QWEN25_LIVE_VOCAB",
]
