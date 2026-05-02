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
# Phase 2 of the torch-replacement roadmap (docs/TORCH_REPLACEMENT_PLAN.md):
# `synapforge.module.Module` is the canonical base class for every block in
# this file, and `synapforge.module.Parameter` replaces ad-hoc
# `nn.Parameter` constructions for the parameters owned by our own classes
# (RMSNorm, SwiGLU, SynapForge100M.pos_embed, optional hp_lambda). The
# `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` modules from torch keep
# their own internal `nn.Parameter` until Phase 3 introduces drop-in
# replacements (`synapforge.module.Linear`, etc).
from .module import Module, Parameter
from .surrogate import PLIFCell
from .thinking.coconut import LatentThinker


def _swiglu_ffn(d: int, ratio: float) -> Module:
    """SwiGLU FFN: silu(W_gate x) * W_up x -> W_down. 3 matrices.

    Returns a :class:`synapforge.module.Module` (Phase 2 — was
    ``nn.Module`` pre-Phase-2). ``nn.Module`` callers continue to
    work because ``synapforge.module.Module`` is a subclass.
    """
    h = int(d * ratio)
    return _SwiGLU(d, h)


class _RMSNorm(Module):
    """Root-mean-square layer norm (no bias, affine scale only).

    Phase 2 base-class swap: was ``nn.Module``, now
    ``synapforge.module.Module``. Bit-exact identical behaviour;
    ``state_dict`` keys (``weight``) unchanged. The ``weight``
    parameter uses :class:`synapforge.module.Parameter` for the same
    bit-exact-but-typed-correctly reason.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _SwiGLU(Module):
    """SwiGLU FFN: ``silu(W_gate x) * W_up x -> W_down``.

    Phase 2 base-class swap: was ``nn.Module``, now
    ``synapforge.module.Module``. ``nn.Linear`` submodules are kept
    as-is — they bring their own ``nn.Parameter`` weights, which our
    ``Module`` parameter-tracking sees through transparently because
    ``Module`` extends ``nn.Module``. Phase 3 of the torch-replacement
    plan introduces ``synapforge.module.Linear``; until then the
    Linear layers stay on torch.
    """

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
        plif_tau_init: "float | str" = 2.5,
        # NeurIPS 2025 (Fang et al., arXiv:2505.18608) A3 -- tri-modal
        # tau init for PLIF.  Pass "trimodal" to split tau channels into
        # ~30% short (0.5), 40% mid (2.0), 30% long (8.0).  Default 2.5
        # preserves the historical uniform-warm tau init bit-for-bit.
        high_pass_residual_weight: float = 0.0,
        # NeurIPS 2025 (Fang et al., arXiv:2505.18608) A2 -- frequency-
        # balancing residual.  When > 0 the block forward becomes
        #     out = block(x) + lambda * (x - LowPass(x))
        # The (x - LowPass(x)) term is a high-pass filter that bypasses
        # both the CfC and PLIF low-pass stages.  ``lambda`` is per-channel
        # learnable (init = scalar passed); LowPass is a depth-wise
        # causal Conv1d (kernel = high_pass_kernel_size).  Default 0.0
        # keeps the legacy code path bit-for-bit.
        high_pass_kernel_size: int = 3,
        # Run 5 PLIF-dead fix #3: SEW (Spike-Element-Wise) shortcut from
        # arXiv:2102.04159.  When True, the spike branch becomes
        #     gated = (synapse(s) + h_pre_plif) * sigmoid(gate(s))
        # i.e. the LiquidCell output ``h`` is added directly to the
        # synapse output, providing a non-zero LM-gradient path even
        # when ``s_t == 0``.  This breaks the dead-PLIF positive
        # feedback loop that collapses LiquidCell weights under weight
        # decay.  Default False keeps Run 5 behaviour bit-identical.
        sew_shortcut: bool = False,
        # 2026-05-02 sparse-spike pack -- exploit the SNN-architecture-
        # unique fact that ``s`` is a *binary sparse* tensor.  When the
        # PLIF spike rate is below ~30%, computing ``synapse(s + h)``
        # via the embedding-bag-style row-gather is asymptotically
        # cheaper than the dense GEMM (O(K * out_dim) vs O(d^2)).
        # See ``synapforge.kernels.sparse_spike_matmul``.  The dispatch
        # auto-falls-back to dense when measured spike density exceeds
        # ``sparse_spike_threshold`` so this flag is safe to leave on
        # even when PLIF is dead/saturated -- in those regimes you just
        # get the dense path back.  Default False = legacy synapse.
        sparse_spike_synapse: bool = False,
        sparse_spike_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.d = int(d)

        self.ln1 = _RMSNorm(d)
        self.liquid = LiquidCell(d, d, init="hasani",
                                 weight_quant=weight_quant)
        # tau_init=2.5 (was 1.5): 1-decay = 0.33 (was 0.49), so per-step
        # input drive is smaller but membrane integrates over more steps,
        # which is the textbook LIF behavior (Fang 2021 uses tau~2-4).
        # When plif_tau_init == "trimodal" the PLIFCell ctor builds a
        # heterogeneous (short/mid/long) tau split per A3.
        self.plif = PLIFCell(d, tau_init=plif_tau_init,
                             threshold_init=plif_threshold,
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

        # ---- A2: optional high-pass residual (NeurIPS 2025 Fang et al.) ----
        # When high_pass_residual_weight == 0.0 this branch is fully off (no
        # extra parameters, no extra modules) and forward() takes the legacy
        # path exactly.  When > 0 we instantiate:
        #   - hp_lowpass: depth-wise causal Conv1d, kernel=k, init = avg.
        #   - hp_lambda:  per-channel learnable scale, init to the scalar.
        self.high_pass_residual_weight = float(high_pass_residual_weight)
        self.high_pass_kernel_size = int(high_pass_kernel_size)
        # Run 5 PLIF-dead fix #3 -- SEW (Spike-Element-Wise) shortcut.
        self.sew_shortcut = bool(sew_shortcut)
        # 2026-05-02 sparse-spike pack -- exploit binary-sparse spikes.
        self.sparse_spike_synapse = bool(sparse_spike_synapse)
        self.sparse_spike_threshold = float(sparse_spike_threshold)
        if not 0.0 <= self.sparse_spike_threshold <= 1.0:
            raise ValueError(
                f"sparse_spike_threshold must be in [0, 1], "
                f"got {sparse_spike_threshold}"
            )
        if self.high_pass_residual_weight != 0.0:
            if self.high_pass_kernel_size < 1:
                raise ValueError(
                    f"high_pass_kernel_size must be >=1; "
                    f"got {high_pass_kernel_size}"
                )
            k = self.high_pass_kernel_size
            self.hp_lowpass = nn.Conv1d(
                in_channels=self.d, out_channels=self.d,
                kernel_size=k, groups=self.d,
                bias=True, padding=0,
            )
            with torch.no_grad():
                self.hp_lowpass.weight.fill_(1.0 / float(k))
                self.hp_lowpass.bias.zero_()
            # Phase 2: use synapforge.module.Parameter (subclass of
            # nn.Parameter; state-dict bit-equivalent).
            self.hp_lambda = Parameter(
                torch.full((self.d,), float(high_pass_residual_weight))
            )
        else:
            self.hp_lowpass = None
            self.hp_lambda = None

    def _high_pass_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute lambda*(x - LowPass(x)).  x: (B, T, d)."""
        if self.hp_lowpass is None or self.hp_lambda is None:
            return torch.zeros_like(x)
        # (B, T, d) -> (B, d, T) for Conv1d.
        xt = x.transpose(1, 2)
        # Causal padding on the LEFT (k-1 zeros) so output length == T.
        pad_left = self.high_pass_kernel_size - 1
        if pad_left > 0:
            xt_padded = F.pad(xt, (pad_left, 0))
        else:
            xt_padded = xt
        lp = self.hp_lowpass(xt_padded.to(self.hp_lowpass.weight.dtype))
        # (B, d, T) -> (B, T, d).  hp_lambda broadcasts across (B, T).
        hp = (xt - lp.to(xt.dtype)).transpose(1, 2)
        lam = self.hp_lambda.to(hp.dtype)
        return hp * lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache the input for the optional high-pass residual *before*
        # the block transforms it (Max-Former: x_hp computed off the
        # BLOCK INPUT, not the post-FFN view).
        x_in = x

        # ---- liquid + spike + sparse synapse ---- (residual #1)
        a = self.ln1(x)
        h = self.liquid(a)              # (B, T, d)
        s, _ = self.plif.forward_seq(h)  # (B, T, d) spikes in {0,1}
        if self.sew_shortcut:
            # arXiv:2102.04159 SEW residual: bypass the spike branch with
            # the LiquidCell output, ensuring the LM gradient still flows
            # to ``self.liquid`` even when ``s == 0`` everywhere.  The
            # binary spike still drives the gate-multiplicative branch
            # so spike-rate stats remain meaningful.  See
            # docs/PLIF_DEAD_DIAGNOSIS.md fix #3.
            spike_input = s + h
        else:
            spike_input = s
        # Synapse path: dense (default) or sparse-spike-aware (opt-in).
        # The sparse-spike kernel exploits the fact that ``s`` is
        # binary AND sparse: instead of casting to fp and running a
        # dense GEMM ``(s + h) @ W.T`` of cost ``O(d^2)``, it
        # computes the spike contribution as a row-gather of
        # ``W[:, k]`` for active spikes (cost ``O(K * out_dim)``,
        # where ``K`` = active spike count per token, typically 5-15%
        # of d for a healthy PLIF).  Auto-falls-back to the dense
        # path when measured density >= ``sparse_spike_threshold`` so
        # this branch is safe on dead/saturated PLIF too.
        # Gate path stays dense (small linear; sparsity wins are on
        # the *synapse* matmul which is the larger and more expensive op).
        if self.sparse_spike_synapse:
            from .kernels.sparse_spike_matmul import sparse_spike_linear
            if self.sew_shortcut:
                # spike_input == s + h; pass s and h separately so the
                # kernel can split the dense ``h @ W.T`` from the sparse
                # spike row-gather contribution.
                synapse_out = sparse_spike_linear(
                    s, h, self.synapse,
                    density_threshold=self.sparse_spike_threshold,
                )
            else:
                # spike_input == s; ``h`` contribution is zero in the
                # synapse-side accounting.  We pass an explicit zeros
                # tensor of h's dtype so the kernel still has a clean
                # reference for its dense bypass GEMM (which collapses
                # to zero when h=0 -- one tiny cuBLAS launch -- so the
                # win is preserved).  Density auto-dispatch tracks ``s``.
                zero_h = torch.zeros_like(s, dtype=h.dtype)
                synapse_out = sparse_spike_linear(
                    s, zero_h, self.synapse,
                    density_threshold=self.sparse_spike_threshold,
                )
        else:
            synapse_out = self.synapse(spike_input)
        gated = synapse_out * torch.sigmoid(self.gate(spike_input))
        x = x + self.drop(gated)

        # ---- SwiGLU FFN ---- (residual #2)
        x = x + self.drop(self.ffn(self.ln2(x)))

        # ---- A2: optional high-pass residual (NeurIPS 2025 §3.2) ----
        if self.hp_lowpass is not None:
            x = x + self._high_pass_residual(x_in)
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
        lm_head_pre_ln: bool = False,
        # T7.3 / P28 primary plan — insert ``nn.LayerNorm(d,
        # elementwise_affine=False)`` immediately BEFORE the final
        # ``lm_head`` projection. The existing ``self.ln_f`` is RMSNorm
        # with an *affine scale parameter* that the optimizer can grow
        # without bound; once it drifts large, post-``ln_f`` hidden
        # magnitudes blow up, logits blow up, and the z-loss term
        # ``log Z = logsumexp(logits)`` drifts linearly upward across
        # training (observed: Run 3b z_loss step 0 -> step 5000 trended
        # +linear despite top-K=2048 sparse penalty in `4d0d2a9`). A
        # parameter-free LayerNorm applied *after* ln_f re-centers and
        # re-scales every row to a fixed norm (mean 0, var 1, no learnable
        # scale), so the lm_head sees a tightly bounded input regardless
        # of how the affine RMSNorm weight evolved. This is the primary
        # P28 plan from docs/TRAINING_ISSUES_RETROSPECTIVE.md §2.d
        # ("nn.LayerNorm(d, elementwise_affine=False) immediately before
        # the final lm_head projection") -- T2.6 spectral_norm is the
        # secondary, opt-in patch (#4 in §3 of the same doc). Default
        # OFF: opt-in via ``--lm-head-pre-ln``, identical baseline path
        # when False (zero new params, zero new modules).
        weight_quant_cfc: str = "none",
        # T2.8 / MATMUL_FREE.md M1 — when "ternary" the LiquidCell input
        # projections inside every HybridBlock are BitNet b1.58 ternary
        # QAT layers. PLIF / Synapse / FFN / LM head untouched. Default
        # "none" preserves the historical fp baseline.
        plif_tau_init: "float | str" = 2.5,
        # NeurIPS 2025 (Fang et al., arXiv:2505.18608) A3 -- "trimodal"
        # splits PLIF tau channels into short (~30%), mid (~40%), long
        # (~30%).  Default 2.5 = historical uniform-warm tau init.
        high_pass_residual_weight: float = 0.0,
        # NeurIPS 2025 (Fang et al., arXiv:2505.18608) A2 -- per-block
        # high-pass residual.  > 0 enables x_hp = lambda * (x -
        # LowPass(x)) at the end of each HybridBlock.  Default 0.0 keeps
        # the legacy code path bit-for-bit.
        latent_k: int = 0,
        # T2.9 / arxiv:2412.06769 — Coconut latent thinking budget. When
        # ``latent_k > 0``, ``encode()`` runs K extra forward passes after
        # the normal stack, feeding the last-token hidden back as a
        # continuous (B, 1, d) input (no token sampling, no embed lookup).
        # The post-thinking hidden replaces the last position. Default 0
        # disables latent thinking entirely (zero-overhead, identity
        # behaviour vs the pre-T2.9 baseline).
        sew_shortcut: bool = False,
        # Run 5 PLIF-dead fix #3 -- SEW (Spike-Element-Wise) shortcut
        # in every HybridBlock.  Default False keeps the Run 5 code path
        # bit-identical.  See HybridBlock ctor for full rationale.
        sparse_spike_synapse: bool = False,
        sparse_spike_threshold: float = 0.30,
        # 2026-05-02 sparse-spike pack -- exploit binary-sparse spikes
        # in every HybridBlock synapse path.  Default False = legacy
        # dense GEMM.  Auto-falls-back to dense when measured spike
        # density >= threshold (default 30%).
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
        self.lm_head_pre_ln = bool(lm_head_pre_ln)
        self.weight_quant_cfc = str(weight_quant_cfc)
        if self.weight_quant_cfc not in ("none", "ternary"):
            raise ValueError(
                f"weight_quant_cfc must be 'none' or 'ternary', "
                f"got {weight_quant_cfc!r}"
            )
        self.plif_tau_init = plif_tau_init
        self.high_pass_residual_weight = float(high_pass_residual_weight)
        self.latent_k = int(latent_k)
        self.sew_shortcut = bool(sew_shortcut)
        self.sparse_spike_synapse = bool(sparse_spike_synapse)
        self.sparse_spike_threshold = float(sparse_spike_threshold)
        if self.latent_k < 0:
            raise ValueError(f"latent_k must be >= 0, got {latent_k}")
        if not 0.0 <= self.sparse_spike_threshold <= 1.0:
            raise ValueError(
                f"sparse_spike_threshold must be in [0, 1], "
                f"got {sparse_spike_threshold}"
            )

        self.tok_embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        # Phase 2: use synapforge.module.Parameter for the
        # block-owned positional embedding. ``nn.Embedding`` keeps its
        # own ``nn.Parameter`` weights for now — they're equivalent
        # under our state-dict contract (Phase 3 introduces
        # ``synapforge.module.Embedding``).
        self.pos_embed = Parameter(torch.zeros(max_seq, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            HybridBlock(d, ffn_ratio=ffn_ratio, sparsity=sparsity,
                        dropout=dropout,
                        weight_quant=self.weight_quant_cfc,
                        plif_tau_init=self.plif_tau_init,
                        high_pass_residual_weight=self.high_pass_residual_weight,
                        sew_shortcut=self.sew_shortcut,
                        sparse_spike_synapse=self.sparse_spike_synapse,
                        sparse_spike_threshold=self.sparse_spike_threshold)
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

        # ---- T7.3 / P28 primary — pre-LM-head affine-free LayerNorm ----
        # See ctor docstring above for full rationale. Module exists only
        # when the flag is on so `state_dict()` is bit-identical to the
        # baseline checkpoint format when off (no new keys, hence no
        # warmstart `missing/unexpected` warnings via P12 strict-load).
        # ``elementwise_affine=False`` -> no learnable gamma/beta -> no
        # parameters added; the layer is pure functional centering +
        # scaling. Spec: out_i = (x_i - mean(x)) / sqrt(var(x) + eps),
        # so per-token L2 norm is exactly sqrt(d) up to eps -- the
        # exact bound we want before the LM projection.
        if self.lm_head_pre_ln:
            self.lm_head_pre_ln_module: Optional[nn.LayerNorm] = nn.LayerNorm(
                self.d, elementwise_affine=False, eps=1e-5
            )
        else:
            self.lm_head_pre_ln_module = None

        # ---- T2.9 — Coconut latent thinker (default off) ----
        # We only build the thinking projections when latent_k>0 so the
        # zero-budget baseline path is bit-identical to pre-T2.9 (no extra
        # parameters, no extra modules in state_dict). When enabled, the
        # thinker holds two linear (d,d) projections initialised to identity
        # so that at step 0 it is a no-op refinement; gradient updates make
        # it learn a useful "thinking transform". The HybridBlock stack is
        # reused as the latent step kernel.
        if self.latent_k > 0:
            self.latent_thinker: Optional[LatentThinker] = LatentThinker(
                hidden=self.d, thinking_proj=True
            )
        else:
            self.latent_thinker = None

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

    def _latent_block_stack(
        self, x: torch.Tensor, _state: Optional[torch.Tensor] = None
    ) -> tuple:
        """Adapter so ``LatentThinker.think_step`` can drive the block stack.

        ``LatentThinker`` expects a callable taking ``(x, state)`` and
        returning ``(out, new_state)``. Our blocks are stateless w.r.t. the
        latent loop (the CfC parallel scan is rerun each step); we just pass
        ``state`` through unchanged.
        """
        return self._run_blocks(x), _state

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> (B, T, d) hidden after backbone (post ln_f).

        When ``self.latent_k > 0`` (T2.9 / arxiv:2412.06769) the last-token
        hidden is then refined by ``latent_k`` extra continuous-thought
        passes (no token sampling) before ``ln_f``.
        """
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T), got {tuple(ids.shape)}")
        B, T = ids.shape
        if T > self.max_seq:
            raise ValueError(f"seq_len {T} > max_seq {self.max_seq}")
        x = self.tok_embed(ids) + self.pos_embed[:T].unsqueeze(0)
        x = self._run_blocks(x)

        # ---- T2.9 — Coconut latent thinking ----
        if self.latent_thinker is not None and self.latent_k > 0:
            # Take the LAST-position hidden as "prefix" for thinking.
            prefix = x[:, -1, :]  # (B, d)
            think_h, _ = self.latent_thinker.think(
                self._latent_block_stack,
                prefix_hidden=prefix,
                k=self.latent_k,
            )
            # Splice the post-think hidden back into the last position.
            x = torch.cat([x[:, :-1, :], think_h.unsqueeze(1)], dim=1)

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
        # T7.3 / P28 — bound the LM-head input norm, regardless of how
        # the affine RMSNorm scale parameter inside ``ln_f`` may have
        # drifted. Affine-free LN re-projects every row onto a fixed
        # ``sqrt(d)``-norm sphere (mean 0, var 1), capping the operator
        # input. The lm_head Lipschitz constant alone (T2.6 spectral_norm)
        # is the orthogonal bound on the OPERATOR; this is the bound on
        # the INPUT. Both can be on simultaneously -- they compose.
        if self.lm_head_pre_ln_module is not None:
            x = self.lm_head_pre_ln_module(x)
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
    lm_head_pre_ln: bool = False,
    weight_quant_cfc: str = "none",
    plif_tau_init: "float | str" = 2.5,
    high_pass_residual_weight: float = 0.0,
    latent_k: int = 0,
    sew_shortcut: bool = False,
    sparse_spike_synapse: bool = False,
    sparse_spike_threshold: float = 0.30,
) -> SynapForge100M:
    return SynapForge100M(
        vocab=vocab, d=d, n_layers=n_layers, loop_depth=loop_depth,
        use_grad_checkpoint=use_grad_checkpoint,
        max_seq=max_seq, ffn_ratio=ffn_ratio, sparsity=sparsity,
        dropout=dropout,
        freeze_vocab_tail=freeze_vocab_tail,
        live_vocab=live_vocab,
        lm_head_spectral_norm=lm_head_spectral_norm,
        lm_head_pre_ln=lm_head_pre_ln,
        weight_quant_cfc=weight_quant_cfc,
        plif_tau_init=plif_tau_init,
        high_pass_residual_weight=high_pass_residual_weight,
        latent_k=latent_k,
        sew_shortcut=sew_shortcut,
        sparse_spike_synapse=sparse_spike_synapse,
        sparse_spike_threshold=sparse_spike_threshold,
    )


__all__ = [
    "SynapForge100M",
    "HybridBlock",
    "build_synapforge_100m",
    "QWEN25_LIVE_VOCAB",
]
