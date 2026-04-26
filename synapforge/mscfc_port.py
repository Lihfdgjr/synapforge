"""mscfc_port — port mscfc.MSpikingCfCLoop architecture to synapforge primitives.

Target: 100M-param language model built EXCLUSIVELY from sf.* primitives,
ready for end-to-end training on WikiText-103 with multi-source plasticity.

Architecture (10 layers, hidden=512, vocab=50257 GPT-2):
    nn.Embedding(50257, 512)                        # 25.7M
    -- per layer (10x) ----------------------------------------- ~7.4M each
    LayerNorm(512)
    sf.LiquidCell(512, 512)                         # ~1.0M (Heinsen scan CfC)
    sf.PLIFCell(512, surrogate='atan')              # ~1k (parametric LIF)
    sf.SparseSynapse(512, 512, sparsity=0.30)       # ~0.26M (bilinear gate)
    LayerNorm(512)
    nn.Linear(512, 6144) + GELU + nn.Linear(6144, 512)  # MLP 6.3M
    -- final ---------------------------------------------------------------
    LayerNorm(512)
    lm_head = tied to embedding                     # 0M (weight-tied)

Total: 25.7M emb + 7.4M*10 layers ≈ 100M

Plasticity wiring:
    block.synapse.weight tagged grad_source=['bp', 'stdp', 'hebb']
    sf.plasticity.STDP attached on (h_pre, spike_post)   # 'stdp' delta
    sf.plasticity.Hebbian attached on (h_pre, x_after)   # 'hebb' delta
    Engine schedule: every 4 steps to keep wall-clock honest.

Forward signature matches mscfc model:
    out = model(tokens)         -> logits (B, T, V)
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import synapforge as sf
from synapforge.cells.liquid import LiquidCell
from synapforge.cells.synapse import SparseSynapse
from synapforge.plasticity import STDP, Hebbian, PlasticityEngine
from synapforge.surrogate import PLIFCell

# ---------------------------------------------------------------------------
# Hybrid block (one layer)
# ---------------------------------------------------------------------------


class HybridBlock(sf.Module):
    """One synapforge transformer-equivalent layer.

    LiquidCell + PLIF + SparseSynapse + MLP wrapped in pre-norm residuals.
    """

    def __init__(
        self,
        hidden: int,
        mlp_dim: int,
        synapse_sparsity: float = 0.30,
        stdp_rule: STDP | None = None,
        hebb_rule: Hebbian | None = None,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        # LNN / SNN block ---------------------------------------------------
        self.norm1 = nn.LayerNorm(hidden)
        self.cfc = LiquidCell(hidden, hidden, init="hasani", bound=True)
        self.plif = PLIFCell(hidden, tau_init=10.0, threshold_init=1.0,
                             surrogate="atan", reset="subtract")
        # Synapse mixing layer with multi-source gradient (BP + STDP + Hebb).
        self.synapse = SparseSynapse(hidden, hidden, sparsity=synapse_sparsity,
                                     bias=False)
        # Tag synapse weight for multi-source optimizer auto-detection.
        self.synapse.weight._sf_grad_source = ["bp", "stdp", "hebb"]
        self.synapse.weight._sf_weight_per_source = {
            "bp": 1.0, "stdp": 0.05, "hebb": 0.02,
        }
        # MLP block ---------------------------------------------------------
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp_in = nn.Linear(hidden, mlp_dim)
        self.mlp_out = nn.Linear(mlp_dim, hidden)
        # Plasticity rules (shared across all forward calls of this block).
        self.stdp = stdp_rule
        self.hebb = hebb_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # Liquid sub-block (pre-norm residual).
        h_in = self.norm1(x)
        h = self.cfc(h_in)                 # (B, T, D), tanh-bounded
        # PLIF sees the liquid current as input current sequence.
        spk, _ = self.plif.forward_seq(h)  # (B, T, D), spikes in {0,1}
        # Synapse mixes spikes; this is the multi-source-grad parameter.
        mixed = self.synapse(spk)
        # Plasticity observation (training only — non-trainable detached path).
        if self.training:
            if self.stdp is not None:
                # STDP wants pre/post on (B*, D); flatten time so the trace
                # accumulates across the sequence.
                with torch.no_grad():
                    pre = h.detach().reshape(-1, self.hidden).float()
                    post = spk.detach().reshape(-1, self.hidden).float()
                self.stdp.observe(pre=pre, post=post, t=1.0)
            if self.hebb is not None:
                with torch.no_grad():
                    pre = h.detach().reshape(-1, self.hidden).float()
                    post = mixed.detach().reshape(-1, self.hidden).float()
                self.hebb.observe(pre=pre, post=post, t=1.0)
        x = x + mixed                       # residual 1
        # MLP sub-block.
        m = self.norm2(x)
        m = self.mlp_out(F.gelu(self.mlp_in(m)))
        x = x + m                           # residual 2
        return x


# ---------------------------------------------------------------------------
# Full LM
# ---------------------------------------------------------------------------


class SynapForgePolicyLM(sf.Module):
    """Decoder-only LM built entirely on synapforge primitives.

    Forward signature mirrors mscfc.MSpikingCfCLoop:
        forward(tokens: LongTensor[B, T]) -> LongTensor[B, T, V]
    """

    def __init__(
        self,
        vocab: int = 50257,
        hidden: int = 512,
        layers: int = 10,
        mlp_dim: int = 6144,
        max_seq_len: int = 1024,
        synapse_sparsity: float = 0.30,
    ) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.max_seq_len = int(max_seq_len)
        # Embedding (vanilla — vocab x hidden = 25.7M for GPT-2 vocab @ 512).
        self.emb = nn.Embedding(vocab, hidden)
        # Learned positional table (modest size; matches mscfc semantics).
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, hidden))
        # Plasticity rules (one shared STDP + one shared Hebbian per model;
        # they accumulate eligibility traces across layers).
        self.stdp = STDP(lr=1e-3, tau_pre=20.0, tau_post=20.0,
                         a_plus=0.01, a_minus=0.012)
        self.hebb = Hebbian(lr=1e-4, max_pending=2048)
        # Layers --------------------------------------------------------
        self.blocks = nn.ModuleList([
            HybridBlock(hidden, mlp_dim, synapse_sparsity=synapse_sparsity,
                        stdp_rule=self.stdp, hebb_rule=self.hebb)
            for _ in range(layers)
        ])
        # Final norm + tied LM head.
        self.norm_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # Tie weights.
        self.lm_head.weight = self.emb.weight
        # Inits ---------------------------------------------------------
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.mlp_in.weight, std=0.02)
            nn.init.zeros_(blk.mlp_in.bias)
            nn.init.normal_(blk.mlp_out.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.zeros_(blk.mlp_out.bias)

    # ------------------------------------------------------------------ API

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B, T); got {tuple(tokens.shape)}")
        B, T = tokens.shape
        if T > self.max_seq_len:
            raise ValueError(f"seq_len {T} > max_seq_len {self.max_seq_len}")
        x = self.emb(tokens)                # (B, T, D)
        x = x + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)            # (B, T, V)
        return logits

    # Convenience.

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def named_synapse_weights(self) -> dict[str, torch.nn.Parameter]:
        """Return dict {plasticity_target_name: weight} for the engine."""
        out: dict[str, torch.nn.Parameter] = {}
        for i, blk in enumerate(self.blocks):
            out[f"blocks.{i}.synapse.weight"] = blk.synapse.weight
        return out


# ---------------------------------------------------------------------------
# Warm-start from adv29 (best-effort)
# ---------------------------------------------------------------------------


def warmstart_from_adv29(
    model: SynapForgePolicyLM,
    ckpt_path: str = "/workspace/runs/step_001250.pt",
    verbose: bool = True,
) -> dict[str, str]:
    """Salvage what's portable from the adv29 mscfc checkpoint.

    adv29 has hidden_size=256, num_layers=5, vocab=50257. Our target has
    hidden=512, layers=10, vocab=50257 — only the GPT-2 embedding is
    directly transferable (vocab dim matches; hidden dim does not).
    For embeddings we project 256 -> 512 via repeat-then-norm so the
    initial LM head still emits sane logits.

    Returns a dict of {key: status} for printing.
    """
    rep: dict[str, str] = {}
    if not os.path.exists(ckpt_path):
        rep["_status"] = f"ckpt not found at {ckpt_path}; random init kept"
        return rep
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model") or ck.get("model_state_dict") or ck.get("state_dict") or ck
    # Strip "wrapped." prefix if present.
    sd = {k.replace("wrapped.", ""): v for k, v in sd.items()
          if isinstance(v, torch.Tensor)}

    # --- Embedding -----------------------------------------------------
    src = sd.get("embed.text_embed.weight")
    if src is not None:
        if src.shape == model.emb.weight.shape:
            with torch.no_grad():
                model.emb.weight.copy_(src)
            rep["emb.weight"] = f"copied {tuple(src.shape)} (exact)"
        elif src.shape[0] == model.emb.weight.shape[0]:
            # Vocab matches but hidden does not. Project 256 -> 512 by
            # tiling (concat self) and rescaling so variance is preserved.
            v_src, d_src = src.shape
            v_dst, d_dst = model.emb.weight.shape
            if d_dst % d_src == 0:
                rep_factor = d_dst // d_src
                projected = src.repeat(1, rep_factor) / math.sqrt(rep_factor)
                with torch.no_grad():
                    model.emb.weight.copy_(projected)
                rep["emb.weight"] = (
                    f"tiled {tuple(src.shape)} -> {tuple(model.emb.weight.shape)} "
                    f"(rep_factor={rep_factor})"
                )
            else:
                rep["emb.weight"] = (
                    f"vocab matches but hidden {d_src} not factor of {d_dst}; init kept"
                )
        else:
            rep["emb.weight"] = (
                f"shape {tuple(src.shape)} vs {tuple(model.emb.weight.shape)}; init kept"
            )
    else:
        rep["emb.weight"] = "key embed.text_embed.weight not in ckpt; init kept"

    # --- Per-layer params: NOT transferred (hidden dim differs) --------
    rep["blocks.*"] = (
        "hidden 256 -> 512 mismatch; per-layer params random-init "
        "(matches adv30 protocol — model still trains down)"
    )
    if verbose:
        print("[warmstart]", flush=True)
        for k, v in rep.items():
            print(f"  {k:40s} {v}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# Plasticity engine factory (call from train script)
# ---------------------------------------------------------------------------


def build_plasticity_engine(model: SynapForgePolicyLM,
                             schedule: str = "every:4") -> PlasticityEngine:
    """Wrap the per-layer synapse weights under STDP + Hebbian rules.

    Both rules are SHARED across layers (one instance per model). They
    accumulate eligibility traces across all layer-forward observations,
    then emit a single delta per call. The engine maps this single delta
    onto every per-layer synapse weight via a name dict.
    """
    rules = {f"blocks.{i}.synapse.weight": model.stdp
             for i in range(model.layers)}
    # Hebbian rule is also offered but, since it shares observations
    # buffers with STDP, we register it under a different name so the
    # engine processes it separately. We use a per-model Hebbian slot
    # to enable distinct schedules in v0.5.
    # For v0.1 we keep a single STDP scheduler.
    return PlasticityEngine(rules, schedule=schedule)


__all__ = [
    "HybridBlock",
    "SynapForgePolicyLM",
    "warmstart_from_adv29",
    "build_plasticity_engine",
]
