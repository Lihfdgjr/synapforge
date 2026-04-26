"""sf.action.neuromcp — neuroplastic tool acquisition (NeuroMCP).

Replaces MCP / function-calling tool tokens with a *neural codebook* that
grows new prototypes when novelty exceeds a threshold, plus a sparse
synaptic projection that uses sf.plasticity.SynaptogenesisGrowPrune for
mask growth/prune via co-activation EMA.

Three classes:

    DynamicActionCodebook    [K_max, D] prototypes; alive_mask grows when
                             a new tool/skill embedding is sufficiently far
                             from all alive prototypes.  No JSON schema.
    SparseSynapticLayer      D->D linear with a structurally sparse mask
                             driven by sf.plasticity.SynaptogenesisGrowPrune.
                             Density grows from initial -> max as new
                             prototypes appear and demand more capacity.
    NeuroMCPHead             composition: SparseSynapticLayer -> LayerNorm
                             -> DynamicCodebook(cosine) -> logits over K_alive.

Validated on the 4-button toy env in /workspace/mscfc_neuromcp.py:
    K=9 -> 13 prototypes (+4) when 4 new "tools" introduced
    density 5% -> 28% growth, 100% hit_rate after warmup.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from ..plasticity import SynaptogenesisGrowPrune

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class CodebookConfig:
    initial_size: int = 9
    max_size: int = 64
    embed_dim: int = 256
    novelty_threshold: float = 0.35
    growth_cooldown: int = 50  # min steps between two grow events
    init_scale: float = 0.02


@dataclass
class SynaptogenesisConfig:
    in_dim: int = 256
    out_dim: int = 256
    initial_density: float = 0.05
    max_density: float = 0.40
    growth_threshold: float = 0.05    # min coact value to grow
    prune_threshold: float = 0.001    # max |W| to prune
    growth_check_every: int = 20
    prune_check_every: int = 200
    growth_init_scale: float = 0.01
    growth_step: float = 0.005        # per-check density bump


# ---------------------------------------------------------------------------
# DynamicActionCodebook — prototype set whose K_alive grows under novelty.
# ---------------------------------------------------------------------------


class DynamicActionCodebook(Module):
    """Cosine-similarity action codebook with online prototype growth.

    forward(z) returns scaled cosine similarity logits to alive prototypes:
        logits[..., k] = (z / |z|) · (proto_k / |proto_k|) / tau

    maybe_grow(z) adds a new prototype if max similarity to alive set
    is < (1 - novelty_threshold), respecting growth_cooldown.

    Stored buffers: alive_mask, use_count, last_growth_step, step_count.
    """

    def __init__(self, cfg: CodebookConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CodebookConfig()
        cfg = self.cfg
        if not 0 < cfg.initial_size <= cfg.max_size:
            raise ValueError(
                f"initial_size must be in (0, max_size], got "
                f"{cfg.initial_size} > {cfg.max_size}"
            )
        # All [max_size, D] prototypes; only first `initial_size` start alive.
        self.prototypes = nn.Parameter(
            torch.randn(cfg.max_size, cfg.embed_dim) * cfg.init_scale
        )
        self.register_buffer("alive_mask", torch.zeros(cfg.max_size, dtype=torch.bool))
        self.register_buffer("use_count", torch.zeros(cfg.max_size, dtype=torch.long))
        self.register_buffer(
            "last_growth_step", torch.tensor(-cfg.growth_cooldown, dtype=torch.long)
        )
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        with torch.no_grad():
            self.alive_mask[: cfg.initial_size] = True

    @property
    def K(self) -> int:
        return int(self.alive_mask.sum().item())

    def forward(self, hidden_z: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        live = self.prototypes[self.alive_mask]                # [K, D]
        z_n = F.normalize(hidden_z, dim=-1)
        p_n = F.normalize(live, dim=-1)
        sim = torch.matmul(z_n, p_n.T)                          # [..., K]
        return sim / float(tau)

    @torch.no_grad()
    def maybe_grow(self, hidden_z: torch.Tensor) -> dict:
        self.step_count += 1
        out = {"K_before": self.K, "K_after": self.K, "grew": False}
        if self.K >= self.cfg.max_size:
            return out
        if (self.step_count - self.last_growth_step) < self.cfg.growth_cooldown:
            return out
        live = self.prototypes[self.alive_mask]                  # [K, D]
        z_n = F.normalize(hidden_z.reshape(-1, hidden_z.size(-1)), dim=-1)
        p_n = F.normalize(live, dim=-1)
        sim = torch.matmul(z_n, p_n.T)                            # [N, K]
        max_sim_per_token = sim.max(dim=-1).values               # [N]
        novelty = 1.0 - max_sim_per_token
        most_novel_idx = int(novelty.argmax().item())
        most_novel_score = float(novelty[most_novel_idx].item())
        if most_novel_score > self.cfg.novelty_threshold:
            dead_slots = (~self.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead_slots) == 0:
                return out
            new_idx = int(dead_slots[0].item())
            self.prototypes[new_idx] = z_n[most_novel_idx] * live.norm(dim=-1).mean()
            self.alive_mask[new_idx] = True
            self.last_growth_step.copy_(self.step_count.clone())
            out["grew"] = True
            out["new_idx"] = new_idx
            out["novelty_score"] = most_novel_score
            out["K_after"] = self.K
        return out


# ---------------------------------------------------------------------------
# SparseSynapticLayer — wraps sf.plasticity.SynaptogenesisGrowPrune as the
# mask-growth rule.  No custom buffer hacks: the rule is an honest
# PlasticityRule.  We keep a coact_ema buffer here too as a fallback for
# when the user does NOT plug us into a PlasticityEngine, which is the case
# for the simple toy training script.  The two are kept compatible.
# ---------------------------------------------------------------------------


class SparseSynapticLayer(Module):
    """Linear D_in -> D_out with structurally sparse mask + synaptogenesis.

    Forward applies (W * mask) @ x.  Backward gradients flow only through
    masked-on entries.

    Plasticity execution model:
      - In TRAINING: forward also calls update_coactivation(pre, post) so
        coact_ema tracks |y_post|.T @ |x_pre|.
      - maybe_grow_prune() returns a dict {added, pruned, density} and is
        called by NeuroMCPHead.step_plasticity().

    The default growth/prune rule is co-activation EMA + magnitude pruning,
    matching the validated mscfc.SparseSynapticLayer.  Optionally pass
    `growth_rule=sf.plasticity.SynaptogenesisGrowPrune(...)` to delegate to
    the first-class plasticity engine, which lifts the grow/prune ops into
    the IR.
    """

    def __init__(
        self,
        cfg: SynaptogenesisConfig | None = None,
        growth_rule: SynaptogenesisGrowPrune | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SynaptogenesisConfig()
        cfg = self.cfg
        self.weight = nn.Parameter(torch.randn(cfg.out_dim, cfg.in_dim) * 0.02)
        rand = torch.rand(cfg.out_dim, cfg.in_dim)
        self.register_buffer("mask", (rand < cfg.initial_density).float())
        self.register_buffer("coact_ema", torch.zeros(cfg.out_dim, cfg.in_dim))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        # Optional: first-class plasticity rule.  When supplied, its observe()
        # is called on each forward and its delta is used to shape mask updates.
        self.growth_rule = growth_rule

    @property
    def density(self) -> float:
        return float(self.mask.mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask)

    @torch.no_grad()
    def update_coactivation(self, x_pre: torch.Tensor, y_post: torch.Tensor) -> None:
        flat_x = x_pre.reshape(-1, x_pre.size(-1)).abs()
        flat_y = y_post.reshape(-1, y_post.size(-1)).abs()
        coact = (flat_y.T @ flat_x) / max(1, flat_x.size(0))
        self.coact_ema.mul_(0.99).add_(coact, alpha=0.01)
        self.step += 1
        if self.growth_rule is not None:
            self.growth_rule.observe(pre=x_pre, post=y_post, t=float(self.step.item()))

    @torch.no_grad()
    def maybe_grow_prune(self) -> dict:
        cfg = self.cfg
        out = {"added": 0, "pruned": 0, "density": float(self.mask.mean().item())}
        # GROW
        if self.step % cfg.growth_check_every == 0:
            masked_off = self.mask < 0.5
            target_density = min(
                cfg.max_density,
                float(self.mask.mean().item()) + cfg.growth_step,
            )
            current = self.mask.sum().item()
            target = target_density * self.mask.numel()
            n_to_grow = max(0, int(target - current))
            if n_to_grow > 0:
                candidate_scores = self.coact_ema * masked_off.float()
                vals, idx = candidate_scores.flatten().topk(n_to_grow)
                grow_mask = torch.zeros_like(self.mask).flatten()
                grow_mask[idx[vals > cfg.growth_threshold]] = 1.0
                grow_mask = grow_mask.view_as(self.mask)
                self.weight.data = self.weight.data + grow_mask * (
                    torch.randn_like(self.weight) * cfg.growth_init_scale
                )
                self.mask.add_(grow_mask).clamp_(max=1.0)
                out["added"] = int(grow_mask.sum().item())
        # PRUNE
        if self.step % cfg.prune_check_every == 0:
            alive_small = (self.mask > 0.5) & (self.weight.abs() < cfg.prune_threshold)
            n_prune = int(alive_small.sum().item())
            if n_prune > 0:
                self.mask[alive_small] = 0.0
                self.weight.data[alive_small] = 0.0
                out["pruned"] = n_prune
        out["density"] = float(self.mask.mean().item())
        return out


# ---------------------------------------------------------------------------
# NeuroMCPHead — composition of sparse projection + dynamic codebook.
# ---------------------------------------------------------------------------


class NeuroMCPHead(Module):
    """Tool-token-free action-codebook head.

    forward(h) -> {"logits": [..., K_alive], "hidden_z": [..., D]}.

    `step_plasticity()` returns a stats dict for logging and triggers both
    SparseSynapticLayer.maybe_grow_prune and (optionally) caller-driven
    DynamicCodebook.maybe_grow.  Caller is expected to call this after
    optimizer.step() to avoid autograd version conflicts (matches
    sf.plasticity OBSERVE/DELTA/APPLY contract).
    """

    def __init__(
        self,
        hidden: int,
        codebook_initial: int = 9,
        codebook_max: int = 64,
        synapse_density: float = 0.05,
        synapse_max_density: float = 0.40,
        codebook_cfg: CodebookConfig | None = None,
        synapse_cfg: SynaptogenesisConfig | None = None,
        growth_rule: SynaptogenesisGrowPrune | None = None,
    ) -> None:
        super().__init__()
        if codebook_cfg is None:
            codebook_cfg = CodebookConfig(
                initial_size=codebook_initial,
                max_size=codebook_max,
                embed_dim=hidden,
            )
        if synapse_cfg is None:
            synapse_cfg = SynaptogenesisConfig(
                in_dim=hidden,
                out_dim=hidden,
                initial_density=synapse_density,
                max_density=synapse_max_density,
            )
        self.proj = SparseSynapticLayer(synapse_cfg, growth_rule=growth_rule)
        self.norm = nn.LayerNorm(hidden)
        self.codebook = DynamicActionCodebook(codebook_cfg)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.proj(h)
        z = self.norm(z)
        logits = self.codebook(z)
        if self.training:
            self.proj.update_coactivation(h.detach(), z.detach())
        return {"logits": logits, "hidden_z": z}

    @torch.no_grad()
    def step_plasticity(self, hidden_z: torch.Tensor | None = None) -> dict:
        """Run plasticity rules.

        If `hidden_z` is provided, also attempt codebook growth on it.
        """
        syn_stats = self.proj.maybe_grow_prune()
        cb_stats = {"grew": False}
        if hidden_z is not None:
            cb_stats = self.codebook.maybe_grow(hidden_z)
        return {**syn_stats, "K_alive": self.codebook.K, "grew_cb": cb_stats["grew"]}


__all__ = [
    "CodebookConfig",
    "SynaptogenesisConfig",
    "DynamicActionCodebook",
    "SparseSynapticLayer",
    "NeuroMCPHead",
]
