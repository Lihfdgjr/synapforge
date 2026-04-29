"""
Per-domain NeuroMCPHead: 4 codebooks (math/chat/code/web) instead of 1.

Why per-domain:
  - One global codebook = catastrophic interference across tasks
  - Math prototypes don't help chat, web tools don't help math
  - Per-domain = Mixture-of-Experts but at the action level

Architecture:
  intent_emb (B, H) -> intent_router (small MLP, 4 logits)
                    -> top-1 domain selection
                    -> route to per-domain NeuroMCPHead
                    -> action_vec (B, A) + entropy bonus

On-demand skill spawn:
  request_skill(intent_emb, urgency) is called at inference when:
    1. The current top-K active prototypes all have low cosine sim to intent
    2. urgency > threshold (user explicitly requests new tool)
  If both: a new prototype is grown immediately and persisted to skill_log.

Persistence (the user's contract):
  On boot, call .restore_from_log(log) to inject all known skills as fixed
  prototypes in the codebook (frozen embedding, learnable usage gate).
  On each forward, .activate(pid) updates skill_log usage stats.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skill_log import SkillEntry, SkillLog


class DynamicCodebook(nn.Module):
    """Growable prototype codebook with usage-driven LTP."""

    def __init__(
        self,
        hidden: int,
        action_dim: int,
        initial: int = 4,
        max_size: int = 64,
        spawn_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.action_dim = action_dim
        self.max_size = max_size
        self.spawn_threshold = spawn_threshold

        self.protos = nn.Parameter(torch.randn(initial, hidden) * 0.02)
        self.actions = nn.Parameter(torch.randn(initial, action_dim) * 0.02)
        self.gates = nn.Parameter(torch.ones(initial))
        self.register_buffer("usage_count", torch.zeros(initial))
        self.register_buffer("frozen_mask", torch.zeros(initial, dtype=torch.bool))

    @property
    def K(self) -> int:
        return self.protos.shape[0]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: (B, hidden) -> action: (B, action_dim), entropy: scalar, top_pids: (B,)"""
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.protos, dim=-1)
        sim = h_n @ p_n.t()
        gated = sim * self.gates.unsqueeze(0)
        weights = F.softmax(gated, dim=-1)

        action = weights @ self.actions

        eps = 1e-8
        ent = -(weights * (weights + eps).log()).sum(-1).mean()

        top_pids = weights.argmax(dim=-1)
        with torch.no_grad():
            self.usage_count.scatter_add_(0, top_pids, torch.ones_like(top_pids, dtype=torch.float))

        return action, ent, top_pids

    def need_spawn(self, intent: torch.Tensor) -> bool:
        if self.K >= self.max_size:
            return False
        with torch.no_grad():
            i_n = F.normalize(intent.detach(), dim=-1)
            p_n = F.normalize(self.protos, dim=-1)
            best = (i_n @ p_n.t()).max().item()
        return best < self.spawn_threshold

    def spawn(self, init_emb: torch.Tensor) -> int:
        """Add a prototype initialized at init_emb. Returns the new local index."""
        if self.K >= self.max_size:
            return -1

        new_proto = init_emb.detach().clone().to(self.protos.device).unsqueeze(0)
        new_action = torch.randn(1, self.action_dim, device=self.actions.device) * 0.02
        new_gate = torch.tensor([1.0], device=self.gates.device)

        protos = torch.cat([self.protos.data, new_proto], dim=0)
        actions = torch.cat([self.actions.data, new_action], dim=0)
        gates = torch.cat([self.gates.data, new_gate], dim=0)

        self.protos = nn.Parameter(protos)
        self.actions = nn.Parameter(actions)
        self.gates = nn.Parameter(gates)
        self.usage_count = torch.cat(
            [self.usage_count, torch.zeros(1, device=self.usage_count.device)]
        )
        self.frozen_mask = torch.cat(
            [self.frozen_mask, torch.zeros(1, dtype=torch.bool, device=self.frozen_mask.device)]
        )
        return self.K - 1

    def freeze(self, local_idx: int) -> None:
        """Mark a prototype as frozen (loaded from log, don't update embedding)."""
        if 0 <= local_idx < self.K:
            self.frozen_mask[local_idx] = True

    def freeze_loaded_grads(self) -> None:
        """Zero out grads for frozen rows. Call before optimizer.step()."""
        if not self.frozen_mask.any():
            return
        with torch.no_grad():
            mask = self.frozen_mask.float().unsqueeze(-1)
            if self.protos.grad is not None:
                self.protos.grad.mul_(1.0 - mask)


class SparseSynapticLayer(nn.Module):
    """Adjacency matrix with synaptogenesis (grow) + pruning."""

    def __init__(
        self,
        n: int,
        density_init: float = 0.05,
        density_max: float = 0.40,
        grow_rate: float = 0.001,
        prune_rate: float = 0.0005,
    ) -> None:
        super().__init__()
        self.n = n
        self.density_max = density_max
        self.grow_rate = grow_rate
        self.prune_rate = prune_rate

        mask = (torch.rand(n, n) < density_init).float()
        mask.fill_diagonal_(0.0)
        self.register_buffer("mask", mask)
        self.W = nn.Parameter(torch.randn(n, n) * 0.02)
        self.register_buffer("coactivity", torch.zeros(n, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.W * self.mask)

    @torch.no_grad()
    def observe_coactivity(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        """Hebbian co-activity tracking. pre/post: (B, n)."""
        ca = (pre.t() @ post) / max(pre.shape[0], 1)
        self.coactivity = 0.99 * self.coactivity + 0.01 * ca.abs()

    @torch.no_grad()
    def step_plasticity(self) -> Dict[str, float]:
        """Grow new edges where coactivity is high; prune weak ones."""
        density = self.mask.mean().item()

        if density < self.density_max:
            unconnected = (self.mask == 0).float()
            unconnected.fill_diagonal_(0.0)
            scores = self.coactivity * unconnected
            n_grow = max(1, int(self.grow_rate * self.n * self.n))
            flat = scores.flatten()
            if flat.numel() > 0 and flat.max() > 0:
                _, top_idx = flat.topk(min(n_grow, flat.numel()))
                self.mask.flatten()[top_idx] = 1.0
                self.W.data.flatten()[top_idx] = (
                    torch.randn(top_idx.numel(), device=self.W.device) * 0.02
                )

        with torch.no_grad():
            connected = self.mask.bool()
            if connected.any():
                weights_abs = self.W.abs() * self.mask
                threshold = weights_abs[connected].quantile(self.prune_rate)
                weak = (weights_abs < threshold) & connected
                self.mask[weak] = 0.0

        return {
            "density": self.mask.mean().item(),
            "n_edges": int(self.mask.sum().item()),
        }


class SingleDomainHead(nn.Module):
    """One NeuroMCPHead = synapse + codebook for a single domain."""

    def __init__(
        self,
        hidden: int,
        action_dim: int = 64,
        codebook_initial: int = 4,
        codebook_max: int = 64,
        synapse_density: float = 0.05,
        synapse_max_density: float = 0.40,
    ) -> None:
        super().__init__()
        self.synapse = SparseSynapticLayer(hidden, synapse_density, synapse_max_density)
        self.codebook = DynamicCodebook(hidden, action_dim, codebook_initial, codebook_max)
        self.proj_in = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_in = self.proj_in(h)
        h_syn = self.synapse(h_in)
        action, ent, pids = self.codebook(h_syn)
        with torch.no_grad():
            self.synapse.observe_coactivity(h_in, h_syn)
        return action, ent, pids

    def step_plasticity(self) -> Dict[str, float]:
        return self.synapse.step_plasticity()


class PerDomainNeuroMCP(nn.Module):
    """4 SingleDomainHeads + a soft router over hidden -> domain."""

    DOMAINS = ("math", "chat", "code", "web")

    def __init__(
        self,
        hidden: int,
        action_dim: int = 64,
        skill_log: Optional[SkillLog] = None,
        codebook_initial_per_domain: int = 4,
        codebook_max_per_domain: int = 64,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.action_dim = action_dim
        self.skill_log = skill_log

        self.heads = nn.ModuleDict({
            d: SingleDomainHead(
                hidden=hidden,
                action_dim=action_dim,
                codebook_initial=codebook_initial_per_domain,
                codebook_max=codebook_max_per_domain,
            )
            for d in self.DOMAINS
        })
        self.router = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, len(self.DOMAINS)),
        )
        self._pid_map: Dict[int, Tuple[str, int]] = {}

        if skill_log is not None:
            self.restore_from_log(skill_log)

    def restore_from_log(self, log: SkillLog) -> int:
        """Restore frozen prototypes from JSON. Returns count restored."""
        restored = 0
        for global_pid, entry in log.skills.items():
            if entry.domain not in self.heads:
                continue
            head = self.heads[entry.domain]
            if head.codebook.K >= head.codebook.max_size:
                continue
            emb = torch.tensor(entry.embedding, dtype=torch.float32)
            local_idx = head.codebook.spawn(emb)
            if local_idx < 0:
                continue
            head.codebook.freeze(local_idx)
            self._pid_map[global_pid] = (entry.domain, local_idx)
            restored += 1
        return restored

    def forward(
        self,
        h: torch.Tensor,
        hard_route: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """h: (B, hidden) -> action: (B, action_dim), entropy: scalar, info dict.

        info: {
          'router_logits': (B, 4),
          'route_probs': (B, 4),
          'domain_pids': {dom: (B,) ints} for activated domains
        }
        """
        logits = self.router(h)
        probs = F.softmax(logits, dim=-1)

        all_actions = []
        all_ents = []
        domain_pids = {}
        for i, dom in enumerate(self.DOMAINS):
            act, ent, pids = self.heads[dom](h)
            all_actions.append(act)
            all_ents.append(ent)
            domain_pids[dom] = pids
        actions_stack = torch.stack(all_actions, dim=1)

        if hard_route:
            top_d = probs.argmax(dim=-1, keepdim=True)
            mixed = actions_stack.gather(
                1, top_d.unsqueeze(-1).expand(-1, 1, self.action_dim)
            ).squeeze(1)
        else:
            mixed = (probs.unsqueeze(-1) * actions_stack).sum(dim=1)

        ent_total = sum(all_ents) / len(all_ents)
        route_ent = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        ent_total = ent_total + 0.1 * route_ent

        return mixed, ent_total, {
            "router_logits": logits,
            "route_probs": probs,
            "domain_pids": domain_pids,
        }

    def request_skill(
        self,
        intent_emb: torch.Tensor,
        domain: str,
        urgency: float = 1.0,
    ) -> Optional[int]:
        """On-demand skill spawn. Returns global prototype_id or None."""
        if domain not in self.heads:
            return None
        head = self.heads[domain]
        intent_1d = intent_emb.detach()
        if intent_1d.dim() > 1:
            intent_1d = intent_1d.mean(dim=0)

        if not head.codebook.need_spawn(intent_1d) and urgency < 1.5:
            return None

        local_idx = head.codebook.spawn(intent_1d)
        if local_idx < 0:
            return None

        if self.skill_log is not None:
            global_pid = self.skill_log.register(
                domain=domain,
                embedding=intent_1d,
            )
            self._pid_map[global_pid] = (domain, local_idx)
            return global_pid
        return local_idx

    def report_activation(self, info: dict, reward: float = 0.0) -> None:
        """Update skill_log with activation stats from a forward pass."""
        if self.skill_log is None:
            return
        domain_pids = info.get("domain_pids", {})
        for dom, pids in domain_pids.items():
            if pids.numel() == 0:
                continue
            for local_idx in pids.unique().tolist():
                global_pid = None
                for gpid, (d, lidx) in self._pid_map.items():
                    if d == dom and lidx == local_idx:
                        global_pid = gpid
                        break
                if global_pid is not None:
                    self.skill_log.activate(global_pid, reward=reward)

    def step_plasticity(self) -> Dict[str, Dict[str, float]]:
        return {dom: self.heads[dom].step_plasticity() for dom in self.DOMAINS}

    def freeze_loaded_grads(self) -> None:
        """Call before optimizer.step() to prevent updating restored prototypes."""
        for head in self.heads.values():
            head.codebook.freeze_loaded_grads()

    def stats(self) -> dict:
        out = {}
        for dom, head in self.heads.items():
            out[dom] = {
                "K": head.codebook.K,
                "synapse_density": head.synapse.mask.mean().item(),
                "max_usage": int(head.codebook.usage_count.max().item()) if head.codebook.K else 0,
            }
        return out
