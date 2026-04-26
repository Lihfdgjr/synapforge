"""SynaptogenesisPass — compile-time op for mask grow/prune.

v1.0 milestone M4: lift mask grow/prune from "hidden in-place buffer
mutation" up to first-class IR ops.

Why this matters
----------------
v0.1 SparseSynapse.grow() / .prune() mutated `self.mask` in-place. The IR
compiler did not see them — the trace just showed a single `synapse` node
with a static `density` attr. Side effects:

  * Lava / edge backends could not export the structural-plasticity
    schedule. They saw a constant mask and got the wrong sparsity at
    deployment.
  * Triton / CUDA-graph backends could not predict mask churn for kernel
    re-tuning. CUDA graphs even silently broke when mask changed shape.
  * Optimizers could not differentiate between "weight zeroed by prune"
    vs "weight zeroed by mask init" — both looked like dead weights.

Lifting these into the IR fixes all three. After SynaptogenesisPass runs:

  IRGraph: 4 nodes
    IRNode(op=module@root,  ...)
    IRNode(op=synapse@layer, attrs={'in_dim':32,'out_dim':16,'density':0.10})
    IRNode(op=grow_op@layer.grow,
           attrs={'rule':'rigl','target_density':0.15,'period':100,
                  'criterion':'gradient_magnitude','synapse':'layer'})
    IRNode(op=prune_op@layer.prune,
           attrs={'rule':'rigl','target_density':0.15,'period':100,
                  'criterion':'weight_magnitude','synapse':'layer'})

Each grow_op / prune_op:
  * carries a reference to its growth rule so backends can dispatch
  * declares the synapse it acts on via `attrs['synapse']` (string ref,
    serializable — the live module is in g.modules[name])
  * is invoked by `runtime.maybe_update_masks(global_step)` between
    forward steps, which is what `cells/synapse.py` plumbs through

RigL (Evci 2020, "Rigging the Lottery") = the canonical research-grade
growth rule we implement here:

  every T steps, magnitude-prune the bottom k% active weights, then
  gradient-magnitude-grow k% inactive weights, holding total density
  fixed at the target sparsity.

LOC budget: this file ~300 LOC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .graph import IRGraph, IRNode

# ---------------------------------------------------------------------------
# Compiler pass base
# ---------------------------------------------------------------------------


class CompilerPass:
    """Base class for IR rewrite passes.

    A pass takes an IRGraph and returns a (potentially new) IRGraph. v0.1
    only had a single pass — `compile_module` itself. v1.0 introduces a
    proper pipeline: SynaptogenesisPass is the first.
    """

    name: str = "compiler_pass"

    def run(self, graph: IRGraph) -> IRGraph:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Growth rule — RigL (Evci et al., 2020)
# ---------------------------------------------------------------------------


@dataclass
class RigL:
    """Rigging the Lottery — canonical sparse-training growth rule.

    Every `period` training steps:
        1. magnitude-prune the smallest |W| of currently-active weights
           down to a fraction `(1 - frac) * active_count`
        2. gradient-magnitude-grow the largest |dL/dW| of currently-
           inactive weights to restore active count to `target * numel`

    This holds active-fraction at `target` while letting connectivity
    drift toward signal. Reference: arXiv:1911.11134.

    Args
    ----
    target : float in (0, 1]
        Target density (fraction of active connections). The mask
        oscillates around this value.
    period : int
        Number of training steps between (prune, grow) cycles.
    drop_frac : float in [0, 1)
        Fraction of currently-active connections to drop per cycle. Decays
        cosine-style toward 0 over `decay_steps` to match the original
        RigL schedule.
    decay_steps : int
        Number of steps over which `drop_frac` decays to 0.
    """

    target: float = 0.10
    period: int = 100
    drop_frac: float = 0.30
    decay_steps: int = 10_000
    name: str = "rigl"

    # ------------------------------------------------------------------ schedule

    def cosine_drop_frac(self, step: int) -> float:
        """RigL's cosine-decay schedule on the drop fraction."""
        if step >= self.decay_steps:
            return 0.0
        return float(0.5 * self.drop_frac * (1.0 + math.cos(math.pi * step / self.decay_steps)))

    def should_fire(self, step: int) -> bool:
        return step > 0 and (step % self.period) == 0

    # ------------------------------------------------------------------ exec

    @torch.no_grad()
    def step(self, synapse, current_step: int) -> dict:
        """Apply one (prune, grow) cycle to the synapse's mask.

        Returns a dict of stats (density before/after, n_pruned, n_grown)
        for the runtime trace.
        """
        mask = synapse.mask
        weight = synapse.weight
        target_active = int(round(self.target * mask.numel()))
        active = mask
        n_active = int(active.sum().item())
        density_before = n_active / mask.numel()

        frac = self.cosine_drop_frac(current_step)
        n_drop = int(round(frac * n_active))
        # Cap drop so we don't go below target (otherwise grow can't catch up
        # if there are no useful gradient signals).
        n_drop = min(n_drop, max(n_active - target_active, 0) + n_active // 4)

        n_pruned = 0
        n_grown = 0

        # ---- prune: smallest |W| of active ----
        if n_drop > 0:
            scores = weight.abs().masked_fill(~active, float("inf"))
            flat = scores.view(-1)
            _, bottom = flat.topk(n_drop, largest=False)
            r = bottom // synapse.in_dim
            c = bottom % synapse.in_dim
            mask[r, c] = False
            weight[r, c] = 0.0
            n_pruned = int(n_drop)

        # ---- grow: largest |grad| (or |W| as fallback) of inactive ----
        # need active count back up to target_active, capped at numel-active.
        active_after_prune = int(mask.sum().item())
        n_grow = max(target_active - active_after_prune, 0)
        if n_grow > 0:
            inactive = ~mask
            grad_src = synapse.weight.grad
            if grad_src is not None and grad_src.numel() == weight.numel():
                scores = grad_src.abs().masked_fill(~inactive, -float("inf"))
            else:
                # No grad yet (e.g. before first backward). Fall back to
                # |W| of inactive — random init ⇒ uniform-ish.
                scores = weight.abs().masked_fill(~inactive, -float("inf"))
            flat = scores.view(-1)
            n_grow_eff = min(n_grow, int(inactive.sum().item()))
            if n_grow_eff > 0:
                _, top = flat.topk(n_grow_eff)
                r = top // synapse.in_dim
                c = top % synapse.in_dim
                mask[r, c] = True
                # Initialize new weight small so it doesn't dominate.
                weight[r, c] = torch.randn(
                    n_grow_eff, device=weight.device, dtype=weight.dtype,
                ) * 0.01
                n_grown = int(n_grow_eff)

        density_after = float(mask.float().mean().item())
        return {
            "step": int(current_step),
            "rule": self.name,
            "density_before": density_before,
            "density_after": density_after,
            "n_pruned": n_pruned,
            "n_grown": n_grown,
        }

    # ---------------------------------------------------------------- IR attrs

    def to_attrs(self) -> dict:
        """Serializable attrs for IRNode (no torch tensors)."""
        return {
            "rule": self.name,
            "target_density": float(self.target),
            "period": int(self.period),
            "drop_frac": float(self.drop_frac),
            "decay_steps": int(self.decay_steps),
        }


# ---------------------------------------------------------------------------
# Pass: walk IR, insert grow_op + prune_op for every synapse w/ growth rule
# ---------------------------------------------------------------------------


class SynaptogenesisPass(CompilerPass):
    """Insert grow_op + prune_op IR nodes for every SparseSynapse with growth.

    Walk strategy: scan `graph.nodes` for `op == "synapse"`. For each, check
    `g.modules[node.name].growth` — if non-None, append two new IRNodes
    after it:

        grow_op@<name>.grow    attrs = rule.to_attrs() | criterion=gradient
        prune_op@<name>.prune  attrs = rule.to_attrs() | criterion=weight

    The new nodes share the same `module_obj` (the synapse) so the runtime
    can call `rule.step(synapse, step)` from a single dispatch.

    Idempotent: running the pass twice is a no-op (we tag inserted nodes
    with `attrs['_inserted_by'] = 'SynaptogenesisPass'` and skip them on
    subsequent runs).
    """

    name = "synaptogenesis"

    def run(self, graph: IRGraph) -> IRGraph:
        new_nodes: list[IRNode] = []
        new_modules: dict = dict(graph.modules)

        # First pass: collect synapses that already have grow_op nodes
        # (so re-running the pass is a no-op).
        already_lifted: set[str] = {
            n.attrs.get("synapse") for n in graph.nodes
            if n.op == "grow_op" and n.attrs.get("synapse")
        }
        for node in graph.nodes:
            new_nodes.append(node)
            if node.op != "synapse":
                continue
            if node.name in already_lifted:
                continue
            mod = graph.modules.get(node.name)
            if mod is None:
                continue
            growth = getattr(mod, "growth", None)
            if growth is None:
                continue

            base_attrs = growth.to_attrs()
            base_attrs["synapse"] = node.name
            base_attrs["_inserted_by"] = "SynaptogenesisPass"

            grow_attrs = dict(base_attrs)
            grow_attrs["criterion"] = "gradient_magnitude"
            grow_node = IRNode(
                op="grow_op",
                name=f"{node.name}.grow" if node.name else "grow",
                attrs=grow_attrs,
                inputs=[node.name],
                outputs=[],
            )

            prune_attrs = dict(base_attrs)
            prune_attrs["criterion"] = "weight_magnitude"
            prune_node = IRNode(
                op="prune_op",
                name=f"{node.name}.prune" if node.name else "prune",
                attrs=prune_attrs,
                inputs=[node.name],
                outputs=[],
            )

            new_nodes.append(grow_node)
            new_nodes.append(prune_node)
            # Both ops dispatch on the same synapse module — the runtime
            # picks rule.step() from the shared `growth` ref.
            new_modules[grow_node.name] = mod
            new_modules[prune_node.name] = mod

        out = IRGraph()
        out.nodes = new_nodes
        out.modules = new_modules
        return out


# ---------------------------------------------------------------------------
# Runtime helper: walk a graph, fire growth rules at the right step
# ---------------------------------------------------------------------------


def maybe_update_masks(graph: IRGraph, global_step: int) -> list[dict]:
    """Walk grow_op nodes, invoke rule.step() if `should_fire(step)`.

    The pair (grow_op, prune_op) is fused into one rule.step() call —
    RigL prunes-then-grows atomically — so we only fire on grow_op nodes
    and skip the prune_op (which is informational for the trace).

    Returns the list of stat dicts produced by each fired rule.
    """
    fired: list[dict] = []
    for node in graph.nodes:
        if node.op != "grow_op":
            continue
        synapse_name = node.attrs.get("synapse")
        synapse = graph.modules.get(synapse_name) if synapse_name else None
        if synapse is None:
            continue
        rule = getattr(synapse, "growth", None)
        if rule is None or not rule.should_fire(global_step):
            continue
        stats = rule.step(synapse, global_step)
        stats["synapse"] = synapse_name
        fired.append(stats)
    return fired


__all__ = [
    "CompilerPass",
    "RigL",
    "SynaptogenesisPass",
    "maybe_update_masks",
]
