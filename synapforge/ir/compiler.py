"""compile_module — walk a sf.Module and emit an IRGraph.

v0.1 produces a flat sequential IR: one node per direct child sf.Module,
in registration order. Nested children are inlined as separate nodes.

v1.0: after the per-module walk, runs a small pass pipeline. Currently
just :class:`SynaptogenesisPass`, which lifts SparseSynapse growth rules
into ``grow_op`` + ``prune_op`` IR nodes.
"""

from __future__ import annotations

import torch.nn as nn

from .graph import IRGraph, IRNode


def _op_for(module: nn.Module) -> str:
    """Map a torch/sf module to an IR op string."""
    cls_name = type(module).__name__
    if cls_name == "LiquidCell":
        return "liquid"
    if cls_name == "PLIF":
        return "plif"
    if cls_name == "SparseSynapse":
        return "synapse"
    if cls_name in ("HebbianPlasticity", "STDP"):
        return "plasticity"
    # Generic torch ops are dense.
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return "dense"
    return "module"  # opaque container


def _attrs_for(module: nn.Module) -> dict:
    """Snapshot key static attributes for the IR (no tensors)."""
    attrs: dict = {}
    for k in ("in_dim", "out_dim", "hidden_dim", "alpha"):
        if hasattr(module, k):
            v = getattr(module, k)
            if isinstance(v, (int, float, str, bool)):
                attrs[k] = v
    if hasattr(module, "reset_by_subtract"):
        attrs["reset_by_subtract"] = bool(module.reset_by_subtract)
    if hasattr(module, "mask"):
        try:
            attrs["density"] = float(module.mask.float().mean().item())
        except Exception:
            pass
    if hasattr(module, "growth") and getattr(module, "growth", None) is not None:
        attrs["has_growth"] = True
    return attrs


def compile_module(module: nn.Module, run_passes: bool = True) -> IRGraph:
    """Walk children and emit one IRNode each, then run pass pipeline.

    For v0.1 we don't try to track tensor dataflow; inputs/outputs of each
    node are left empty and the runtime executes the module via forward().

    v1.0: ``run_passes=True`` (default) runs SynaptogenesisPass, lifting
    ``SparseSynapse(growth=...)`` rules into ``grow_op`` / ``prune_op`` IR
    nodes. Pass ``run_passes=False`` to inspect the raw walk-only IR
    (useful for diffing pass outputs in tests).
    """
    g = IRGraph()
    g.add(
        IRNode(op="module", name="root",
               attrs={"class": type(module).__name__}),
        module_obj=module,
    )
    for name, child in module.named_modules():
        if child is module:
            continue
        op = _op_for(child)
        node = IRNode(op=op, name=name, attrs=_attrs_for(child))
        g.add(node, module_obj=child)

    if run_passes:
        # Lazy import — avoids circular issues if user imports
        # ``ir.compiler`` directly.
        from .synaptogenesis import SynaptogenesisPass
        g = SynaptogenesisPass().run(g)

    return g


__all__ = ["compile_module"]
