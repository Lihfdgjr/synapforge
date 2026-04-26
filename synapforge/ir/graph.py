"""IRGraph — minimal event-graph IR.

v0.1 doesn't actually USE the IR for execution (the GPU-dense backend just
calls model.forward directly). The IR exists to lock in the data structure
so v0.2 / v0.3 backends (Triton, event-driven CPU) can consume the same shape.

A node has:
    op:    str       — one of: "liquid", "plif", "synapse", "plasticity", "dense"
    name:  str       — fully-qualified module name (e.g., "block.cfc")
    attrs: dict      — op-specific attributes (hidden_dim, threshold, ...)
    inputs:  list[str]  — names of upstream nodes
    outputs: list[str]  — names of downstream nodes

Edges are implicit via input/output names. A graph is a topologically sorted
list of nodes (v0.1 produces sequential graphs from sf.Module.children()).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IRNode:
    op: str
    name: str
    attrs: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"IRNode({self.op}@{self.name}, in={self.inputs}, out={self.outputs})"


@dataclass
class IRGraph:
    nodes: list[IRNode] = field(default_factory=list)
    # Map module name -> the actual nn.Module instance, for dense backend
    # to look up and dispatch. Kept separate from IRNode so the IR is
    # serializable in v1.0 without the live torch module.
    modules: dict = field(default_factory=dict)

    def add(self, node: IRNode, module_obj=None) -> None:
        self.nodes.append(node)
        if module_obj is not None:
            self.modules[node.name] = module_obj

    def by_name(self, name: str) -> IRNode | None:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def summary(self) -> str:
        lines = [f"IRGraph: {len(self.nodes)} nodes"]
        for n in self.nodes:
            lines.append(f"  {n}")
        return "\n".join(lines)


__all__ = ["IRNode", "IRGraph"]
