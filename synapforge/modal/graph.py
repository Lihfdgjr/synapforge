"""GraphEmbed -- vanilla message-passing GNN embedding (no torch_geometric).

Inputs (graph_dict)
-------------------
nodes:    (B, N, F_n)  float -- node features (padded to N).
edges:    (B, E, 2)    long  -- (src, dst) node indices, padded to E.
edge_feat:(B, E, F_e)  float -- optional edge features.
node_mask:(B, N)       bool  -- True for valid nodes (excludes padding).
edge_mask:(B, E)       bool  -- True for valid edges.

Pipeline
--------
1. Linear-project node features F_n -> hidden.
2. Linear-project edge features F_e -> hidden (zero proj if absent).
3. K rounds (default K=3) of message passing:
     m_e = MLP_edge( cat( h[src], h[dst], edge_feat ) )    # (B, E, hidden)
     agg[v] = sum_{e where dst==v} m_e   (scatter-add, masked)
     h[v]   = h[v] + GRU_or_MLP( agg[v] )
4. Optional pool: if pool="set", keep all N node tokens.
                  if pool="readout", produce a single graph token.
5. Add node-positional encoding (degree-based or learned per-index).
6. Prepend learned <|graph|> marker.

Returns
-------
(B, 1 + N, hidden) (default pool="set") OR (B, 1 + 1, hidden) (pool="readout").

Vanilla torch only — implemented via scatter_add on flat (B*E, hidden) tensors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module


class _MPLayer(nn.Module):
    """Single round of message passing. Edge MLP + node-update MLP."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        # Edge MLP: cat(h_src, h_dst, h_e) -> hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        for m in self.edge_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Node update.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        for m in self.node_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,            # (B, N, hidden)
        edges: torch.Tensor,        # (B, E, 2) long
        edge_feat: torch.Tensor,    # (B, E, hidden)
        edge_mask: torch.Tensor,    # (B, E) bool
        node_mask: torch.Tensor,    # (B, N) bool
    ) -> torch.Tensor:
        B, N, H = h.shape
        E = edges.shape[1]
        # Gather h[src] and h[dst]: (B, E, H)
        src = edges[..., 0]   # (B, E)
        dst = edges[..., 1]   # (B, E)
        # Clamp invalid edges to 0 to avoid OOB; we'll mask below.
        src_safe = src.clamp(0, N - 1)
        dst_safe = dst.clamp(0, N - 1)
        h_src = torch.gather(
            h, 1, src_safe.unsqueeze(-1).expand(-1, -1, H)
        )
        h_dst = torch.gather(
            h, 1, dst_safe.unsqueeze(-1).expand(-1, -1, H)
        )
        # Compute edge messages.
        cat = torch.cat([h_src, h_dst, edge_feat], dim=-1)  # (B, E, 3H)
        msg = self.edge_mlp(cat)                            # (B, E, H)
        # Mask invalid edges to zero.
        msg = msg * edge_mask.unsqueeze(-1).to(msg.dtype)
        # Scatter-add into agg[B, N, H] via dst index.
        agg = torch.zeros_like(h)
        agg = agg.scatter_add(
            dim=1,
            index=dst_safe.unsqueeze(-1).expand(-1, -1, H),
            src=msg,
        )
        # Update nodes.
        cat_node = torch.cat([h, agg], dim=-1)              # (B, N, 2H)
        upd = self.node_mlp(cat_node)                       # (B, N, H)
        h_new = h + upd
        # Mask invalid nodes -> keep at 0.
        h_new = h_new * node_mask.unsqueeze(-1).to(h_new.dtype)
        return h_new


class GraphEmbed(Module):
    """Vanilla message-passing GNN that produces a token sequence.

    Forward
    -------
    graph: dict with keys
        nodes      (B, N, F_n)  float
        edges      (B, E, 2)    long, optional
        edge_feat  (B, E, F_e)  float, optional
        node_mask  (B, N)       bool, optional (default: all True)
        edge_mask  (B, E)       bool, optional (default: all True)
    Returns (B, 1 + N, hidden) when pool='set'.
    """

    def __init__(
        self,
        hidden: int = 512,
        node_feat: int = 32,
        edge_feat: int = 0,
        rounds: int = 3,
        pool: str = "set",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.node_feat = int(node_feat)
        self.edge_feat = int(edge_feat)
        self.rounds = int(rounds)
        if pool not in ("set", "readout"):
            raise ValueError(f"pool must be 'set' or 'readout'; got {pool!r}")
        self.pool = pool
        self.node_proj = nn.Linear(node_feat, hidden, bias=False)
        nn.init.normal_(self.node_proj.weight, std=0.02)
        if edge_feat > 0:
            self.edge_proj = nn.Linear(edge_feat, hidden, bias=False)
            nn.init.normal_(self.edge_proj.weight, std=0.02)
        else:
            self.edge_proj = None
        # Bias-less learned vector for "no edge feature".
        self.empty_edge = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.empty_edge, std=0.02)
        self.layers = nn.ModuleList([_MPLayer(hidden) for _ in range(self.rounds)])
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)

    def forward(self, graph: dict) -> torch.Tensor:
        if not isinstance(graph, dict):
            raise TypeError(f"GraphEmbed expects a dict; got {type(graph)}")
        nodes = graph.get("nodes")
        if nodes is None:
            raise ValueError("graph dict missing 'nodes'")
        if nodes.dim() != 3:
            raise ValueError(f"nodes must be (B,N,F); got {tuple(nodes.shape)}")
        B, N, F_n = nodes.shape
        if F_n != self.node_feat:
            if F_n < self.node_feat:
                pad = torch.zeros(
                    B, N, self.node_feat - F_n,
                    device=nodes.device, dtype=nodes.dtype,
                )
                nodes = torch.cat([nodes, pad], dim=-1)
            else:
                nodes = nodes[..., : self.node_feat]
        edges = graph.get("edges")
        if edges is None:
            edges = torch.zeros(B, 0, 2, dtype=torch.long, device=nodes.device)
        E = edges.shape[1]
        edge_feat = graph.get("edge_feat")
        node_mask = graph.get("node_mask")
        edge_mask = graph.get("edge_mask")
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=nodes.device)
        if edge_mask is None:
            edge_mask = torch.ones(B, E, dtype=torch.bool, device=nodes.device)
        # Project.
        h = self.node_proj(nodes.to(self.node_proj.weight.dtype))
        # Edge features.
        if self.edge_proj is not None and edge_feat is not None:
            ef = self.edge_proj(edge_feat.to(self.edge_proj.weight.dtype))
        else:
            ef = self.empty_edge.to(h.dtype).view(1, 1, -1).expand(B, E, self.hidden)
        # Message passing.
        for layer in self.layers:
            h = layer(h, edges, ef, edge_mask, node_mask)
        # Token sequence.
        if self.pool == "readout":
            # Mean over valid nodes -> single graph token.
            denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype).unsqueeze(-1)
            graph_tok = (h * node_mask.unsqueeze(-1).to(h.dtype)).sum(dim=1, keepdim=True) / denom
            tokens = graph_tok          # (B, 1, hidden)
        else:
            tokens = h                  # (B, N, hidden)
        marker = self.marker.to(h.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, tokens], dim=1)
        return z

    @staticmethod
    def expected_token_count(N: int, pool: str = "set") -> int:
        return 1 + (1 if pool == "readout" else N)
