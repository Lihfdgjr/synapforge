"""TritonBlockBackend -- v0.2 fused LNN+SNN backend.

Replaces the v0.1 GPUDenseBackend pass-through. On `compile(model)` we walk
the model graph and detect every `(LiquidCell -> PLIF)` adjacency and
replace those two modules with a single `TritonHybridBlock` that fuses
CfC scan + PLIF spike + subtract-on-spike reset + STDP eligibility into ONE
Triton kernel launch per layer.

This v0.3 also handles `(LiquidCell -> PLIFCell)` pairs (the variant used by
`sf.model_100m.SynapForge100M`), where PLIFCell exposes a `forward_seq(x)`
API returning `(spikes, v_final)` rather than PLIF's `(spk, mem)` tuple.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ir.graph import IRGraph
from .base import Backend
from .triton_block_kernel import (
    _HAS_TRITON,
    TritonHybridBlock,
)

# ---------------------------------------------------------------------------
# Adapter modules
# ---------------------------------------------------------------------------


class _SharedTritonBlock(nn.Module):
    """Holds the actual TritonHybridBlock and a per-call output buffer.

    The (_FusedLiquid, _FusedPLIF*) pair both reference this same object. The
    Liquid call runs the kernel and stores spikes; the PLIF call reads them.
    """

    def __init__(self, d_in: int, d_hidden: int, alpha: float, enable_stdp: bool):
        super().__init__()
        self.block = TritonHybridBlock(
            d_in=d_in,
            d_hidden=d_hidden,
            alpha=alpha,
            enable_stdp=enable_stdp,
        )
        # Last call cache (spikes, mem) so the FusedPLIF can return them.
        self._last_spikes: torch.Tensor | None = None
        self._last_y: torch.Tensor | None = None


class _FusedLiquid(nn.Module):
    """Stand-in for sf.LiquidCell that calls the shared fused Triton block."""

    def __init__(self, shared: _SharedTritonBlock):
        super().__init__()
        self.shared = shared

    def forward(self, x: torch.Tensor, h0=None) -> torch.Tensor:
        y_norm, spikes = self.shared.block(x, h0=h0)
        # Stash for the PLIF stand-in.
        self.shared._last_spikes = spikes
        self.shared._last_y = y_norm
        return y_norm


class _FusedPLIF(nn.Module):
    """Stand-in for sf.PLIF (cells.plif.PLIF) that returns (spikes, mem)."""

    def __init__(self, shared: _SharedTritonBlock):
        super().__init__()
        self.shared = shared

    def forward(self, current: torch.Tensor, membrane=None, dt: float = 1.0):
        spk = self.shared._last_spikes
        mem = self.shared._last_y
        if spk is None or mem is None:
            raise RuntimeError(
                "_FusedPLIF called before the upstream _FusedLiquid produced "
                "spikes. Did the fusion replace the wrong module pair?"
            )
        return spk, mem


class _FusedPLIFCellSeq(nn.Module):
    """Stand-in for sf.surrogate.PLIFCell.

    Mimics both the per-step `forward(x_t, v_prev)` and the sequence
    `forward_seq(x_seq, v0)` API. Returns spikes from the cached fused
    block call (run by `_FusedLiquid`).

    PLIFCell.forward_seq returns (spike_seq, v_final). We return:
        spike_seq = shared._last_spikes      # (B, T, D)
        v_final   = shared._last_y[:, -1]    # (B, D)  (last membrane post-norm)
    """

    def __init__(self, shared: _SharedTritonBlock, hidden: int):
        super().__init__()
        self.shared = shared
        self.hidden = int(hidden)

    def forward(self, x_t: torch.Tensor, v_prev=None):
        spk = self.shared._last_spikes
        mem = self.shared._last_y
        if spk is None or mem is None:
            raise RuntimeError(
                "_FusedPLIFCellSeq called before _FusedLiquid produced spikes."
            )
        # PLIFCell.forward returns (s_t, v_t) for a single step. Caller is
        # almost always forward_seq, which we override below; this branch
        # only exists as a defensive fallback.
        if x_t.dim() == 2:
            return spk[:, -1], mem[:, -1]
        return spk, mem

    def forward_seq(self, x_seq: torch.Tensor, v0=None):
        spk = self.shared._last_spikes
        mem = self.shared._last_y
        if spk is None or mem is None:
            raise RuntimeError(
                "_FusedPLIFCellSeq.forward_seq called before _FusedLiquid."
            )
        # Match PLIFCell.forward_seq return: (spikes_BTH, v_final_BH)
        return spk, mem[:, -1]


# ---------------------------------------------------------------------------
# Fusion pass
# ---------------------------------------------------------------------------


# Class names we treat as "PLIF-like" siblings of a LiquidCell.
_PLIF_LIKE_NAMES = ("PLIF", "PLIFCell")


def _find_pairs(root: nn.Module) -> list[tuple[nn.Module, str, str, str]]:
    """Return list of (parent, liquid_attr, plif_attr, plif_kind).

    plif_kind in {"PLIF", "PLIFCell"} so the fusion knows which adapter to
    install. We require both children to share the same hidden dim.
    """
    pairs = []
    for parent in root.modules():
        liquid_attr = None
        plif_attr = None
        liquid_mod = None
        plif_mod = None
        plif_kind = None
        for name, child in parent.named_children():
            cls = type(child).__name__
            if cls == "LiquidCell":
                liquid_attr = name
                liquid_mod = child
            elif cls in _PLIF_LIKE_NAMES:
                plif_attr = name
                plif_mod = child
                plif_kind = cls
        if liquid_attr is not None and plif_attr is not None:
            l_h = getattr(liquid_mod, "hidden_dim", None)
            # PLIFCell uses .hidden, PLIF uses .hidden_dim
            p_h = getattr(plif_mod, "hidden_dim", None)
            if p_h is None:
                p_h = getattr(plif_mod, "hidden", None)
            if l_h is not None and p_h is not None and l_h == p_h:
                pairs.append((parent, liquid_attr, plif_attr, plif_kind))
    return pairs


def _fuse_one_pair(
    parent: nn.Module,
    liquid_attr: str,
    plif_attr: str,
    plif_kind: str,
) -> _SharedTritonBlock:
    """Replace parent.<liquid_attr> and parent.<plif_attr> with fused stand-ins."""
    liquid = getattr(parent, liquid_attr)
    plif = getattr(parent, plif_attr)

    d_in = int(liquid.in_dim)
    d_hidden = int(liquid.hidden_dim)
    alpha = float(getattr(plif, "alpha", 2.0))

    shared = _SharedTritonBlock(
        d_in=d_in,
        d_hidden=d_hidden,
        alpha=alpha,
        enable_stdp=False,  # off by default; user can flip via .enable_stdp
    )
    # Move shared to the device/dtype of the original modules.
    try:
        dev = next(liquid.parameters()).device
        dt = next(liquid.parameters()).dtype
        shared = shared.to(device=dev, dtype=dt)
    except StopIteration:
        pass

    # Copy the weights from the originals so a warm-start checkpoint works.
    with torch.no_grad():
        shared.block.delta_proj.weight.copy_(liquid.delta_proj.weight)
        shared.block.delta_proj.bias.copy_(liquid.delta_proj.bias)
        shared.block.b_proj.weight.copy_(liquid.b_proj.weight)
        shared.block.b_proj.bias.copy_(liquid.b_proj.bias)
        shared.block.A_log.copy_(liquid.A_log)
        # PLIF threshold may be a Parameter or non-learnable buffer/scalar.
        thr = plif.threshold
        if torch.is_tensor(thr):
            if thr.dim() == 0:
                shared.block.threshold.fill_(float(thr.item()))
            else:
                shared.block.threshold.copy_(thr)
        else:
            shared.block.threshold.fill_(float(thr))

    setattr(parent, liquid_attr, _FusedLiquid(shared))
    if plif_kind == "PLIFCell":
        setattr(parent, plif_attr, _FusedPLIFCellSeq(shared, hidden=d_hidden))
    else:
        setattr(parent, plif_attr, _FusedPLIF(shared))
    return shared


def _apply_fusion(root: nn.Module) -> dict:
    """Walk the model and rewrite every Liquid->PLIF/PLIFCell pair."""
    pairs = _find_pairs(root)
    fused_blocks: list[_SharedTritonBlock] = []
    for parent, l_attr, p_attr, kind in pairs:
        sb = _fuse_one_pair(parent, l_attr, p_attr, kind)
        fused_blocks.append(sb)
    return {
        "n_pairs_fused": len(pairs),
        "fused_blocks": fused_blocks,
        "triton_available": _HAS_TRITON,
    }


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TritonBlockBackend(Backend):
    """v0.3 backend: fuse Liquid+PLIF/PLIFCell pairs into Triton launches."""

    name = "triton_block"

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.device = device
        self._compiled_root_id: int | None = None
        self._fusion_stats: dict | None = None

    # ---- public surface -------------------------------------------------

    def compile(self, root: nn.Module) -> dict:
        """Apply the fusion pass to `root` in-place. Idempotent per root."""
        rid = id(root)
        if self._compiled_root_id == rid:
            return self._fusion_stats or {}
        stats = _apply_fusion(root)
        self._compiled_root_id = rid
        self._fusion_stats = stats
        return stats

    # ---- Backend ABC ----------------------------------------------------

    def run(self, graph: IRGraph, *inputs, **kwargs):
        root = graph.modules.get("root")
        if root is None:
            raise RuntimeError("graph has no root module")
        if id(root) != self._compiled_root_id:
            self.compile(root)
        if self.device is not None:
            inputs = tuple(
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            )
            kwargs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }
        return root(*inputs, **kwargs)

    def warmup(self, graph: IRGraph, *inputs, **kwargs) -> None:
        with torch.no_grad():
            _ = self.run(graph, *inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


__all__ = [
    "TritonBlockBackend",
    "TritonHybridBlock",
]
