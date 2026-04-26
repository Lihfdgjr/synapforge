"""TritonBlockBackend -- v0.2 fused LNN+SNN backend.

Replaces the v0.1 GPUDenseBackend pass-through. On `compile(model)` we walk
the model graph and detect every `(LiquidCell -> PLIF)` adjacency
(either as direct siblings inside an `nn.Sequential`/`sf.Module` parent, or
as the `cfc + plif` attribute pair inside a HybridBlock-style container)
and replace those two modules with a single `TritonHybridBlock` that fuses
CfC scan + PLIF spike + subtract-on-spike reset + STDP eligibility into ONE
Triton kernel launch per layer.

If Triton is not available (e.g. Windows / CPU-only CI) or compilation
fails at runtime, the kernel internally falls back to the pure-PyTorch
reference path. Modules that are not part of a Liquid->PLIF pair are left
untouched and execute via the standard `torch.nn.Module.forward`.

Mixed models (some Liquid->PLIF pairs + plain modules) are fine: the
backend only rewrites the matched pairs.

Usage:

    rt = sf.compile(model, backend="triton_block")
    y  = rt(x)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base import Backend
from ..ir.graph import IRGraph
from .triton_block_kernel import (
    TritonHybridBlock,
    _HAS_TRITON,
)


# ---------------------------------------------------------------------------
# Adapter modules: bridge a TritonHybridBlock to the (LiquidCell, PLIF) pair
# API expected by user models (LiquidCell.forward(x) -> h ; PLIF.forward(h) ->
# (spk, mem)). We expose two facets of the fused block:
#
#   _FusedLiquid  -- intercepts the LiquidCell call site, runs the fused
#                    kernel, caches (y_norm, spikes), and returns y_norm so
#                    the next module in the chain (PLIF) sees the right shape.
#   _FusedPLIF    -- intercepts the PLIF call site and returns the cached
#                    (spikes, y_norm) tuple, matching PLIF's (spk, mem) API.
#
# This keeps the user's HybridBlock.forward() body unchanged while routing
# both submodules through one fused Triton launch.
# ---------------------------------------------------------------------------


class _SharedTritonBlock(nn.Module):
    """Holds the actual TritonHybridBlock and a per-call output buffer.

    The (_FusedLiquid, _FusedPLIF) pair both reference this same object. The
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
    """Stand-in for sf.PLIF that returns the cached (spikes, mem) from the
    shared fused block. Same return shape as the real PLIF: (spk, mem).
    """

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


# ---------------------------------------------------------------------------
# Fusion pass: walk the module tree, find (LiquidCell, PLIF) attribute pairs
# inside the same parent and rewrite them.
# ---------------------------------------------------------------------------


def _find_pairs(root: nn.Module) -> List[Tuple[nn.Module, str, str]]:
    """Return list of (parent, liquid_attr_name, plif_attr_name) where the
    parent module has both a LiquidCell and a PLIF child. We require the two
    to share `hidden_dim == LiquidCell.hidden_dim == PLIF.hidden_dim`.
    """
    pairs = []
    for parent in root.modules():
        liquid_attr = None
        plif_attr = None
        liquid_mod = None
        plif_mod = None
        for name, child in parent.named_children():
            cls = type(child).__name__
            if cls == "LiquidCell":
                liquid_attr = name
                liquid_mod = child
            elif cls == "PLIF":
                plif_attr = name
                plif_mod = child
        if liquid_attr is not None and plif_attr is not None:
            # Compatibility check: same hidden dim.
            l_h = getattr(liquid_mod, "hidden_dim", None)
            p_h = getattr(plif_mod, "hidden_dim", None)
            if l_h is not None and p_h is not None and l_h == p_h:
                pairs.append((parent, liquid_attr, plif_attr))
    return pairs


def _fuse_one_pair(
    parent: nn.Module,
    liquid_attr: str,
    plif_attr: str,
) -> _SharedTritonBlock:
    """Replace parent.<liquid_attr> and parent.<plif_attr> with fused stand-ins."""
    liquid = getattr(parent, liquid_attr)
    plif = getattr(parent, plif_attr)

    d_in = int(getattr(liquid, "in_dim"))
    d_hidden = int(getattr(liquid, "hidden_dim"))
    alpha = float(getattr(plif, "alpha", 2.0))

    shared = _SharedTritonBlock(
        d_in=d_in,
        d_hidden=d_hidden,
        alpha=alpha,
        enable_stdp=False,  # off by default; user can flip via .enable_stdp
    )
    # Move shared to the device of the original modules.
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
        # PLIF threshold may be a Parameter or a non-learnable buffer/scalar.
        thr = plif.threshold
        if torch.is_tensor(thr):
            if thr.dim() == 0:
                shared.block.threshold.fill_(float(thr.item()))
            else:
                shared.block.threshold.copy_(thr)
        else:
            shared.block.threshold.fill_(float(thr))
        # out_norm starts as identity-ish LayerNorm; nothing to copy from
        # the originals (Liquid has no LN, PLIF has none).

    setattr(parent, liquid_attr, _FusedLiquid(shared))
    setattr(parent, plif_attr, _FusedPLIF(shared))
    return shared


def _apply_fusion(root: nn.Module) -> dict:
    """Walk the model and rewrite every Liquid->PLIF pair. Returns stats."""
    pairs = _find_pairs(root)
    fused_blocks: List[_SharedTritonBlock] = []
    for parent, l_attr, p_attr in pairs:
        sb = _fuse_one_pair(parent, l_attr, p_attr)
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
    """v0.2 backend: fuse Liquid+PLIF pairs into single Triton kernel launches.

    On the first `run()` call (or via explicit `.compile()`) we walk the
    graph's root module and rewrite matched pairs in place. Subsequent
    calls are pure forward dispatch -- no per-call rewrite cost.

    Mixed models work: any module that is NOT part of a Liquid->PLIF pair
    is left as-is and executes via the standard PyTorch path -- effectively
    the same as gpu_dense for those nodes.
    """

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

        # Lazy fuse on first call.
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
