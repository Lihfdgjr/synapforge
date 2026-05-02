"""sf.modal.byte_patch -- frequency-aware byte-patch tokeniser primitive.

Reusable patch-tokeniser block: takes ``(B, L, F_in)`` and produces a
``(B, T, hidden)`` patch-token tensor.  Each patch is a non-overlapping
window of length ``patch`` that is *pooled* along the patch axis before
projection.  The pooling branch is configurable -- this is the knob the
NeurIPS 2025 paper of Fang et al. (arXiv:2505.18608) flags as the cheapest
way to inject high-frequency content into an SNN/LNN front end.

Choice of pool
--------------
``BytePatch`` exposes the choice as an explicit knob:

    pool="avg"     -> classic mean over the patch axis (low-pass).
    pool="max"     -> max over the patch axis (high-pass / edge detector).
    pool="max+avg" -> concat(mean, max) then 1x1 mix to ``hidden``.

"max+avg" is the Max-Former recipe (paper Section 3.2): concat both
branches and let a 1x1 conv learn how to mix them.

Defaults
--------
- ``pool="avg"`` reproduces the classic low-pass byte-patch behaviour.
- The internal projection initialises ``std=0.02``.

Output shape
------------
``(B, T, hidden)`` where ``T = ceil(L / patch)``.  Padding to multiple
of ``patch`` happens internally.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


PoolKind = Literal["avg", "max", "max+avg"]


class BytePatch(Module):
    """Byte-patch tokeniser with selectable pool branch.

    Args:
        in_feat: feature width of the per-time-step input (F_in).
        hidden: output token width.
        patch: window size along the time axis (kernel == stride).
        pool: "avg" (default), "max", or "max+avg".
        bias: whether the projection layer carries a bias.

    Forward:
        x: ``(B, L, F_in)`` float; padded to multiple of ``patch`` on
           the right with zeros.

    Returns:
        ``(B, T, hidden)`` token tensor where ``T = ceil(L / patch)``.

    Notes
    -----
    "avg" path: window-flatten then Linear (Conv1d-equivalent with
    kernel == stride == patch).  "max" path: max over the patch axis,
    yielding F_in features per token, projected to hidden.  "max+avg"
    path: concat(mean, max) then a single Linear of width
    ``2 * F_in -> hidden``.
    """

    VALID_POOLS = ("avg", "max", "max+avg")

    def __init__(
        self,
        in_feat: int,
        hidden: int,
        patch: int = 8,
        pool: PoolKind = "avg",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if patch <= 0:
            raise ValueError(f"patch must be >0; got {patch}")
        if in_feat <= 0:
            raise ValueError(f"in_feat must be >0; got {in_feat}")
        if hidden <= 0:
            raise ValueError(f"hidden must be >0; got {hidden}")
        if pool not in self.VALID_POOLS:
            raise ValueError(
                f"pool must be one of {self.VALID_POOLS}; got {pool!r}"
            )
        self.in_feat = int(in_feat)
        self.hidden = int(hidden)
        self.patch = int(patch)
        self.pool = pool

        if pool == "avg":
            self.proj = nn.Linear(self.patch * self.in_feat, self.hidden, bias=bias)
            nn.init.normal_(self.proj.weight, std=0.02)
            self.proj_max = None
            self.proj_concat = None
        elif pool == "max":
            self.proj = None
            self.proj_max = nn.Linear(self.in_feat, self.hidden, bias=bias)
            nn.init.normal_(self.proj_max.weight, std=0.02)
            self.proj_concat = None
        else:  # "max+avg"
            self.proj = None
            self.proj_max = None
            self.proj_concat = nn.Linear(2 * self.in_feat, self.hidden, bias=bias)
            nn.init.normal_(self.proj_concat.weight, std=0.02)

    def expected_token_count(self, L: int) -> int:
        return (L + self.patch - 1) // self.patch

    @staticmethod
    def _pad_to_patch(x: torch.Tensor, patch: int) -> torch.Tensor:
        L = x.shape[1]
        if L % patch == 0:
            return x
        pad = patch - (L % patch)
        return F.pad(x, (0, 0, 0, pad))

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_patch(x, self.patch)
        B, L_pad, F_in = x.shape
        T = L_pad // self.patch
        return x.reshape(B, T, self.patch, F_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 3:
            raise ValueError(
                f"BytePatch expects (B, L, F_in); got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_feat:
            raise ValueError(
                f"BytePatch expects F_in={self.in_feat}; got {x.shape[-1]}"
            )
        windows = self._windows(x)  # (B, T, patch, F_in)

        if self.pool == "avg":
            B, T, P, F_in = windows.shape
            flat = windows.reshape(B, T, P * F_in)
            return self.proj(flat.to(self.proj.weight.dtype))

        if self.pool == "max":
            pooled = windows.amax(dim=2)
            return self.proj_max(pooled.to(self.proj_max.weight.dtype))

        # "max+avg"
        avg = windows.mean(dim=2)
        mx = windows.amax(dim=2)
        concat = torch.cat([avg, mx], dim=-1)
        return self.proj_concat(concat.to(self.proj_concat.weight.dtype))


__all__ = ["BytePatch", "PoolKind"]
