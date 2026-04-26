"""sf.action.head — structured action vector head (RT-2 / OpenVLA-style).

Replaces tool-token / MCP-JSON output with a direct neural projection:
hidden state [B,T,D] -> {action_type, xy, scroll, key, text_trigger}.

Ported from mscfc.action_head (mscfc_action_head.py) onto sf.Module so it
plugs into synapforge's Triton/numba backends, IR compiler, and
PlasticityEngine on equal footing with sf.LiquidCell / sf.PLIF.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module

# ---------------------------------------------------------------------------
# Action-type vocabulary and key vocabulary (default values, override via
# OSActionSpec).
# ---------------------------------------------------------------------------

ACTION_TYPES: tuple[str, ...] = (
    "click",         # 0
    "double_click",  # 1
    "right_click",   # 2
    "type",          # 3 -> emit text via backbone LM head
    "key",           # 4 -> single named key press
    "scroll",        # 5
    "drag",          # 6
    "wait",          # 7
    "done",          # 8
)

# 80-key desktop vocabulary (letters + digits + modifiers + nav + chords).
KEY_VOCAB: tuple[str, ...] = (
    *"abcdefghijklmnopqrstuvwxyz",
    *"0123456789",
    "ctrl", "shift", "alt", "meta",
    "enter", "tab", "space", "backspace", "delete", "esc",
    "up", "down", "left", "right",
    "home", "end", "pageup", "pagedown", "insert",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+s",
    "alt+tab", "cmd+space",
)


@dataclass
class OSActionSpec:
    """Defines the structured action vector contract.

    Field-level shape contract (last dim only; B/T are leading):
        action_type:  [num_action_types]      categorical logits
        xy:           [2]                     sigmoid -> [0,1]^2
        scroll:       [2]                     tanh    -> [-1,1]^2
        key:          [num_keys]              categorical logits
        text_trigger: [1]                     sigmoid -> [0,1]

    The actuator semantically interprets fields based on `action_type`:
      click/double_click/right_click/drag -> use xy
      scroll                              -> use scroll
      key                                 -> use key
      type                                -> use text_trigger>0.5; backbone LM
                                              head emits the token stream

    `extra_keys` lets callers add tool-specific keys (e.g. "win+r") without
    modifying the global KEY_VOCAB tuple.
    """

    action_types: tuple[str, ...] = ACTION_TYPES
    key_vocab: tuple[str, ...] = KEY_VOCAB
    extra_keys: tuple[str, ...] = field(default_factory=tuple)
    mlp_hidden: int | None = None
    dropout: float = 0.0

    @classmethod
    def default(cls) -> OSActionSpec:
        return cls()

    @property
    def num_action_types(self) -> int:
        return len(self.action_types)

    @property
    def all_keys(self) -> tuple[str, ...]:
        return self.key_vocab + self.extra_keys

    @property
    def num_keys(self) -> int:
        return len(self.all_keys)

    @property
    def total_out(self) -> int:
        # logits + xy + scroll + key + text_trigger
        return self.num_action_types + 2 + 2 + self.num_keys + 1

    @property
    def xy_action_ids(self) -> frozenset:
        names = ("click", "double_click", "right_click", "drag")
        return frozenset(self.action_types.index(n) for n in names if n in self.action_types)

    @property
    def scroll_action_ids(self) -> frozenset:
        return frozenset(
            {self.action_types.index("scroll")} if "scroll" in self.action_types else set()
        )

    @property
    def key_action_ids(self) -> frozenset:
        return frozenset(
            {self.action_types.index("key")} if "key" in self.action_types else set()
        )

    @property
    def text_action_ids(self) -> frozenset:
        return frozenset(
            {self.action_types.index("type")} if "type" in self.action_types else set()
        )


@dataclass
class ActionOutput:
    """Structured forward output of ActionHead.  All tensors keep [..., *]."""

    action_type_logits: torch.Tensor   # [..., num_action_types]
    xy: torch.Tensor                    # [..., 2]   sigmoid -> [0,1]
    scroll: torch.Tensor                # [..., 2]   tanh    -> [-1,1]
    key_logits: torch.Tensor            # [..., num_keys]
    text_trigger: torch.Tensor          # [..., 1]   sigmoid -> [0,1]


@dataclass
class ActionTargets:
    """Ground-truth tensors aligned with ActionOutput."""

    action_type: torch.Tensor   # long [B, T]
    xy: torch.Tensor            # [B, T, 2] in [0,1]
    scroll: torch.Tensor        # [B, T, 2] in [-1,1]
    key: torch.Tensor           # long [B, T]
    text_trigger: torch.Tensor  # [B, T] in {0,1}


class ActionHead(Module):
    """MLP head that projects hidden state into a structured action vector.

    Layer:    LayerNorm -> Linear(D, mlp_hidden) -> GELU -> [Dropout] ->
              Linear(mlp_hidden, total_out).

    Output is split into 5 sub-fields per OSActionSpec.  Output layer is
    zero-initialised so early in training xy ≈ 0.5 (sigmoid(0)) and the
    type prior is uniform — same as mscfc.ActionHead so the validation
    runs reproduce.
    """

    def __init__(
        self,
        hidden: int,
        spec: OSActionSpec | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        self.spec = spec or OSActionSpec.default()
        mlp_hidden = self.spec.mlp_hidden or self.hidden
        self.mlp_hidden = mlp_hidden

        self.ln = nn.LayerNorm(hidden)
        layers: list[nn.Module] = [
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
        ]
        if self.spec.dropout > 0.0:
            layers.append(nn.Dropout(self.spec.dropout))
        layers.append(nn.Linear(mlp_hidden, self.spec.total_out))
        self.trunk = nn.Sequential(*layers)

        # Zero-init final layer for stable early training.
        final = self.trunk[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _split(self, flat: torch.Tensor) -> ActionOutput:
        spec = self.spec
        off = 0
        logits = flat[..., off:off + spec.num_action_types]
        off += spec.num_action_types
        xy_raw = flat[..., off:off + 2]
        off += 2
        scr_raw = flat[..., off:off + 2]
        off += 2
        key_logits = flat[..., off:off + spec.num_keys]
        off += spec.num_keys
        txt_raw = flat[..., off:off + 1]
        off += 1
        assert off == spec.total_out, f"split mismatch: {off} vs {spec.total_out}"
        return ActionOutput(
            action_type_logits=logits,
            xy=torch.sigmoid(xy_raw),
            scroll=torch.tanh(scr_raw),
            key_logits=key_logits,
            text_trigger=torch.sigmoid(txt_raw),
        )

    def forward(self, h: torch.Tensor) -> ActionOutput:
        if h.dim() < 2:
            raise ValueError(f"h must have >=2 dims, got {tuple(h.shape)}")
        if h.shape[-1] != self.hidden:
            raise ValueError(
                f"last dim mismatch: expected {self.hidden}, got {h.shape[-1]}"
            )
        x = self.ln(h)
        flat = self.trunk(x)
        return self._split(flat)

    # ------------------------------------------------------------------ utilities

    @torch.no_grad()
    def to_dict(self, out: ActionOutput, batch_idx: int = 0, t: int = -1) -> dict:
        """Slice an ActionOutput to a single dict consumable by OSActuator."""
        spec = self.spec
        type_id = int(out.action_type_logits[batch_idx, t].argmax().item())
        type_id = max(0, min(type_id, len(spec.action_types) - 1))
        type_name = spec.action_types[type_id]
        xy = out.xy[batch_idx, t].detach().cpu().tolist()
        scr = out.scroll[batch_idx, t].detach().cpu().tolist()
        key_id = int(out.key_logits[batch_idx, t].argmax().item())
        key_id = max(0, min(key_id, spec.num_keys - 1))
        text_trigger = bool(
            float(out.text_trigger[batch_idx, t].detach().cpu().item()) > 0.5
        )
        d: dict = {
            "type": type_name,
            "x": None, "y": None,
            "scroll_dx": None, "scroll_dy": None,
            "key": None,
            "text_trigger": text_trigger,
        }
        if type_id in spec.xy_action_ids:
            d["x"], d["y"] = float(xy[0]), float(xy[1])
        if type_id in spec.scroll_action_ids:
            d["scroll_dx"], d["scroll_dy"] = float(scr[0]), float(scr[1])
        if type_id in spec.key_action_ids:
            d["key"] = spec.all_keys[key_id]
        return d


# ---------------------------------------------------------------------------
# ActionLoss — weighted sum of CE/MSE/BCE per sub-field with type-masking.
# ---------------------------------------------------------------------------


class ActionLoss(Module):
    """Weighted multi-head loss with per-action-type masking.

    Default weights match mscfc.ActionLoss:
        1.0 * CE(type)   + 2.0 * MSE(xy on xy-types) +
        1.0 * MSE(scroll on scroll-type) + 1.0 * CE(key on key-type) +
        0.5 * BCE(text_trigger on text-type)
    """

    def __init__(
        self,
        spec: OSActionSpec | None = None,
        w_type: float = 1.0,
        w_xy: float = 2.0,
        w_scroll: float = 1.0,
        w_key: float = 1.0,
        w_text: float = 0.5,
    ) -> None:
        super().__init__()
        self.spec = spec or OSActionSpec.default()
        self.w_type = float(w_type)
        self.w_xy = float(w_xy)
        self.w_scroll = float(w_scroll)
        self.w_key = float(w_key)
        self.w_text = float(w_text)

    @staticmethod
    def _type_mask(targets: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
        ids = list(ids)
        if not ids:
            return torch.zeros_like(targets, dtype=torch.bool)
        m = targets == ids[0]
        for i in ids[1:]:
            m = m | (targets == i)
        return m

    def forward(
        self,
        out: ActionOutput,
        targets: ActionTargets,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        spec = self.spec
        device = out.action_type_logits.device
        dtype = out.action_type_logits.dtype

        if action_mask is None:
            action_mask = torch.ones_like(targets.action_type, dtype=torch.bool)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        B, T = action_mask.shape

        def _zero() -> torch.Tensor:
            return torch.zeros((), device=device, dtype=dtype)

        flat_logits = out.action_type_logits.reshape(B * T, spec.num_action_types)
        flat_tgt = targets.action_type.reshape(B * T).long()
        flat_tgt = flat_tgt.masked_fill(~action_mask.reshape(B * T), -100)
        type_ce = F.cross_entropy(flat_logits, flat_tgt, ignore_index=-100)
        if not torch.isfinite(type_ce):
            type_ce = _zero()

        xy_mask = action_mask & self._type_mask(targets.action_type, spec.xy_action_ids)
        xy_mse = F.mse_loss(out.xy[xy_mask], targets.xy[xy_mask]) if xy_mask.any() else _zero()

        scr_mask = action_mask & self._type_mask(targets.action_type, spec.scroll_action_ids)
        scroll_mse = (
            F.mse_loss(out.scroll[scr_mask], targets.scroll[scr_mask])
            if scr_mask.any() else _zero()
        )

        key_mask = action_mask & self._type_mask(targets.action_type, spec.key_action_ids)
        key_ce = (
            F.cross_entropy(out.key_logits[key_mask], targets.key[key_mask].long())
            if key_mask.any() else _zero()
        )

        txt_mask = action_mask & self._type_mask(targets.action_type, spec.text_action_ids)
        if txt_mask.any():
            txt_pred = out.text_trigger[txt_mask].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            text_bce = F.binary_cross_entropy(txt_pred, targets.text_trigger[txt_mask])
        else:
            text_bce = _zero()

        total = (
            self.w_type * type_ce
            + self.w_xy * xy_mse
            + self.w_scroll * scroll_mse
            + self.w_key * key_ce
            + self.w_text * text_bce
        )
        return {
            "total": total,
            "type_ce": type_ce.detach(),
            "xy_mse": xy_mse.detach(),
            "scroll_mse": scroll_mse.detach(),
            "key_ce": key_ce.detach(),
            "text_bce": text_bce.detach(),
        }


__all__ = [
    "ACTION_TYPES",
    "KEY_VOCAB",
    "OSActionSpec",
    "ActionOutput",
    "ActionTargets",
    "ActionHead",
    "ActionLoss",
]
