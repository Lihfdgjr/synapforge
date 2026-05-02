"""synapforge.neuromcp.primitives -- the fixed OS primitive vocabulary.

NeuroMCP closed-loop computer use.  These primitives are the *only*
things the actuator can dispatch.  Compounds (Layer-4 in the stack) are
NOT new primitives -- they are *sequences* of these primitive ids glued
by the SparseSynapticLayer, so the action vocabulary is bounded and
auditable.

User铁律 (memory ``feedback_neural_action_no_token_no_mcp``):
  - No ``<tool_call>`` token, no MCP JSON schema registration.
  - Hidden state -> primitive_id -> params -> OSActuator.execute.
  - New tools = new compound prototypes added by Hebbian wire-together,
    NOT JSON entries in a registry.

Each primitive has an explicit signature:

    Primitive(
        id              : int,           # 0..N-1
        name            : str,           # human-readable
        param_slots     : tuple[str,...],# which of 8 ParamHead slots it uses
        sandbox_guard   : bool,          # True => requires sandbox confirmation
        category        : str,           # "pointer" | "keyboard" | "screen" | "system"
        description     : str,
    )

The 8 ParamHead slots are universal across primitives:

    0: x        (float, [0,1] normalised)
    1: y        (float, [0,1] normalised)
    2: x2       (float, [0,1])  -- drag end-x
    3: y2       (float, [0,1])  -- drag end-y
    4: token_id (int, vocab id) -- typed text passed via backbone LM head
    5: keysym   (int, KEY_VOCAB index)
    6: dx       (float, [-1,1])  -- scroll dx
    7: dy       (float, [-1,1])  -- scroll dy

This file is **torch-free** so it can be imported by linting / docs jobs
without spinning up CUDA.  All the 24 primitives below are purely
declarative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Primitive:
    """A single OS primitive the actuator can dispatch."""

    id: int
    name: str
    param_slots: Tuple[str, ...]
    sandbox_guard: bool
    category: str
    description: str


# ---------------------------------------------------------------------------
# Param slot names -- keep in sync with ActionHead's 8-slot ParamHead.
# ---------------------------------------------------------------------------

PARAM_SLOTS: Tuple[str, ...] = (
    "x", "y", "x2", "y2", "token_id", "keysym", "dx", "dy",
)
NUM_PARAM_SLOTS: int = len(PARAM_SLOTS)


# ---------------------------------------------------------------------------
# 24 fixed primitives.  Order is the canonical id assignment -- DO NOT
# reorder: training checkpoints encode primitive_id by index.
# ---------------------------------------------------------------------------

PRIMITIVES: Tuple[Primitive, ...] = (
    # ----- 0..7  pointer primitives ------------------------------------
    Primitive(0, "click_at",
              ("x", "y"), False, "pointer",
              "Single left click at normalised (x,y)."),
    Primitive(1, "double_click",
              ("x", "y"), False, "pointer",
              "Double left click at normalised (x,y)."),
    Primitive(2, "right_click",
              ("x", "y"), False, "pointer",
              "Right click at normalised (x,y) -- opens context menu."),
    Primitive(3, "middle_click",
              ("x", "y"), False, "pointer",
              "Middle (wheel) click at normalised (x,y)."),
    Primitive(4, "drag",
              ("x", "y", "x2", "y2"), False, "pointer",
              "Click-and-drag from (x,y) to (x2,y2)."),
    Primitive(5, "move_mouse",
              ("x", "y"), False, "pointer",
              "Move pointer without clicking."),
    Primitive(6, "hover",
              ("x", "y"), False, "pointer",
              "Hover (move + dwell 100 ms) at (x,y)."),
    Primitive(7, "scroll",
              ("dx", "dy"), False, "pointer",
              "Scroll wheel by (dx, dy) in [-1,1] (units of ~200 lines)."),

    # ----- 8..15  keyboard primitives ----------------------------------
    Primitive(8, "type_text",
              ("token_id",), False, "keyboard",
              "Emit a text token from backbone LM head."),
    Primitive(9, "press_key",
              ("keysym",), False, "keyboard",
              "Press a single named key (KEY_VOCAB index)."),
    Primitive(10, "key_chord",
              ("keysym",), False, "keyboard",
              "Press a chord like ctrl+s (chord names live in KEY_VOCAB)."),
    Primitive(11, "key_down",
              ("keysym",), False, "keyboard",
              "Press-and-hold a key (release via key_up)."),
    Primitive(12, "key_up",
              ("keysym",), False, "keyboard",
              "Release a previously held key."),
    Primitive(13, "select_all",
              (), False, "keyboard",
              "Convenience for ctrl+a (kept primitive for compound seeding)."),
    Primitive(14, "copy",
              (), False, "keyboard",
              "ctrl+c (kept primitive for compound seeding)."),
    Primitive(15, "paste",
              (), False, "keyboard",
              "ctrl+v (kept primitive for compound seeding)."),

    # ----- 16..21  screen / observation -------------------------------
    Primitive(16, "screenshot",
              (), False, "screen",
              "Capture a screen frame and return PIL/png bytes."),
    Primitive(17, "wait",
              ("dx",), False, "screen",
              "Sleep for ~|dx| * 1000 ms (max 1.0 s)."),
    Primitive(18, "focus_window",
              ("keysym",), False, "screen",
              "Bring window with title hash=keysym to foreground."),
    Primitive(19, "minimize_window",
              ("keysym",), False, "screen",
              "Minimize window with title hash=keysym."),
    Primitive(20, "get_active_window",
              (), False, "screen",
              "Return the currently focused window's metadata."),
    Primitive(21, "get_screen_size",
              (), False, "screen",
              "Return (W, H) of primary display."),

    # ----- 22..23  system (sandbox-guarded) ----------------------------
    # These DO NOT execute in default sandbox-by-default mode.  They
    # require the explicit --neuromcp-real-os flag at runtime.
    Primitive(22, "file_delete",
              ("token_id",), True, "system",
              "Delete a file at path indexed by token_id (sandbox-guarded)."),
    Primitive(23, "exec_shell",
              ("token_id",), True, "system",
              "Run a shell command from token_id (sandbox-guarded)."),
)


# ---------------------------------------------------------------------------
# Public lookup helpers.
# ---------------------------------------------------------------------------

NUM_PRIMITIVES: int = len(PRIMITIVES)

_BY_ID = {p.id: p for p in PRIMITIVES}
_BY_NAME = {p.name: p for p in PRIMITIVES}


def by_id(primitive_id: int) -> Primitive:
    """Look up a Primitive by integer id.  Raises KeyError on invalid id."""
    return _BY_ID[int(primitive_id)]


def by_name(name: str) -> Primitive:
    """Look up a Primitive by canonical name.  Raises KeyError on miss."""
    return _BY_NAME[str(name)]


def names() -> Tuple[str, ...]:
    """Return the tuple of all primitive names in id order."""
    return tuple(p.name for p in PRIMITIVES)


def sandbox_guarded_ids() -> Tuple[int, ...]:
    """Subset of primitive ids that need sandbox confirmation."""
    return tuple(p.id for p in PRIMITIVES if p.sandbox_guard)


def param_slot_index(slot_name: str) -> int:
    """Return the index of a param slot inside the 8-slot ParamHead vector."""
    if slot_name not in PARAM_SLOTS:
        raise KeyError(f"unknown param slot {slot_name!r}; valid: {PARAM_SLOTS}")
    return PARAM_SLOTS.index(slot_name)


def slot_indices(primitive_id: int) -> Tuple[int, ...]:
    """Return the param-vector indices the given primitive actually reads."""
    p = by_id(primitive_id)
    return tuple(param_slot_index(s) for s in p.param_slots)


__all__ = [
    "Primitive",
    "PARAM_SLOTS",
    "NUM_PARAM_SLOTS",
    "PRIMITIVES",
    "NUM_PRIMITIVES",
    "by_id",
    "by_name",
    "names",
    "sandbox_guarded_ids",
    "param_slot_index",
    "slot_indices",
]
