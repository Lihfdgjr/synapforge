"""synapforge.neuromcp -- closed-loop neural computer-use stack.

The 5-layer NeuroMCP architecture, end-to-end:

    Layer 0: primitives.PRIMITIVES                  -- 24 fixed OS primitives
    Layer 1: action_head.NeuroActionHead.primitive  -- hidden -> primitive_id
    Layer 2: action_head.NeuroActionHead.params     -- hidden -> 8-slot params
    Layer 3: action_head.NeuroActionHead.codebook   -- DynamicCodebook (delegated
                                                       to sf.action.neuromcp)
    Layer 4: compound_growth.CompoundGrowth         -- Hebbian compound emergence
    Layer 5: os_actuator.OSActuator                 -- sandbox / win32 / mcp

The closed-loop env that ties it all together is in
``closed_loop.ClosedLoopEnv``.  Demo recording / replay (the imitation
seed) lives in ``demo_record``.

User铁律 enforced
-----------------
* No ``<tool_call>`` token, no MCP JSON schema registration anywhere.
* Sandbox by default; real-OS opt-in via explicit flag.
* No torch import at top-level so the package imports cleanly without
  GPU stack present (lazy imports inside the action_head class).

Public surface
--------------

>>> from synapforge.neuromcp import (
...     # Layer 0
...     PRIMITIVES, NUM_PRIMITIVES, by_id, by_name,
...     # Layer 1/2/3
...     NeuroActionHead, ActionHeadConfig,
...     # Layer 4
...     CompoundGrowth, CompoundPrototype,
...     # Layer 5
...     OSActuator, ObservationDict,
...     SandboxBackend, Win32Backend, McpControlBackend,
...     # closed-loop
...     ClosedLoopEnv, StepResult,
...     # demo
...     DemoRecorder, DemoReplayer, DemoEvent,
...     # sandbox primitives
...     VirtualDesktop,
... )
"""
from __future__ import annotations

from .primitives import (
    PRIMITIVES,
    NUM_PRIMITIVES,
    NUM_PARAM_SLOTS,
    PARAM_SLOTS,
    Primitive,
    by_id,
    by_name,
    names,
    sandbox_guarded_ids,
    param_slot_index,
    slot_indices,
)
from .compound_growth import CompoundGrowth, CompoundPrototype
from .os_actuator import (
    OSActuator,
    ObservationDict,
    SandboxBackend,
    Win32Backend,
    McpControlBackend,
)
from .sandbox import VirtualDesktop, Button, TextInput
from .closed_loop import ClosedLoopEnv, StepResult
from .demo_record import DemoRecorder, DemoReplayer, DemoEvent

# action_head pulls torch lazily inside the class -- importing the
# *module* is torch-free, but to keep this package's __init__ also
# torch-free at import time we expose the module not the class.
# Users can do ``from synapforge.neuromcp import action_head; head = action_head.NeuroActionHead(...)``
# OR ``from synapforge.neuromcp.action_head import NeuroActionHead``.
from . import action_head  # noqa: F401

# Provide ActionHeadConfig at top level since it's a torch-free dataclass.
from .action_head import ActionHeadConfig

__all__ = [
    # Layer 0
    "PRIMITIVES",
    "NUM_PRIMITIVES",
    "NUM_PARAM_SLOTS",
    "PARAM_SLOTS",
    "Primitive",
    "by_id",
    "by_name",
    "names",
    "sandbox_guarded_ids",
    "param_slot_index",
    "slot_indices",
    # Layer 1/2/3
    "ActionHeadConfig",
    "action_head",
    # Layer 4
    "CompoundGrowth",
    "CompoundPrototype",
    # Layer 5
    "OSActuator",
    "ObservationDict",
    "SandboxBackend",
    "Win32Backend",
    "McpControlBackend",
    # Sandbox
    "VirtualDesktop",
    "Button",
    "TextInput",
    # Closed loop
    "ClosedLoopEnv",
    "StepResult",
    # Demo
    "DemoRecorder",
    "DemoReplayer",
    "DemoEvent",
]


def NeuroActionHead(*args, **kwargs):
    """Lazy factory wrapper around :class:`action_head.NeuroActionHead`.

    Importing the package does NOT import torch.  Calling this factory
    DOES (because it constructs a torch.nn.Module child inside).
    """
    return action_head.NeuroActionHead(*args, **kwargs)


__all__.append("NeuroActionHead")
