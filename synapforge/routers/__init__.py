"""synapforge.routers — depth/width routing primitives for sf.Module stacks.

Four building blocks ported from mscfc, refactored to be backend-agnostic
(work with both the v0.1 ``gpu_dense`` backend and the v0.2 ``triton_block``
backend) and bf16-friendly.

Public API
----------
    >>> from synapforge.routers import MoRStack, ChainOfExperts, DeepSeekMoE, RDTLoop
    >>> body = MyBlock(d=256)                       # any sf.Module
    >>> mor = MoRStack(body, hidden=256, max_depth=4)
    >>> y, depth_loss = mor(x)
    >>> rdt = RDTLoop(body, hidden=256, max_loops_train=4, max_loops_infer=8)
    >>> y = rdt(x)

Naming + design
---------------
- All four are plain ``sf.Module`` subclasses (no monkey patching of the
  body's forward — that legacy mscfc pattern broke autograd graph sharing
  and made backend rewrites hairy).
- Each router accepts an arbitrary ``body: nn.Module`` and only assumes its
  output's first element is the feature tensor (so they work with both
  ``HybridBlock(x, h, mem) -> (x, h, mem, sr, W)`` and ``Block(x) -> x``).
- ``RDTLoop`` is the most opinionated: it wraps a body and runs ``R``
  recurrent passes with per-step embedding/LoRA/gate-bias/early-exit.

Author note: routers don't replace ``sf.HybridBlock`` — they wrap it.
That's why every API takes a ``body=`` arg. CfC+PLIF stays untouched.
"""

from __future__ import annotations

from .mor import MoRStack, MoRRouter
from .coe import ChainOfExperts, attach_coe_to_block
from .moe import (
    DeepSeekMoE,
    FineGrainedExpert,
    SharedExpertGroup,
    TopKRouter,
    RouterOutput,
    MoELoadBalanceLoss,
    attach_moe_to_block,
)
from .rdt import (
    RDTLoop,
    RDTConfig,
    LoopIndexEmbedding,
    LayerScale,
    ResidualGateBias,
    DepthLoRAAdapter,
    AccelExit,
)

__all__ = [
    # MoR
    "MoRStack",
    "MoRRouter",
    # CoE
    "ChainOfExperts",
    "attach_coe_to_block",
    # MoE
    "DeepSeekMoE",
    "FineGrainedExpert",
    "SharedExpertGroup",
    "TopKRouter",
    "RouterOutput",
    "MoELoadBalanceLoss",
    "attach_moe_to_block",
    # RDT
    "RDTLoop",
    "RDTConfig",
    "LoopIndexEmbedding",
    "LayerScale",
    "ResidualGateBias",
    "DepthLoRAAdapter",
    "AccelExit",
]
