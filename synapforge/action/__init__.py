"""sf.action — direct neural control for OS/UI actions, no tokens, no MCP.

The model never emits a JSON-schema tool call or a `<tool_call>` token; the
hidden state goes straight into a structured action vector that the OS
actuator agent consumes.

Three building blocks:

    ActionHead       hidden -> {action_type, xy, scroll, key, text} dict
                     (RT-2 / OpenVLA-style structured action vector)
    NeuroMCPHead     SparseSynapse + DynamicActionCodebook for *neuroplastic
                     tool acquisition* — neurons grow new prototypes and
                     synapses for each new tool/skill, no schema required
    OSActuator       executes the action dict on Windows/Mac/Linux via
                     pyautogui (or safe_mode print) — closes the loop
    ScreenObservation captures pixels into a (3,H,W) torch.Tensor

Usage
-----
    >>> import synapforge as sf
    >>> import synapforge.action as sfa
    >>>
    >>> class Agent(sf.Module):
    ...     def __init__(self, hidden=256):
    ...         super().__init__()
    ...         self.encoder  = sfa.PatchEncoder(patch=8, hidden=hidden)
    ...         self.block    = sf.LiquidCell(hidden, hidden)
    ...         self.neuromcp = sfa.NeuroMCPHead(hidden,
    ...                                         codebook_initial=9,
    ...                                         codebook_max=64,
    ...                                         synapse_density=0.05,
    ...                                         synapse_max_density=0.4)
    ...         self.action_head = sfa.ActionHead(hidden, sfa.OSActionSpec.default())
    ...     def forward(self, screen):
    ...         z = self.encoder(screen).mean(dim=1, keepdim=True)
    ...         h = self.block(z)
    ...         act_logits = self.neuromcp(h)
    ...         actions    = self.action_head(h)
    ...         return {"actions": actions, "action_logits": act_logits}
    >>> agent = Agent().cuda()
    >>> actuator = sfa.OSActuator(safe_mode=True)
"""

from __future__ import annotations

from .actuator import OSActuator, ScreenObservation
from .envs import FourButtonEnv, PatchEncoder, SpatialXYHead
from .head import (
    ACTION_TYPES,
    KEY_VOCAB,
    ActionHead,
    ActionLoss,
    ActionOutput,
    ActionTargets,
    OSActionSpec,
)
from .neuromcp import (
    CodebookConfig,
    DynamicActionCodebook,
    NeuroMCPHead,
    SparseSynapticLayer,
    SynaptogenesisConfig,
)
from .skill_log import SkillEntry, SkillLog
from .per_domain_neuromcp import PerDomainNeuroMCP, SingleDomainHead

__all__ = [
    # head.py
    "ActionHead",
    "ActionOutput",
    "ActionLoss",
    "ActionTargets",
    "OSActionSpec",
    "ACTION_TYPES",
    "KEY_VOCAB",
    # neuromcp.py
    "NeuroMCPHead",
    "DynamicActionCodebook",
    "SparseSynapticLayer",
    "CodebookConfig",
    "SynaptogenesisConfig",
    # actuator.py
    "OSActuator",
    "ScreenObservation",
    # envs.py
    "FourButtonEnv",
    "PatchEncoder",
    "SpatialXYHead",
    # v4.2: persistent skill memory + per-domain heads
    "SkillEntry",
    "SkillLog",
    "PerDomainNeuroMCP",
    "SingleDomainHead",
]
