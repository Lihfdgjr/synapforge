"""sf.action — direct neural control for OS/UI actions, no tokens, no MCP.

The model never emits a JSON-schema tool call or a `<tool_call>` token; the
hidden state goes straight into a structured action vector that the OS
actuator agent consumes.

Building blocks:

    ActionHead       hidden -> {action_type, xy, scroll, key, text} dict
    NeuroMCPHead     SparseSynapse + DynamicActionCodebook for *neuroplastic
                     tool acquisition* — neurons grow new prototypes and
                     synapses for each new tool/skill, no schema required
    PerDomainNeuroMCP  4 codebooks (math/chat/code/web) + intent router
    HNSWSkillIndex   O(log K) lookup over 100k+ persistent prototypes
    HierarchicalCodebook  L1 primitives + L2 compounds (Hebbian co-firing)
    OSActuator       executes the action dict on Windows/Mac/Linux via
                     pyautogui (or safe_mode print) — closes the loop
    SkillLog         JSON-backed persistent prototype memory (LTP/LTD)
    ScreenObservation captures pixels into a (3,H,W) torch.Tensor
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
from .per_domain_neuromcp import (
    PerDomainNeuroMCP,
    SingleDomainHead,
    DynamicCodebook,
)
from .hnsw_skill_index import (
    HNSWSkillIndex,
    SkillRecord,
    migrate_from_skill_log,
)
from .compositional_codebook import (
    CompoundPrototype,
    TemporalAttentionPooler,
    CoFiringDetector,
    HierarchicalCodebook,
)

__all__ = [
    # head.py
    "ActionHead",
    "ActionOutput",
    "ActionLoss",
    "ActionTargets",
    "OSActionSpec",
    "ACTION_TYPES",
    "KEY_VOCAB",
    # neuromcp.py (legacy)
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
    # skill_log.py
    "SkillEntry",
    "SkillLog",
    # per_domain_neuromcp.py
    "PerDomainNeuroMCP",
    "SingleDomainHead",
    "DynamicCodebook",
    # hnsw_skill_index.py (NEW)
    "HNSWSkillIndex",
    "SkillRecord",
    "migrate_from_skill_log",
    # compositional_codebook.py (NEW)
    "CompoundPrototype",
    "TemporalAttentionPooler",
    "CoFiringDetector",
    "HierarchicalCodebook",
]
