"""synapforge — purpose-built LNN+SNN framework (v0.1).

v0.1 surfaces the public API (sf.Module / sf.LiquidCell / sf.PLIF /
sf.SparseSynapse / sf.HebbianPlasticity / sf.STDP) on a GPU-dense backend
that delegates to PyTorch. Numerical-equivalent to mscfc.LiquidS4Cell.

Roadmap
-------
v0.1  PyTorch-backed dense backend, API surface, correctness tests   <- HERE
v0.2  Triton-fused parallel scan (3-5x forward speedup target)
v0.3  Event-driven CPU backend (numba) for inference
v0.4  Lava export for neuromorphic deployment
v1.0  Distributed training + plasticity-aware autograd
"""

from __future__ import annotations

__version__ = "0.1.0"

from .module import Module
from .cells.liquid import LiquidCell
from .cells.plif import PLIF
from .cells.synapse import SparseSynapse
from .plasticity import (
    HebbianPlasticity, STDP,
    PlasticityRule, Hebbian, BCM, SynaptogenesisGrowPrune, PlasticityEngine,
)
from .runtime import compile, Runtime
from .optim import build_optimizer, MultiSourceParam, PlasticityAwareAdamW, Param
from .surrogate import spike, PLIFCell, register as register_surrogate
from .train import train
from .data import ParquetTokenStream
from . import distributed
from .distributed import init_dist, wrap_model, PlasticBufferSync
from . import interop_torch
from .interop_torch import (
    SFAsTorchModule, TorchAsSFModule,
    replace_linear_with_sparse, replace_relu_with_plif,
    convert_sparse_to_linear,
)
# sf.action — direct neural OS/UI control (NeuroMCP + ActionHead).
from . import action
from .action import (
    ActionHead,
    NeuroMCPHead,
    OSActuator,
    OSActionSpec,
    DynamicActionCodebook,
)


# sf.modal -- first-class multimodal embedding (text + image + audio + video).
from . import modal


def tied_lm_head(hidden, vocab, embedding=None):
    """Build an LM head whose .weight is tied to a token embedding."""
    import torch.nn as _nn
    head = _nn.Linear(hidden, vocab, bias=False)
    if embedding is not None:
        if embedding.embedding_dim != hidden or embedding.num_embeddings != vocab:
            raise ValueError(
                f"embedding shape ({embedding.num_embeddings}, {embedding.embedding_dim}) "
                f"does not match (vocab={vocab}, hidden={hidden})"
            )
        head.weight = embedding.weight
    return head


def __getattr__(name):
    if name in ("hf_trainer", "SFTrainer", "PlasticityCallback"):
        from . import hf_trainer as _hf
        if name == "hf_trainer":
            return _hf
        return getattr(_hf, name)
    raise AttributeError(f"module 'synapforge' has no attribute {name!r}")

__all__ = [
    "__version__",
    "Module", "LiquidCell", "PLIF", "SparseSynapse",
    "HebbianPlasticity", "STDP",
    "PlasticityRule", "Hebbian", "BCM", "SynaptogenesisGrowPrune", "PlasticityEngine",
    "compile", "Runtime",
    "build_optimizer", "MultiSourceParam", "PlasticityAwareAdamW", "Param",
    "spike", "PLIFCell", "register_surrogate",
    "train", "ParquetTokenStream",
    "distributed", "init_dist", "wrap_model", "PlasticBufferSync",
    "interop_torch",
    "SFAsTorchModule", "TorchAsSFModule",
    "replace_linear_with_sparse", "replace_relu_with_plif", "convert_sparse_to_linear",
    "hf_trainer", "SFTrainer", "PlasticityCallback",
    # action (sf.action.*)
    "action",
    "ActionHead",
    "NeuroMCPHead",
    "OSActuator",
    "OSActionSpec",
    "DynamicActionCodebook",
    # modal (sf.modal.*)
    "modal",
    "tied_lm_head",
]


# Bio-inspired modules (sub-package).
from . import bio  # re-exported as sf.bio.{KWTA, TauSplit, ...}

# Wave / depth mixers (sequence-axis spectral).
from .wave_mixer import (
    WaveFormer1D, Hyena1D, FNet1D, attach_wave_mixer_to_block,
)

# World-model head + hypothesis search.
from .world_model import (
    WorldModelHead, WorldModelLoss, WorldModelOutput,
    HypothesisGenerator, HypothesisOutput, WorldModelCritic,
)

# Latent-space thinking (Coconut / Quiet-STaR style).
from .latent_thinking import (
    ThinkingTokens, ThinkingActionTokens,
    LatentLoopController, LatentConsistencyLoss, LatentSearchBeam,
)

# Intrinsic-motivation (curiosity / homeostasis / self-goals).
from .intrinsic import (
    FreeEnergySurprise, SelfGoalProposer, ImaginationRollout,
    NoveltyDrive, HomeostaticRegulator, IdleLoop, GoalMemory,
    IntrinsicReward,
)

# Long-context (5-tier memory hierarchy).
from .infinite import (
    RotaryPositionEncoding, LocalGQAttention,
    HierarchicalMemoryConfig, HierarchicalMemory, DeltaCompress,
    AdaptiveSlowTau, SSMDiagScan,
    ExternalVectorMemory, DiskMemmapArchive,
    InfiniteReaderConfig, InfiniteContextReader,
    ChunkedStateCarry, LongContextMonitor, StreamingInfiniteEvaluator,
)

__all__ += [
    "bio",
    # wave_mixer
    "WaveFormer1D", "Hyena1D", "FNet1D", "attach_wave_mixer_to_block",
    # world_model
    "WorldModelHead", "WorldModelLoss", "WorldModelOutput",
    "HypothesisGenerator", "HypothesisOutput", "WorldModelCritic",
    # latent_thinking
    "ThinkingTokens", "ThinkingActionTokens",
    "LatentLoopController", "LatentConsistencyLoss", "LatentSearchBeam",
    # intrinsic
    "FreeEnergySurprise", "SelfGoalProposer", "ImaginationRollout",
    "NoveltyDrive", "HomeostaticRegulator", "IdleLoop", "GoalMemory",
    "IntrinsicReward",
    # infinite
    "RotaryPositionEncoding", "LocalGQAttention",
    "HierarchicalMemoryConfig", "HierarchicalMemory", "DeltaCompress",
    "AdaptiveSlowTau", "SSMDiagScan",
    "ExternalVectorMemory", "DiskMemmapArchive",
    "InfiniteReaderConfig", "InfiniteContextReader",
    "ChunkedStateCarry", "LongContextMonitor", "StreamingInfiniteEvaluator",
]


# Routers (sub-package: MoR / CoE / MoE / RDT depth-control primitives).
from . import routers  # re-exported as sf.routers.{ChainOfExperts, RDTLoop, ...}
from .routers import (
    MoRStack, ChainOfExperts, RDTLoop, RDTConfig,
    LoopIndexEmbedding, LayerScale, ResidualGateBias, DepthLoRAAdapter, AccelExit,
    DeepSeekMoE, FineGrainedExpert, SharedExpertGroup, TopKRouter,
    RouterOutput, MoELoadBalanceLoss,
    attach_coe_to_block, attach_moe_to_block,
)

# Aliases preserving the original mscfc class names so legacy callers /
# task briefs that referenced ``ChainOfExpertsMoE`` / ``EnhancedLoopStack``
# / ``RDTLoopConfig`` continue to import without rewriting call sites.
ChainOfExpertsMoE = ChainOfExperts
EnhancedLoopStack = RDTLoop
RDTLoopConfig = RDTConfig
# Surface the same aliases under sf.routers.* for one-stop import.
routers.ChainOfExpertsMoE = ChainOfExperts
routers.EnhancedLoopStack = RDTLoop
routers.RDTLoopConfig = RDTConfig

__all__ += [
    "routers",
    "MoRStack", "ChainOfExperts", "RDTLoop", "RDTConfig",
    "LoopIndexEmbedding", "LayerScale", "ResidualGateBias",
    "DepthLoRAAdapter", "AccelExit",
    "DeepSeekMoE", "FineGrainedExpert", "SharedExpertGroup", "TopKRouter",
    "RouterOutput", "MoELoadBalanceLoss",
    "attach_coe_to_block", "attach_moe_to_block",
    # Mscfc-naming aliases.
    "ChainOfExpertsMoE", "EnhancedLoopStack", "RDTLoopConfig",
]


# ---------------------------------------------------------------------------
# Round 9: self-learning / defense / tool-use / adversarial / distillation
# (per memory feedback_self_learning.md, feedback_self_learn_poison_defense.md,
#  feedback_tool_use_web_access.md — these are MANDATORY bedrock features).
# ---------------------------------------------------------------------------

from . import self_learn  # noqa: E402
from .self_learn import (  # noqa: E402
    ExperienceReplayBuffer, TestTimeTraining, SelfPlayLoop,
    MAMLAdapter, SelfLearnEngine,
)

from . import defense  # noqa: E402
from .defense import (  # noqa: E402
    PoisonDetector, ProvenanceTracker, ProvenanceRecord,
    WeightFirewall, AdversarialRedTeam,
    DefenseConfig, DefenseStack,
)

from . import tools  # noqa: E402
from .tools import (  # noqa: E402
    ToolSpec, ToolRegistry,
    WebSearchTool, WebFetchTool, WebScrapeTool,
    ShellTool, CodeExecTool,
    ToolCaller, ToolLearningLoop,
)

from . import adversarial  # noqa: E402
from .adversarial import (  # noqa: E402
    FastGradientSignAttack, ProjectedGradientDescentAttack,
    TokenDiscriminator, adversarial_losses,
    AdversarialTrainerCfg, AdversarialTrainer,
)

from . import distill  # noqa: E402
from .distill import (  # noqa: E402
    TeacherCache, DistillationLoss, DistillConfig, DistillTrainer,
)

__all__ += [
    "self_learn",
    "ExperienceReplayBuffer", "TestTimeTraining", "SelfPlayLoop",
    "MAMLAdapter", "SelfLearnEngine",
    "defense",
    "PoisonDetector", "ProvenanceTracker", "ProvenanceRecord",
    "WeightFirewall", "AdversarialRedTeam",
    "DefenseConfig", "DefenseStack",
    "tools",
    "ToolSpec", "ToolRegistry",
    "WebSearchTool", "WebFetchTool", "WebScrapeTool",
    "ShellTool", "CodeExecTool",
    "ToolCaller", "ToolLearningLoop",
    "adversarial",
    "FastGradientSignAttack", "ProjectedGradientDescentAttack",
    "TokenDiscriminator", "adversarial_losses",
    "AdversarialTrainerCfg", "AdversarialTrainer",
    "distill",
    "TeacherCache", "DistillationLoss", "DistillConfig", "DistillTrainer",
]
