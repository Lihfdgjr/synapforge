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

__version__ = "1.0.0"

# sf.action — direct neural OS/UI control (NeuroMCP + ActionHead).
# sf.modal -- first-class multimodal embedding (text + image + audio + video).
from . import action, distributed, interop_torch, modal
from .action import (
    ActionHead,
    DynamicActionCodebook,
    NeuroMCPHead,
    OSActionSpec,
    OSActuator,
)
from .cells.liquid import LiquidCell
from .cells.plif import PLIF
from .cells.synapse import SparseSynapse
# 'ParquetTokenStream' available via synapforge.data (requires pyarrow)
from .distributed import PlasticBufferSync, init_dist, wrap_model
from .interop_torch import (
    SFAsTorchModule,
    TorchAsSFModule,
    convert_sparse_to_linear,
    replace_linear_with_sparse,
    replace_relu_with_plif,
)
from .module import Module
from .optim import MultiSourceParam, Param, PlasticityAwareAdamW, build_optimizer
from .plasticity import (
    BCM,
    STDP,
    Hebbian,
    HebbianPlasticity,
    PlasticityEngine,
    PlasticityRule,
    SynaptogenesisGrowPrune,
)
from .runtime import Runtime, compile
from .surrogate import PLIFCell, spike
from .surrogate import register as register_surrogate

# `synapforge.train` was relocated to `legacy/synapforge_train.py` in
# commit cb571f9 (P15+P16+P17 cleanup). The hard import here was missed
# by that commit. Make it defensive so `import synapforge.action.*` and
# friends keep working — `train` becomes None unless the legacy script
# is wired back in.
try:
    from .train import train  # type: ignore
except Exception:  # pragma: no cover - legacy module relocated
    train = None  # type: ignore[assignment]


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

# Long-context (5-tier memory hierarchy).
from .infinite import (
    AdaptiveSlowTau,
    ChunkedStateCarry,
    DeltaCompress,
    DiskMemmapArchive,
    ExternalVectorMemory,
    HierarchicalMemory,
    HierarchicalMemoryConfig,
    InfiniteContextReader,
    InfiniteReaderConfig,
    LocalGQAttention,
    LongContextMonitor,
    RotaryPositionEncoding,
    SSMDiagScan,
    StreamingInfiniteEvaluator,
)

# Intrinsic-motivation (curiosity / homeostasis / self-goals).
from .intrinsic import (
    FreeEnergySurprise,
    GoalMemory,
    HomeostaticRegulator,
    IdleLoop,
    ImaginationRollout,
    IntrinsicReward,
    NoveltyDrive,
    SelfGoalProposer,
)

# Latent-space thinking (Coconut / Quiet-STaR style).
from .latent_thinking import (
    LatentConsistencyLoss,
    LatentLoopController,
    LatentSearchBeam,
    ThinkingActionTokens,
    ThinkingTokens,
)

# Wave / depth mixers (sequence-axis spectral).
from .wave_mixer import (
    FNet1D,
    Hyena1D,
    WaveFormer1D,
    attach_wave_mixer_to_block,
)

# World-model head + hypothesis search.
from .world_model import (
    HypothesisGenerator,
    HypothesisOutput,
    WorldModelCritic,
    WorldModelHead,
    WorldModelLoss,
    WorldModelOutput,
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
    AccelExit,
    ChainOfExperts,
    DeepSeekMoE,
    DepthLoRAAdapter,
    FineGrainedExpert,
    LayerScale,
    LoopIndexEmbedding,
    MoELoadBalanceLoss,
    MoRStack,
    RDTConfig,
    RDTLoop,
    ResidualGateBias,
    RouterOutput,
    SharedExpertGroup,
    TopKRouter,
    attach_coe_to_block,
    attach_moe_to_block,
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

from . import (
    adversarial,  # noqa: E402
    defense,  # noqa: E402
    distill,  # noqa: E402
    self_learn,  # noqa: E402
    tools,  # noqa: E402
)
from .adversarial import (  # noqa: E402
    AdversarialTrainer,
    AdversarialTrainerCfg,
    FastGradientSignAttack,
    ProjectedGradientDescentAttack,
    TokenDiscriminator,
    adversarial_losses,
)
from .defense import (  # noqa: E402
    AdversarialRedTeam,
    DefenseConfig,
    DefenseStack,
    PoisonDetector,
    ProvenanceRecord,
    ProvenanceTracker,
    WeightFirewall,
)
from .distill import (  # noqa: E402
    DistillationLoss,
    DistillConfig,
    DistillTrainer,
    TeacherCache,
)
from .self_learn import (  # noqa: E402
    ExperienceReplayBuffer,
    MAMLAdapter,
    SelfLearnEngine,
    SelfPlayLoop,
    TestTimeTraining,
)
from .tools import (  # noqa: E402
    CodeExecTool,
    ShellTool,
    ToolCaller,
    ToolLearningLoop,
    ToolRegistry,
    ToolSpec,
    WebFetchTool,
    WebScrapeTool,
    WebSearchTool,
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


# ---------------------------------------------------------------------------
# Synap-1 named configs (synapforge.configs) - Base 100M / Pro 300M.
# Re-exported at top-level so callers can write::
#
#     from synapforge import SYNAP1_PRO, build_from_config
#     model = build_from_config("synap1_pro")
#
# The factory builds a `SynapForge100M` (the same class for both sizes -
# Pro is just bigger d / n_layers / ffn_ratio).
# ---------------------------------------------------------------------------
from . import configs  # noqa: E402  re-exported as sf.configs.*
from .configs import (  # noqa: E402
    SYNAP1_BASE,
    SYNAP1_PRO,
    Synap1Config,
)
from .configs import get_config as _get_synap1_config  # noqa: E402
from .configs import list_configs as list_synap1_configs  # noqa: E402


def build_from_config(name, **overrides):
    """Build a ``SynapForge100M`` from a named Synap-1 config.

    Parameters
    ----------
    name : str | Synap1Config
        Either the canonical name (``"synap1_base"`` / ``"synap1_pro"``,
        case- and separator-insensitive) or an already-resolved
        :class:`Synap1Config` instance.
    **overrides
        Any keyword that ``build_synapforge_100m`` accepts (e.g.
        ``loop_depth=4``, ``max_seq=512``, ``latent_k=2``). Overrides win
        over the named config.

    Returns
    -------
    synapforge.model_100m.SynapForge100M
    """
    from .model_100m import build_synapforge_100m
    cfg = name if isinstance(name, Synap1Config) else _get_synap1_config(name)
    kwargs = cfg.kwargs()
    kwargs.update(overrides)
    return build_synapforge_100m(**kwargs)


__all__ += [
    "configs",
    "Synap1Config",
    "SYNAP1_BASE",
    "SYNAP1_PRO",
    "build_from_config",
    "list_synap1_configs",
]
