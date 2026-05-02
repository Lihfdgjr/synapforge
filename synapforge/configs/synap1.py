"""Synap-1 model configurations.

Three named configs are exported here so launch scripts, demos, paper
benchmarks, and unit tests can reference a model size by **name**
(e.g. ``SYNAP1_ULTRA``) rather than passing a half-dozen ``--d / --n-layers /
--ffn-ratio`` flags around.

All configs use the same ``synapforge.model_100m.SynapForge100M`` class
- only the hyperparameters differ. The architecture is **100% LNN+SNN**
(LiquidCell + PLIF + SparseSynapse + SwiGLU FFN, RDT-loopable). No LoRA,
no transformer fallback - per the project iron rule (memory key
``feedback_no_lora_no_transformer_fallback``).

Naming
------
- ``SYNAP1_BASE``  - the historical "Synap-1" 100M-class model
  (d=512, n_layers=10). Backbone ~73.5M, total ~151M (the embedding
  alone is 77.8M because vocab=151936 from Qwen 2.5).
- ``SYNAP1_PRO``   - the 300M-class scaled variant
  (d=1024, n_layers=14, ffn_ratio=2.5). Backbone ~169M, total ~325M.
  Backbone parameter count is **~2.3x BASE** at the same vocab.
- ``SYNAP1_ULTRA`` - the 500M-class scaled variant
  (d=1280, n_layers=16, ffn_ratio=3.0, loop_depth=2). Backbone ~341M,
  total ~536M. Backbone is **~4.6x BASE** (~13.7x the ~25M useful 100M
  baseline backbone after stripping the dominant embedding table).

The Pro variant trades FFN width (8 -> 2.5) for hidden width (512 -> 1024)
and depth (10 -> 14). At ffn_ratio=8.0 a d=1024 n=14 model is 567M total
which would dwarf the embedding bias and overshoot the 300M target by
nearly 2x; ffn_ratio=2.5 keeps total within 10% of 300M and backbone
within 10% of 175M as required by the variant spec.

The Ultra variant scales hidden width again (1024 -> 1280) and depth
(14 -> 16) and uses ffn_ratio=3.0 (slightly above Pro's 2.5). The Pro
ratio of 2.5 was tuned to fit the 300M total target with d=1024; at
d=1280 / n_layers=16 we need ~3.0 to push the **backbone** budget into
the 350M target band (a 14x multiplier over the 100M's ~25M backbone)
without overshooting the 500M total too far. RDT loop_depth=2 doubles
the effective compute path without adding parameters (weight-shared
loop), giving Ultra a "wider plus deeper plus more recurrent" profile
that matches the d=1280 / n=16 / ffn=3 spec to ~535.8M total / ~341.3M
backbone.

Usage
-----
::

    from synapforge.configs import SYNAP1_BASE, SYNAP1_PRO
    from synapforge import build_from_config

    model = build_from_config("synap1_pro")
    # or pass kwargs through the existing factory:
    from synapforge.model_100m import build_synapforge_100m
    model = build_synapforge_100m(**SYNAP1_PRO.kwargs())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Synap1Config:
    """Frozen hyperparameter bag for a Synap-1 size class.

    Mirrors the ``build_synapforge_100m`` keyword arguments 1:1 so that
    ``build_synapforge_100m(**cfg.kwargs())`` is the canonical instantiation
    path. Anything not in this dataclass keeps its ``model_100m`` default.
    """

    name: str
    # Tokenizer / vocabulary.
    vocab: int = 151936
    # Hidden width (per HybridBlock) and stack depth.
    d: int = 512
    n_layers: int = 10
    # RDT depth-loop count (1 = no recurrence; >1 = weight-shared loop).
    loop_depth: int = 1
    # Max sequence length the position-embed table is sized for.
    max_seq: int = 256
    # SwiGLU FFN expansion factor (hidden = ratio * d).
    ffn_ratio: float = 8.0
    # SparseSynapse mask density (0.95 = 5 % connections kept).
    sparsity: float = 0.95
    # Per-block dropout. Always 0 in pretraining; tune in SFT/RL only.
    dropout: float = 0.0
    # T2.4 - freeze rows [live_vocab, vocab) of tok_embed (Qwen 2.5 padding).
    freeze_vocab_tail: bool = True
    live_vocab: int = 151643
    # T2.6 / T7.3 / T2.8 / T2.9 - all opt-in, default off so warmstart
    # checkpoints from earlier runs continue to load strictly.
    lm_head_spectral_norm: bool = False
    lm_head_pre_ln: bool = False
    weight_quant_cfc: str = "none"
    latent_k: int = 0

    def kwargs(self) -> dict:
        """Return kwargs dict ready for ``build_synapforge_100m``.

        Drops the ``name`` field (only used for logging / config-by-name
        dispatch) so the result is exactly the factory signature.
        """
        d = asdict(self)
        d.pop("name", None)
        return d


# ---------------------------------------------------------------------------
# Synap-1 Base - 100M class, d=512 n=10 ffn=8.
# Backbone ~73.5M, total ~151M (embed 77.8M dominates because vocab=151936).
# Identical to the historical default of `build_synapforge_100m()`.
# ---------------------------------------------------------------------------
SYNAP1_BASE = Synap1Config(
    name="synap1_base",
    vocab=151936,
    d=512,
    n_layers=10,
    loop_depth=1,
    max_seq=256,
    ffn_ratio=8.0,
    sparsity=0.95,
    dropout=0.0,
)


# ---------------------------------------------------------------------------
# Synap-1 Pro - 300M class, d=1024 n=14 ffn=2.5.
# Backbone ~169M (~2.3x BASE backbone), total ~325M (within 10 % of 300M).
# Same LNN+SNN primitives, just wider hidden + deeper stack.
# ---------------------------------------------------------------------------
SYNAP1_PRO = Synap1Config(
    name="synap1_pro",
    vocab=151936,
    d=1024,
    n_layers=14,
    # loop_depth=2 matches docs/NAMING.md "Synap-1 Pro spec" - the Pro
    # tier is RDT-recurred 2x to compensate for fewer dense FFN params.
    # `loop_depth` does NOT change parameter count (weight-shared), so
    # the 300M total / 175M backbone budget still holds.
    loop_depth=2,
    max_seq=256,
    ffn_ratio=2.5,
    sparsity=0.95,
    dropout=0.0,
)


# ---------------------------------------------------------------------------
# Synap-1 Ultra - 500M class, d=1280 n=16 ffn=3.0 loop_depth=2.
# Backbone ~341M (~4.6x BASE, ~13.7x the 100M useful-backbone of ~25M),
# total ~536M (within 10 % of 500M target). Same LNN+SNN primitives,
# wider hidden + deeper stack + RDT recurrence.
# ---------------------------------------------------------------------------
SYNAP1_ULTRA = Synap1Config(
    name="synap1_ultra",
    vocab=151936,
    d=1280,
    n_layers=16,
    # loop_depth=2 weight-shared RDT recurrence -- doubles effective
    # compute depth (16 layers x 2 = 32 effective layer-passes) without
    # adding any parameters. Matches Pro's RDT recipe but on a deeper
    # stack so the recurrent compute path is meaningfully larger.
    loop_depth=2,
    max_seq=256,
    # ffn_ratio=3.0 (vs Pro's 2.5) is the sweet spot at d=1280 / n=16:
    # ratio=2.5 leaves backbone at ~302M (under the 315M lower bound for
    # the 350M +/- 10% target); ratio=3.5 overshoots total to 575M
    # (above 550M cap); ratio=3.0 lands at 535.8M total / 341.3M backbone,
    # both inside the +/- 10% bands for 500M total / 350M backbone.
    ffn_ratio=3.0,
    sparsity=0.95,
    dropout=0.0,
)


# ---------------------------------------------------------------------------
# Registry for dispatch-by-name (case-insensitive, dashes/underscores OK).
# Aliases let callers say "synap1-base" / "Synap-1 Pro" / "pro" too.
# ---------------------------------------------------------------------------
_REGISTRY: dict = {
    "synap1_base": SYNAP1_BASE,
    "synap1-base": SYNAP1_BASE,
    "base": SYNAP1_BASE,
    "synap1_pro": SYNAP1_PRO,
    "synap1-pro": SYNAP1_PRO,
    "pro": SYNAP1_PRO,
    "synap1_ultra": SYNAP1_ULTRA,
    "synap1-ultra": SYNAP1_ULTRA,
    "ultra": SYNAP1_ULTRA,
}


def get_config(name: str) -> Synap1Config:
    """Look a config up by name. Case- and separator-insensitive.

    Accepts any of: ``synap1_pro`` / ``synap1-pro`` / ``Synap-1 Pro`` /
    ``SynapPro`` / ``pro`` (and analogous for Base). Internally we
    normalise by lowercasing, mapping spaces and dashes to underscores,
    then stripping the explicit ``"1"`` between ``synap`` and the tier
    suffix so ``synap-1-pro`` and ``synap1_pro`` collapse to the same key.
    """
    if name is None:
        raise KeyError("config name is required (got None)")
    raw = str(name).strip().lower()
    # Try the registry as-is first (covers "pro", "base", "synap1_pro").
    if raw in _REGISTRY:
        return _REGISTRY[raw]
    # Normalise separators: spaces and dashes -> underscores.
    key = raw.replace(" ", "_").replace("-", "_")
    if key in _REGISTRY:
        return _REGISTRY[key]
    # Collapse "synap_1_*" -> "synap1_*" (handle "synap-1 pro").
    key2 = key.replace("synap_1_", "synap1_").replace("synap1__", "synap1_")
    if key2 in _REGISTRY:
        return _REGISTRY[key2]
    # Final attempt: collapse all underscores ("synappro" / "synap1pro").
    key3 = key2.replace("_", "")
    for k, cfg in _REGISTRY.items():
        if k.replace("_", "").replace("-", "") == key3:
            return cfg
    raise KeyError(
        f"unknown synap1 config {name!r}; "
        f"known: {sorted(set(_REGISTRY.keys()))}"
    )


def list_configs() -> list:
    """Return canonical config names (deduped, sorted)."""
    return sorted({c.name for c in _REGISTRY.values()})


__all__ = [
    "Synap1Config",
    "SYNAP1_BASE",
    "SYNAP1_PRO",
    "SYNAP1_ULTRA",
    "get_config",
    "list_configs",
]
