"""Integration tests for Synap-1 named configs (Base 100M / Pro 300M).

Verifies that:
- ``SYNAP1_BASE`` matches the historical ``model_100m`` factory defaults
  (d=512, n_layers=10) so swapping the config-by-name path in is a no-op
  for existing 100M training jobs.
- ``SYNAP1_PRO`` produces a model whose **total** parameter count is
  within 10 % of 300M and whose **backbone** (everything except the
  ``tok_embed`` table) is within 10 % of 175M, per the Synap-1 Pro
  variant spec in ``docs/NAMING.md``.
- ``build_from_config(name)`` returns a real ``nn.Module`` for either
  name, with the right hidden width / layer count, so launch scripts
  can rely on it.

CPU-only and no GPU dependencies. Uses the project venv's installed
torch; skips the entire module if torch is unavailable (matches the
pattern in ``test_freeze_vocab_tail.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def _count_params(model):
    """Return (total, embed, backbone) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    embed = model.tok_embed.weight.numel()
    backbone = total - embed
    return total, embed, backbone


def test_base_unchanged(torch_mod):
    """``SYNAP1_BASE`` matches the historical 100M factory defaults.

    If this test ever fails, the named config has drifted from the
    canonical ``build_synapforge_100m()`` defaults - either fix the
    config or update the BASE record in ``configs/synap1.py``.
    """
    from synapforge import SYNAP1_BASE, build_from_config
    from synapforge.model_100m import build_synapforge_100m

    # Mirror the keyword arguments BASE claims.
    assert SYNAP1_BASE.d == 512
    assert SYNAP1_BASE.n_layers == 10
    assert SYNAP1_BASE.vocab == 151936
    assert SYNAP1_BASE.ffn_ratio == 8.0

    torch_mod.manual_seed(0)
    m_named = build_from_config("synap1_base")
    torch_mod.manual_seed(0)
    m_default = build_synapforge_100m(
        vocab=SYNAP1_BASE.vocab,
        d=SYNAP1_BASE.d,
        n_layers=SYNAP1_BASE.n_layers,
        loop_depth=SYNAP1_BASE.loop_depth,
        max_seq=SYNAP1_BASE.max_seq,
        ffn_ratio=SYNAP1_BASE.ffn_ratio,
        sparsity=SYNAP1_BASE.sparsity,
        dropout=SYNAP1_BASE.dropout,
    )

    # Architectural identity: same hidden width, same depth, same vocab.
    assert m_named.d == m_default.d == 512
    assert m_named.n_layers == m_default.n_layers == 10
    assert m_named.vocab == m_default.vocab == 151936

    # Same total parameter count - the structural invariant. We don't
    # require bitwise identical weights because nn.init RNG state can
    # differ even with the same seed when sub-modules are constructed
    # in a different call order.
    n_named = sum(p.numel() for p in m_named.parameters())
    n_default = sum(p.numel() for p in m_default.parameters())
    assert n_named == n_default, (
        f"BASE param count drift: named={n_named} default={n_default}"
    )


def test_pro_param_count_about_300M(torch_mod):
    """Pro total params within 10 % of 300M (spec target)."""
    from synapforge import build_from_config

    model = build_from_config("synap1_pro")
    total, embed, backbone = _count_params(model)

    target = 300_000_000
    lo, hi = int(target * 0.9), int(target * 1.1)
    assert lo <= total <= hi, (
        f"PRO total params {total/1e6:.2f}M outside [{lo/1e6:.0f}M, "
        f"{hi/1e6:.0f}M] (target 300M +/- 10 %); embed={embed/1e6:.2f}M, "
        f"backbone={backbone/1e6:.2f}M"
    )


def test_pro_backbone_about_175M(torch_mod):
    """Pro backbone (everything except tok_embed) within 10 % of 175M.

    The embedding (151 936 x 1024) eats ~155.6M, so the backbone is what
    actually does the learning. Spec: ~175M, so ~7x BASE backbone.
    """
    from synapforge import build_from_config

    model = build_from_config("synap1_pro")
    total, embed, backbone = _count_params(model)

    target = 175_000_000
    lo, hi = int(target * 0.9), int(target * 1.1)
    assert lo <= backbone <= hi, (
        f"PRO backbone params {backbone/1e6:.2f}M outside "
        f"[{lo/1e6:.0f}M, {hi/1e6:.0f}M] (target 175M +/- 10 %); "
        f"total={total/1e6:.2f}M, embed={embed/1e6:.2f}M"
    )


def test_build_from_config_returns_module(torch_mod):
    """``build_from_config`` returns a real torch ``nn.Module`` for both
    sizes, with the architecture knobs from the named config and a
    callable forward path on a tiny CPU input.
    """
    from synapforge import build_from_config

    nn = torch_mod.nn

    base = build_from_config("synap1_base")
    pro = build_from_config("synap1_pro")

    # Both must be nn.Modules.
    assert isinstance(base, nn.Module)
    assert isinstance(pro, nn.Module)

    # Architecture knobs propagate from the named config.
    assert base.d == 512 and base.n_layers == 10
    assert pro.d == 1024 and pro.n_layers == 14
    assert pro.vocab == base.vocab == 151936

    # build_from_config also accepts overrides, so launch scripts can
    # tweak loop_depth / max_seq / latent_k without redefining the config.
    pro_lk = build_from_config("synap1_pro", latent_k=0)
    assert pro_lk.latent_k == 0

    # Aliases work (case-/separator-insensitive).
    pro_alias = build_from_config("Synap-1 Pro")
    assert pro_alias.d == 1024 and pro_alias.n_layers == 14

    # Forward smoke on Base (Pro forward is too memory-heavy for CI on
    # CPU when run alongside Base; the architectural identity test above
    # already verifies Pro instantiates correctly).
    ids = torch_mod.zeros((1, 4), dtype=torch_mod.long)
    with torch_mod.no_grad():
        logits = base(ids)
    assert logits.shape == (1, 4, base.vocab)
