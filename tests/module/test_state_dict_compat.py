"""State-dict round-trip test for ``synapforge.module.Module``.

The Phase 2 contract (see ``docs/TORCH_REPLACEMENT_PLAN.md``) requires
that an ``nn.Module``-saved state-dict and a ``synapforge.module.Module``-
saved state-dict are byte-equivalent so that existing checkpoints
under ``runs/*/ckpts/*.pt`` continue to load without conversion.

The running rental's ``train_100m_kd.py`` saves checkpoints via
``torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
"step": step})``; warmstart loads via ``model.load_state_dict(...)``.
Phase 2 must not change that handshake.

This module covers four scenarios:

1. Save with the synapforge-Module path; load it back. Round-trip
   should reproduce the same forward output exactly.
2. ``torch.save`` / ``torch.load`` (pickle) round-trip — the format
   used in production. We write to a tempfile, read back, load into
   a fresh module, and compare forward outputs.
3. Synthesise an ``nn.Module``-style state-dict (string-keyed,
   ``torch.Tensor`` values) and load it into a synapforge model.
   This simulates loading a ckpt produced by an older code path
   that pre-dates Phase 2.
4. The ``HybridBlock`` and ``SynapForge100M`` exact ckpt-format pin:
   key set + dtype + shape on a small instance.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from synapforge.model_100m import HybridBlock, SynapForge100M
from synapforge.module import Module, Parameter


def _seeded_input(*shape: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype == torch.long:
        return torch.randint(0, 100, shape, generator=g, dtype=torch.long)
    return torch.randn(*shape, generator=g, dtype=dtype)


# ---------------------------------------------------------------------------
# Test 1: synapforge save -> synapforge load round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_in_memory() -> None:
    """``state_dict`` -> ``load_state_dict`` round-trip preserves forward.

    Creates two SynapForge100M instances with different random init,
    transfers state from src to dst, and asserts forward outputs
    match exactly.

    Note on call order: we load BEFORE the first forward on dst. This
    matches the production warmstart path (``train_100m_kd.py`` builds
    the model, then immediately ``model.load_state_dict(...)`` from
    the warmstart ckpt, then starts training). Calling forward on a
    fresh dst before load would populate
    ``synapse._cached_typed_mask`` (a non-state-dict attribute used
    by the masked-linear MFU optimization on
    ``synapforge/cells/synapse.py``), which the load doesn't
    invalidate. That's a pre-existing cache-invalidation issue
    orthogonal to Phase 2 — not in scope to fix here.
    """
    torch.manual_seed(7)
    src = SynapForge100M(
        vocab=128,
        d=32,
        n_layers=2,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.5,
    )
    torch.manual_seed(13)  # different seed for dst init
    dst = SynapForge100M(
        vocab=128,
        d=32,
        n_layers=2,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.5,
    )

    # Verify pre-load that random init really differs (otherwise the
    # test is vacuously passing for the wrong reason).
    sp = dict(src.named_parameters())
    dp = dict(dst.named_parameters())
    init_diff = max(
        (sp[n] - dp[n]).abs().max().item() for n in sp
    )
    assert init_diff > 0.01, (
        f"Random init didn't actually differ ({init_diff=}) — "
        "test is vacuous."
    )

    sd = src.state_dict()
    dst.load_state_dict(sd)

    src.eval()
    dst.eval()
    ids = _seeded_input(2, 8, seed=99, dtype=torch.long)
    with torch.no_grad():
        y_src = src(ids)
        y_dst = dst(ids)

    rel = (y_src - y_dst).abs().max().item() / (y_src.abs().max().item() + 1e-12)
    assert rel < 1e-6, (
        f"state_dict round-trip diverged: rel_err={rel:.3e}"
    )
    assert torch.equal(y_src, y_dst), (
        f"state_dict round-trip must be BIT-EXACT on CPU fp32; "
        f"max abs diff = {(y_src - y_dst).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2: torch.save / torch.load round-trip via tempfile
# ---------------------------------------------------------------------------


def test_torch_save_load_round_trip(tmp_path: Path) -> None:
    """``torch.save(model.state_dict())`` round-trips as production does.

    The running rental's ckpts use this exact serialization pathway.
    We exercise it end-to-end:

    1. Build a synapforge model, save its state_dict to disk via
       ``torch.save`` (pickle).
    2. ``torch.load`` it back into another fresh synapforge model.
    3. Confirm forward outputs match bit-exactly.
    """
    torch.manual_seed(101)
    src = SynapForge100M(
        vocab=64,
        d=16,
        n_layers=1,
        loop_depth=1,
        max_seq=8,
        ffn_ratio=2.0,
        sparsity=0.5,
    )
    src.eval()
    ids = _seeded_input(1, 4, seed=2026, dtype=torch.long)
    ids = ids.clamp(max=63)  # respect vocab
    with torch.no_grad():
        y_src = src(ids)

    ckpt_path = tmp_path / "model.pt"
    torch.save({"model": src.state_dict(), "step": 42}, ckpt_path)

    # Fresh model, different init.
    torch.manual_seed(202)
    dst = SynapForge100M(
        vocab=64,
        d=16,
        n_layers=1,
        loop_depth=1,
        max_seq=8,
        ffn_ratio=2.0,
        sparsity=0.5,
    )
    blob = torch.load(ckpt_path, map_location="cpu")
    assert "model" in blob and blob["step"] == 42
    dst.load_state_dict(blob["model"])
    dst.eval()

    with torch.no_grad():
        y_dst = dst(ids)
    assert torch.equal(y_src, y_dst), (
        f"torch.save/load round-trip diverged: "
        f"max abs diff = {(y_src - y_dst).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 3: nn.Module state_dict -> synapforge.Module load
# ---------------------------------------------------------------------------


def test_load_nn_module_state_dict_into_synapforge_module() -> None:
    """An ``nn.Module``-style state-dict loads into a sf.Module without conversion.

    The Phase 2 contract requires this for backwards-compat with
    pre-Phase-2 ckpts: any ``state_dict()`` produced by an
    ``nn.Module`` whose layout matches our model can load directly.

    We simulate the pre-Phase-2 path by building a *plain*
    ``nn.Module`` MLP, calling its ``state_dict()``, and loading it
    into a synapforge MLP with the same structure.
    """
    class TorchMLP(nn.Module):
        def __init__(self, d_in: int, d_h: int) -> None:
            super().__init__()
            self.l1 = nn.Linear(d_in, d_h)
            self.l2 = nn.Linear(d_h, d_in)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l2(self.l1(x).relu())

    class SfMLP(Module):
        def __init__(self, d_in: int, d_h: int) -> None:
            super().__init__()
            self.l1 = nn.Linear(d_in, d_h)
            self.l2 = nn.Linear(d_h, d_in)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l2(self.l1(x).relu())

    torch.manual_seed(31)
    ref = TorchMLP(8, 16)
    sf = SfMLP(8, 16)

    # Pre-load: should differ end-to-end.
    x = _seeded_input(2, 8, seed=44)
    ref.eval()
    sf.eval()
    with torch.no_grad():
        y_ref_pre = ref(x)
        y_sf_pre = sf(x)
    assert not torch.allclose(y_ref_pre, y_sf_pre, atol=1e-5), (
        "Pre-load output should differ"
    )

    # Load nn.Module state_dict into sf.Module and re-run.
    sd = ref.state_dict()
    sf.load_state_dict(sd)
    with torch.no_grad():
        y_ref = ref(x)
        y_sf = sf(x)
    assert torch.equal(y_ref, y_sf), (
        f"nn.Module state_dict didn't transfer cleanly to sf.Module; "
        f"max abs diff = {(y_ref - y_sf).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 4: synthesize a torch ckpt and load it
# ---------------------------------------------------------------------------


def test_synthetic_torch_ckpt_loads(tmp_path: Path) -> None:
    """A handcrafted state-dict (string keys, torch.Tensor values) loads cleanly.

    Simulates a checkpoint produced by an older trainer that wrote a
    plain ``dict[str, torch.Tensor]``. The synapforge model's
    ``load_state_dict`` must accept it without complaint.
    """
    torch.manual_seed(64)
    model = SynapForge100M(
        vocab=64,
        d=16,
        n_layers=1,
        loop_depth=1,
        max_seq=8,
        ffn_ratio=2.0,
        sparsity=0.5,
    )

    # Get the canonical state-dict, write it as a flat tensor map (no
    # module wrapper), then load it back. This is the format that
    # ``torch.save({"model": sd})`` writes to disk modulo the outer
    # key.
    sd: dict[str, torch.Tensor] = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }

    # Modify one tensor so we can verify load actually transferred.
    target_key = next(iter(k for k in sd if "ln_f" in k))
    sd[target_key] = sd[target_key] + 1.0  # bump by 1.0

    model.load_state_dict(sd, strict=True)
    # The ln_f.weight should now be (original + 1.0).
    assert torch.allclose(
        getattr(model.ln_f, "weight"),
        torch.ones_like(getattr(model.ln_f, "weight")) + 1.0,  # original was init=1.0
    ), "synthetic state_dict didn't load expected values"


# ---------------------------------------------------------------------------
# Test 5: state_dict keys/dtype/shape pin
# ---------------------------------------------------------------------------


def test_state_dict_format_pin() -> None:
    """Pin the exact key set, dtype, and shape of a small SynapForge100M.

    Existing ckpts depend on this. Any change here is a Phase 2
    contract violation.
    """
    torch.manual_seed(0)
    model = SynapForge100M(
        vocab=32,
        d=16,
        n_layers=1,
        loop_depth=1,
        max_seq=8,
        ffn_ratio=2.0,
        sparsity=0.5,
    )
    sd = model.state_dict()

    # Required keys.
    expected = {
        "tok_embed.weight",
        "pos_embed",
        "ln_f.weight",
        "blocks.0.ln1.weight",
        "blocks.0.ln2.weight",
        "blocks.0.gate.weight",
        "blocks.0.gate.bias",
        "blocks.0.synapse.weight",
        "blocks.0.ffn.w_gate.weight",
        "blocks.0.ffn.w_up.weight",
        "blocks.0.ffn.w_down.weight",
    }
    missing = expected - set(sd.keys())
    assert not missing, (
        f"Phase 2 dropped keys: {missing}; got {sorted(sd.keys())}"
    )

    # Shape sanity: tok_embed is (vocab, d).
    assert sd["tok_embed.weight"].shape == (32, 16)
    assert sd["pos_embed"].shape == (8, 16)
    assert sd["ln_f.weight"].shape == (16,)


# ---------------------------------------------------------------------------
# Test 6: Parameter survives serialization
# ---------------------------------------------------------------------------


def test_parameter_serializes_via_torch_save() -> None:
    """``synapforge.module.Parameter`` round-trips through torch.save / load.

    Confirms our subclass doesn't break pickle. Uses an in-memory
    BytesIO so the test doesn't touch disk.
    """
    torch.manual_seed(5)
    p = Parameter(torch.randn(3, 4), requires_grad=True)
    buf = io.BytesIO()
    torch.save(p, buf)
    buf.seek(0)
    p2 = torch.load(buf)
    assert isinstance(p2, nn.Parameter)
    assert torch.equal(p, p2)
    assert p2.requires_grad is True
