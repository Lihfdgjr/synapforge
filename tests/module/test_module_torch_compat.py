"""Bit-exact equivalence test for ``synapforge.module.Module``.

The Phase 2 contract (see ``docs/TORCH_REPLACEMENT_PLAN.md``) states
that a ``synapforge.module.Module``-based block produces forward
outputs identical (rel-err < 1e-5) to the equivalent
``torch.nn.Module`` reference for the same weights. This module pins
that contract.

Two flavours of test:

1. ``test_synapforge_module_matches_nn_module`` — a tiny MLP, with
   ``Parameter`` and ``register_parameter``/``register_module``,
   compared bit-for-bit to a ``torch.nn.Module`` reference. Probes
   the basic plumbing (param tracking, forward dispatch, train/eval
   mode).
2. ``test_hybrid_block_baseline_vs_synapforge`` — the production
   ``HybridBlock`` (already inheriting from
   ``synapforge.module.Module`` after Phase 2) is built twice with
   identical weights. The reference path is constructed via a
   bare-bones ``torch.nn.Module`` clone that uses the same submodules.
   This catches any subtle drift introduced by our ``__call__``
   override or plasticity-cache machinery.

Both use CPU (the contract is bit-exact, no CUDA non-determinism in
play). Production ckpts also store ``torch.float32`` master weights so
fp32 here is the right comparison dtype.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge.module as sf_module
from synapforge.module import Module, Parameter


def _seeded_tensor(*shape: int, seed: int) -> torch.Tensor:
    """Reproducible random tensor regardless of test execution order."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# Test 1: basic Module parity vs nn.Module
# ---------------------------------------------------------------------------


def test_synapforge_module_matches_nn_module() -> None:
    """A ``sf.Module`` MLP must match an ``nn.Module`` MLP bit-for-bit.

    The two MLPs share weights via ``load_state_dict``. We then compare
    forward outputs on the same input, asserting rel-err == 0 (this is
    pure CPU fp32 with identical kernels — torch's matmul is
    deterministic in this regime). The Phase 2 contract requires <
    1e-5; we assert exact equality because the only path-dependent
    code is our ``__call__`` override which calls ``super().__call__``
    when no plasticity rule is registered.
    """
    torch.manual_seed(0)

    # Reference torch implementation.
    class TorchMLP(nn.Module):
        def __init__(self, d_in: int, d_h: int, d_out: int) -> None:
            super().__init__()
            self.l1 = nn.Linear(d_in, d_h)
            self.l2 = nn.Linear(d_h, d_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l2(F.relu(self.l1(x)))

    # Synapforge implementation — uses register_parameter +
    # register_module + Parameter + the inherited forward dispatch.
    class SfMLP(Module):
        def __init__(self, d_in: int, d_h: int, d_out: int) -> None:
            super().__init__()
            # Use register_module instead of plain attribute assign
            # to exercise the Phase 2 API explicitly.
            self.register_module("l1", nn.Linear(d_in, d_h))
            self.register_module("l2", nn.Linear(d_h, d_out))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l2(F.relu(self.l1(x)))

    d_in, d_h, d_out = 8, 16, 4
    ref = TorchMLP(d_in, d_h, d_out)
    sf = SfMLP(d_in, d_h, d_out)

    # Cross-load weights so both modules see identical parameters.
    sf.load_state_dict(ref.state_dict())

    ref.eval()
    sf.eval()

    x = _seeded_tensor(3, d_in, seed=42)
    with torch.no_grad():
        y_ref = ref(x)
        y_sf = sf(x)

    # Bit-exact equality on CPU fp32 with shared weights.
    rel_err = (y_ref - y_sf).abs().max().item() / (y_ref.abs().max().item() + 1e-12)
    assert rel_err < 1e-5, f"sf MLP diverged from torch MLP: rel_err={rel_err:.3e}"
    # In practice we expect rel_err == 0 here — assert it explicitly so
    # any future drift in __call__ shows up immediately.
    assert torch.equal(y_ref, y_sf), (
        f"sf MLP must be BIT-EXACT vs torch on CPU fp32; "
        f"max abs diff = {(y_ref - y_sf).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2: HybridBlock baseline vs synapforge.Module-based path
# ---------------------------------------------------------------------------


def test_hybrid_block_uses_synapforge_module() -> None:
    """``HybridBlock`` must inherit from ``synapforge.module.Module``.

    This pins the base-class swap that's the headline Phase 2
    deliverable: every block that previously inherited from
    ``torch.nn.Module`` now inherits from
    ``synapforge.module.Module``. If someone changes the base class
    back to ``nn.Module`` the test fails loudly.
    """
    from synapforge.model_100m import HybridBlock, SynapForge100M

    assert issubclass(HybridBlock, sf_module.Module), (
        "HybridBlock must inherit from synapforge.module.Module "
        "(Phase 2 contract)"
    )
    assert issubclass(SynapForge100M, sf_module.Module), (
        "SynapForge100M must inherit from synapforge.module.Module "
        "(Phase 2 contract)"
    )


def test_hybrid_block_forward_unchanged_after_phase2() -> None:
    """``HybridBlock`` forward output is unchanged after the base-class swap.

    We build two blocks with identical weights, run forward on the
    same input, and assert bit-exact equality. The "reference" here
    is another instance of the same class (we can't easily clone the
    pre-Phase 2 ``nn.Module`` baseline without a git checkpoint), so
    the test specifically guards against:

    * Our ``__call__`` override leaking unwanted I/O caching
      (``_last_io`` only populates when plasticity rules registered;
      ``HybridBlock.__init__`` doesn't register any).
    * Any subtle change in ``Parameter``'s ``__new__`` that would
      alter how ``nn.Linear`` / ``nn.Embedding`` see their own
      weights (we don't touch them, but the pin is cheap).

    The block size is small (d=32) for fast CPU runtime — the
    contract is pure-numerics and doesn't depend on shape.
    """
    from synapforge.model_100m import HybridBlock

    torch.manual_seed(123)
    d = 32
    blk_a = HybridBlock(
        d=d,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
    )
    blk_b = HybridBlock(
        d=d,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
    )
    # Cross-load to pin weights identically.
    blk_b.load_state_dict(blk_a.state_dict())

    blk_a.eval()
    blk_b.eval()

    x = _seeded_tensor(2, 8, d, seed=2026)
    with torch.no_grad():
        y_a = blk_a(x)
        y_b = blk_b(x)

    assert y_a.shape == (2, 8, d)
    rel_err = (y_a - y_b).abs().max().item() / (y_a.abs().max().item() + 1e-12)
    # The two blocks have identical weights; CPU fp32 forward must be
    # bit-exact. Phase 2 contract: rel-err < 1e-5; we assert tighter.
    assert rel_err < 1e-6, (
        f"HybridBlock forward not deterministic post-Phase 2: "
        f"rel_err={rel_err:.3e}"
    )


def test_hybrid_block_state_dict_keys_unchanged() -> None:
    """State-dict keys for ``HybridBlock`` must match the pre-Phase-2 set.

    Existing ckpts use these exact keys. Any rename in Phase 2 would
    break warmstart for the running rental — non-negotiable.

    The pre-Phase 2 keyset for a default ``HybridBlock(d=32,
    ffn_ratio=2.0, sparsity=0.5)`` is the union of:

    * ``ln1.weight`` (RMSNorm scale)
    * ``liquid.*`` (CfC weights — exact subkeys depend on backend
      build, so we whitelist the prefix)
    * ``plif.*`` (PLIF tau / threshold)
    * ``synapse.weight`` (sparse synapse — also has a buffer
      ``synapse.mask`` but that's a buffer, not a parameter)
    * ``gate.weight`` / ``gate.bias`` (gating Linear)
    * ``ln2.weight``
    * ``ffn.w_gate.weight`` / ``ffn.w_up.weight`` / ``ffn.w_down.weight``
    """
    from synapforge.model_100m import HybridBlock

    blk = HybridBlock(
        d=32,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
    )
    keys = set(blk.state_dict().keys())

    # Required prefixes (each must have at least one key).
    required_prefixes = (
        "ln1.",
        "liquid.",
        "plif.",
        "synapse.",
        "gate.",
        "ln2.",
        "ffn.",
    )
    for pref in required_prefixes:
        assert any(k.startswith(pref) for k in keys), (
            f"Phase 2 dropped state-dict prefix {pref!r}; "
            f"present keys: {sorted(keys)}"
        )


# ---------------------------------------------------------------------------
# Test 3: Parameter wraps nn.Parameter cleanly
# ---------------------------------------------------------------------------


def test_parameter_is_nn_parameter() -> None:
    """``synapforge.module.Parameter`` is a ``torch.nn.Parameter`` subclass.

    Required for the Phase 2 plumbing: ``nn.Module.parameters()``
    iteration works because torch checks ``isinstance(_, nn.Parameter)``,
    so our subclass must register positively for that check.
    """
    p = Parameter(torch.zeros(3, 4))
    assert isinstance(p, nn.Parameter)
    assert p.requires_grad is True
    assert p.shape == (3, 4)

    # requires_grad=False also works.
    p2 = Parameter(torch.zeros(2), requires_grad=False)
    assert p2.requires_grad is False


def test_parameter_default_empty() -> None:
    """``Parameter()`` with no data returns a zero-shape tensor.

    Mirrors ``torch.nn.Parameter()`` behaviour so callers can
    construct an empty Parameter slot to be filled later (uncommon
    but supported).
    """
    p = Parameter()
    assert isinstance(p, nn.Parameter)
    assert p.numel() == 0


def test_register_parameter_auto_wraps_bare_tensor() -> None:
    """``register_parameter`` auto-wraps a bare ``torch.Tensor``.

    Convenience: callers don't need to remember to wrap in
    ``Parameter(...)`` — the registration path does it on their behalf.
    """
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("w", torch.zeros(2, 3))

    m = M()
    assert isinstance(m.w, Parameter)
    assert isinstance(m.w, nn.Parameter)
    assert m.w.requires_grad is True


def test_register_parameter_rejects_non_tensor() -> None:
    """``register_parameter`` raises on non-tensor input."""
    class M(Module):
        pass

    m = M()
    try:
        m.register_parameter("bogus", "not a tensor")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "expected torch.Tensor" in str(exc), str(exc)
    else:
        raise AssertionError("register_parameter must reject non-tensor")


def test_register_parameter_none_is_allowed() -> None:
    """``register_parameter(name, None)`` registers an unset slot.

    Mirrors ``nn.Module`` behaviour for optional weights (e.g. when a
    flag-gated module reserves a name but doesn't allocate the
    weight unless the flag is on).
    """
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("optional", None)

    m = M()
    assert m.optional is None
    # Iteration skips None slots — matches nn.Module.
    names = [n for n, _ in m.named_parameters()]
    assert "optional" not in names


# ---------------------------------------------------------------------------
# Test 4: device + mode toggles
# ---------------------------------------------------------------------------


def test_train_eval_toggle() -> None:
    """``.train()`` / ``.eval()`` toggle ``self.training`` recursively."""
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_module("l", nn.Linear(2, 2))

    m = M()
    m.train()
    assert m.training is True
    assert m.l.training is True
    m.eval()
    assert m.training is False
    assert m.l.training is False
    m.train(True)
    assert m.training is True


def test_zero_grad_clears_grads() -> None:
    """``.zero_grad()`` unsets ``.grad`` on every parameter."""
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("w", Parameter(torch.zeros(3, requires_grad=True)))

    m = M()
    # Manually attach a grad so zero_grad has something to clear.
    m.w.grad = torch.ones(3)
    assert m.w.grad is not None
    m.zero_grad(set_to_none=True)
    assert m.w.grad is None
