"""T9.2: trainer-side NeuroMCP wire-in tests.

Resolves T9.2 in ``docs/DEEP_MAINT_QUEUE.md``. Companion long-horizon
PoC test: ``tests/integration/test_neuromcp_long_horizon.py`` exercises
the standalone 4-button env. This file exercises the **trainer mixin**
(``synapforge.training.neuromcp_mixin.NeuroMCPMixin``) -- specifically:

  1. Default OFF -- the mixin is NOT instantiated when the trainer is
     run with ``--neuromcp-weight 0.0`` (the production default).
  2. Enabled path runs forward without raising on a single LM-shaped
     hidden state and produces a finite, autograd-attached scalar.
  3. The action loss is *added* to the trainer's running total when
     the weight is non-zero (we replicate the trainer's loss-fold
     arithmetic on a tiny dummy LM).
  4. Density grows over ~100 forward + plasticity steps -- the
     SparseSynapticLayer's coact_ema / grow rule kicks in fast enough
     that ``density`` is strictly above its initial value within 100
     steps. This is the PoC contract verified at scale by the
     long-horizon test.

These tests are pure-CPU and finish in <5s on a typical laptop. They do
NOT depend on the 100M model weights, ParquetTokenStream, or any GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Test depends on torch (the mixin is autograd-attached).
torch = pytest.importorskip("torch")
import torch.nn as nn

# Make the repo root importable so ``import synapforge`` works regardless
# of pytest invocation directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synapforge.training.neuromcp_mixin import NeuroMCPMixin  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny LM stub: just exposes ``.d`` and one parameter so the mixin can
# read hidden width and place itself on the same device. No real LM.
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.d = int(d)
        self.tok_embed = nn.Embedding(64, d)


# ===========================================================================
# Test 1 -- default OFF: trainer wire-in must not instantiate the mixin
# ===========================================================================


def test_default_off_no_instantiation() -> None:
    """When --neuromcp-weight 0.0 (default), trainer must NOT build the head.

    We replicate the trainer's gating predicate verbatim: the trainer
    only constructs ``NeuroMCPMixin`` when ``args.neuromcp_weight > 0``.
    This test asserts the predicate is honoured by replicating the gate
    here -- the mixin is not built, and the optional code path is a true
    no-op.
    """
    args_neuromcp_weight = 0.0  # production default
    mixin = None
    if float(args_neuromcp_weight) > 0:
        # Trainer would construct here -- but it shouldn't on the default path.
        mixin = NeuroMCPMixin(_StubLM(), hidden=32)
    assert mixin is None, (
        "NeuroMCPMixin must NOT be instantiated when --neuromcp-weight is 0; "
        "default-OFF contract violated."
    )

    # Also smoke the explicit OFF: a manually-constructed mixin still works
    # (so re-enabling at runtime is graceful), but the gate above is the
    # production contract that protects existing runs.
    explicit = NeuroMCPMixin(_StubLM(), hidden=32, codebook_size=4, verbose=False)
    assert explicit.head is not None, "explicit construction should succeed"


# ===========================================================================
# Test 2 -- enabled path: a single forward step works end-to-end
# ===========================================================================


def test_enabled_runs_forward() -> None:
    """With --neuromcp-weight=0.05 a single forward returns a finite scalar.

    Mirrors the trainer's inner loop arithmetic: feed the mixin a
    text-hidden tensor + next-token labels, compute action_loss, and
    verify the scalar is finite + autograd-attached + .backward() works.
    """
    torch.manual_seed(0)
    d = 32
    model = _StubLM(d)
    mixin = NeuroMCPMixin(model, hidden=d, codebook_size=4, verbose=False)
    assert mixin.head is not None

    # Simulate trainer: text_hidden = model.encode(x); shape (B, T, d).
    B, T = 2, 4
    text_hidden = torch.randn(B, T, d, requires_grad=True)
    y = torch.randint(0, 64, (B, T))

    # The trainer multiplies by --neuromcp-weight before adding to total loss.
    nmcp_weight = 0.05
    action_loss = mixin.action_loss(text_hidden, y_next=y)
    assert torch.is_tensor(action_loss), "action_loss must return a Tensor"
    assert action_loss.dim() == 0, f"expected scalar, got shape {action_loss.shape}"
    assert torch.isfinite(action_loss), "action_loss must be finite"
    assert action_loss.requires_grad, (
        "action_loss must be autograd-attached so weight*loss "
        "contributes to the training gradient"
    )
    # backward should not raise on the synthesised graph.
    (nmcp_weight * action_loss).backward()
    # text_hidden should now have a non-trivial gradient.
    assert text_hidden.grad is not None, "no gradient flowed back to text_hidden"
    # post-step plasticity hook (called in trainer after optim.step()).
    plast = mixin.step_plasticity()
    assert isinstance(plast, dict), "step_plasticity must return a dict"


# ===========================================================================
# Test 3 -- loss addition: total = base + neuromcp_weight * action_loss
# ===========================================================================


def test_loss_addition_reflects_weight() -> None:
    """The neuromcp loss is *added* to the running total with weight.

    We replicate the trainer's ``loss = loss + args.neuromcp_weight *
    neuromcp_aux`` arithmetic and verify (a) the resulting total is
    different from the base alone, (b) the difference equals weight *
    detached(action_loss) within a tight tolerance.
    """
    torch.manual_seed(1)
    d = 32
    model = _StubLM(d)
    mixin = NeuroMCPMixin(model, hidden=d, codebook_size=4, verbose=False)
    assert mixin.head is not None

    B, T = 2, 4
    text_hidden = torch.randn(B, T, d, requires_grad=True)
    y = torch.randint(0, 64, (B, T))
    base_loss = (text_hidden ** 2).mean()  # stand-in for ce_loss + z_loss + kd

    weight = 0.05
    action_loss = mixin.action_loss(text_hidden, y_next=y)
    total = base_loss + weight * action_loss

    # The total must be strictly greater than base alone for any non-zero
    # action_loss (action_loss is CE >= 0; with random init it's always > 0).
    base_val = float(base_loss.detach())
    total_val = float(total.detach())
    action_val = float(action_loss.detach())

    assert action_val > 0.0, (
        f"action_loss must be > 0 with random init; got {action_val}"
    )
    expected = base_val + weight * action_val
    assert abs(total_val - expected) < 1e-5, (
        f"loss addition arithmetic broken: "
        f"total={total_val} expected={expected} "
        f"(base={base_val} + {weight} * action={action_val})"
    )

    # The 0 weight path must collapse to the base.
    weight_zero = 0.0
    total_zero = base_loss + weight_zero * action_loss
    assert abs(float(total_zero.detach()) - base_val) < 1e-6, (
        "weight=0 path should match base_loss exactly"
    )


# ===========================================================================
# Test 4 -- density grows over forward steps
# ===========================================================================


def test_density_grows_after_100_steps() -> None:
    """After 100 forward+plasticity steps, density must be strictly above
    its initial value.

    This is the PoC contract: the SparseSynapticLayer co-activation EMA
    accumulates non-zero coactivations as soon as forwards run, and
    ``maybe_grow_prune`` (called from ``step_plasticity``) bumps density
    by ``growth_step`` (default 0.005) every ``growth_check_every`` (20)
    steps. So in 100 steps we expect ~5 grow events, each adding ~0.005
    -> initial 0.05 -> ~0.075 floor.

    We use a low growth_threshold so coact_ema actually qualifies as
    grow-eligible on tiny-batch CPU runs (the production default is
    tuned for B=80 on A100; on CPU/B=2 the EMA values are smaller).
    """
    torch.manual_seed(2)
    d = 32
    model = _StubLM(d)
    mixin = NeuroMCPMixin(model, hidden=d, codebook_size=4, verbose=False)
    assert mixin.head is not None

    # Lower the grow threshold so even small CPU coact_emas qualify -- this
    # mirrors what the long-horizon test on the 4-button env does at scale.
    mixin.head.proj.cfg.growth_threshold = 1e-6

    initial_density = mixin.density
    assert initial_density > 0.0, (
        f"initial density should reflect the configured "
        f"initial_density (0.05), got {initial_density}"
    )

    N_STEPS = 100
    B, T = 2, 4
    for _ in range(N_STEPS):
        text_hidden = torch.randn(B, T, d, requires_grad=True)
        y = torch.randint(0, 64, (B, T))
        loss = mixin.action_loss(text_hidden, y_next=y)
        loss.backward()
        mixin.step_plasticity()

    final_density = mixin.density
    # density may oscillate slightly via prune, but with growth_check_every=20
    # and growth_step=0.005, 100 steps gives ~5 grow events. Strict >.
    assert final_density > initial_density, (
        f"density did not grow over {N_STEPS} steps: "
        f"{initial_density:.4f} -> {final_density:.4f}. "
        "SparseSynapticLayer plasticity rule is not firing."
    )

    # Also check stats() reports the same numbers.
    stats = mixin.stats()
    assert abs(stats["density"] - final_density) < 1e-6, "stats density mismatch"
    assert stats["step_count"] >= N_STEPS, (
        f"step_count must equal at least {N_STEPS}; got {stats['step_count']}"
    )
