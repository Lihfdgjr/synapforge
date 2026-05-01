"""T2.5: spike-rate-target auxiliary loss term test.

Resolves §6 P25 (PLIF dead 10/10 throughout all runs). The loss term
is:

    spike_target_loss = sum_layers (
        (rate - high).clamp(min=0) ** 2
      + (low  - rate).clamp(min=0) ** 2
    )

with default ``low=0.05``, ``high=0.20``. When the PLIF spike rate is
below ``low`` the term is positive and its gradient (computed through
the surrogate-gradient path of ``spike()`` in
:mod:`synapforge.surrogate`) flows back to ``threshold`` and
``log_tau``. Optimising against this term should therefore *push* the
spike rate up when it starts below ``low``.

We verify exactly that with a toy 2-PLIF model whose threshold is
intentionally too high (yields a 0% start spike rate). After 50
optimiser steps with ``spike_target_loss_weight=0.1``, the spike rate
must increase. With weight 0.0 the rate must stay essentially
unchanged (sanity for the no-op path).

Adam is used as the optimiser (not SGD) to mirror the live trainer's
choice — SGD with the small ATan-tail surrogate gradient on the dead
side of the threshold needs an extreme LR to escape, and that LR
isn't representative of the trainer hyperparameters this loss is
designed to work under.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from synapforge.surrogate import PLIFCell  # noqa: E402

# bounds match the trainer defaults (--spike-target-loss-low / -high)
LOW = 0.05
HIGH = 0.20


class _Toy2PLIF(nn.Module):
    """Two stacked Linear+PLIFCell blocks at intentionally too-high thr.

    Mirrors the ``test_plif_homeostasis._toy_two_block_model`` shape so
    we exercise the same ``modules()`` walk the trainer uses to collect
    PLIF cells via ``isinstance(m, PLIFCell)``.
    """

    def __init__(self, hidden: int = 16, threshold_init: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.plif1 = PLIFCell(hidden, threshold_init=threshold_init)
        self.fc2 = nn.Linear(hidden, hidden)
        self.plif2 = PLIFCell(hidden, threshold_init=threshold_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.plif1(self.fc1(x))
        h, _ = self.plif2(self.fc2(h))
        return h


def _spike_target_term(plifs, low: float = LOW, high: float = HIGH) -> torch.Tensor:
    """Mirror of trainer block: build the same loss term to drive optim."""
    terms = []
    for m in plifs:
        rate = m.last_spike_rate()  # T2.5: live, autograd-attached
        if rate.dim() > 0:
            rate = rate.squeeze()
        rate = rate.float()
        over = (rate - high).clamp(min=0.0).pow(2)
        under = (low - rate).clamp(min=0.0).pow(2)
        terms.append(over + under)
    return torch.stack(terms).sum()


def _measure_rate(model: nn.Module, plifs, n_batches: int = 8,
                  hidden: int = 16, batch: int = 8,
                  seed: int | None = 123) -> float:
    """Run a few forward passes and return mean spike rate over PLIF cells.

    Uses a fixed measurement seed so the comparison between initial and
    final rates is decoupled from training-loop seed advancement.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    accum = []
    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randn(batch, hidden)
            _ = model(x)
            for p in plifs:
                accum.append(float(p.last_spike_rate().detach().mean()))
    model.train()
    return sum(accum) / max(len(accum), 1)


def test_last_spike_rate_method_returns_tensor() -> None:
    """``last_spike_rate()`` is callable and returns a 1-element tensor."""
    cell = PLIFCell(hidden=8, threshold_init=0.3)
    # Without any forward, the method falls back to the buffer (zeros).
    rate = cell.last_spike_rate()
    assert torch.is_tensor(rate)
    assert rate.numel() == 1
    # After a forward, the live tensor should have an autograd graph.
    x = torch.randn(2, 8, requires_grad=True)
    s, _ = cell(x)
    rate2 = cell.last_spike_rate()
    assert torch.is_tensor(rate2)
    assert rate2.numel() == 1
    # Surrogate gradient flows through `(v - threshold)`, so the rate
    # tensor must have a grad_fn back to the autograd graph.
    assert rate2.requires_grad or rate2.grad_fn is not None


def test_spike_target_loss_pushes_dead_rate_up() -> None:
    """With weight=0.1 + dead PLIF, 50 steps must raise the spike rate."""
    torch.manual_seed(0)
    hidden = 16
    # threshold_init=0.3 is intentionally too high: spike rate starts at
    # 0% under N(0, 1) inputs (verified empirically), well below
    # ``low=0.05``, so the under-band term is active and pushing.
    model = _Toy2PLIF(hidden=hidden, threshold_init=0.3)
    plifs = [m for m in model.modules() if isinstance(m, PLIFCell)]
    assert len(plifs) == 2

    initial_rate = _measure_rate(model, plifs, hidden=hidden)
    # rate must start in the dead zone for the test to be meaningful.
    assert initial_rate < LOW, (
        f"toy model not actually 'dead' at start (rate={initial_rate:.3f} "
        f">= low={LOW}); raise threshold_init to recreate the P25 condition"
    )

    weight = 0.1
    # Adam mirrors the live trainer (train_100m_kd.py uses Adam-family
    # optimisers). lr=0.1 gives the spike-target loss enough authority
    # over threshold / log_tau to escape the dead surrogate tail in 50
    # steps. SGD would need lr~2.0 because ATan surrogate decays as
    # 1/x^2 once the membrane drifts far below threshold.
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(50):
        x = torch.randn(8, hidden)
        _ = model(x)  # populates _last_spike_rate_live on each PLIFCell
        loss = weight * _spike_target_term(plifs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    final_rate = _measure_rate(model, plifs, hidden=hidden)
    # The auxiliary loss must monotonically push the dead rate upward
    # toward the active band [low, high]. Strict: rate must rise from
    # the dead-zone start (not necessarily land inside the band in 50
    # steps, but it should be visibly higher).
    assert final_rate > initial_rate, (
        f"weight=0.1 did not push spike rate up: "
        f"initial={initial_rate:.4f} -> final={final_rate:.4f}"
    )


def test_spike_target_loss_weight_zero_is_noop() -> None:
    """With weight=0, spike rate stays essentially unchanged.

    Sanity check that the no-op path (the default behaviour expected
    from a default-OFF flag during phase-0 launches) doesn't move
    parameters even when other code paths are wired to read the loss.
    """
    torch.manual_seed(0)
    hidden = 16
    model = _Toy2PLIF(hidden=hidden, threshold_init=0.3)
    plifs = [m for m in model.modules() if isinstance(m, PLIFCell)]

    initial_rate = _measure_rate(model, plifs, hidden=hidden)

    weight = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(50):
        x = torch.randn(8, hidden)
        _ = model(x)
        # ``loss = weight * term`` with weight=0 is an all-zero-grad
        # tensor (not a no-grad scalar). Adam treats zero grad as a
        # no-op, so parameters must not move.
        loss = weight * _spike_target_term(plifs)
        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
        opt.step()

    final_rate = _measure_rate(model, plifs, hidden=hidden)
    # No driving force on threshold/tau, so the rate must stay flat.
    # Tight tolerance because the measurement seed is fixed (see
    # ``_measure_rate``) and Adam.zero-grad path is exactly idempotent
    # in fp32.
    assert abs(final_rate - initial_rate) < 1e-6, (
        f"weight=0 path moved spike rate: "
        f"initial={initial_rate:.6f} -> final={final_rate:.6f}"
    )
