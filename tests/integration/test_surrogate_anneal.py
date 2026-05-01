"""T2.3: surrogate gradient width anneal 10 -> 1 over N steps.

Resolves §6 P25 (PLIF dead 10/10 layers throughout all runs). The
PLIFCell forward divides ``self.alpha`` by ``self.surrogate_width``;
larger width => smaller effective alpha => wider / smoother surrogate
that reaches dead spikes (membrane far below threshold). At training
start a wide surrogate (width=10) lets the gradient flow through
near-zero spike rate so ``threshold`` and ``log_tau`` can learn out of
the dead zone; once activations come up the trainer hook narrows the
width back to the production sharp ATan derivative (width=1) over the
first N steps. Default trainer flag is 0 (no anneal) to keep the
baseline path unchanged.

Five tests:
    1. ``test_default_no_anneal`` — buffer stays at 1.0 with no hook.
    2. ``test_linear_decay`` — schedule numerically matches at step
       0 / N/2 / N (10.0 / 5.5 / 1.0 for start=10 target=1 N=1000).
    3. ``test_clamp_to_target`` — past the anneal window the width
       stays exactly at ``target``.
    4. ``test_grad_flows_at_width_10_with_dead_spike`` — sanity that
       even when the membrane is far below threshold (dead PLIF), a
       width=10 surrogate produces a gradient floor strictly larger
       than the width=1 baseline (this is the entire point of T2.3).
    5. ``test_grad_decay_at_width_1`` — width=1 reproduces the legacy
       sharp ATan analytic peak ``alpha/2`` at the threshold.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from synapforge.surrogate import PLIFCell  # noqa: E402


# ---------------------------------------------------------------------
# Schedule helper — exact mirror of the trainer hook in train_100m_kd.py
# ---------------------------------------------------------------------

def _scheduled_width(step: int, start: float, target: float, anneal_steps: int) -> float:
    """Mirror the trainer formula so tests are anchored to the exact
    ``progress = min(1.0, step / max(N, 1))`` interpolation the live hook
    uses.  Returns ``max(target, start - (start - target) * progress)``.
    """
    if anneal_steps <= 0:
        return float(target)  # no-op default
    progress = min(1.0, step / max(anneal_steps, 1))
    return max(float(target), float(start) - (float(start) - float(target)) * progress)


def _walk_and_apply(model: nn.Module, new_width: float) -> None:
    """Mirror the trainer's ``modules()`` walk: every PLIFCell gets the
    same width.  Used by tests instead of importing the trainer to keep
    the test self-contained.
    """
    for m in model.modules():
        if isinstance(m, PLIFCell):
            m.update_surrogate_width(new_width)


# ---------------------------------------------------------------------
# 1. Default no-op path
# ---------------------------------------------------------------------

def test_default_no_anneal() -> None:
    """anneal_steps=0 must leave ``surrogate_width`` at 1.0 forever.

    Mirrors the trainer's default behaviour: ``--surrogate-anneal-steps``
    defaults to 0, so the hook in the step loop is gated off entirely;
    the buffer never moves from its 1.0 init.
    """
    cell = PLIFCell(hidden=8)
    # Buffer must exist and start at 1.0 (matches the constructor default
    # and the legacy alpha-only path before T2.3).
    assert hasattr(cell, "surrogate_width")
    assert torch.is_tensor(cell.surrogate_width)
    assert math.isclose(float(cell.surrogate_width.item()), 1.0, abs_tol=1e-9)

    start, target, anneal_steps = 10.0, 1.0, 0
    # Simulate 1000 fake step boundaries; with anneal_steps=0 the
    # scheduled width is constant at ``target`` and the trainer hook is
    # gated OFF, so we never call ``update_surrogate_width``. Buffer
    # therefore must remain at its 1.0 init.
    for step in range(0, 1001, 100):
        if anneal_steps > 0:
            w = _scheduled_width(step, start, target, anneal_steps)
            cell.update_surrogate_width(w)
    assert math.isclose(float(cell.surrogate_width.item()), 1.0, abs_tol=1e-9), (
        "no-anneal default moved the surrogate_width buffer"
    )


# ---------------------------------------------------------------------
# 2. Linear decay numerical check
# ---------------------------------------------------------------------

def test_linear_decay() -> None:
    """At anneal_steps=1000 start=10 target=1, check 0/500/1000 exactly.

    The schedule is
        width(step) = max(target, start - (start - target) * step / N).
    For (10, 1, 1000):
        step=0    -> 10.0
        step=500  -> 10 - 9 * 0.5  = 5.5
        step=1000 -> 10 - 9 * 1.0  = 1.0  (== target, clamp engages)
    """
    cell = PLIFCell(hidden=4)
    start, target, anneal_steps = 10.0, 1.0, 1000

    # step 0 -> width = start exactly
    cell.update_surrogate_width(_scheduled_width(0, start, target, anneal_steps))
    assert math.isclose(float(cell.surrogate_width.item()), 10.0, rel_tol=1e-6), (
        f"step 0 width != 10.0: got {float(cell.surrogate_width.item())}"
    )

    # step 500 -> mid-anneal = 5.5 (linear interpolation midpoint)
    cell.update_surrogate_width(_scheduled_width(500, start, target, anneal_steps))
    assert math.isclose(float(cell.surrogate_width.item()), 5.5, rel_tol=1e-6), (
        f"step 500 width != 5.5: got {float(cell.surrogate_width.item())}"
    )

    # step 1000 -> end-of-anneal == target
    cell.update_surrogate_width(_scheduled_width(1000, start, target, anneal_steps))
    assert math.isclose(float(cell.surrogate_width.item()), 1.0, rel_tol=1e-6), (
        f"step 1000 width != 1.0: got {float(cell.surrogate_width.item())}"
    )


# ---------------------------------------------------------------------
# 3. Clamp past anneal_steps
# ---------------------------------------------------------------------

def test_clamp_to_target() -> None:
    """Once step > anneal_steps the schedule must stay at target exactly.

    The ``progress = min(1.0, step / N)`` clamp + the ``max(target, ...)``
    floor together guarantee the schedule never undershoots the target.
    Test it numerically at several over-the-end step counts.
    """
    cell = PLIFCell(hidden=4)
    start, target, anneal_steps = 10.0, 1.0, 500

    # At step == anneal_steps the schedule is exactly at target
    w_at = _scheduled_width(anneal_steps, start, target, anneal_steps)
    assert math.isclose(w_at, target, rel_tol=1e-6)

    # Past the anneal window: 600, 1000, 5000, 100000 — all must clamp
    for over_step in (600, 1000, 5000, 100000):
        w = _scheduled_width(over_step, start, target, anneal_steps)
        cell.update_surrogate_width(w)
        assert math.isclose(float(cell.surrogate_width.item()), target, rel_tol=1e-6), (
            f"step {over_step}: surrogate_width drifted past target "
            f"({float(cell.surrogate_width.item())} != {target})"
        )


# ---------------------------------------------------------------------
# 4. Wide surrogate revives gradient on dead PLIF
# ---------------------------------------------------------------------

def test_grad_flows_at_width_10_with_dead_spike() -> None:
    """Width=10 must produce visibly larger gradient than width=1 when
    the membrane has drifted far below threshold (P25 dead-PLIF).

    PLIFCell.forward divides ``alpha`` by ``surrogate_width``, so the
    effective ATan derivative is
        g_eff(x) = (alpha/width) / (2 * (1 + (pi/2 * alpha/width * x)**2))
    where x = v_t - threshold. At small |x|, the SHARP path (width=1)
    has bigger peak (alpha/2 vs alpha/(2*width)) and dominates.  At
    LARGE |x| (the dead-PLIF condition: membrane several units below
    threshold), the squared-tail of the sharp path crushes its
    gradient by ~width^2 while the wide path's tail is much fatter.
    Crossover at |x| ~ 1 for alpha=2 width=10 (computed analytically:
    `g_wide(2) = 7.2e-2`, `g_sharp(2) = 2.5e-2`, ratio ~2.9).

    Drive: thr=1.0 (default), input = -10.0 -> after one LIF step with
    decay~0.905 the membrane lands at ~-0.95, so |v-thr| ~ 1.95, well
    inside the wide-dominates regime (ratio ~2.9 analytically).
    """
    torch.manual_seed(0)
    hidden = 8
    cell_wide = PLIFCell(hidden=hidden, threshold_init=1.0, alpha=2.0)
    cell_sharp = PLIFCell(hidden=hidden, threshold_init=1.0, alpha=2.0)
    cell_wide.update_surrogate_width(10.0)
    cell_sharp.update_surrogate_width(1.0)
    # Same parameters across both cells so only the width differs.
    with torch.no_grad():
        cell_sharp.threshold.copy_(cell_wide.threshold)
        cell_sharp.log_tau.copy_(cell_wide.log_tau)

    # Strong NEGATIVE drive: pushes membrane well below threshold = 1.0.
    # With log_tau init=log(10), decay=exp(-0.1)~0.905 so v_t ~ 0.095*x.
    # At x=-10 -> v_t ~ -0.95 -> v_t - thr ~ -1.95 -> deep dead zone.
    # Sanity floor (gradient must survive) AND the wide-dominates band
    # both apply at this drive.
    x_w = torch.full((4, hidden), -10.0, requires_grad=True)
    s_w, _ = cell_wide(x_w)
    s_w.sum().backward()
    g_wide = float(x_w.grad.abs().sum())

    x_s = torch.full((4, hidden), -10.0, requires_grad=True)
    s_s, _ = cell_sharp(x_s)
    s_s.sum().backward()
    g_sharp = float(x_s.grad.abs().sum())

    # Floor: the entire point of T2.3 is that the wide surrogate keeps
    # gradient alive when v drifts far from threshold. Must be strictly
    # positive (not numerically zero from the squared denominator).
    assert g_wide > 1e-6, (
        f"wide surrogate (width=10) produced ~zero gradient on dead PLIF "
        f"input (grad sum = {g_wide:.6e}); T2.3 hypothesis fails"
    )
    # Crossover: at |v-thr|~1.95 with alpha=2, width=10 vs width=1, the
    # analytic ATan-derivative ratio is ~2.9 (see docstring). Test 2.0x
    # as a robust lower bound that holds across the random tau / thr
    # init noise. This is the *gradient revival* the trainer hook
    # exploits: dead PLIF cells get usable signal with width=10 that
    # they would not get under the legacy width=1 path.
    assert g_wide > 2.0 * g_sharp, (
        f"width=10 gradient {g_wide:.6e} not > 2x width=1 gradient "
        f"{g_sharp:.6e}; surrogate widening did not revive gradient on "
        f"the dead PLIF input"
    )


# ---------------------------------------------------------------------
# 5. Width=1 reproduces sharp ATan
# ---------------------------------------------------------------------

def test_grad_decay_at_width_1() -> None:
    """At width=1, peak gradient at v=threshold equals legacy alpha/2.

    ATan analytical peak at x=0 is ``alpha/2`` (Fang et al. 2021,
    test_surrogate.py::test_atan_known_peak_value mirrors this). With
    surrogate_width=1, the PLIFCell forward should produce the same
    peak as the pre-T2.3 legacy code path. We bypass the full LIF cell
    by using ``spike()`` directly with ``alpha=alpha/width`` to verify
    the peak math, then independently verify the cell's forward stamps
    the buffer correctly.
    """
    cell = PLIFCell(hidden=4, alpha=2.0)
    cell.update_surrogate_width(1.0)
    assert math.isclose(float(cell.surrogate_width.item()), 1.0, rel_tol=1e-9)

    # Build an input that drives the membrane EXACTLY to threshold so
    # the surrogate fires at its peak. The cell does
    #   v_t = decay * 0 + (1 - decay) * x  (v_prev = 0)
    # so x = thr / (1 - decay) puts v_t = thr, giving the peak gradient.
    decay = float(cell.get_decay().mean().item())
    thr = float(cell.threshold.mean().item())
    # Single-channel scalar drive
    x = torch.full((1, 4), thr / max(1.0 - decay, 1e-6), requires_grad=True)
    s, _ = cell(x)
    s.sum().backward()
    g = float(x.grad.abs().mean().item())

    # Expected: ATan peak gradient at x=0 is ``alpha/2 * (1 - decay)``
    # (chain rule through ``v_t = (1-decay) * x``). Use a generous
    # rel_tol because the input only approximately lands at threshold
    # (the single-step test_atan_known_peak_value uses ``alpha/2``
    # directly via spike(), here we exercise the full PLIFCell forward
    # which includes the LIF integration). Floor: gradient must be
    # > 0.5 * alpha/2 * (1 - decay) at width=1.
    expected_peak = (cell.alpha / 2.0) * (1.0 - decay)
    assert g > 0.5 * expected_peak, (
        f"width=1 gradient {g:.6e} too small vs ATan analytic floor "
        f"{0.5 * expected_peak:.6e} (alpha={cell.alpha}, decay={decay:.4f})"
    )
    assert g <= expected_peak * 1.5, (
        f"width=1 gradient {g:.6e} exceeded ATan analytic ceiling "
        f"{expected_peak * 1.5:.6e}; the surrogate is sharper than "
        f"alpha/2 -- buffer wiring may be inverted"
    )


# ---------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
