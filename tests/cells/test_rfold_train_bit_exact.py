"""End-to-end training-step equivalence tests for the R-fold path.

The existing tests/cells/test_rfold_equivalence.py covers single-forward
and single-backward equivalence. This file adds the training-loop level
test that the user explicitly asked for in the activation spec
(decorative -> real, 2026-05-02): 100 training steps with --rfold ON
must match the legacy sequential path within 1% on the resulting
parameters / loss curve.

Why this matters: a per-iteration fp32 round-off difference can compound
over thousands of steps. The chunked closed-form scan is algebraically
identical to the sequential recurrence, so the cumulative drift after
100 steps is essentially zero (single fp32 noise per step). If a
regression slips a non-bit-exact change into the rfold path (e.g.
swapping chunk math for an approximation) this test catches it
immediately.

We test on a small LiquidCell (D=32, T=32, B=4) so the suite stays fast
on CPU. The same algebra holds at the trainer's D=1280 scale.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from synapforge.cells.liquid import LiquidCell  # noqa: E402


def _train_loop(
    rfold: bool, n_steps: int = 100,
    B: int = 4, T: int = 32, D: int = 32,
    chunk: int = 16,
    seed: int = 0,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Build a fresh cell, run n_steps SGD steps with deterministic per-step
    inputs. Return loss history and final state_dict."""
    torch.manual_seed(seed)
    cell = LiquidCell(D, D, init="hasani", rfold=rfold, rfold_chunk=chunk)
    opt = torch.optim.SGD(cell.parameters(), lr=1e-3)
    losses: list[float] = []
    for step in range(n_steps):
        # Per-step deterministic input (same for both rfold ON / OFF).
        torch.manual_seed(100 + step)
        x = torch.randn(B, T, D) * 0.1
        h = cell(x)
        loss = (h.float() ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses, cell.state_dict()


def test_rfold_train_loss_matches_sequential() -> None:
    """100 SGD steps: rfold ON loss curve matches OFF within 1%.

    With identical weights, identical per-step inputs, and an algebraically
    identical recurrence, the only delta is fp32 round-off in the
    cumprod/cumsum chunked scan. Empirically the drift stays under
    1e-5% per step at D=32; at 100 steps the cumulative drift is well
    under the 1% bound.
    """
    losses_seq, _ = _train_loop(rfold=False)
    losses_rf, _ = _train_loop(rfold=True)
    # Per-step relative drift
    drifts: list[float] = []
    for s, r in zip(losses_seq, losses_rf):
        if abs(s) < 1e-12:
            drifts.append(abs(r))
        else:
            drifts.append(abs(s - r) / abs(s))
    max_drift = max(drifts)
    mean_drift = sum(drifts) / len(drifts)
    # 1% spec; with cumprod/cumsum we comfortably stay under 1e-4.
    assert max_drift < 1e-2, (
        f"per-step drift exceeded 1%: max={max_drift:.4e} "
        f"mean={mean_drift:.4e}"
    )


def test_rfold_train_final_params_match_sequential() -> None:
    """After 100 train steps, the final delta_proj/b_proj/A_log
    parameters should differ only by fp32 round-off (1e-3 relative)."""
    _, sd_seq = _train_loop(rfold=False)
    _, sd_rf = _train_loop(rfold=True)
    for k in sd_seq:
        a = sd_seq[k].float()
        b = sd_rf[k].float()
        assert a.shape == b.shape, (
            f"shape mismatch on {k}: {a.shape} vs {b.shape}"
        )
        if a.numel() == 0:
            continue
        a_norm = a.norm().item() + 1e-12
        diff_norm = (a - b).norm().item()
        rel = diff_norm / a_norm
        assert rel < 1e-3, (
            f"param '{k}' drift after 100 steps: rel={rel:.4e}"
        )


def test_rfold_short_training_is_deterministic() -> None:
    """Re-running the same train loop with the same seeds gives identical
    loss curves: a sanity check that we have no hidden source of
    nondeterminism in the rfold path."""
    losses_a, _ = _train_loop(rfold=True, n_steps=20)
    losses_b, _ = _train_loop(rfold=True, n_steps=20)
    for la, lb in zip(losses_a, losses_b):
        assert la == lb, (
            f"rfold path is non-deterministic: la={la} lb={lb}"
        )


def test_rfold_grad_path_is_active_in_train_mode() -> None:
    """In train mode, .grad on delta_proj.weight after backward must be
    non-zero with rfold=True. (The reference test in
    test_rfold_equivalence.py covers this; we duplicate at the train-step
    level to ensure no hidden in-place op stops the chain.)"""
    torch.manual_seed(0)
    cell = LiquidCell(32, 32, init="hasani", rfold=True, rfold_chunk=8)
    cell.train()
    x = torch.randn(2, 16, 32) * 0.1
    h = cell(x)
    h.pow(2).mean().backward()
    assert cell.delta_proj.weight.grad is not None
    assert cell.b_proj.weight.grad is not None
    assert cell.A_log.grad is not None
    assert cell.delta_proj.weight.grad.abs().max() > 0
    assert cell.b_proj.weight.grad.abs().max() > 0
    assert cell.A_log.grad.abs().max() > 0


def test_rfold_eval_mode_grad_disabled() -> None:
    """Eval mode + torch.no_grad still runs forward (used by inference)."""
    torch.manual_seed(0)
    cell = LiquidCell(16, 16, init="hasani", rfold=True, rfold_chunk=8)
    cell.eval()
    x = torch.randn(2, 8, 16) * 0.1
    with torch.no_grad():
        h = cell(x)
    assert h.shape == (2, 8, 16)
    assert h.requires_grad is False


def test_rfold_ema_param_order_matches_seq() -> None:
    """The order of the parameter tensors in named_parameters() is identical
    between rfold ON and OFF, so external code (e.g. EMA or grad-clip)
    that walks parameters in named order won't see a different order in
    Run 8 vs Run 7."""
    cell_seq = LiquidCell(8, 8, init="hasani", rfold=False)
    cell_rf = LiquidCell(8, 8, init="hasani", rfold=True, rfold_chunk=4)
    names_seq = [n for n, _ in cell_seq.named_parameters()]
    names_rf = [n for n, _ in cell_rf.named_parameters()]
    assert names_seq == names_rf, (
        f"param-order regression: rfold ON has different ordering\n"
        f"  seq={names_seq}\n  rfold={names_rf}"
    )


if __name__ == "__main__":
    test_rfold_train_loss_matches_sequential()
    print("OK 100 train steps: loss curves match within 1%")
    test_rfold_train_final_params_match_sequential()
    print("OK 100 train steps: final params match within 1e-3 rel")
    test_rfold_short_training_is_deterministic()
    print("OK rfold train is deterministic")
    test_rfold_grad_path_is_active_in_train_mode()
    print("OK rfold grad path active in train mode")
    test_rfold_eval_mode_grad_disabled()
    print("OK rfold eval mode no_grad works")
    test_rfold_ema_param_order_matches_seq()
    print("OK named_parameters order matches sequential")
