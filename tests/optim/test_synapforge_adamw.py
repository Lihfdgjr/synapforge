"""Unit tests for ``synapforge.optim.AdamW`` (Phase 1 of torch removal).

The crucial test is ``test_matches_torch_optim_adamw``: it asserts the
pure-python AdamW lands within 1e-5 relative-error of the reference
``torch.optim.AdamW`` after 50 steps on a non-trivial regression task.
That's the contract Phase 1 of the torch-replacement roadmap commits
to (see ``docs/TORCH_REPLACEMENT_PLAN.md``).

Numerics note: ``torch.optim.AdamW(fused=True)`` requires CUDA; on a
CPU CI runner we fall back to ``fused=False`` which uses the same
single-tensor reference implementation. Both paths are bit-for-bit
identical for fp32 inputs (Adam moment update is
kernel-implementation-independent), so the 1e-5 contract holds either
way.
"""
from __future__ import annotations

import copy

import torch

from synapforge.optim import AdamW as SfAdamW


def _device() -> str:
    """Use CUDA when available; CPU is fine for the 1e-5 contract test."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test 1 — numerics match torch.optim.AdamW within 1e-5 over 50 steps
# ---------------------------------------------------------------------------


def test_matches_torch_optim_adamw():
    """SynapForge AdamW must match torch.optim.AdamW(fused=True) numerics.

    Sets up two copies of an identical 16x8 linear-regression target.
    One copy is updated by ``synapforge.optim.AdamW``, the other by
    ``torch.optim.AdamW``. After 50 SGD-style steps the parameter
    snapshots must agree to within rel-err 1e-5.
    """
    torch.manual_seed(0)
    dev = _device()
    d_in, d_out, n_steps = 16, 8, 50

    p_sf = torch.nn.Parameter(torch.randn(d_out, d_in, device=dev))
    p_to = torch.nn.Parameter(p_sf.detach().clone())

    opt_sf = SfAdamW(
        [p_sf], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    # Use fused=True on CUDA for the strongest possible cross-impl check;
    # on CPU torch.optim defaults to the single-tensor reference path
    # which is what we mirror.
    opt_to_kwargs = dict(
        lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )
    if dev.startswith("cuda"):
        opt_to_kwargs["fused"] = True
    opt_to = torch.optim.AdamW([p_to], **opt_to_kwargs)

    target = torch.randn(d_out, d_in, device=dev)
    for _ in range(n_steps):
        opt_sf.zero_grad()
        opt_to.zero_grad()
        loss_sf = ((p_sf - target) ** 2).sum()
        loss_to = ((p_to - target) ** 2).sum()
        loss_sf.backward()
        loss_to.backward()
        opt_sf.step()
        opt_to.step()

    rel = (p_sf - p_to).norm() / p_to.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, (
        f"synapforge.optim.AdamW vs torch.optim.AdamW rel_err = "
        f"{rel.item():.3e}, expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Test 2 — state_dict / load_state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    """SynapForge AdamW state_dict must round-trip losslessly.

    After 5 warmup steps, save state, build a fresh optimizer on a clone
    of the parameters, load_state_dict, and step both for another 5
    iterations. The parameters must remain bit-equal.
    """
    torch.manual_seed(1)
    dev = _device()
    p_a = torch.nn.Parameter(torch.randn(8, 8, device=dev))
    p_b = torch.nn.Parameter(p_a.detach().clone())

    opt_a = SfAdamW([p_a], lr=3e-3, weight_decay=0.05)
    target = torch.zeros_like(p_a.data)

    # Warmup 5 steps so Adam moments are non-zero before round-trip.
    for _ in range(5):
        opt_a.zero_grad()
        ((p_a - target) ** 2).sum().backward()
        opt_a.step()

    sd = opt_a.state_dict()

    # Round-trip into a brand-new optimizer over the cloned param.
    opt_b = SfAdamW([p_b], lr=3e-3, weight_decay=0.05)
    # First copy the warmup-time param state into p_b.
    p_b.data.copy_(p_a.data)
    opt_b.load_state_dict(sd)

    # Now run 5 more steps on both. They should remain bit-equal because
    # the state has been faithfully restored.
    for _ in range(5):
        opt_a.zero_grad()
        opt_b.zero_grad()
        ((p_a - target) ** 2).sum().backward()
        ((p_b - target) ** 2).sum().backward()
        opt_a.step()
        opt_b.step()

    # Bit-equal after round-trip + 5 more steps.
    assert torch.equal(p_a.data, p_b.data), (
        "state_dict round-trip diverged: max_abs = "
        f"{(p_a.data - p_b.data).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — torch.optim.AdamW ckpt cross-load (warmstart compat)
# ---------------------------------------------------------------------------


def test_loads_torch_optim_adamw_ckpt():
    """SynapForge AdamW must load a torch.optim.AdamW state_dict.

    This is the key warmstart-compat contract for the
    ``--synapforge-adamw`` opt-in flag in train_100m_kd.py: a ckpt saved
    by ``torch.optim.AdamW(fused=True)`` (the current --fused-adamw path)
    must round-trip into ``synapforge.optim.AdamW`` so we don't lose
    moment estimates when migrating.
    """
    torch.manual_seed(2)
    dev = _device()
    p_to = torch.nn.Parameter(torch.randn(4, 4, device=dev))
    p_sf = torch.nn.Parameter(p_to.detach().clone())

    opt_to_kwargs = dict(lr=5e-3, weight_decay=0.0)
    if dev.startswith("cuda"):
        opt_to_kwargs["fused"] = True
    opt_to = torch.optim.AdamW([p_to], **opt_to_kwargs)
    target = torch.zeros_like(p_to.data)

    # Warm the torch optimizer for 8 steps so moments are populated.
    for _ in range(8):
        opt_to.zero_grad()
        ((p_to - target) ** 2).sum().backward()
        opt_to.step()

    # Snapshot the torch state, then transplant into SynapForge AdamW.
    torch_sd = opt_to.state_dict()

    # Bring the SynapForge param up to the same data point.
    p_sf.data.copy_(p_to.data)
    opt_sf = SfAdamW([p_sf], lr=5e-3, weight_decay=0.0)
    opt_sf.load_state_dict(torch_sd)

    # 5 more steps on each. They should stay within the cross-impl 1e-5
    # rel-err contract (same as Test 1, but starting from a transferred
    # moment state rather than zero-init).
    for _ in range(5):
        opt_to.zero_grad()
        opt_sf.zero_grad()
        ((p_to - target) ** 2).sum().backward()
        ((p_sf - target) ** 2).sum().backward()
        opt_to.step()
        opt_sf.step()

    rel = (p_to - p_sf).norm() / p_to.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, (
        f"torch ckpt cross-load rel_err = {rel.item():.3e}, "
        "expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Test 4 (bonus) — NaN/Inf grad is silently skipped (does not poison moments)
# ---------------------------------------------------------------------------


def test_nan_grad_skipped():
    """A NaN/Inf gradient must NOT update the parameter or its moments.

    Mirrors the ``not torch.isfinite(grad).all()`` short-circuit in
    ``AdamW.step``; if this regresses, a single bad grad would
    permanently poison Adam's m/v with NaN.
    """
    dev = _device()
    p = torch.nn.Parameter(torch.ones(3, 3, device=dev))
    opt = SfAdamW([p], lr=1e-2, weight_decay=0.0)
    p_before = p.data.clone()

    # Manually attach a NaN gradient (simulating a numerical blow-up).
    p.grad = torch.full_like(p.data, float("nan"))
    opt.step()

    assert torch.equal(p.data, p_before), "NaN grad mutated the parameter"
    # And state should still be empty (no moment init on a skipped step).
    assert id(p) not in opt.state, "NaN grad initialized optim state"
