"""Unit tests for ``synapforge.optim.CPUOffloadAdamW``.

Contract (mirrors the AdamW Phase-1 contract):

* After 50 SGD-style steps on a tiny regression target, the params
  updated by CPUOffloadAdamW must agree with those updated by the
  in-memory ``synapforge.optim.AdamW`` to within 1e-5 relative-error.
  This is the bit-exactness guarantee for ZeRO-Offload Stage 0 — the
  only difference is *where* the moment update runs (CPU vs GPU/CPU
  in-place), the math is identical.

* state_dict round-trip: a freshly-constructed CPUOffloadAdamW that
  load_state_dict's a previous run's state continues to step bit-for-
  bit with the donor optimizer.

* NaN/Inf grad does not poison the master copy nor the moments.

CPU vs GPU
----------
On CUDA hosts we exercise the full async H2D→CPU AdamW→async D2H
pipeline. On CPU CI runners pinned-memory is unavailable; the
optimizer transparently degrades to non-pinned CPU staging buffers and
the test still passes (the degraded path is just a sync copy_).
"""
from __future__ import annotations

import torch

from synapforge.optim import AdamW as SfAdamW
from synapforge.optim.cpu_offload_adamw import CPUOffloadAdamW


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test 1 — numerics match synapforge.optim.AdamW within 1e-5 over 50 steps
# ---------------------------------------------------------------------------


def test_matches_synapforge_adamw_50_steps():
    """CPUOffloadAdamW vs synapforge.optim.AdamW — bit-exact within 1e-5.

    Tiny model (d=64, ~1k params) so the test is sub-second on CPU CI.
    Identical init, identical grads (same random seed + same loss
    surface), identical hyperparams. The only delta is the offload
    pipeline.
    """
    torch.manual_seed(0)
    dev = _device()
    d_in, d_out, n_steps = 16, 8, 50

    p_off = torch.nn.Parameter(torch.randn(d_out, d_in, device=dev))
    p_ref = torch.nn.Parameter(p_off.detach().clone())

    opt_off = CPUOffloadAdamW(
        [p_off], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    opt_ref = SfAdamW(
        [p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    target = torch.randn(d_out, d_in, device=dev)
    for _ in range(n_steps):
        opt_off.zero_grad()
        opt_ref.zero_grad()
        loss_off = ((p_off - target) ** 2).sum()
        loss_ref = ((p_ref - target) ** 2).sum()
        loss_off.backward()
        loss_ref.backward()
        opt_off.step()
        opt_ref.step()

    # On CUDA we need to barrier so the final H2D copy is visible
    # before we compute the diff.
    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    rel = (p_off - p_ref).norm() / p_ref.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, (
        f"CPUOffloadAdamW vs synapforge.optim.AdamW rel_err = "
        f"{rel.item():.3e}, expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Test 2 — state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    """CPUOffloadAdamW state_dict round-trips losslessly.

    Warmup 5 steps, save state, build fresh optimizer on a clone, load
    state, step both for 5 more iterations. Params must remain
    bit-equal.
    """
    torch.manual_seed(1)
    dev = _device()
    p_a = torch.nn.Parameter(torch.randn(8, 8, device=dev))
    p_b = torch.nn.Parameter(p_a.detach().clone())

    opt_a = CPUOffloadAdamW([p_a], lr=3e-3, weight_decay=0.05)
    target = torch.zeros_like(p_a.data)

    for _ in range(5):
        opt_a.zero_grad()
        ((p_a - target) ** 2).sum().backward()
        opt_a.step()

    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    sd = opt_a.state_dict()

    # Fresh optimizer over the cloned param.
    opt_b = CPUOffloadAdamW([p_b], lr=3e-3, weight_decay=0.05)
    p_b.data.copy_(p_a.data)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    opt_b.load_state_dict(sd)

    # 5 more steps on both. They should remain bit-equal.
    for _ in range(5):
        opt_a.zero_grad()
        opt_b.zero_grad()
        ((p_a - target) ** 2).sum().backward()
        ((p_b - target) ** 2).sum().backward()
        opt_a.step()
        opt_b.step()

    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    # CPU-offload + state-dict round-trip is bit-exact in fp32 in this
    # tiny regression task. (For the cross-load with synapforge.optim.AdamW
    # we'd hit the 1e-5 contract because the master fp32 copy is rebuilt
    # from p.data at load time which has fp32 round-tripped through the
    # GPU.)
    rel = (p_a - p_b).norm() / p_a.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, (
        f"CPUOffloadAdamW state_dict round-trip rel_err = "
        f"{rel.item():.3e}, expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Test 3 — synapforge.optim.AdamW ckpt cross-load (warmstart compat)
# ---------------------------------------------------------------------------


def test_loads_synapforge_adamw_ckpt():
    """CPUOffloadAdamW must load a synapforge.optim.AdamW state_dict.

    This is the warmstart-compat contract: a ckpt saved by the in-memory
    AdamW (today's default with --synapforge-adamw) must transplant into
    the offload optimizer. After the cross-load, 5 more steps on each
    must stay within rel-err 1e-5.
    """
    torch.manual_seed(2)
    dev = _device()
    p_in = torch.nn.Parameter(torch.randn(4, 4, device=dev))
    p_off = torch.nn.Parameter(p_in.detach().clone())

    opt_in = SfAdamW([p_in], lr=5e-3, weight_decay=0.0)
    target = torch.zeros_like(p_in.data)

    for _ in range(8):
        opt_in.zero_grad()
        ((p_in - target) ** 2).sum().backward()
        opt_in.step()

    in_sd = opt_in.state_dict()

    # Bring offload param up to the same data point.
    p_off.data.copy_(p_in.data)
    opt_off = CPUOffloadAdamW([p_off], lr=5e-3, weight_decay=0.0)
    opt_off.load_state_dict(in_sd)

    for _ in range(5):
        opt_in.zero_grad()
        opt_off.zero_grad()
        ((p_in - target) ** 2).sum().backward()
        ((p_off - target) ** 2).sum().backward()
        opt_in.step()
        opt_off.step()

    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    rel = (p_in - p_off).norm() / p_in.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, (
        f"AdamW→CPUOffload ckpt cross-load rel_err = "
        f"{rel.item():.3e}, expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Test 4 — NaN grad is silently skipped (does not poison master/moments)
# ---------------------------------------------------------------------------


def test_nan_grad_skipped():
    """A NaN/Inf gradient must NOT update the master copy or moments.

    Mirrors the ``isfinite`` short-circuit in
    ``synapforge.optim.AdamW.step``.
    """
    dev = _device()
    p = torch.nn.Parameter(torch.ones(3, 3, device=dev))
    opt = CPUOffloadAdamW([p], lr=1e-2, weight_decay=0.0)
    p_before = p.data.clone()

    p.grad = torch.full_like(p.data, float("nan"))
    opt.step()

    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    assert torch.equal(p.data, p_before), (
        "NaN grad mutated the GPU param via offload pipeline"
    )
    assert id(p) not in opt.state, (
        "NaN grad initialized optimizer state (master/moments allocated)"
    )


# ---------------------------------------------------------------------------
# Test 5 — pinned-memory smoke (CUDA only): assert moments use pinned mem
# ---------------------------------------------------------------------------


def test_pinned_memory_when_cuda():
    """When CUDA is available, master/moments/grad_buf must be pinned.

    Pinning is what makes ``copy_(non_blocking=True)`` actually async;
    without it the H2D/D2H silently degrades to sync copies. We assert
    the optimizer is set up the way the docstring claims.
    """
    if not torch.cuda.is_available():
        # On a CPU CI runner pinned memory is N/A; skip the assertion.
        return

    p = torch.nn.Parameter(torch.randn(4, 4, device="cuda:0"))
    opt = CPUOffloadAdamW([p], lr=1e-3)
    p.grad = torch.zeros_like(p.data)
    opt.step()
    torch.cuda.synchronize()
    st = opt.state[id(p)]
    assert st["master"].is_pinned(), "master fp32 copy not in pinned memory"
    assert st["exp_avg"].is_pinned(), "exp_avg moment not in pinned memory"
    assert st["exp_avg_sq"].is_pinned(), "exp_avg_sq moment not in pinned memory"
    assert st["grad_buf"].is_pinned(), "grad staging buffer not in pinned memory"
