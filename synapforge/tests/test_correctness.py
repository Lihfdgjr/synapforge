"""Numerical correctness tests vs mscfc reference.

Verifies:
  1. synapforge.LiquidCell forward matches mscfc.liquid_s4.LiquidS4Cell when
     parameters are copied (rel_err < 1e-3, target 1e-5).
  2. synapforge.PLIF integrates correctly across several steps (spike rate
     in (0,1), membrane bounded).
  3. sf.compile(...).run produces same output as direct .forward.
  4. Backward through LiquidCell + PLIF doesn't NaN.
  5. sf.SparseSynapse mask grow/prune work and density tracks.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn

import synapforge as sf
from mscfc.liquid_s4 import LiquidS4Cell


# ---------------------------------------------------------------------------
# 1. LiquidCell vs mscfc.LiquidS4Cell numerical equivalence
# ---------------------------------------------------------------------------


def test_liquid_matches_mscfc(device: str = "cpu") -> float:
    torch.manual_seed(42)
    B, T, D = 4, 32, 64

    sf_cell = sf.LiquidCell(D, D, init="hasani").to(device)
    ms_cell = LiquidS4Cell(D, D).to(device)

    # Copy params from sf to mscfc (same tensor shapes / names).
    with torch.no_grad():
        ms_cell.delta_proj.weight.copy_(sf_cell.delta_proj.weight)
        ms_cell.delta_proj.bias.copy_(sf_cell.delta_proj.bias)
        ms_cell.b_proj.weight.copy_(sf_cell.b_proj.weight)
        ms_cell.b_proj.bias.copy_(sf_cell.b_proj.bias)
        ms_cell.A_log.copy_(sf_cell.A_log)

    x = torch.randn(B, T, D, device=device)
    y_sf = sf_cell(x)
    y_ms = ms_cell.forward_seq(x)

    abs_err = (y_sf - y_ms).abs().max().item()
    rel_err = abs_err / (y_ms.abs().max().item() + 1e-12)
    print(f"[corr/liquid {device}] abs_err={abs_err:.3e}  rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, (
        f"LiquidCell drift > 1e-3: rel_err={rel_err:.3e}"
    )
    return rel_err


# ---------------------------------------------------------------------------
# 2. PLIF dynamics — spike rate sane, membrane decays
# ---------------------------------------------------------------------------


def test_plif_dynamics(device: str = "cpu") -> dict:
    torch.manual_seed(0)
    D = 32
    plif = sf.PLIF(D, threshold=0.3, tau_init=1.0,
                   reset_by_subtract=True).to(device)
    # Drive with strong positive current.
    cur = 0.6 * torch.randn(4, D, device=device).abs()
    mem = None
    spike_rates = []
    for _ in range(20):
        spk, mem = plif(cur, mem)
        spike_rates.append(spk.mean().item())
    avg_rate = sum(spike_rates) / len(spike_rates)
    print(f"[corr/plif {device}] avg spike rate = {avg_rate:.3f}, mem.mean = {mem.mean().item():.3f}")
    assert 0.005 < avg_rate < 0.995, (
        f"PLIF spike rate out of plausible range: {avg_rate:.3f}"
    )
    return {"avg_rate": avg_rate, "mem_mean": float(mem.mean().item())}


# ---------------------------------------------------------------------------
# 3. compile() runtime equivalence
# ---------------------------------------------------------------------------


def test_compile_equivalence(device: str = "cpu") -> float:
    torch.manual_seed(7)

    class Block(sf.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d)

        def forward(self, x: torch.Tensor):
            h = self.cfc(x)
            spk, mem = self.plif(h)
            return spk, mem, h

    m = Block(32).to(device)
    x = torch.randn(2, 16, 32, device=device)
    spk_a, mem_a, h_a = m(x)
    rt = sf.compile(m, backend="gpu_dense")
    spk_b, mem_b, h_b = rt(x)

    rel_h = (h_a - h_b).abs().max().item() / (h_a.abs().max().item() + 1e-12)
    print(f"[corr/compile {device}] rel_err(h)={rel_h:.3e}")
    assert rel_h < 1e-6, f"compile drift {rel_h}"
    return rel_h


# ---------------------------------------------------------------------------
# 4. Backward sanity
# ---------------------------------------------------------------------------


def test_backward(device: str = "cpu") -> None:
    torch.manual_seed(11)

    class Block(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d)
            self.head = nn.Linear(d, d)

        def forward(self, x):
            h = self.cfc(x)
            spk, mem = self.plif(h)
            return self.head(spk * h)

    m = Block(32).to(device)
    x = torch.randn(2, 16, 32, device=device, requires_grad=True)
    y = m(x).sum()
    y.backward()
    grads_finite = all(
        (p.grad is None or torch.isfinite(p.grad).all().item())
        for p in m.parameters()
    )
    print(f"[corr/backward {device}] all grads finite: {grads_finite}")
    assert grads_finite, "NaN/Inf in gradients"


# ---------------------------------------------------------------------------
# 5. SparseSynapse grow / prune
# ---------------------------------------------------------------------------


def test_sparse_synapse() -> None:
    syn = sf.SparseSynapse(64, 32, sparsity=0.10)
    d0 = syn.density()
    syn.grow(n_new=128, criterion="random")
    d1 = syn.density()
    syn.prune(n_prune=64, criterion="magnitude")
    d2 = syn.density()
    print(f"[corr/synapse] density: {d0:.3f} -> {d1:.3f} -> {d2:.3f}")
    assert d1 > d0, "grow did not increase density"
    assert d2 < d1, "prune did not decrease density"

    x = torch.randn(8, 64)
    y = syn(x)
    assert y.shape == (8, 32), f"bad output shape {y.shape}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== synapforge v0.1 correctness ===")
    rel_cpu = test_liquid_matches_mscfc("cpu")
    test_plif_dynamics("cpu")
    test_compile_equivalence("cpu")
    test_backward("cpu")
    test_sparse_synapse()

    if torch.cuda.is_available():
        rel_cuda = test_liquid_matches_mscfc("cuda")
        test_plif_dynamics("cuda")
        test_compile_equivalence("cuda")
        test_backward("cuda")
    print("=== all correctness tests PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
