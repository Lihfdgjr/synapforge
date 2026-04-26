"""Numerical correctness tests for synapforge core API.

Verifies (CPU-friendly; CUDA tests run when available):
  1. LiquidCell forward matches a numpy Heinsen-scan reference (rel_err < 1e-3).
  2. PLIF spike-rate is sane and membrane stays bounded over many steps.
  3. sf.compile(..., backend='gpu_dense') is numerically identical to direct forward.
  4. Backward through LiquidCell + PLIF produces finite gradients.
  5. SparseSynapse grow/prune mutate density correctly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import synapforge as sf


# ---------------------------------------------------------------------------
# Reference Heinsen scan in pure numpy (no PyTorch graph) — eliminates the
# previous mscfc dependency.
# ---------------------------------------------------------------------------
def _liquid_reference(
    x: np.ndarray,
    delta_w: np.ndarray, delta_b: np.ndarray,
    b_w: np.ndarray, b_b: np.ndarray,
    A_log: np.ndarray,
    bound: bool = True,
) -> np.ndarray:
    """Reference forward pass mirroring synapforge.LiquidCell exactly."""
    delta_pre = x @ delta_w.T + delta_b                   # (B, T, D)
    delta = np.log1p(np.exp(-np.abs(delta_pre))) + np.maximum(delta_pre, 0.0)
    b_t = delta * (x @ b_w.T + b_b)
    A = np.exp(A_log)
    A_t = np.exp(-delta * A)                              # (B, T, D)
    log_A = np.log(np.maximum(A_t, 1e-30))
    S = np.cumsum(log_A, axis=1)
    inner = np.exp(-S) * b_t
    inner_sum = np.cumsum(inner, axis=1)
    h = np.exp(S) * inner_sum
    return np.tanh(h) if bound else h


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_liquid_matches_reference(device: str) -> None:
    torch.manual_seed(42)
    B, T, D = 4, 32, 64
    cell = sf.LiquidCell(D, D, init="hasani").to(device)
    x = torch.randn(B, T, D, device=device)

    y = cell(x).detach().cpu().numpy()
    y_ref = _liquid_reference(
        x.detach().cpu().numpy(),
        cell.delta_proj.weight.detach().cpu().numpy(),
        cell.delta_proj.bias.detach().cpu().numpy(),
        cell.b_proj.weight.detach().cpu().numpy(),
        cell.b_proj.bias.detach().cpu().numpy(),
        cell.A_log.detach().cpu().numpy(),
        bound=cell.bound,
    )

    rel_err = float(np.abs(y - y_ref).max() / (np.abs(y_ref).max() + 1e-12))
    assert rel_err < 1e-3, f"LiquidCell drift > 1e-3 on {device}: rel_err={rel_err:.3e}"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_plif_dynamics(device: str) -> None:
    torch.manual_seed(0)
    D = 32
    plif = sf.PLIF(D, threshold=0.3, tau_init=1.0, reset_by_subtract=True).to(device)
    cur = 0.6 * torch.randn(4, D, device=device).abs()
    mem = None
    rates = []
    for _ in range(20):
        spk, mem = plif(cur, mem)
        rates.append(spk.mean().item())
    avg = sum(rates) / len(rates)
    assert 0.005 < avg < 0.995, f"PLIF avg spike rate {avg:.3f} out of plausible range"
    assert torch.isfinite(mem).all().item(), "PLIF membrane went non-finite"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_compile_equivalence(device: str) -> None:
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
    _, _, h_a = m(x)
    rt = sf.compile(m, backend="gpu_dense")
    _, _, h_b = rt(x)
    rel_h = (h_a - h_b).abs().max().item() / (h_a.abs().max().item() + 1e-12)
    assert rel_h < 1e-6, f"compile drift {rel_h:.3e}"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_backward_finite(device: str) -> None:
    torch.manual_seed(11)

    class Block(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d)
            self.head = nn.Linear(d, d)

        def forward(self, x):
            h = self.cfc(x)
            spk, _ = self.plif(h)
            return self.head(spk * h)

    m = Block(32).to(device)
    x = torch.randn(2, 16, 32, device=device, requires_grad=True)
    y = m(x).sum()
    y.backward()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all().item(), \
            f"non-finite grad in {n!r}"


def test_sparse_synapse_grow_prune() -> None:
    syn = sf.SparseSynapse(64, 32, sparsity=0.10)
    d0 = syn.density()
    n_grown = syn.grow(n_new=128, criterion="random")
    d1 = syn.density()
    n_pruned = syn.prune(n_prune=64, criterion="magnitude")
    d2 = syn.density()
    assert n_grown > 0 and d1 > d0, "grow did not increase density"
    assert n_pruned > 0 and d2 < d1, "prune did not decrease density"
    y = syn(torch.randn(8, 64))
    assert y.shape == (8, 32)


def test_module_is_nn_module_subclass() -> None:
    """Public sf.Module must subclass torch.nn.Module so .to/.cuda/.eval/etc. work."""
    assert issubclass(sf.Module, nn.Module)


def test_version_string() -> None:
    assert isinstance(sf.__version__, str)
    parts = sf.__version__.split(".")
    assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2])
