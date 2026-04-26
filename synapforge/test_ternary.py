"""Tests for sf.quantize — BitNet b1.58 ternary QAT.

Run:
    /opt/conda/bin/python /workspace/synapforge/test_ternary.py

Tests
-----
1. quantize_ternary returns values in {-gamma, 0, +gamma}.
2. STE backward: gradient passes straight through (= grad_output).
3. TernaryLinear: gamma EMA warmup, freeze after N steps.
4. convert_model_to_ternary: respects exclude list, leaves emb/lm_head fp32.
5. End-to-end QAT: train a tiny mscfc-like model 100 steps on random data,
   verify loss decreases monotonically (with some tolerance), no NaN, and
   that final ternary weights live exactly in {-gamma, 0, +gamma}.
6. State-dict roundtrip works.
7. Plasticity / sparse synapse weights are NOT touched by the converter.
"""

from __future__ import annotations

import math
import os
import sys
import time

# Make synapforge importable from /workspace.
sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapforge.quantize import (
    DEFAULT_GAMMA_WARMUP_STEPS,
    TernaryLinear,
    TernaryQuantizer,
    convert_model_to_ternary,
    count_ternary_params,
    freeze_gamma,
    quantize_ternary,
)


# Tests run on GPU 1 to avoid colliding with other agents on GPU 0.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 1 mapped via CUDA_VISIBLE_DEVICES=1
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Ternary buckets
# ---------------------------------------------------------------------------


def test_ternary_buckets() -> None:
    w = torch.randn(64, 128, device=DEVICE) * 0.5
    wq = quantize_ternary(w)
    gamma = w.abs().mean()
    # All values are in {-gamma, 0, +gamma}.
    unique = torch.unique(wq).tolist()
    assert len(unique) <= 3, f"expected <=3 unique values, got {len(unique)}: {unique}"
    for u in unique:
        # Allow a small numerical tolerance because gamma is fp32.
        assert (
            abs(u - 0.0) < 1e-6
            or abs(u - float(gamma.item())) < 1e-5
            or abs(u + float(gamma.item())) < 1e-5
        ), f"unexpected ternary code value: {u} (gamma={gamma.item()})"
    print(f"[1] ternary buckets OK -- {len(unique)} unique values (subset of "
          f"{{-gamma={-gamma.item():.4f}, 0, +gamma={gamma.item():.4f}}})")


# ---------------------------------------------------------------------------
# 2. STE backward = pass-through
# ---------------------------------------------------------------------------


def test_ste_backward() -> None:
    w = torch.randn(8, 16, device=DEVICE, requires_grad=True)
    gamma = w.detach().abs().mean()
    wq = TernaryQuantizer.apply(w, gamma)
    # Pretend upstream gradient is some pattern.
    grad_out = torch.arange(w.numel(), device=DEVICE, dtype=torch.float32).reshape_as(w) * 0.01
    wq.backward(grad_out)
    assert w.grad is not None
    assert torch.allclose(w.grad, grad_out), "STE: grad_w must equal grad_out"
    print("[2] STE backward = pass-through OK")


# ---------------------------------------------------------------------------
# 3. TernaryLinear: gamma EMA + freeze
# ---------------------------------------------------------------------------


def test_gamma_ema_and_freeze() -> None:
    layer = TernaryLinear(32, 64, gamma_warmup_steps=10).to(DEVICE)
    layer.train()
    x = torch.randn(4, 32, device=DEVICE)

    # Step 0: gamma initialized on first forward.
    assert not bool(layer.ternary_initialized.item())
    _ = layer(x)
    assert bool(layer.ternary_initialized.item())
    g0 = float(layer.gamma.item())

    # Drive 10 more forwards with weights mutated in between -> gamma EMA-tracks.
    with torch.no_grad():
        layer.weight.mul_(2.0)
    for _ in range(20):
        _ = layer(x)

    # After warmup, ternary_step should be capped (we did >warmup steps).
    assert int(layer.ternary_step.item()) >= 10
    g1 = float(layer.gamma.item())
    # gamma should have moved toward the new larger weight scale.
    assert g1 > g0, f"gamma should grow after weight scaling: {g0} -> {g1}"

    # freeze_gamma() locks it.
    n_frozen = freeze_gamma(layer)
    assert n_frozen == 1
    g_frozen = float(layer.gamma.item())
    with torch.no_grad():
        layer.weight.mul_(0.5)
    for _ in range(5):
        _ = layer(x)
    assert math.isclose(float(layer.gamma.item()), g_frozen, rel_tol=1e-9), \
        "gamma must not change after freeze_gamma"
    print(f"[3] gamma EMA: {g0:.4f} -> {g1:.4f}, frozen at {g_frozen:.4f} OK")


# ---------------------------------------------------------------------------
# 4. convert_model_to_ternary respects exclude list
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    """A tiny LM-shaped model: emb -> stack of Linear -> lm_head."""

    def __init__(self, vocab: int = 64, dim: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for blk in self.layers:
            x = x + blk(x)
        x = self.norm(x)
        return self.lm_head(x)


def test_convert_respects_exclude() -> None:
    m = TinyLM().to(DEVICE)
    n_replaced = convert_model_to_ternary(m, exclude=("emb", "lm_head"))
    # 3 layers x 2 linears each = 6.
    assert n_replaced == 6, f"expected 6 replacements, got {n_replaced}"
    # lm_head and emb still original.
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, TernaryLinear)
    assert isinstance(m.emb, nn.Embedding)
    # Inner layers replaced.
    for blk in m.layers:
        assert isinstance(blk[0], TernaryLinear)
        assert isinstance(blk[2], TernaryLinear)
    print(f"[4] convert_model_to_ternary: {n_replaced} replaced, emb/lm_head preserved OK")


# ---------------------------------------------------------------------------
# 5. End-to-end QAT: 100 steps, loss decreases, no NaN, ternary buckets
# ---------------------------------------------------------------------------


def test_qat_train_100_steps() -> None:
    torch.manual_seed(42)
    model = TinyLM(vocab=128, dim=64, n_layers=4).to(DEVICE)
    n_replaced = convert_model_to_ternary(model, exclude=("emb", "lm_head"))
    n_tern, n_total = count_ternary_params(model)
    print(f"    pre-train: replaced {n_replaced} layers, "
          f"{n_tern / 1e3:.1f}K ternary / {n_total / 1e3:.1f}K total params "
          f"({n_tern / n_total:.0%} ternarizable)")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    bs, T, vocab = 8, 16, 128
    losses: list[float] = []
    for step in range(100):
        ids = torch.randint(0, vocab, (bs, T), device=DEVICE)
        # Toy task: predict ids[..., 1:] from ids[..., :-1] (next-token).
        logits = model(ids)[..., :-1, :]
        target = ids[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        assert math.isfinite(losses[-1]), f"NaN/Inf loss at step {step}: {losses[-1]}"

    first_avg = sum(losses[:10]) / 10.0
    last_avg = sum(losses[-10:]) / 10.0
    print(f"    losses: first10 avg={first_avg:.4f}, last10 avg={last_avg:.4f}")
    assert last_avg < first_avg, \
        f"loss should decrease over 100 QAT steps, got {first_avg:.4f} -> {last_avg:.4f}"

    # After training, *quantized* weights (what forward actually uses) live
    # in {-gamma, 0, +gamma} per layer.
    model.eval()
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            qw = mod.quantized_weight()
            unique = torch.unique(qw).tolist()
            assert len(unique) <= 3, f"{name}: too many unique quantized values ({len(unique)})"
            gamma = float(mod.gamma.item())
            for u in unique:
                assert (
                    abs(u) < 1e-6 or abs(abs(u) - gamma) < 1e-5
                ), f"{name}: bad ternary code {u} vs gamma={gamma}"
    print("[5] 100 QAT steps: loss decreasing, no NaN, ternary buckets clean OK")


# ---------------------------------------------------------------------------
# 6. State-dict roundtrip
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip() -> None:
    m1 = TinyLM(vocab=32, dim=16, n_layers=2).to(DEVICE)
    convert_model_to_ternary(m1, exclude=("emb", "lm_head"))
    m1.train()
    x = torch.randint(0, 32, (2, 8), device=DEVICE)
    for _ in range(5):
        _ = m1(x)
    sd = m1.state_dict()

    m2 = TinyLM(vocab=32, dim=16, n_layers=2).to(DEVICE)
    convert_model_to_ternary(m2, exclude=("emb", "lm_head"))
    m2.load_state_dict(sd)
    m2.eval()
    m1.eval()
    out1 = m1(x)
    out2 = m2(x)
    assert torch.allclose(out1, out2, atol=1e-5), "state-dict roundtrip mismatch"
    print("[6] state-dict roundtrip OK")


# ---------------------------------------------------------------------------
# 7. Plasticity-style fast weights left alone
# ---------------------------------------------------------------------------


def test_plasticity_buffers_untouched() -> None:
    class WithPlasticity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 32)
            # "Fast weight" -- a Parameter, not inside an nn.Linear.
            self.hebb_fast = nn.Parameter(torch.zeros(32, 32))

    m = WithPlasticity().to(DEVICE)
    n = convert_model_to_ternary(m, exclude=("emb", "lm_head"))
    assert n == 1
    assert isinstance(m.proj, TernaryLinear)
    assert isinstance(m.hebb_fast, nn.Parameter) and not isinstance(m.hebb_fast, TernaryLinear)
    print("[7] plasticity / hebb_fast left untouched OK")


# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE}")
    t0 = time.time()
    test_ternary_buckets()
    test_ste_backward()
    test_gamma_ema_and_freeze()
    test_convert_respects_exclude()
    test_qat_train_100_steps()
    test_state_dict_roundtrip()
    test_plasticity_buffers_untouched()
    print(f"all 7 tests passed in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
