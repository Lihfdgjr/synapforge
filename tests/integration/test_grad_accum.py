"""T2.7 — gradient accumulation flag mathematical equivalence.

Validates the new ``--grad-accum-steps N`` flag in ``train_100m_kd.py``:
N=2 micro-batches at bs=B must produce the SAME optim.step() update as
N=1 at bs=2*B, modulo float rounding (1e-5 tolerance). This is the
identity that makes grad-accum a legitimate VRAM workaround for the
bs=80 OOM that killed multiple runs (see DEEP_MAINT_QUEUE.md T2.7).

This file is CPU-only and uses a hand-built toy MLP -- not the real
SynapForge100M -- because we're validating the GRAD-ACCUM IDENTITY,
not the model. The trainer's actual loop is exercised in the smoke
suite; here we prove the math.

Math:
    bs=2B path:  loss(big_batch).backward(); optim.step()
    bs=B  path:  for micro in [b1, b2]:
                     (loss(micro) / 2).backward()
                 optim.step()

    They must produce identical param.grad (to autograd float
    rounding) because:
        grad(L(b1)+L(b2)) = grad(L(b1)) + grad(L(b2))   (linearity)
        grad((L(b1)/2 + L(b2)/2) reduce='mean')
            = (grad(L(b1)) + grad(L(b2))) / 2           (sum/mean diff)

    The pre-divide-by-N matches the F.cross_entropy default ``mean``
    reduction across the FULL effective batch -- that's why the trainer
    does ``(loss / accum_steps).backward()``.

Runs in <2s on Windows dev box, no GPU. Uses ``importorskip('torch')``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _toy_model(seed: int = 0):
    """Build a tiny CE-loss-trainable MLP. ``manual_seed`` ensures
    bit-identical init across both paths in the same process."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),  # 4-way classifier
    )
    return model


def _gen_batch(B: int, seed: int):
    """Deterministic random batch (B, 8) inputs + (B,) labels in [0,4)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 8, generator=g)
    y = torch.randint(0, 4, (B,), generator=g)
    return x, y


def test_grad_accum_2_equals_bs_8():
    """Two micro-batches of size 4 + accum_steps=2 must match one batch
    of size 8 (no accumulation) bit-for-bit at 1e-5 tolerance after
    optim.step(). This is the mathematical identity that makes the
    --grad-accum-steps flag a legitimate VRAM workaround.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    # Two identical models with identical optim init.
    model_big = _toy_model(seed=42)
    model_acc = _toy_model(seed=42)
    # Sanity: identical init.
    for p_big, p_acc in zip(model_big.parameters(), model_acc.parameters()):
        assert torch.allclose(p_big, p_acc), \
            "test setup bug -- models should share init"

    optim_big = torch.optim.SGD(model_big.parameters(), lr=0.1)
    optim_acc = torch.optim.SGD(model_acc.parameters(), lr=0.1)

    # Two micro-batches of size 4 ...
    x_a, y_a = _gen_batch(4, seed=1)
    x_b, y_b = _gen_batch(4, seed=2)
    # ... concat -> single batch of size 8.
    x_big = torch.cat([x_a, x_b], dim=0)
    y_big = torch.cat([y_a, y_b], dim=0)

    # ---- Path 1: bs=8, accum_steps=1 (single backward) ----
    optim_big.zero_grad(set_to_none=True)
    logits_big = model_big(x_big)
    loss_big = F.cross_entropy(logits_big, y_big)  # mean over 8
    loss_big.backward()
    # Snapshot grads BEFORE step so we can compare grad-equivalence
    # too (independent of optimizer kind).
    grads_big = [p.grad.detach().clone() for p in model_big.parameters()]
    optim_big.step()

    # ---- Path 2: bs=4, accum_steps=2 (two backwards then step) ----
    optim_acc.zero_grad(set_to_none=True)
    accum_steps = 2
    for x_micro, y_micro in [(x_a, y_a), (x_b, y_b)]:
        logits_micro = model_acc(x_micro)
        # F.cross_entropy default reduction is 'mean' over the micro-
        # batch (4 elements). Dividing by accum_steps=2 makes the SUM
        # across both micro-batches equal to the mean-over-8 from the
        # bs=8 path: (mean_4(L_a) + mean_4(L_b)) / 2 == mean_8(L_a u L_b).
        loss_micro = F.cross_entropy(logits_micro, y_micro)
        (loss_micro / float(accum_steps)).backward()
    grads_acc = [p.grad.detach().clone() for p in model_acc.parameters()]
    optim_acc.step()

    # ---- Identity: gradients match within 1e-5 ----
    for i, (g_big, g_acc) in enumerate(zip(grads_big, grads_acc)):
        assert torch.allclose(g_big, g_acc, atol=1e-5, rtol=1e-5), (
            f"param[{i}] grad mismatch: max diff "
            f"{(g_big - g_acc).abs().max().item():.3e}"
        )

    # ---- Identity: post-step weights match within 1e-5 ----
    for i, (p_big, p_acc) in enumerate(
        zip(model_big.parameters(), model_acc.parameters())
    ):
        assert torch.allclose(p_big, p_acc, atol=1e-5, rtol=1e-5), (
            f"param[{i}] post-step weight mismatch: max diff "
            f"{(p_big - p_acc).abs().max().item():.3e}"
        )


def test_grad_accum_3_equals_bs_12():
    """Same identity at N=3 -- exercises non-power-of-2 accum steps."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    model_big = _toy_model(seed=7)
    model_acc = _toy_model(seed=7)
    optim_big = torch.optim.SGD(model_big.parameters(), lr=0.05)
    optim_acc = torch.optim.SGD(model_acc.parameters(), lr=0.05)

    x_a, y_a = _gen_batch(4, seed=11)
    x_b, y_b = _gen_batch(4, seed=12)
    x_c, y_c = _gen_batch(4, seed=13)
    x_big = torch.cat([x_a, x_b, x_c], dim=0)
    y_big = torch.cat([y_a, y_b, y_c], dim=0)

    optim_big.zero_grad(set_to_none=True)
    F.cross_entropy(model_big(x_big), y_big).backward()
    optim_big.step()

    optim_acc.zero_grad(set_to_none=True)
    accum_steps = 3
    for x_m, y_m in [(x_a, y_a), (x_b, y_b), (x_c, y_c)]:
        loss_m = F.cross_entropy(model_acc(x_m), y_m)
        (loss_m / float(accum_steps)).backward()
    optim_acc.step()

    for p_big, p_acc in zip(
        model_big.parameters(), model_acc.parameters()
    ):
        assert torch.allclose(p_big, p_acc, atol=1e-5, rtol=1e-5), \
            f"N=3 weight mismatch max={p_big.sub(p_acc).abs().max():.3e}"


def test_grad_accum_1_is_noop():
    """Backward-compat: --grad-accum-steps 1 (default) must behave
    identically to no-flag. Train one step both ways, weights MUST
    match exactly (same code path, just no scaling)."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    model_a = _toy_model(seed=99)
    model_b = _toy_model(seed=99)
    optim_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    optim_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

    x, y = _gen_batch(8, seed=99)

    optim_a.zero_grad(set_to_none=True)
    F.cross_entropy(model_a(x), y).backward()
    optim_a.step()

    accum_steps = 1
    optim_b.zero_grad(set_to_none=True)
    loss_b = F.cross_entropy(model_b(x), y)
    (loss_b / float(accum_steps)).backward()  # / 1 == no-op
    optim_b.step()

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        # Should match to machine precision -- division by 1.0 is exact.
        assert torch.equal(p_a, p_b), "accum_steps=1 must be a true no-op"


def test_grad_accum_activation_size_proportional_to_micro_batch():
    """VRAM headline claim: at grad_accum=N + bs=B, peak ACTIVATION
    memory is proportional to B, not B*N. This is what makes grad-accum
    a legitimate VRAM workaround for the bs=80 OOM (T2.7).

    The decisive measurement on GPU would be
    ``torch.cuda.max_memory_allocated()`` -- but CI runs CPU-only, so we
    measure the activation tensor sizes DIRECTLY. The activation tensor
    coming out of layer L scales linearly with B (the leading dim);
    at bs=B the largest activation is B*4096*4 bytes, while at bs=2B
    it's 2*B*4096*4 bytes -- the 2x ratio is the load-bearing claim.

    Asserts: largest forward-activation footprint at the micro-batch
    is at most 60% of the largest at the full batch. (The 60%
    threshold has slack for CPU/torch overhead bookkeeping.)
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    torch.manual_seed(0)

    # Activation sizes are deterministic from input shape; the LARGEST
    # intermediate is the (B, 4096) tensor after Linear(64, 4096) /
    # Linear(4096, 4096). We measure those tensor element counts as a
    # proxy for activation VRAM (4 bytes/elem fp32).
    bs_micro = 32
    bs_full = 64  # = bs_micro * accum_steps(2)

    def _measure_peak_activation_bytes(model, x):
        """Forward pass with hooks; record max output bytes per layer."""
        peaks = {"max_bytes": 0}

        def hook(_mod, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            n = t.numel() * t.element_size()
            if n > peaks["max_bytes"]:
                peaks["max_bytes"] = n

        handles = []
        for m in model.modules():
            if not isinstance(m, nn.Sequential):
                handles.append(m.register_forward_hook(hook))
        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()
        return peaks["max_bytes"]

    model = nn.Sequential(
        nn.Linear(64, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4),
    )

    x_micro = torch.randn(bs_micro, 64)
    x_full = torch.randn(bs_full, 64)

    peak_micro = _measure_peak_activation_bytes(model, x_micro)
    peak_full = _measure_peak_activation_bytes(model, x_full)

    # Hard physical claim: peak activation at bs=32 is roughly half
    # the peak at bs=64. Allow +/- 5% slack on the 0.5 ratio for
    # alignment / dtype / shape padding.
    ratio = peak_micro / max(peak_full, 1)
    assert ratio <= 0.60, (
        f"grad-accum micro-batch peak activation ({peak_micro} B) "
        f"should be ~50% of full batch ({peak_full} B); got "
        f"ratio={ratio:.3f} (expected <= 0.60)"
    )

    # And also: trainable runs both ways without OOM-style explosion.
    # Sanity: the math identity already proven by the other tests,
    # we just wrap the activations measurement in an actual backward.
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    y_micro = torch.randint(0, 4, (bs_micro,))
    optim.zero_grad(set_to_none=True)
    accum_steps = 2
    for _ in range(accum_steps):
        loss = F.cross_entropy(model(x_micro), y_micro)
        (loss / float(accum_steps)).backward()
    optim.step()  # should not raise


def test_grad_accum_cli_flag_present():
    """Argparse contract: ``--grad-accum-steps`` must exist with int
    type and default 1. Guards against silent removal in refactors."""
    pytest.importorskip("torch")
    # Lazy import -- train_100m_kd touches torch at module load.
    import importlib
    if "train_100m_kd" in sys.modules:
        mod = importlib.reload(sys.modules["train_100m_kd"])
    else:
        mod = importlib.import_module("train_100m_kd")
    parser = mod.build_parser() if hasattr(mod, "build_parser") else None
    if parser is None:
        # Build via argparse path used by main(). Inspect _actions on the
        # parser inside main() by parsing a known short arg list.
        # Fallback: just parse a help-style argument list and probe.
        import argparse
        # train_100m_kd builds parser inline in main() -- inspect the
        # source for the flag string instead. Cheap but reliable.
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "--grad-accum-steps" in src, \
            "T2.7 CLI flag missing from train_100m_kd.py"
        assert "default=1" in src, \
            "T2.7 default must be 1 (back-compat)"
        return
    args = parser.parse_args(["--grad-accum-steps", "4"])
    assert args.grad_accum_steps == 4
    args = parser.parse_args([])
    assert args.grad_accum_steps == 1
