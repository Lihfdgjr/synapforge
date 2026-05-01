"""T2.8 — BitNet b1.58 ternary QAT on CfC input projections.

Validates the new ``--quant-cfc-weights ternary`` flag in
``train_100m_kd.py`` and the underlying ``LiquidCell(weight_quant=...)``
constructor arg + ``synapforge.quantize.apply_ternary_to_cfc`` helper.
This is the M1 milestone of ``docs/MATMUL_FREE.md``.

Coverage
--------
1. ``test_quant_default_off`` — default args build a model with fp
   ``nn.Linear`` input projections (back-compat).
2. ``test_quant_on_produces_ternary_values`` — with ``weight_quant=
   'ternary'``, the quantized forward weight has ``unique values
   subset {-gamma, 0, +gamma}`` per layer.
3. ``test_state_dict_roundtrip`` — save_state / load_state preserves
   the quantization status, gamma buffers, and ternary_step counter.
4. ``test_ste_backward_flows`` — after ``loss.backward()`` the
   ``TernaryLinear.weight`` gradient is non-zero, finite, and matches
   the fp weight shape (the straight-through identity).
5. ``test_loss_within_1pct_of_fp`` — 100 step synthetic train, the
   ternary final loss is within 1% of the fp baseline (BitNet b1.58
   acceptance bar from MATMUL_FREE.md M1).

CPU-only, <30s wall on a dev box. Uses ``importorskip("torch")`` so
the file skips cleanly when torch is absent (matches the pattern in
``test_kd_topk_softmax.py`` / ``test_grad_accum.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures: tiny model factory so every test runs in seconds on CPU.
# ---------------------------------------------------------------------------


def _build_tiny_model(weight_quant: str = "none", seed: int = 0):
    """Tiny SynapForge100M: vocab=64, d=16, 1 layer, 1 loop, seq=8.

    ~40k params. Fits in <100ms on CPU per forward pass; suitable for
    the 100-step parity test below to run in <15s wall.
    """
    import torch
    from synapforge.model_100m import build_synapforge_100m
    torch.manual_seed(seed)
    return build_synapforge_100m(
        vocab=64,
        d=16,
        n_layers=1,
        loop_depth=1,
        max_seq=8,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        # T2.4: live_vocab is internal default; the tail-freeze hook is
        # off here because vocab=64 < QWEN25_LIVE_VOCAB so all rows are
        # "live". freeze_vocab_tail flag has no effect at this scale.
        freeze_vocab_tail=False,
        lm_head_spectral_norm=False,
        weight_quant_cfc=weight_quant,
    )


# ---------------------------------------------------------------------------
# 1. Default off — fp nn.Linear input projections.
# ---------------------------------------------------------------------------


def test_quant_default_off():
    """Default ``weight_quant_cfc='none'`` must keep CfC input projections
    as plain ``nn.Linear`` (not ``TernaryLinear``). This guards against
    silent activation of the QAT path on existing trainings.
    """
    pytest.importorskip("torch")
    import torch.nn as nn
    from synapforge.cells.liquid import LiquidCell
    from synapforge.quantize import TernaryLinear

    model = _build_tiny_model(weight_quant="none", seed=0)

    # Find every LiquidCell in the model and verify their input
    # projections are plain nn.Linear, NOT TernaryLinear.
    cells = [m for m in model.modules() if isinstance(m, LiquidCell)]
    assert len(cells) >= 1, (
        "test setup bug -- tiny model should contain >=1 LiquidCell"
    )
    n_linear = 0
    for cell in cells:
        for proj_name in ("delta_proj", "b_proj"):
            proj = getattr(cell, proj_name)
            assert isinstance(proj, nn.Linear), (
                f"LiquidCell.{proj_name} must be nn.Linear when "
                f"weight_quant='none'; got {type(proj).__name__}"
            )
            assert not isinstance(proj, TernaryLinear), (
                f"LiquidCell.{proj_name} must NOT be TernaryLinear "
                f"under weight_quant='none' (got TernaryLinear)"
            )
            n_linear += 1
    # Belt-and-suspenders: zero TernaryLinear modules anywhere.
    n_ternary = sum(
        1 for m in model.modules() if isinstance(m, TernaryLinear)
    )
    assert n_ternary == 0, (
        f"weight_quant='none' must leave zero TernaryLinear in the "
        f"model; found {n_ternary}"
    )
    assert n_linear == 2 * len(cells), (
        f"each LiquidCell should expose delta_proj + b_proj as nn.Linear; "
        f"found {n_linear} for {len(cells)} cells"
    )


# ---------------------------------------------------------------------------
# 2. Ternary on — quantized weight values lie in {-gamma, 0, +gamma}.
# ---------------------------------------------------------------------------


def test_quant_on_produces_ternary_values():
    """With ``weight_quant='ternary'``, every CfC input projection is a
    ``TernaryLinear`` whose ``quantized_weight()`` returns values
    drawn from the ternary set ``{-gamma, 0, +gamma}`` (per-tensor).
    Also asserts the recurrent decay parameter ``A_log`` is NOT
    quantized (still fp32, drawn from a continuous distribution).
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn  # noqa: F401
    from synapforge.cells.liquid import LiquidCell
    from synapforge.quantize import TernaryLinear

    model = _build_tiny_model(weight_quant="ternary", seed=1)

    # Walk one forward pass so gamma initialises (TernaryLinear inits
    # gamma on first call, EMA-updates while training).
    model.train()
    x = torch.randint(0, 64, (2, 8), dtype=torch.long)
    _ = model(x)

    cells = [m for m in model.modules() if isinstance(m, LiquidCell)]
    assert len(cells) >= 1
    n_ternary = 0
    for cell in cells:
        for proj_name in ("delta_proj", "b_proj"):
            proj = getattr(cell, proj_name)
            assert isinstance(proj, TernaryLinear), (
                f"LiquidCell.{proj_name} must be TernaryLinear under "
                f"weight_quant='ternary'; got {type(proj).__name__}"
            )
            n_ternary += 1
            wq = proj.quantized_weight()
            gamma = float(proj.gamma.item())
            assert gamma > 0, (
                f"gamma must be > 0 after first forward; got {gamma}"
            )
            unique = torch.unique(wq).tolist()
            # The bucketing yields at most 3 unique values per layer.
            assert len(unique) <= 3, (
                f"TernaryLinear must produce <=3 unique values; "
                f"got {len(unique)}: {unique}"
            )
            for u in unique:
                # Allow fp32 epsilon for round-trip via gamma multiply.
                assert (
                    abs(u - 0.0) < 1e-6
                    or abs(u - gamma) < 1e-5
                    or abs(u + gamma) < 1e-5
                ), (
                    f"TernaryLinear weight values must be in "
                    f"{{-gamma, 0, +gamma}} (gamma={gamma}); got {u}"
                )

        # A_log must NOT be ternary -- it's the recurrent decay (the
        # "tau"), which is the most sensitive parameter in a CfC.
        a_log = cell.A_log
        n_unique_a = int(torch.unique(a_log.detach()).numel())
        assert n_unique_a > 3, (
            f"A_log must remain fp (not quantized); got "
            f"{n_unique_a} unique values, expected >3"
        )

    assert n_ternary == 2 * len(cells), (
        f"weight_quant='ternary' should wire 2 TernaryLinears per cell; "
        f"got {n_ternary} for {len(cells)} cells"
    )


# ---------------------------------------------------------------------------
# 3. State-dict roundtrip — save+load preserves quantization status.
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip(tmp_path: Path):
    """Save an in-train ternary model's ``state_dict`` and reload it
    into a freshly-built ternary model. Must preserve:
        * the fp Parameter ``weight`` of every TernaryLinear,
        * the ``gamma`` buffer (which is what makes the forward
          deterministic across save/load),
        * the ``ternary_step`` counter (so warmup-then-frozen status
          survives the roundtrip).
    """
    torch = pytest.importorskip("torch")
    from synapforge.quantize import TernaryLinear

    # Train the source model for 5 forward passes so gamma + step
    # counter are non-default.
    src = _build_tiny_model(weight_quant="ternary", seed=2)
    src.train()
    torch.manual_seed(99)
    x = torch.randint(0, 64, (2, 8), dtype=torch.long)
    for _ in range(5):
        _ = src(x)

    # Save.
    ckpt_path = tmp_path / "src.pt"
    torch.save({"model": src.state_dict()}, ckpt_path)

    # Build dst with identical arch but DIFFERENT init seed so we
    # know the load is what's restoring the state (not random match).
    dst = _build_tiny_model(weight_quant="ternary", seed=99)

    # Sanity: pre-load, dst params differ from src.
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    differ_pre = False
    for k in src_sd:
        if k in dst_sd and src_sd[k].shape == dst_sd[k].shape:
            if not torch.equal(src_sd[k], dst_sd[k]):
                differ_pre = True
                break
    assert differ_pre, (
        "test setup bug -- src and dst should differ before load_state_dict"
    )

    # Load.
    blob = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = dst.load_state_dict(blob["model"], strict=False)
    assert not unexpected, (
        f"unexpected keys on load: {unexpected[:5]}"
    )

    # Walk every TernaryLinear pair and confirm bit-identical buffers.
    src_layers = [
        (n, m) for n, m in src.named_modules() if isinstance(m, TernaryLinear)
    ]
    dst_layers = [
        (n, m) for n, m in dst.named_modules() if isinstance(m, TernaryLinear)
    ]
    assert len(src_layers) == len(dst_layers) > 0, (
        f"layer counts must match: {len(src_layers)} vs {len(dst_layers)}"
    )
    for (sn, sm), (dn, dm) in zip(src_layers, dst_layers):
        assert sn == dn, f"layer name mismatch: {sn} vs {dn}"
        # Param: fp weight
        assert torch.equal(sm.weight.data, dm.weight.data), (
            f"{sn}.weight not preserved across save/load"
        )
        # Buffer: gamma
        assert torch.equal(sm.gamma, dm.gamma), (
            f"{sn}.gamma not preserved (src={sm.gamma.item()} "
            f"dst={dm.gamma.item()})"
        )
        # Buffer: step counter (warmup status)
        assert int(sm.ternary_step.item()) == int(dm.ternary_step.item()), (
            f"{sn}.ternary_step not preserved "
            f"(src={int(sm.ternary_step.item())} "
            f"dst={int(dm.ternary_step.item())})"
        )
        # Buffer: initialised flag
        assert bool(sm.ternary_initialized.item()) == bool(
            dm.ternary_initialized.item()
        ), f"{sn}.ternary_initialized not preserved"

    # Final guarantee: identical forward post-load.
    src.eval()
    dst.eval()
    with torch.no_grad():
        out_src = src(x)
        out_dst = dst(x)
    assert torch.allclose(out_src, out_dst, atol=1e-5, rtol=1e-5), (
        "post-load forward differs from src "
        f"(max diff {(out_src - out_dst).abs().max().item():.3e})"
    )


# ---------------------------------------------------------------------------
# 4. STE backward — gradient flows through the round/clamp.
# ---------------------------------------------------------------------------


def test_ste_backward_flows():
    """A backward pass through a ternary model must populate
    ``TernaryLinear.weight.grad`` with a non-zero, finite tensor of the
    same shape as the fp weight. This validates the straight-through
    estimator: the round + clamp in forward is treated as identity for
    gradient purposes.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812
    from synapforge.quantize import TernaryLinear

    model = _build_tiny_model(weight_quant="ternary", seed=3)
    model.train()

    torch.manual_seed(0)
    x = torch.randint(0, 64, (2, 8), dtype=torch.long)
    y = torch.randint(0, 64, (2, 8), dtype=torch.long)

    # Zero any pre-existing grad (paranoia -- should be None on a fresh
    # build, but guards against rerun-state contamination).
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    logits = model(x)
    # CE over (B*T, V) is the same loss the trainer uses.
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()

    # Walk every TernaryLinear and confirm its weight has a populated
    # grad with the right shape and finite values.
    n_checked = 0
    for name, m in model.named_modules():
        if not isinstance(m, TernaryLinear):
            continue
        n_checked += 1
        w = m.weight
        assert w.grad is not None, (
            f"{name}.weight.grad is None after backward (STE broken?)"
        )
        assert w.grad.shape == w.shape, (
            f"{name}.weight.grad shape {tuple(w.grad.shape)} != "
            f"weight shape {tuple(w.shape)}"
        )
        assert torch.isfinite(w.grad).all().item(), (
            f"{name}.weight.grad contains non-finite values "
            f"(NaN/inf -- STE broken?)"
        )
        # The grad MUST have at least one non-zero element. With a
        # CE loss on random tokens through a 16-dim model, all grads
        # are dense to fp32 precision -- a zero grad would mean the
        # forward path skipped this layer entirely.
        assert w.grad.abs().sum().item() > 0, (
            f"{name}.weight.grad is identically zero (forward path "
            f"didn't reach this layer)"
        )
    assert n_checked > 0, (
        "test setup bug -- no TernaryLinear layers found in ternary model"
    )


# ---------------------------------------------------------------------------
# 5. 100-step parity — ternary loss within 1% of fp baseline.
# ---------------------------------------------------------------------------


def test_loss_within_1pct_of_fp():
    """100 steps of synthetic training: ternary CfC loss must end
    within 1% of fp baseline. This is the BitNet b1.58 acceptance
    criterion documented in ``docs/MATMUL_FREE.md`` M1: discretizing
    the CfC input projections to {-1,0,+1} costs <=1% perplexity vs
    the fp16 baseline after a short fine-tune.

    Setup: identical init seed for both models; identical data; same
    optimizer; same LR; same batch order. The only difference is
    whether the CfC input projections are TernaryLinear or nn.Linear.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    n_steps = 100
    bs = 2
    seq = 8
    vocab = 64
    lr = 5e-3

    # Identical init seed for both models -- the fp Parameter weight
    # of TernaryLinear uses kaiming_uniform_ with the same a=sqrt(5)
    # default that nn.Linear uses, so seed parity gives identical
    # underlying fp weights at step 0. The ternary path then
    # discretises those same weights.
    torch.manual_seed(7)
    model_fp = _build_tiny_model(weight_quant="none", seed=7)
    torch.manual_seed(7)
    model_tern = _build_tiny_model(weight_quant="ternary", seed=7)

    opt_fp = torch.optim.SGD(model_fp.parameters(), lr=lr)
    opt_tern = torch.optim.SGD(model_tern.parameters(), lr=lr)

    # Pre-generate the data stream so both models see the same batches
    # in the same order. Using a fixed seed for reproducibility.
    g = torch.Generator().manual_seed(31)
    batches = []
    for _ in range(n_steps):
        x = torch.randint(0, vocab, (bs, seq), generator=g, dtype=torch.long)
        y = torch.randint(0, vocab, (bs, seq), generator=g, dtype=torch.long)
        batches.append((x, y))

    def _step(model, opt, x, y):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        opt.step()
        return float(loss.item())

    model_fp.train()
    model_tern.train()
    final_fp = None
    final_tern = None
    # Average the last 5 step losses to reduce single-step noise.
    last_fp_window = []
    last_tern_window = []
    for i, (x, y) in enumerate(batches):
        l_fp = _step(model_fp, opt_fp, x, y)
        l_tern = _step(model_tern, opt_tern, x, y)
        if i >= n_steps - 5:
            last_fp_window.append(l_fp)
            last_tern_window.append(l_tern)
        final_fp = l_fp
        final_tern = l_tern

    avg_fp = sum(last_fp_window) / max(len(last_fp_window), 1)
    avg_tern = sum(last_tern_window) / max(len(last_tern_window), 1)
    rel_diff = abs(avg_tern - avg_fp) / max(abs(avg_fp), 1e-6)

    # Print measurements so the run log captures the actual gap (this
    # is what the report-back will quote -- per the honesty rules, we
    # do NOT silently relax the threshold below).
    msg = (
        f"\n[T2.8 100-step parity]"
        f"\n  fp final loss   (last={final_fp:.4f}, avg5={avg_fp:.4f})"
        f"\n  ternary loss    (last={final_tern:.4f}, avg5={avg_tern:.4f})"
        f"\n  rel_diff        {rel_diff*100:.3f}% (threshold 1.000%)"
    )
    print(msg)

    assert rel_diff < 0.01, (
        f"ternary CfC final loss must be within 1% of fp baseline "
        f"after 100 steps. Got rel_diff={rel_diff*100:.3f}% "
        f"(fp avg5={avg_fp:.4f}, tern avg5={avg_tern:.4f})."
        f" If this regression persists, do NOT relax the threshold; "
        f"investigate (BitNet bar: <=1% by construction)."
    )
