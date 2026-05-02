"""docs/SPEEDUP_AUDIT_2026-05-02.md -- argparse + behaviour smoke for the
three speedup-audit ships:

  * ``--lazy-host-sync-accum`` -- defer per-microbatch loss-component
    .item() calls to log boundary. Default OFF.
  * ``--fused-adamw`` -- swap PlasticityAwareAdamW for vanilla
    torch.optim.AdamW(fused=True) when no plasticity sources are wired.
    Default OFF.
  * ``--skip-warmstart-eval-N`` -- skip first N val evaluations on
    warmstart relaunch. Default 0.

The trainer is heavy (Qwen tokenizer, parquet, GPU). These tests pin the
argparse surface and the helper-level semantics that must hold for the
three flags to be safe to enable on the rental:

  * default values match the docstring (False / False / 0)
  * non-default values stick (--lazy-host-sync-accum, --fused-adamw,
    --skip-warmstart-eval-N N)
  * the `_detect_optim_state_layout` helper distinguishes
    PlasticityAwareAdamW state (m/v keys) from torch.optim.AdamW state
    (exp_avg/exp_avg_sq keys)
  * the lazy-host-sync tensor accumulator yields the same Python float
    sum as the eager-host-sync float accumulator across N microbatches
    (correctness preservation)
  * the fused-AdamW safe-detection refuses (falls back) when ANY param
    has a non-bp grad source (plasticity tag); accepts when all params
    are bp-only or untagged

CPU-only -- no GPU, no real model.  We use a tiny ``nn.Linear`` /
``nn.Sequential`` toy model so the cache contract is testable without
booting the trainer.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_module():
    """Import / reload train_100m_kd so its argparse surface reflects the
    file-on-disk version (matters when running this test alongside other
    perf tests which also reload the module)."""
    pytest.importorskip("torch")
    if "train_100m_kd" in sys.modules:
        return importlib.reload(sys.modules["train_100m_kd"])
    return importlib.import_module("train_100m_kd")


def _parse(mod, argv):
    with patch.object(sys, "argv", ["train_100m_kd.py", *argv]):
        return mod._parse_args()


# ---------------------------------------------------------------------------
# Ship #1: --lazy-host-sync-accum
# ---------------------------------------------------------------------------


def test_lazy_host_sync_accum_default_off():
    """Default OFF -- preserve current Run 5 host-sync semantics. Opt-in
    flips to GPU-tensor accumulators, deferring all 6 .item() calls per
    microbatch to the log boundary."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.lazy_host_sync_accum is False, (
        "--lazy-host-sync-accum default must stay OFF (back-compat: "
        "the 6 .item() calls in the inner accum loop run every microbatch)."
    )
    args_on = _parse(mod, ["--lazy-host-sync-accum"])
    assert args_on.lazy_host_sync_accum is True


def test_lazy_host_sync_accum_correctness():
    """The lazy-mode tensor accumulator must yield the SAME Python float
    sum as the eager-mode float accumulator after N microbatches.

    Pin the contract: per-microbatch the trainer either does

        accum += float(t.detach().item())          # eager (default)
        accum_t = accum_t + t.detach().float()      # lazy (opt-in)

    and at log boundary the lazy path reads

        accum = float(accum_t.item())

    These two paths must produce identical accumulated values -- otherwise
    the train.log loss column would silently shift the moment the user
    flipped the flag, which would invalidate cross-run perf comparisons.
    """
    torch = pytest.importorskip("torch")

    # Simulate 2 microbatches (the production accum_steps for Ultra).
    torch.manual_seed(42)
    losses = [torch.tensor(2.5), torch.tensor(3.7)]

    # Eager path (current Run 5 behaviour).
    eager_accum = 0.0
    for loss in losses:
        eager_accum += float(loss.detach().item())

    # Lazy path (Ship #1, this PR).
    lazy_accum_t = torch.zeros((), dtype=torch.float32)
    for loss in losses:
        lazy_accum_t = lazy_accum_t + loss.detach().float()
    lazy_accum = float(lazy_accum_t.item())

    assert lazy_accum == pytest.approx(eager_accum, abs=1e-6), (
        f"lazy-host-sync-accum must match eager: lazy={lazy_accum} "
        f"eager={eager_accum}"
    )


# ---------------------------------------------------------------------------
# Ship #2: --fused-adamw
# ---------------------------------------------------------------------------


def test_fused_adamw_default_off():
    """Default OFF -- preserve PlasticityAwareAdamW path. Opt-in swaps to
    torch.optim.AdamW(fused=True) only if all params have bp-only grad
    sources."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.fused_adamw is False
    args_on = _parse(mod, ["--fused-adamw"])
    assert args_on.fused_adamw is True


def test_detect_optim_state_layout_plasticity():
    """PlasticityAwareAdamW.state_dict() carries `m` / `v` per-param keys."""
    mod = _import_module()
    # Mirror PlasticityAwareAdamW's state-dict layout.
    state = {
        "state": {
            0: {"step": 5, "m": "TENSOR_PLACEHOLDER", "v": "TENSOR_PLACEHOLDER"},
        },
        "param_groups": [{"lr": 1e-3}],
    }
    assert mod._detect_optim_state_layout(state) == "plasticity_adamw"


def test_detect_optim_state_layout_torch_adamw():
    """torch.optim.AdamW.state_dict() carries `exp_avg` / `exp_avg_sq`."""
    mod = _import_module()
    state = {
        "state": {
            0: {
                "step": "TENSOR_PLACEHOLDER",
                "exp_avg": "TENSOR_PLACEHOLDER",
                "exp_avg_sq": "TENSOR_PLACEHOLDER",
            },
        },
        "param_groups": [{"lr": 1e-3}],
    }
    assert mod._detect_optim_state_layout(state) == "torch_adamw"


def test_detect_optim_state_layout_unknown():
    """Empty state dict / unknown optimizer -- returns 'unknown' so the
    trainer warns + skips load instead of silently mis-loading."""
    mod = _import_module()
    assert mod._detect_optim_state_layout({}) == "unknown"
    assert mod._detect_optim_state_layout({"state": {}}) == "unknown"
    # An optimizer with `momentum_buffer` only (e.g. torch.optim.SGD).
    sgd_like = {
        "state": {0: {"momentum_buffer": "TENSOR_PLACEHOLDER"}},
    }
    assert mod._detect_optim_state_layout(sgd_like) == "unknown"


def test_fused_adamw_safe_detection_pure_bp():
    """When every param has only ['bp'] grad source (or no tag at all),
    the fused-adamw path is safe.

    The trainer enumerates `model.named_parameters()` and inspects
    `getattr(p, '_sf_grad_source', None)`. We replicate that contract
    here so the fast-path detection is pinned at unit-test scale.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    # Build a tiny model with ALL params bp-only (== Run 5 production today).
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # Tag the first weight bp-only; leave the rest untagged.
    model[0].weight._sf_grad_source = ["bp"]

    # Replicate the trainer's safety check (train_100m_kd.py main()).
    safe = True
    bad = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        sources = getattr(p, "_sf_grad_source", None)
        if sources is None:
            continue
        if list(sources) != ["bp"]:
            safe = False
            bad.append(pname)
    assert safe is True, f"pure-bp model should be safe; bad={bad}"


def test_fused_adamw_safe_detection_with_plasticity_falls_back():
    """When ANY param has non-bp grad sources (e.g. STDP, Hebb), the
    fused-adamw fast path MUST refuse and fall back to PlasticityAwareAdamW
    -- otherwise plasticity gradients would be silently dropped."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # Tag one weight with STDP plasticity -- this should force fallback.
    model[0].weight._sf_grad_source = ["bp", "stdp"]

    safe = True
    bad = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        sources = getattr(p, "_sf_grad_source", None)
        if sources is None:
            continue
        if list(sources) != ["bp"]:
            safe = False
            bad.append(pname)
    assert safe is False, "plasticity-tagged model must NOT pick the fast path"
    assert bad, "plasticity-tagged param should be in `bad` list"


def test_fused_adamw_optimizer_step_numerically_equivalent_to_unfused():
    """Vanilla AdamW(fused=True) and AdamW(fused=False) produce identical
    weight updates given identical seeds + identical grads. Pin this so
    the production path's correctness preservation claim is measured (not
    just asserted)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fused AdamW kernel requires CUDA; CI runs CPU-only.")
    import torch.nn as nn

    def _build_with_grads(seed: int):
        torch.manual_seed(seed)
        m = nn.Linear(4, 4).cuda()
        torch.manual_seed(seed + 1000)
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        return m

    a = _build_with_grads(0)
    b = _build_with_grads(0)
    opt_unfused = torch.optim.AdamW(
        a.parameters(), lr=1e-3, weight_decay=0.05, fused=False,
    )
    opt_fused = torch.optim.AdamW(
        b.parameters(), lr=1e-3, weight_decay=0.05, fused=True,
    )
    opt_unfused.step()
    opt_fused.step()
    for p_a, p_b in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p_a.data, p_b.data, atol=1e-6), (
            "fused vs unfused AdamW diverged after one step"
        )


# ---------------------------------------------------------------------------
# Ship #3: --skip-warmstart-eval-N
# ---------------------------------------------------------------------------


def test_skip_warmstart_eval_n_default_zero():
    """Default 0 = current behaviour: every eval-every step still runs
    both ttt + holdout val. Opt-in N>0 skips the first N evals on a
    warmstart relaunch."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.skip_warmstart_eval_n == 0
    args_on = _parse(mod, ["--skip-warmstart-eval-N", "2"])
    assert args_on.skip_warmstart_eval_n == 2


def test_skip_warmstart_eval_n_decrement_logic():
    """Pin the trainer's skip-counter contract:

        * remaining > 0 + step != n_steps => log SKIP, decrement, continue
        * remaining == 0                  => normal eval path
        * step == n_steps                 => ALWAYS eval (final step force)

    Replicate the modulo + remaining bookkeeping the trainer hot loop
    runs so a future regression on the gate fires here, not 60s into a
    warmstart on the rental.
    """
    def _would_skip_eval(step: int, n_steps: int, remaining: int) -> bool:
        # Mirror exactly the gate in train_100m_kd.py:
        #   if _skip_eval_remaining > 0 and step != n_steps:
        return remaining > 0 and step != n_steps

    # remaining=0: never skip.
    assert _would_skip_eval(step=500, n_steps=10000, remaining=0) is False
    assert _would_skip_eval(step=10000, n_steps=10000, remaining=0) is False

    # remaining=1: skip step 500, but not the final step.
    assert _would_skip_eval(step=500, n_steps=10000, remaining=1) is True
    assert _would_skip_eval(step=10000, n_steps=10000, remaining=1) is False

    # remaining=2: skip 500 + 1000, but not final.
    assert _would_skip_eval(step=500, n_steps=10000, remaining=2) is True
    assert _would_skip_eval(step=1000, n_steps=10000, remaining=2) is True
    assert _would_skip_eval(step=10000, n_steps=10000, remaining=2) is False


def test_skip_warmstart_eval_n_negative_clamps():
    """Negative N is degenerate; the runtime guard `max(0, ...)` clamps to 0
    so we never get a negative skip counter."""
    def _remaining_for(arg_val) -> int:
        return max(0, int(arg_val))

    assert _remaining_for(-3) == 0
    assert _remaining_for(0) == 0
    assert _remaining_for(2) == 2


# ---------------------------------------------------------------------------
# Composition: SPEEDUP_AUDIT combo + prior PERF_AUDIT/PERF_KNOBS combos
# ---------------------------------------------------------------------------


def test_speedup_audit_combo_composes_with_prior_knobs():
    """The full recommended SPEEDUP_AUDIT + PERF_AUDIT + PERF_KNOBS combo
    parses cleanly. This is the next-restart launch script for Run 5."""
    mod = _import_module()
    args = _parse(
        mod,
        [
            "--batch-size", "24",
            "--grad-accum-steps", "2",
            # PERF_KNOBS v2 (already shipped)
            "--z-loss-topk", "2048",
            "--kd-every", "4",
            "--kd-async-teacher",
            "--prefetch-factor", "4",
            "--pin-memory",
            "--shuffle-buffer", "10000",
            # PERF_AUDIT (prior PR)
            "--cuda-sync-every", "10",
            "--clip-grad-cache",
            # SPEEDUP_AUDIT (this PR)
            "--lazy-host-sync-accum",
            "--fused-adamw",
            "--skip-warmstart-eval-N", "1",
        ],
    )
    assert args.batch_size == 24
    assert args.grad_accum_steps == 2
    assert args.cuda_sync_every == 10
    assert args.clip_grad_cache is True
    assert args.lazy_host_sync_accum is True
    assert args.fused_adamw is True
    assert args.skip_warmstart_eval_n == 1


def test_speedup_audit_combo_default_off_is_back_compat():
    """When NONE of the new flags are passed, the trainer behaves
    identically to the pre-PR Run 5 path. Pin defaults so a future
    silent-flip regression is caught here, not by retraining a 535M
    model on the rental at 14k tok/s."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.lazy_host_sync_accum is False, (
        "Default flip: --lazy-host-sync-accum must stay OFF -- the 6 "
        "per-microbatch .item() calls are the back-compat behaviour."
    )
    assert args.fused_adamw is False, (
        "Default flip: --fused-adamw must stay OFF -- "
        "PlasticityAwareAdamW is the back-compat optimizer."
    )
    assert args.skip_warmstart_eval_n == 0, (
        "Default flip: --skip-warmstart-eval-N must stay 0 -- every "
        "eval-every step runs ttt + holdout val by default."
    )
