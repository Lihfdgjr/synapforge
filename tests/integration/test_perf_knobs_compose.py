"""docs/PERF_KNOBS.md -- argparse compose smoke for the 4 new perf knobs.

This test asserts that the perf-knob CLI surface composes without errors,
so the next trainer restart picks them up cleanly. It does NOT run the
trainer (a real trainer run would need a GPU + parquet data + a
multi-GiB Qwen tokenizer dir).

What we check:
  * BATCH_SIZE module constant bumped to 80.
  * --batch-size default == BATCH_SIZE (no drift between docstring and code).
  * --z-loss-topk default == 2048.
  * --kd-async-teacher default OFF, store_true.
  * --torch-compile default 'off', choices include 'reduce-overhead' /
    'max-autotune'.
  * The full A800-80GB recommended combo composes:
      --batch-size 80 --z-loss-topk 2048 --kd-every 4 --torch-compile off
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
    pytest.importorskip("torch")
    if "train_100m_kd" in sys.modules:
        return importlib.reload(sys.modules["train_100m_kd"])
    return importlib.import_module("train_100m_kd")


def _parse(mod, argv):
    """Run train_100m_kd._parse_args with a synthetic sys.argv."""
    with patch.object(sys, "argv", ["train_100m_kd.py", *argv]):
        return mod._parse_args()


def test_batch_size_default_is_80():
    mod = _import_module()
    assert mod.BATCH_SIZE == 80, (
        f"module constant BATCH_SIZE should be 80; got {mod.BATCH_SIZE}"
    )
    args = _parse(mod, [])
    assert args.batch_size == 80, (
        f"--batch-size default should be 80 (matches BATCH_SIZE); "
        f"got {args.batch_size}"
    )


def test_z_loss_topk_default_is_2048():
    mod = _import_module()
    args = _parse(mod, [])
    assert args.z_loss_topk == 2048, (
        f"--z-loss-topk default should be 2048; got {args.z_loss_topk}"
    )


def test_kd_async_teacher_default_off():
    mod = _import_module()
    args = _parse(mod, [])
    assert args.kd_async_teacher is False
    args_on = _parse(mod, ["--kd-async-teacher"])
    assert args_on.kd_async_teacher is True


def test_torch_compile_default_off_with_choices():
    mod = _import_module()
    args = _parse(mod, [])
    assert args.torch_compile == "off"
    # Valid choices compose
    for c in ("off", "reduce-overhead", "max-autotune"):
        args_c = _parse(mod, ["--torch-compile", c])
        assert args_c.torch_compile == c, f"choice {c!r} did not stick"
    # Invalid choice rejected via SystemExit (argparse error path)
    with pytest.raises(SystemExit):
        _parse(mod, ["--torch-compile", "bogus"])


def test_recommended_a800_combo_composes():
    """The recommended A800-80GB combo composes without argparse error."""
    mod = _import_module()
    args = _parse(
        mod,
        [
            "--batch-size", "80",
            "--z-loss-topk", "2048",
            "--kd-every", "4",
            "--torch-compile", "off",
        ],
    )
    assert args.batch_size == 80
    assert args.z_loss_topk == 2048
    assert args.kd_every == 4
    assert args.torch_compile == "off"
    # The async-teacher knob must compose with the rest.
    args_async = _parse(
        mod,
        [
            "--batch-size", "80",
            "--z-loss-topk", "2048",
            "--kd-async-teacher",
            "--kd-every", "4",
            "--torch-compile", "reduce-overhead",
        ],
    )
    assert args_async.kd_async_teacher is True
    assert args_async.torch_compile == "reduce-overhead"


def test_z_loss_topk_zero_disables_sparse():
    """--z-loss-topk 0 must be accepted (caller falls through to full)."""
    mod = _import_module()
    args = _parse(mod, ["--z-loss-topk", "0"])
    assert args.z_loss_topk == 0


def test_sparse_z_loss_helper_is_module_level():
    """The helper must be importable for the perf doc + downstream tests."""
    mod = _import_module()
    assert callable(mod._sparse_z_loss), (
        "_sparse_z_loss should be exposed at module scope"
    )


# ---------------------------------------------------------------------------
# Perf v2 knobs (2026-05-01): prefetch + pin-memory + adaptive KD-every +
# triton fused backward stub. See ``docs/PERF_KNOBS.md`` v2 section.
# ---------------------------------------------------------------------------


def test_prefetch_factor_default_is_2():
    """``--prefetch-factor`` defaults to 2 (background producer thread on)."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.prefetch_factor == 2, (
        f"--prefetch-factor default should be 2; got {args.prefetch_factor}"
    )
    args_off = _parse(mod, ["--prefetch-factor", "0"])
    assert args_off.prefetch_factor == 0


def test_pin_memory_default_on_with_negation():
    """``--pin-memory`` defaults ON; ``--no-pin-memory`` flips it off."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.pin_memory is True
    args_off = _parse(mod, ["--no-pin-memory"])
    assert args_off.pin_memory is False


def test_kd_every_adaptive_default_off():
    """Adaptive KD-every must be explicit opt-in (default OFF)."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.kd_every_adaptive is False
    args_on = _parse(mod, ["--kd-every-adaptive"])
    assert args_on.kd_every_adaptive is True


def test_kd_teacher_ce_estimate_default_4_5():
    mod = _import_module()
    args = _parse(mod, [])
    assert args.kd_teacher_ce_estimate == 4.5
    args_low = _parse(mod, ["--kd-teacher-ce-estimate", "3.5"])
    assert args_low.kd_teacher_ce_estimate == 3.5


def test_triton_fused_backward_default_off():
    """Speculative knob — flag plumbed but kernel stubbed."""
    mod = _import_module()
    args = _parse(mod, [])
    assert args.triton_fused_backward is False
    args_on = _parse(mod, ["--triton-fused-backward"])
    assert args_on.triton_fused_backward is True


def test_adaptive_kd_every_helper_schedule():
    """``_adaptive_kd_every`` produces the documented schedule for the
    four gap regions. Pin the math so a future regression on the
    schedule surfaces here, not at training time."""
    mod = _import_module()
    fn = mod._adaptive_kd_every
    teacher = 4.5
    base = 4
    # gap > 3.0 -> base // 2 = 2 (early, more KD)
    assert fn(student_ce=8.0, teacher_ce_estimate=teacher, base=base) == 2
    # gap in (1.5, 3.0] -> base = 4
    assert fn(student_ce=6.5, teacher_ce_estimate=teacher, base=base) == 4
    # gap in (0.5, 1.5] -> base * 2 = 8
    assert fn(student_ce=5.5, teacher_ce_estimate=teacher, base=base) == 8
    # gap <= 0.5 -> base * 4 = 16 (KD nearly converged)
    assert fn(student_ce=4.7, teacher_ce_estimate=teacher, base=base) == 16
    # student CE under teacher CE clamps gap to 0 (still in last bucket)
    assert fn(student_ce=2.0, teacher_ce_estimate=teacher, base=base) == 16


def test_recommended_a800_v2_combo_composes():
    """The 2026-05-01 recommended A800-80GB combo composes:

      bs=80 --prefetch-factor 4 --pin-memory --kd-every-adaptive
      --shuffle-buffer 10000
    """
    mod = _import_module()
    args = _parse(
        mod,
        [
            "--batch-size", "80",
            "--prefetch-factor", "4",
            "--pin-memory",
            "--kd-every-adaptive",
            "--shuffle-buffer", "10000",
        ],
    )
    assert args.batch_size == 80
    assert args.prefetch_factor == 4
    assert args.pin_memory is True
    assert args.kd_every_adaptive is True
    assert args.shuffle_buffer == 10000


def test_triton_fused_backward_stub_raises():
    """The fused-backward kernel is stubbed; ``enable_fused_backward``
    raises ``NotImplementedError`` so the trainer falls back gracefully.
    """
    pytest.importorskip("torch")
    from synapforge.backends.triton_fused_backward import (
        enable_fused_backward,
        is_available,
    )
    assert is_available() is False
    with pytest.raises(NotImplementedError):
        enable_fused_backward()
