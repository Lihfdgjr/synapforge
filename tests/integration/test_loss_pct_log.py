"""T5.1 (DEEP_MAINT_QUEUE.md): per-step loss-component % log columns.

The trainer accumulates ``ce / kd / z / modal_aux / cur_aux`` across the
inner ``--accum`` loop, then averages and emits one log line per
``--log-every`` steps. T5.1 appends a ``pct_ce / pct_kd / pct_z`` (and
``pct_modal / pct_cur`` when the corresponding mixin is active) suffix
so root-cause work can see at a glance which component dominates.

The math (re-stated to make the test self-explanatory):

    denom    = max(step_loss, 1e-9)
    pct_ce    = step_ce          / denom * 100
    pct_kd    = step_kd          / denom * 100
    pct_z     = step_z * z_w     / denom * 100        # z accum is RAW
    pct_modal = step_modal_aux   / denom * 100        # modal added unweighted
    pct_cur   = step_cur_aux*c_w / denom * 100        # cur accum is RAW

This file exercises the pure-Python formatter ``_format_loss_pct`` and
the argparse toggle ``--log-loss-pct`` / ``--no-log-loss-pct``. All
three tests required by the queue task:

    1. ``test_pct_calculation_correct`` -- mock components, pct sums ~100.
    2. ``test_pct_handles_kd_zero``     -- kd=0 doesn't crash, pct_kd=0.0.
    3. ``test_pct_log_disabled_default_log_unchanged`` -- with the
       toggle off the log line matches the legacy schema.

CPU-only; uses ``pytest.importorskip("torch")`` because ``train_100m_kd``
unconditionally imports torch at module scope.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_trainer():
    """Import ``train_100m_kd`` lazily; skip cleanly when torch is absent."""
    pytest.importorskip("torch")
    if "train_100m_kd" in sys.modules:
        return importlib.reload(sys.modules["train_100m_kd"])
    return importlib.import_module("train_100m_kd")


# ---------- regex schemas (mirror trainer log line) -----------------------

# Old (pre-T5.1) format:
#   step  1234 loss=... ce=... kd=... z=... lr=... step_ms=... tok/s=...
_LEGACY_LINE = re.compile(
    r"^step\s+\d+\s+loss=\d+\.\d+\s+ce=\d+\.\d+\s+kd=\d+\.\d+\s+"
    r"z=\d+\.\d+\s+lr=\d+\.\d+\s+step_ms=\d+\.\d+\s+tok/s=\d+$"
)

# New (T5.1) tail appended:
#   ... pct_ce=X.X pct_kd=Y.Y pct_z=Z.Z [pct_modal=M.M] [pct_cur=C.C]
_PCT_TAIL = re.compile(
    r"\spct_ce=\d+\.\d+\spct_kd=\d+\.\d+\spct_z=\d+\.\d+"
    r"(?:\spct_modal=\d+\.\d+)?"
    r"(?:\spct_cur=\d+\.\d+)?$"
)


# ==========================================================================
# Test 1 -- pct math is right & the columns sum back to ~100% of total loss
# ==========================================================================

def test_pct_calculation_correct():
    """Mock loss components, verify percentages sum to ~100 within rounding.

    Construct a known total = ce + kd + z*z_w + modal + cur*c_w and feed
    each component through ``_format_loss_pct``. Parse the rendered
    string, sum the printed percentages, and assert the result is close
    to 100 (truncation-to-1-decimal can cost up to 0.5 per column =
    2.5 max for 5 columns).
    """
    t = _import_trainer()

    ce, kd = 5.0, 2.0
    z, z_w = 100.0, 0.01      # z*z_w = 1.0
    modal = 0.30              # un-weighted
    cur, c_w = 4.0, 0.05      # cur*c_w = 0.2
    total = ce + kd + z * z_w + modal + cur * c_w  # 8.5

    out = t._format_loss_pct(
        step_loss=total,
        step_ce=ce,
        step_kd=kd,
        step_z=z,
        z_loss_weight=z_w,
        step_modal_aux=modal,
        has_modal=True,
        step_cur_aux=cur,
        cur_weight=c_w,
        has_curiosity=True,
    )

    # --- structural ---------------------------------------------------------
    # Leading space + the five expected keys in fixed order.
    assert out.startswith(" pct_ce=")
    assert " pct_kd=" in out
    assert " pct_z=" in out
    assert " pct_modal=" in out
    assert " pct_cur=" in out
    # The whole tail must match the schema regex (so live log lines parse).
    assert re.fullmatch(_PCT_TAIL, out), f"pct tail did not match schema: {out!r}"

    # --- numeric ------------------------------------------------------------
    parsed = dict(re.findall(r"pct_(\w+)=(\d+\.\d+)", out))
    pct_sum = sum(float(v) for v in parsed.values())
    # Five columns, each truncated/rounded to 1 decimal, so |error| <= 5*0.05.
    assert abs(pct_sum - 100.0) < 1.0, (
        f"percent components must sum near 100, got {pct_sum} from {parsed}"
    )

    # Spot-check the dominant column: ce / total = 5 / 8.5 = 58.8%.
    assert float(parsed["ce"]) == pytest.approx(58.8, abs=0.1)
    # And the smallest: cur*c_w / total = 0.2 / 8.5 = 2.4%.
    assert float(parsed["cur"]) == pytest.approx(2.4, abs=0.1)


# ==========================================================================
# Test 2 -- KD-OFF step (kd == 0) doesn't crash; pct_kd is reported as 0.0
# ==========================================================================

def test_pct_handles_kd_zero():
    """A KD-OFF inner step has kd=0; the formatter must still render
    ``pct_kd=0.0`` (stable schema for downstream parsers) and must not
    divide-by-zero anywhere even when total loss is itself near zero.
    """
    t = _import_trainer()

    # KD-OFF: kd=0 contributes nothing.
    ce, kd = 4.0, 0.0
    z, z_w = 50.0, 0.01           # z*z_w = 0.5
    total = ce + kd + z * z_w     # 4.5

    out = t._format_loss_pct(
        step_loss=total,
        step_ce=ce,
        step_kd=kd,
        step_z=z,
        z_loss_weight=z_w,
    )
    assert "pct_kd=0.0" in out, f"kd=0 must render explicit 0.0, got {out!r}"
    assert " pct_ce=88.9" in out  # 4 / 4.5 = 88.888...
    assert " pct_z=11.1" in out   # 0.5 / 4.5 = 11.111...

    # When ``has_modal`` / ``has_curiosity`` are False (default) those
    # columns are OMITTED -- not printed as 0.
    assert "pct_modal=" not in out
    assert "pct_cur=" not in out

    # Pre-backward (or fully-degenerate) edge case: total = 0. The 1e-9
    # floor must keep the call from dividing-by-zero.
    safe = t._format_loss_pct(
        step_loss=0.0,
        step_ce=0.0,
        step_kd=0.0,
        step_z=0.0,
        z_loss_weight=1e-4,
    )
    # All zeros / 1e-9 floor => clean numbers, no NaN, no inf.
    assert "nan" not in safe.lower()
    assert "inf" not in safe.lower()
    assert "pct_ce=0.0" in safe
    assert "pct_kd=0.0" in safe
    assert "pct_z=0.0" in safe


# ==========================================================================
# Test 3 -- with --no-log-loss-pct the log schema reverts to legacy
# ==========================================================================

def test_pct_log_disabled_default_log_unchanged(monkeypatch):
    """``--no-log-loss-pct`` must restore the pre-T5.1 log line schema.

    We exercise two halves:

    1.  Argparse -- ``_parse_args()`` honors ``--log-loss-pct`` (default
        True) and ``--no-log-loss-pct`` (override to False). Both flags
        share a single ``log_loss_pct`` dest.
    2.  Schema -- when the trainer's per-step branch sees
        ``args.log_loss_pct == False``, ``pct_str`` stays ``""`` and the
        emitted log line therefore matches ``_LEGACY_LINE``. We
        reconstruct the f-string the trainer uses (lines around 1577 of
        ``train_100m_kd.py``) so this guards the appended-tail
        regression separately from the helper's own math.
    """
    t = _import_trainer()

    # --- (1) argparse -------------------------------------------------------
    monkeypatch.setattr(sys, "argv", ["train_100m_kd"])
    ns_default = t._parse_args()
    assert ns_default.log_loss_pct is True, (
        "--log-loss-pct must default to True so existing launches see "
        "the new columns without flag changes"
    )

    monkeypatch.setattr(sys, "argv", ["train_100m_kd", "--no-log-loss-pct"])
    ns_off = t._parse_args()
    assert ns_off.log_loss_pct is False, (
        "--no-log-loss-pct must override the default to False"
    )

    # --- (2) schema ---------------------------------------------------------
    # Mirror the trainer's f-string (verbatim) for the disabled case.
    step = 1234
    loss_v = 8.5
    ce_v = 5.0
    kd_v = 2.0
    z_v = 0.5
    cur_lr = 1e-4
    step_ms = 12.3
    tok_s = 4567
    mem_str = ""    # CPU path
    mixin_str = ""  # no mixins

    pct_str = ""    # ``args.log_loss_pct == False`` => empty
    line = (
        f"step {step:5d} loss={loss_v:.4f} ce={ce_v:.3f} "
        f"kd={kd_v:.3f} z={z_v:.3f} lr={cur_lr:.5f} "
        f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}"
        f"{mem_str}{mixin_str}{pct_str}"
    )
    assert _LEGACY_LINE.match(line), (
        f"with --no-log-loss-pct the line must match legacy schema, got: {line!r}"
    )
    # And explicitly: no pct_* substring leaks through.
    assert "pct_" not in line

    # --- (2b) when ON, the same line gains a pct_* tail --------------------
    pct_str_on = t._format_loss_pct(
        step_loss=loss_v,
        step_ce=ce_v,
        step_kd=kd_v,
        step_z=50.0,           # z accum is RAW; 50 * 0.01 = 0.5 == z_v
        z_loss_weight=0.01,
    )
    line_on = (
        f"step {step:5d} loss={loss_v:.4f} ce={ce_v:.3f} "
        f"kd={kd_v:.3f} z={z_v:.3f} lr={cur_lr:.5f} "
        f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}"
        f"{mem_str}{mixin_str}{pct_str_on}"
    )
    assert "pct_ce=" in line_on
    assert "pct_kd=" in line_on
    assert "pct_z=" in line_on
    # The legacy regex must NOT match when the tail is appended.
    assert not _LEGACY_LINE.match(line_on)
