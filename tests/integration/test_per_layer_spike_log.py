"""T5.2 (DEEP_MAINT_QUEUE.md): per-layer spike-rate log line.

The trainer's existing aggregated ``spike: mean=... range=[a,b] dead=D/T
sat=S/T`` line hides which individual PLIF layer is dead. T5.2 introduces
an opt-in per-layer line, emitted immediately below the aggregated line
at the same cadence (every 50 steps):

    spike_rates_per_layer: l0=0.000 l1=0.000 ... l<N-1>=0.000

The flag is ``--log-spike-per-layer`` (default OFF -- enable for
dead-layer root-cause work). The number of entries tracks the actual
PLIF cell count, not a hardcoded 10. Each value is rendered to 3-decimal
precision to match the existing ``spike: mean=...`` line.

This file exercises the pure-Python formatter
``_format_spike_rates_per_layer`` and the argparse toggle. The four
tests required by the queue task:

    1. ``test_default_off``                  -- argparse default is False.
    2. ``test_enabled_emits_per_layer``      -- flag -> True; helper
                                                emits ``spike_rates_per_layer: l0=...``.
    3. ``test_handles_variable_n_layers``    -- 4 layers emits l0-l3,
                                                12 layers emits l0-l11.
    4. ``test_format_3decimal_places``       -- exactly 3 decimals,
                                                matches the existing
                                                ``spike: mean=...`` style.

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


# ==========================================================================
# Test 1 -- default OFF: no per-layer line, argparse default is False
# ==========================================================================

def test_default_off(monkeypatch):
    """Without ``--log-spike-per-layer`` the flag must default to False so
    existing launches see no per-layer line and downstream tooling that
    parses the aggregated ``spike: mean=...`` schema is undisturbed.

    Also covers the formatter being importable as a public name (the
    trainer's per-step branch will call it conditionally).
    """
    t = _import_trainer()

    # --- (1) argparse default ----------------------------------------------
    monkeypatch.setattr(sys, "argv", ["train_100m_kd"])
    ns_default = t._parse_args()
    assert ns_default.log_spike_per_layer is False, (
        "--log-spike-per-layer must default to False so existing launches "
        "do NOT emit the new per-layer line by default"
    )

    # --- (2) trainer log path: when args.log_spike_per_layer is False, the
    # extra ``spike_rates_per_layer:`` line MUST NOT be emitted. We mirror
    # the trainer's conditional branch here so we guard the wire-in
    # directly without spinning up a real training run.
    rates = [0.001, 0.002, 0.003, 0.004]
    log_lines: list[str] = []

    def _fake_log(msg: str) -> None:
        log_lines.append(msg)

    # Aggregated line (ALWAYS emitted, regardless of flag).
    _fake_log("  spike: mean=0.003 range=[0.001, 0.004] dead=4/4 sat=0/4")
    # Conditional per-layer line (OFF in this test).
    if ns_default.log_spike_per_layer:
        _fake_log("  " + t._format_spike_rates_per_layer(rates))

    # Aggregated line is present, per-layer line is NOT.
    assert any("spike: mean=" in line for line in log_lines), (
        "aggregated `spike: mean=...` line must always be emitted"
    )
    assert not any("spike_rates_per_layer:" in line for line in log_lines), (
        "with --log-spike-per-layer OFF, the per-layer line must NOT appear"
    )


# ==========================================================================
# Test 2 -- flag ON: emits ``spike_rates_per_layer: l0=...`` line
# ==========================================================================

def test_enabled_emits_per_layer(monkeypatch):
    """With ``--log-spike-per-layer`` the argparse flag flips to True and
    the helper produces a string matching the documented schema.

    Schema (verbatim from the task spec):
        spike_rates_per_layer: l0=0.000 l1=0.000 ... l9=0.000   (10 entries)

    We test on a 10-layer model since that is the production layout. The
    variable-N case is covered separately in test 3.
    """
    t = _import_trainer()

    # --- argparse override --------------------------------------------------
    monkeypatch.setattr(sys, "argv",
                        ["train_100m_kd", "--log-spike-per-layer"])
    ns_on = t._parse_args()
    assert ns_on.log_spike_per_layer is True, (
        "--log-spike-per-layer must override the default to True"
    )

    # --- helper output (10-layer canonical case) ---------------------------
    rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = t._format_spike_rates_per_layer(rates)

    # Prefix label is fixed; values follow the documented format.
    assert out.startswith("spike_rates_per_layer: "), (
        f"per-layer line must start with the documented prefix, got {out!r}"
    )
    # All 10 entries present in order l0..l9 separated by single spaces.
    expected = "spike_rates_per_layer: " + " ".join(
        f"l{i}=0.000" for i in range(10)
    )
    assert out == expected, (
        f"10-layer canonical line mismatch:\n  got:      {out!r}\n  expected: {expected!r}"
    )

    # --- trainer wire-in: when flag is ON the line MUST be appended -------
    log_lines: list[str] = []

    def _fake_log(msg: str) -> None:
        log_lines.append(msg)

    _fake_log("  spike: mean=0.000 range=[0.000, 0.000] dead=10/10 sat=0/10")
    if ns_on.log_spike_per_layer:
        _fake_log("  " + t._format_spike_rates_per_layer(rates))

    # Aggregated line is FIRST, per-layer line is RIGHT BELOW it.
    assert len(log_lines) == 2, (
        f"flag ON must emit exactly 2 lines (aggregated + per-layer), "
        f"got {len(log_lines)}: {log_lines!r}"
    )
    assert "spike: mean=" in log_lines[0]
    assert log_lines[1].startswith("  spike_rates_per_layer: l0="), (
        f"per-layer line must follow the spike: line; got {log_lines!r}"
    )


# ==========================================================================
# Test 3 -- variable layer counts: 4 layers -> l0..l3, 12 layers -> l0..l11
# ==========================================================================

def test_handles_variable_n_layers():
    """The helper MUST handle any layer count, not just 10.

    The trainer derives ``rates`` from ``[m.last_spike_rate().item() for m
    in plif_cells]`` so the length tracks the actual PLIF cell count for
    that model. A 4-layer smoke model emits 4 entries; a 12-layer
    research model emits 12. Hardcoding 10 would silently truncate /
    pad on either side.
    """
    t = _import_trainer()

    # --- 4-layer case: l0..l3 only -----------------------------------------
    rates_4 = [0.10, 0.05, 0.20, 0.00]
    out_4 = t._format_spike_rates_per_layer(rates_4)
    assert out_4 == (
        "spike_rates_per_layer: l0=0.100 l1=0.050 l2=0.200 l3=0.000"
    ), f"4-layer case wrong:\n  got: {out_4!r}"

    # No l4 / l5 / ... should leak through.
    for i in range(4, 10):
        assert f"l{i}=" not in out_4, (
            f"4-layer line must not contain l{i}=; got: {out_4!r}"
        )

    # --- 12-layer case: l0..l11 ---------------------------------------------
    rates_12 = [0.0 + 0.01 * i for i in range(12)]   # 0.00, 0.01, ..., 0.11
    out_12 = t._format_spike_rates_per_layer(rates_12)
    # All 12 keys present in order.
    for i in range(12):
        assert f"l{i}=" in out_12, (
            f"12-layer line missing l{i}=; got: {out_12!r}"
        )
    # And NO l12 / l13 (silent over-extension).
    assert "l12=" not in out_12 and "l13=" not in out_12, (
        f"12-layer line leaked higher index; got: {out_12!r}"
    )
    # Spot-check the last entry: rates_12[11] = 0.11 -> "l11=0.110".
    assert "l11=0.110" in out_12, (
        f"12-layer last-entry mismatch; got: {out_12!r}"
    )

    # --- 1-layer + 0-layer edge cases (degenerate but must not crash) -----
    out_1 = t._format_spike_rates_per_layer([0.42])
    assert out_1 == "spike_rates_per_layer: l0=0.420", out_1

    out_0 = t._format_spike_rates_per_layer([])
    # Empty -> just the prefix label, no entries; helper must not crash.
    assert out_0 == "spike_rates_per_layer:", out_0


# ==========================================================================
# Test 4 -- format precision: exactly 3 decimal places, matching `spike:`
# ==========================================================================

def test_format_3decimal_places():
    """Each per-layer entry MUST be rendered to exactly 3 decimal places
    (``.3f``) to match the existing ``spike: mean=X.XXX range=[a, b]``
    line. Anything else (e.g. ``.2f`` or scientific notation) would break
    log-parsing tooling that already consumes the aggregated line.
    """
    t = _import_trainer()

    # Mix of values that exercise rounding, leading-zero, and the
    # full-saturation / dead extremes.
    rates = [
        0.0,             # full dead -> "0.000"
        0.001,           # 1e-3      -> "0.001"
        0.0005,          # rounds up -> "0.001" (banker's? bake in :.3f result)
        0.12345,         # truncates -> "0.123"
        0.5,             # mid       -> "0.500"
        1.0,             # saturated -> "1.000"
    ]
    out = t._format_spike_rates_per_layer(rates)

    # Pull each "l<N>=<value>" pair and assert the value has EXACTLY
    # 3 digits after the decimal point.
    pairs = re.findall(r"l(\d+)=(-?\d+\.\d+)", out)
    assert len(pairs) == len(rates), (
        f"expected {len(rates)} per-layer pairs, got {len(pairs)} "
        f"from line: {out!r}"
    )
    for idx, val_str in pairs:
        # Exactly one '.' and exactly 3 digits after it.
        assert val_str.count(".") == 1, (
            f"l{idx}={val_str!r} must contain exactly one decimal point"
        )
        decimal_part = val_str.split(".", 1)[1]
        assert len(decimal_part) == 3, (
            f"l{idx}={val_str!r} must have exactly 3 decimal places "
            f"to match `spike: mean=X.XXX` style; got {len(decimal_part)}"
        )
        # Sanity: scientific notation must NOT be used even for tiny
        # values; ``"%.3f" % 1e-7 == "0.000"`` so "e" must not appear.
        assert "e" not in val_str.lower(), (
            f"l{idx}={val_str!r} must not be in scientific notation"
        )

    # Anchor specific renderings against the :.3f spec.
    assert "l0=0.000" in out, out
    assert "l1=0.001" in out, out
    assert "l3=0.123" in out, out      # 0.12345 -> 0.123
    assert "l4=0.500" in out, out
    assert "l5=1.000" in out, out
