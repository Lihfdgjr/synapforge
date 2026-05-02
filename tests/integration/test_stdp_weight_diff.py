"""T8.1 — Tests for ``scripts/measure_stdp_weight_diff.py``.

Verifies the headline paper claim end-to-end:

  1. ``test_smoke_writes_json`` — n=64 smoke run; JSON + PNG written,
     timeline well-formed.
  2. ``test_w_changes_during_inference`` — after 64 forward passes the
     STDP fast-weight buffer ``W`` differs from its initial snapshot.
     This is the gate-removal verification (``stdp_fast.py:127``).
  3. ``test_monotonic_increase_loose`` — for each STDP layer, the
     ``layer_<i>_delta`` series is non-decreasing across measurement
     points (Frobenius norm of ``W - W_initial`` cannot shrink — adding
     LTP/LTD at non-conflicting indices grows it; collisions might keep
     it flat momentarily but never shrink to negative).
  4. ``test_no_grad_path`` — running the inference path under
     ``torch.no_grad()`` must STILL mutate the STDP buffers (Hebbian
     rule needs forward-only state mutation, no autograd dependency).

All tests run on CPU in <10s with the smoke config (n=64 tokens, d=128,
2 layers).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Repo root + scripts/ on path for both ``import scripts.X`` and
# ``import measure_stdp_weight_diff``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _load_script_module():
    """Import ``scripts/measure_stdp_weight_diff.py`` as a module.

    Falls back to spec-based import in case sys.path order picks the
    wrong ``measure_stdp_weight_diff`` first.
    """
    spec = importlib.util.spec_from_file_location(
        "measure_stdp_weight_diff_mod",
        str(_SCRIPTS_DIR / "measure_stdp_weight_diff.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1 — smoke: JSON + PNG written, structure correct
# ---------------------------------------------------------------------------


def test_smoke_writes_json(tmp_path: Path):
    """64-token smoke run produces valid JSON + PNG; timeline well-formed."""
    if not _torch_available():
        pytest.skip("torch not installed")

    sm = _load_script_module()
    out_json = tmp_path / "timeline.json"
    out_png = tmp_path / "delta_W.png"
    result = sm.run(
        ckpt=None,
        n_tokens=64,  # ignored when smoke=True
        out_json=str(out_json),
        out_png=str(out_png),
        seed=11,
        smoke=True,
    )

    assert out_json.exists(), "JSON not written"
    assert out_png.exists(), "PNG not written"
    assert out_json.stat().st_size > 0
    assert out_png.stat().st_size > 0

    # JSON parses + has expected keys
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_tokens"] == 64, "smoke should clamp n_tokens to 64"
    assert payload["n_layers"] >= 1
    assert "timeline" in payload and len(payload["timeline"]) >= 2
    # First entry is baseline 0, last is at n_tokens.
    first = payload["timeline"][0]
    last = payload["timeline"][-1]
    assert first["token_idx"] == 0
    assert last["token_idx"] == payload["n_tokens"]
    # Per-layer keys present everywhere.
    n_layers = payload["n_layers"]
    for row in payload["timeline"]:
        for i in range(n_layers):
            assert f"layer_{i}_delta" in row
    # Result returned mirrors what was saved.
    assert result["n_tokens"] == 64
    assert result["n_layers"] == n_layers


# ---------------------------------------------------------------------------
# Test 2 — W mutates during inference (gate removal verification)
# ---------------------------------------------------------------------------


def test_w_changes_during_inference():
    """After 64 forward passes the STDP buffer W has moved off zero.

    This is THE headline paper claim: forward-only Hebbian, no
    optimizer, no loss. Confirms ``self.training`` gate removal at
    ``synapforge/bio/stdp_fast.py:127`` — STDP fires at
    ``model.eval()`` time.
    """
    if not _torch_available():
        pytest.skip("torch not installed")

    import torch

    sm = _load_script_module()
    model = sm._build_model(ckpt=None)
    stdp_layers = sm._attach_stdp_layers(model)
    assert len(stdp_layers) >= 1

    # Snapshot initial Ws (zero by construction in STDPFastWeight ctor).
    W0 = [layer.W.detach().clone() for layer in stdp_layers]
    for w in W0:
        assert torch.allclose(w, torch.zeros_like(w)), (
            "STDPFastWeight should init W to zeros"
        )

    sm.measure_stdp_weight_diff(
        model=model,
        stdp_layers=stdp_layers,
        n_tokens=64,
        probe_every=16,
        seed=11,
    )

    # At least ONE layer's W must have changed off zero.
    any_changed = False
    for i, (layer, w0) in enumerate(zip(stdp_layers, W0)):
        if not torch.allclose(layer.W, w0):
            any_changed = True
            # Frobenius norm strictly positive.
            norm = float(torch.linalg.norm((layer.W - w0).float()).item())
            assert norm > 0.0, f"layer {i} ||W - W0|| should be > 0, got {norm}"
    assert any_changed, (
        "no STDP layer mutated W during inference — gate removal failed "
        "(check stdp_fast.py:127 still defaults to 'on')"
    )


# ---------------------------------------------------------------------------
# Test 3 — monotonic non-decreasing per-layer delta
# ---------------------------------------------------------------------------


def test_monotonic_increase_loose():
    """Each layer's ‖ΔW‖ at probe points is non-decreasing.

    Loose because at very early steps a layer might not see structured
    input yet (zero update, flat). The strict version (strict-increase)
    is too strong for tiny n_tokens; non-decrease across the run is the
    paper claim and what the headline plot will show.
    """
    if not _torch_available():
        pytest.skip("torch not installed")

    sm = _load_script_module()
    model = sm._build_model(ckpt=None)
    stdp_layers = sm._attach_stdp_layers(model)
    result = sm.measure_stdp_weight_diff(
        model=model,
        stdp_layers=stdp_layers,
        n_tokens=128,
        probe_every=16,
        seed=11,
    )
    timeline = result["timeline"]
    n_layers = result["n_layers"]
    assert len(timeline) >= 3, "need >=3 probe points for monotonicity check"

    for i in range(n_layers):
        prev = -1.0
        for row in timeline:
            v = row[f"layer_{i}_delta"]
            # Non-decreasing within tiny float epsilon.
            assert v >= prev - 1e-6, (
                f"layer {i} delta is not non-decreasing at "
                f"token_idx={row['token_idx']}: {v} < prev {prev}"
            )
            prev = v
    # Bonus: at least ONE layer must show actual growth (>0 final delta).
    final = timeline[-1]
    assert any(
        final[f"layer_{i}_delta"] > 0.0 for i in range(n_layers)
    ), "no layer's ||DW|| moved off zero — paper claim not verified"


# ---------------------------------------------------------------------------
# Test 4 — no_grad path still mutates W (Hebbian rule needs no autograd)
# ---------------------------------------------------------------------------


def test_no_grad_path():
    """Running the entire inference under ``torch.no_grad()`` STILL mutates W.

    The STDP rule mutates buffers in-place inside its own ``no_grad``
    block — it should be unaffected by an outer ``no_grad`` context.
    This is the "forward-only Hebbian, no autograd needed" guarantee.
    """
    if not _torch_available():
        pytest.skip("torch not installed")

    import torch

    sm = _load_script_module()
    model = sm._build_model(ckpt=None)
    stdp_layers = sm._attach_stdp_layers(model)
    assert len(stdp_layers) >= 1

    W0 = [layer.W.detach().clone() for layer in stdp_layers]

    # Wrap the entire generation/measurement in torch.no_grad().
    with torch.no_grad():
        sm.measure_stdp_weight_diff(
            model=model,
            stdp_layers=stdp_layers,
            n_tokens=64,
            probe_every=16,
            seed=11,
        )

    # Same assertion as test_w_changes_during_inference: at least one
    # layer mutated even though the outer scope is no_grad.
    any_changed = False
    for layer, w0 in zip(stdp_layers, W0):
        if not torch.allclose(layer.W, w0):
            any_changed = True
            break
    assert any_changed, (
        "STDP did not mutate W under outer torch.no_grad() — Hebbian rule "
        "should be autograd-independent (only mutates buffers in-place)"
    )
