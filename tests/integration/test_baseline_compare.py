"""T6.7 — tests for the SmolLM2-360M vs Synap-1 baseline harness.

Validates ``scripts/baseline_smollm2_compare`` end-to-end without actually
downloading the 700MB SmolLM2 weights. Three checks:

* ``test_smollm2_load_mock``      — ``measure_smollm2(mock=True)`` returns a
  shape-correct dict; verifies the FLOPs-per-token formula doesn't NaN.
* ``test_baseline_table_emits``   — full ``main()`` with ``--mock`` writes a
  markdown file containing both rows, ``Synap-1`` and ``SmolLM2-360M``.
* ``test_energy_proxy_calculation`` — sanity: dense FLOPs > sparse FLOPs
  at spike_rate=0.1 (the headline neuromorphic claim).

All tests run in <2 seconds, pure CPU, no HF download.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make repo + scripts importable so ``import baseline_smollm2_compare`` works
# even when pytest is invoked from a different CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _import_module():
    """Import ``baseline_smollm2_compare`` lazily; reload on each test."""
    if "baseline_smollm2_compare" in sys.modules:
        return importlib.reload(sys.modules["baseline_smollm2_compare"])
    return importlib.import_module("baseline_smollm2_compare")


def test_smollm2_load_mock(monkeypatch):
    """``measure_smollm2(mock=True)`` returns full dict, no HF download."""
    mod = _import_module()
    # Defensive: also monkeypatch transformers so even if mock=True ever leaks,
    # we don't accidentally hit the network.
    monkeypatch.setitem(sys.modules, "transformers", None)
    out = mod.measure_smollm2(model_id="HuggingFaceTB/SmolLM2-360M",
                              n_samples=10,
                              seq_len=256,
                              device="cpu",
                              mock=True)
    assert out["mocked"] is True
    assert out["model"] == "HuggingFaceTB/SmolLM2-360M"
    assert out["n_samples"] == 10
    assert out["seq_len"] == 256
    assert isinstance(out["ppl"], float) and out["ppl"] > 0
    assert isinstance(out["tok_per_s"], float) and out["tok_per_s"] > 0
    # FLOPs/token must be a finite positive number ~ 2N.
    fpt = out["flops_per_token"]
    assert fpt > 0 and fpt < 1e15  # SmolLM2-360M @ seq=256 ~ 7e8
    # Sanity: 2N lower bound holds.
    N = mod.SMOLLM2_360M_PUBLISHED["params"]
    assert fpt >= 2.0 * N


def test_baseline_table_emits(tmp_path, monkeypatch):
    """Full ``main(--mock --output X)`` writes a markdown with both rows."""
    mod = _import_module()
    out_path = tmp_path / "BASELINE_COMPARISON_LIVE.md"
    rc = mod.main([
        "--mock",
        "--n-samples", "10",
        "--seq-len", "128",
        "--output", str(out_path),
        # No --synap-ckpt -> Synap-1 leg is mocked TBD.
    ])
    assert rc == 0
    assert out_path.exists(), f"expected markdown at {out_path}"
    body = out_path.read_text(encoding="utf-8")
    # Required sections + rows.
    assert "BASELINE_COMPARISON_LIVE" in body, "missing title"
    assert "Synap-1 (us)" in body, "missing Synap-1 row"
    assert "SmolLM2-360M" in body, "missing SmolLM2 row"
    assert "Energy proxy" in body, "missing energy proxy line"
    # Synap-1 leg should be TBD because no --synap-ckpt was given.
    assert "TBD" in body, "Synap-1 leg should be TBD without --synap-ckpt"
    # Reproduction commands present.
    assert "scripts/baseline_smollm2_compare.py" in body
    assert "ssh root@rental" in body, "missing rental-side reproduction cmd"
    # Datestamp present.
    assert "UTC" in body, "missing datestamp"


def test_energy_proxy_calculation():
    """Dense (SmolLM2) FLOPs > sparse (Synap-1) FLOPs at spike_rate=0.1.

    Sanity check on the headline neuromorphic claim. The claim only holds at
    matched seq_len because the lm_head amortization differs (Synap-1 has
    151k vocab, SmolLM2 has 49k); we assert the ratio is < 1 at spike_rate=0.1.
    """
    mod = _import_module()
    seq_len = 1024
    smol_flops = mod._smollm2_dense_flops_per_token(seq_len=seq_len)
    syn_flops_at_10pct = mod._synap1_sparse_flops_per_token(
        seq_len=seq_len, spike_rate=0.10)
    syn_flops_at_100pct = mod._synap1_sparse_flops_per_token(
        seq_len=seq_len, spike_rate=1.00)

    # Hard sanity: spike rate matters — sparser is cheaper.
    assert syn_flops_at_10pct < syn_flops_at_100pct, \
        "sparser firing must mean fewer FLOPs"
    # Headline claim: at spike_rate=0.10 the backbone is sparse, but the
    # dense lm_head dominates so we don't claim 0.05x — just <1x.
    ratio = syn_flops_at_10pct / smol_flops
    assert 0.0 < ratio < 1.0, \
        f"Synap-1 (10% spikes) should be cheaper than SmolLM2 dense; ratio={ratio:.3f}"


def test_synap1_leg_tbd_when_no_ckpt():
    """``measure_synap1`` with empty ckpt returns TBD-shaped dict, not crash."""
    mod = _import_module()
    out = mod.measure_synap1(
        ckpt_path="",
        tokenizer_path="Qwen/Qwen2.5-0.5B",
        n_samples=10,
        seq_len=256,
        device="cpu",
        mock=True,
    )
    assert out["mocked"] is True
    assert out["ppl"] is None, "TBD ppl when no ckpt"
    assert out["tok_per_s"] is None, "TBD tok/s when no ckpt"
    assert out["params"] == 100_000_000
    assert out["flops_per_token"] > 0  # FLOPs formula independent of measurement
    assert "Awaits rental run" in out["note"]
