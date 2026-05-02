"""tests/native/test_native_demo.py -- gates for the no-torch MVP.

Three gates -- all three must pass for the MVP to be accepted:

(a) STATIC: ``synapforge/native_demo.py`` MUST NOT contain
    ``import torch`` or ``from torch ...``.

(b) MONOTONICITY: rolling-10 mean of the loss curve must be lower
    in the second half of the run than in the first half.

(c) TORCH-PARITY: native final loss must be within 5% of the torch
    reference final loss on the same seed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
NATIVE_PY = REPO / "synapforge" / "native_demo.py"
TORCH_REF_PY = REPO / "synapforge" / "native_demo_torch_ref.py"
NATIVE_JSON = REPO / "synapforge" / "_native_demo_results.json"
TORCH_JSON = REPO / "synapforge" / "_native_demo_torch_results.json"


@pytest.fixture(scope="module")
def native_results() -> dict:
    """Run native_demo.py end-to-end and return the JSON dict."""
    if NATIVE_JSON.exists():
        NATIVE_JSON.unlink()
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    proc = subprocess.run(
        [sys.executable, str(NATIVE_PY)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"native_demo.py exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}\n"
    )
    assert NATIVE_JSON.exists(), "native_demo.py did not write JSON"
    return json.loads(NATIVE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def torch_ref_results() -> dict:
    """Run torch reference and return the JSON dict."""
    if TORCH_JSON.exists():
        TORCH_JSON.unlink()
    proc = subprocess.run(
        [sys.executable, str(TORCH_REF_PY)],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"native_demo_torch_ref.py exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}\n"
    )
    return json.loads(TORCH_JSON.read_text(encoding="utf-8"))


def test_a_no_torch_import_in_native_demo() -> None:
    """STATIC gate: 'import torch' MUST NOT appear in native_demo.py."""
    src = NATIVE_PY.read_text(encoding="utf-8")
    bad_lines = []
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if (s.startswith("import torch") or s.startswith("from torch")
                or re.match(r"^\s*import\s+torch(\s|,|$)", line)
                or re.match(r"^\s*from\s+torch(\.|\s)", line)):
            bad_lines.append((i, line))
    assert not bad_lines, (
        "native_demo.py must not import torch.  Found:\n"
        + "\n".join(f"  L{ln}: {txt}" for ln, txt in bad_lines)
    )


def test_b_loss_monotonic(native_results: dict) -> None:
    """MONOTONICITY gate: rolling-10 mean second-half < first-half."""
    assert native_results["loss_decreased_monotonically"] is True, (
        f"Loss curve did not decrease monotonically: "
        f"first_half={native_results['first_half_mean']:.4f} "
        f"second_half={native_results['second_half_mean']:.4f}"
    )
    # Sanity: final loss must beat random init (log(V)=log(256)~=5.545)
    assert native_results["final_loss"] < 5.45, (
        f"final_loss={native_results['final_loss']:.4f} did not beat "
        "random-init baseline log(256)=5.545"
    )


def test_c_torch_parity(native_results: dict, torch_ref_results: dict
                         ) -> None:
    """TORCH-PARITY gate: native final loss within 5% of torch reference.

    Uses a soft tolerance because the native and torch runs aren't
    bit-identical (different RNG paths, different math kernels), but
    on the same seed at the same scale the final loss should land
    within ~5%.
    """
    nat = native_results["final_loss"]
    ref = torch_ref_results["final_loss"]
    rel_diff = abs(nat - ref) / max(1e-6, ref)
    assert rel_diff < 0.05, (
        f"native final loss {nat:.4f} differs from torch ref {ref:.4f} "
        f"by {rel_diff * 100:.2f}% (gate: <5%)"
    )


def test_d_native_self_check_flag(native_results: dict) -> None:
    """The script's own self-check must report no torch import."""
    assert native_results["has_import_torch"] is False, (
        "native_demo.py self-check reported it imports torch -- "
        "the static-grep gate would also fail"
    )


def test_e_speed_budget(native_results: dict) -> None:
    """Per-step wall time should be < 100 ms on local CPU at d=64.

    Soft gate: warns rather than fails on slow CI machines, but the
    median must beat 200 ms.
    """
    p50 = native_results["ms_per_step_p50"]
    assert p50 < 200.0, (
        f"ms_per_step_p50={p50:.1f} exceeded 200 ms soft cap"
    )
