"""T5.4 (DEEP_MAINT_QUEUE.md) — best-val ckpt symlink/copy tracker.

The trainer evaluates ``val_ppl_holdout`` every ``--eval-every`` steps. T5.4
adds a side-effect after each eval: if this val improves on the running min,
update a single ``best_step_<N>.pt`` link/copy in the run directory pointing
at the matching ``step_<N>.pt`` ckpt that was just saved. Warmstart code
can then resume from ``best_step_*.pt`` without grepping the log.

These five tests required by the queue task spec all exercise the pure-Python
helper ``train_100m_kd._update_best_ckpt`` directly (no torch, no GPU,
no real data), so they run in seconds on CPU CI:

    1. ``test_first_eval_creates_best_link``       — first val sets the floor.
    2. ``test_better_val_replaces_best_link``      — strictly-lower val swaps.
    3. ``test_worse_val_keeps_old_best``           — equal-or-worse is no-op.
    4. ``test_disabled_no_links_created``          — ``enabled=False`` skips.
    5. ``test_windows_fallback_uses_copy``         — ``os_name='nt'`` -> copy.

CPU-only; uses ``pytest.importorskip("torch")`` because ``train_100m_kd``
unconditionally imports torch at module scope.
"""
from __future__ import annotations

import importlib
import os
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


def _make_dummy_step_ckpt(out_dir: Path, step: int, payload: bytes = b"x") -> Path:
    """Drop a non-empty file at ``step_<N:06>.pt`` so the helper can link it."""
    p = out_dir / f"step_{step:06d}.pt"
    p.write_bytes(payload)
    return p


def _list_best_links(out_dir: Path) -> list[str]:
    """Return all ``best_step_*.pt`` filenames currently in ``out_dir``."""
    return sorted(
        f.name
        for f in out_dir.iterdir()
        if f.name.startswith("best_step_") and f.name.endswith(".pt")
    )


# ==========================================================================
# Test 1 -- first eval seeds the floor and creates the link
# ==========================================================================

def test_first_eval_creates_best_link(tmp_path):
    """First call with finite val + finite-or-inf prior should create a link.

    The ckpt file ``step_000500.pt`` is written manually before invocation so
    the helper has a real source to link/copy.
    """
    t = _import_trainer()
    _make_dummy_step_ckpt(tmp_path, 500, payload=b"step500-bytes")

    captured: list[str] = []
    new_best, link = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=500,
        val_ppl=42.5,
        best_val_ppl=float("inf"),
        enabled=True,
        log_fn=captured.append,
    )

    assert new_best == pytest.approx(42.5), \
        f"running min should advance from inf to 42.5; got {new_best}"
    assert link is not None and Path(link).name == "best_step_000500.pt"
    assert _list_best_links(tmp_path) == ["best_step_000500.pt"]
    # Log line must mention 'improved from' with prev=inf rendering.
    assert any(
        "improved from inf" in m and "step 500" in m and "val=42.5" in m.lower()
        for m in captured
    ), f"expected 'improved from inf ... step 500' log line; got {captured!r}"


# ==========================================================================
# Test 2 -- a strictly-better val swaps the link to the new step
# ==========================================================================

def test_better_val_replaces_best_link(tmp_path):
    """Two evals; the second has lower val -> the OLD link must be removed."""
    t = _import_trainer()
    _make_dummy_step_ckpt(tmp_path, 500, payload=b"step500-bytes")
    _make_dummy_step_ckpt(tmp_path, 1000, payload=b"step1000-bytes")

    # First eval seeds best=88.0 -> best_step_000500.pt
    best, _link1 = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=500,
        val_ppl=88.0,
        best_val_ppl=float("inf"),
        enabled=True,
        log_fn=lambda _m: None,
    )
    assert _list_best_links(tmp_path) == ["best_step_000500.pt"]
    assert best == pytest.approx(88.0)

    # Second eval improves to 42.0 at step 1000 -> link must move.
    captured: list[str] = []
    new_best, link2 = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=1000,
        val_ppl=42.0,
        best_val_ppl=best,
        enabled=True,
        log_fn=captured.append,
    )

    assert new_best == pytest.approx(42.0), "running min should drop to 42.0"
    assert link2 is not None and Path(link2).name == "best_step_001000.pt"
    # CRITICAL: exactly ONE best_step_*.pt -- the old one must be removed.
    assert _list_best_links(tmp_path) == ["best_step_001000.pt"], \
        f"old best_step_000500.pt should be deleted; got {_list_best_links(tmp_path)!r}"
    assert any(
        "improved from 88.00" in m and "step 1000" in m for m in captured
    ), f"expected 'improved from 88.00 ... step 1000' log line; got {captured!r}"


# ==========================================================================
# Test 3 -- equal/worse val: the old link & best are kept; idempotent
# ==========================================================================

def test_worse_val_keeps_old_best(tmp_path):
    """Re-running with a worse OR equal val must leave state untouched.

    Idempotency check: if the trainer fires val twice in a row with the same
    number (e.g. retry / restart artifact), we should NOT recreate the link.
    """
    t = _import_trainer()
    _make_dummy_step_ckpt(tmp_path, 500, payload=b"step500-bytes")
    _make_dummy_step_ckpt(tmp_path, 1000, payload=b"step1000-bytes")

    # Seed best=42.0 at step 500.
    best, _ = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=500,
        val_ppl=42.0,
        best_val_ppl=float("inf"),
        enabled=True,
        log_fn=lambda _m: None,
    )
    assert _list_best_links(tmp_path) == ["best_step_000500.pt"]

    # Snapshot the link target so we can prove it didn't change.
    old_link = tmp_path / "best_step_000500.pt"
    snap = old_link.read_bytes() if old_link.is_file() else None

    # Worse val at step 1000 -> no-op.
    captured_worse: list[str] = []
    new_best, link_worse = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=1000,
        val_ppl=99.9,
        best_val_ppl=best,
        enabled=True,
        log_fn=captured_worse.append,
    )
    assert new_best == pytest.approx(42.0), "best must not regress"
    assert link_worse is None
    assert _list_best_links(tmp_path) == ["best_step_000500.pt"]
    assert not any("improved from" in m for m in captured_worse), \
        f"worse val must not log an improvement; got {captured_worse!r}"

    # Equal val at step 1000 -> still no-op (strict-less-than rule).
    captured_eq: list[str] = []
    new_best_eq, link_eq = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=1000,
        val_ppl=42.0,
        best_val_ppl=new_best,
        enabled=True,
        log_fn=captured_eq.append,
    )
    assert new_best_eq == pytest.approx(42.0)
    assert link_eq is None
    assert _list_best_links(tmp_path) == ["best_step_000500.pt"]
    if snap is not None and old_link.is_file():
        assert old_link.read_bytes() == snap, \
            "best_step_000500.pt was rewritten -- helper is not idempotent"


# ==========================================================================
# Test 4 -- disabled flag short-circuits everything
# ==========================================================================

def test_disabled_no_links_created(tmp_path):
    """``enabled=False`` -> no link, no log, return ``best_val_ppl`` unchanged."""
    t = _import_trainer()
    _make_dummy_step_ckpt(tmp_path, 500, payload=b"step500-bytes")

    captured: list[str] = []
    new_best, link = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=500,
        val_ppl=10.0,                # would be a huge improvement
        best_val_ppl=float("inf"),
        enabled=False,
        log_fn=captured.append,
    )

    assert new_best == float("inf"), \
        "with enabled=False the running min must not advance"
    assert link is None
    assert _list_best_links(tmp_path) == [], \
        f"no best_step_*.pt should exist; got {_list_best_links(tmp_path)!r}"
    assert captured == [], f"no log lines should be emitted; got {captured!r}"


# ==========================================================================
# Test 5 -- Windows fallback (os_name='nt') uses file copy, not symlink
# ==========================================================================

def test_windows_fallback_uses_copy(tmp_path):
    """On Windows ``os.name == 'nt'``, fall back to ``shutil.copyfile``.

    Symlinks on Windows require admin / dev-mode; the helper must detect
    that path and copy instead. The injected ``os_name='nt'`` argument
    forces the Windows branch even when the test runs on Linux CI, so this
    one assertion exercises both platforms.

    We verify:
        * the new ``best_step_<N>.pt`` is a *regular file* (not a symlink),
        * its bytes equal the source ckpt's bytes,
        * the log line includes the literal ``(copy)`` method tag so log
          parsers can tell which branch was taken.
    """
    t = _import_trainer()
    src_payload = b"windows-fallback-payload-bytes"
    src = _make_dummy_step_ckpt(tmp_path, 750, payload=src_payload)

    captured: list[str] = []
    new_best, link = t._update_best_ckpt(
        out_dir=str(tmp_path),
        step=750,
        val_ppl=33.3,
        best_val_ppl=float("inf"),
        enabled=True,
        log_fn=captured.append,
        os_name="nt",  # force Windows branch
    )

    assert new_best == pytest.approx(33.3)
    assert link is not None
    link_p = Path(link)
    assert link_p.name == "best_step_000750.pt"
    assert link_p.exists(), "best_step_000750.pt must be on disk"
    # The copy branch must NOT create a symlink -- it must be a regular file.
    assert not link_p.is_symlink(), \
        "Windows branch should fall back to a copy, not a symlink"
    assert link_p.read_bytes() == src_payload == src.read_bytes(), \
        "copied best_step_*.pt bytes must match the source step_*.pt"
    # Log line should make the method explicit so log parsers can tell.
    assert any("(copy)" in m and "best_step_000750.pt" in m for m in captured), \
        f"Windows fallback must log '(copy)' method tag; got {captured!r}"
    # Sanity: the source is NOT mutated.
    assert src.read_bytes() == src_payload
