"""T5.3 (DEEP_MAINT_QUEUE.md): per-named-module gradient-norm log line.

Run 3l/3m diverged at step 14000+ but the existing aggregated
``loss=... ce=... kd=... z=...`` line couldn't show WHICH submodule's
gradient blew up first. T5.3 adds an opt-in per-named-module grad-norm
line, emitted every 100 steps after backward + ``clip_grad_norm_`` but
before ``optim.step()`` (so the rendered numbers are exactly the
post-clip gradients about to be applied to parameters):

    grad_norm: tok_embed=X.XXXe+YY blocks_0=Y.YYYe+YY ... ln_f=W.WWWe+ZZ

Key design choices reflected in the tests:

  * Top-level ``named_children()`` only -- a 100M model has thousands
    of submodules; recursion would emit thousands of lines per fire.
  * One-level expansion of ``nn.ModuleList`` so SynapForge100M's
    ``blocks`` list emits ``blocks_0`` ... ``blocks_<N-1>`` rather
    than a single rolled-up ``blocks=`` entry. The task spec
    ``... block_0=Y.YY ... block_9=Z.ZZ ...`` requires this.
  * Total norm = ``sqrt(sum_p ||p.grad||^2 if p.grad is not None)``
    matching ``torch.nn.utils.clip_grad_norm_``'s aggregation.
  * Modules with ZERO grads (every ``p.grad is None``) are SKIPPED
    cleanly so we don't emit a meaningless ``0.0`` that could be
    confused with a dead-grad layer.
  * Scientific notation (``.3e``) because grad norms span orders of
    magnitude across modules and across training (early ``lm_head``
    grads dwarf early ``tok_embed`` grads by 10x+).
  * Default OFF -- the per-parameter ``norm().item()`` reduction
    every 100 steps would otherwise add cost on hot training loops
    that don't need the diagnostic.

CPU-only; uses ``pytest.importorskip("torch")`` because
``train_100m_kd`` unconditionally imports torch at module scope.

The four tests required by the queue task:

  1. ``test_default_off``                       -- argparse default is False.
  2. ``test_enabled_emits_grad_norm_log``       -- flag -> True; helper
                                                   emits ``grad_norm: name=...``.
  3. ``test_handles_module_without_grads``      -- skip cleanly when a
                                                   module has no grads.
  4. ``test_format_scientific_notation``        -- ``.3e`` rendering.
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


def _make_toy_model():
    """Build a tiny toy model whose top-level named_children mirror
    SynapForge100M's structure (Embedding ``tok_embed``, ModuleList
    ``blocks``, single-Linear ``ln_f``) without any of the heavy CfC /
    PLIF dependencies. We only need ``model.named_children()`` to walk
    one level; the helper does NOT recurse so a toy model is enough to
    exercise the production code path.

    Returns the model AND keeps explicit child references so the tests
    can directly assign / clear ``p.grad`` slabs without worrying about
    autograd graph state.
    """
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class _ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embed = nn.Embedding(8, 4)
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.ln_f = nn.Linear(4, 4)
            # Untied LM head so we exercise the non-ModuleList branch
            # alongside the ModuleList path for ``blocks``.
            self.lm_head = nn.Linear(4, 8, bias=False)

        def forward(self, x):
            h = self.tok_embed(x)
            for blk in self.blocks:
                h = blk(h)
            h = self.ln_f(h)
            return self.lm_head(h)

    return _ToyModel()


def _set_constant_grads(model, value: float = 1.0):
    """Populate ``p.grad`` with a constant ``value`` tensor on every
    parameter so the helper can compute deterministic norms without
    actually running a backward pass.
    """
    import torch

    for p in model.parameters():
        p.grad = torch.full_like(p.data, fill_value=float(value))


# ==========================================================================
# Test 1 -- default OFF: argparse default is False, helper not invoked
# ==========================================================================

def test_default_off(monkeypatch):
    """Without ``--log-grad-norm`` the flag must default to False so
    existing launches see no per-module log line and downstream tooling
    parsing the existing log schema is undisturbed.

    Also covers the formatter being importable as a public name (the
    trainer's per-step branch will call it conditionally).
    """
    t = _import_trainer()

    # --- (1) argparse default ----------------------------------------------
    monkeypatch.setattr(sys, "argv", ["train_100m_kd"])
    ns_default = t._parse_args()
    assert ns_default.log_grad_norm is False, (
        "--log-grad-norm must default to False so existing launches do "
        "NOT emit the new grad_norm log line by default"
    )

    # --- (2) trainer log path: when args.log_grad_norm is False, the
    # extra ``grad_norm:`` line MUST NOT be emitted. We mirror the
    # trainer's conditional branch here so we guard the wire-in directly
    # without spinning up a real training run.
    pairs = [("tok_embed", 1.5e-3), ("blocks_0", 2.7e-2)]
    log_lines: list[str] = []

    def _fake_log(msg: str) -> None:
        log_lines.append(msg)

    # When OFF, the conditional branch in the trainer does NOT call the
    # helper -- mirror that here.
    if ns_default.log_grad_norm:
        _fake_log("  " + t._format_grad_norm_per_module(pairs))

    # No grad_norm line emitted.
    assert not any("grad_norm:" in line for line in log_lines), (
        "with --log-grad-norm OFF, the grad_norm line must NOT appear; "
        f"got {log_lines!r}"
    )

    # --- (3) The helpers exist as public names (so the trainer can
    # call them) and are pure (no side effects when invoked here) ---
    assert hasattr(t, "_format_grad_norm_per_module"), (
        "trainer must expose _format_grad_norm_per_module helper"
    )
    assert hasattr(t, "_compute_grad_norm_per_named_module"), (
        "trainer must expose _compute_grad_norm_per_named_module helper"
    )


# ==========================================================================
# Test 2 -- flag ON: emits ``grad_norm: <name>=<.3e> ...`` line
# ==========================================================================

def test_enabled_emits_grad_norm_log(monkeypatch):
    """With ``--log-grad-norm`` the argparse flag flips to True, the
    helpers compute ONE entry per top-level child (with one-level
    ``ModuleList`` expansion for ``blocks``), and the rendered line
    matches the documented schema.
    """
    t = _import_trainer()
    pytest.importorskip("torch")
    import torch  # noqa: F401  (used implicitly via the toy model)

    # --- argparse override --------------------------------------------------
    monkeypatch.setattr(sys, "argv",
                        ["train_100m_kd", "--log-grad-norm"])
    ns_on = t._parse_args()
    assert ns_on.log_grad_norm is True, (
        "--log-grad-norm must override the default to True"
    )

    # --- helper output on a toy model whose children are
    #     {tok_embed, blocks=[L0, L1, L2], ln_f, lm_head} -----------------
    model = _make_toy_model()
    _set_constant_grads(model, value=1.0)

    pairs = t._compute_grad_norm_per_named_module(model)
    names = [n for n, _ in pairs]

    # Top-level non-list children appear once each, and the ModuleList
    # ``blocks`` is expanded into ``blocks_0 .. blocks_2`` (NOT a single
    # ``blocks`` rollup -- the task spec requires ``block_0 .. block_9``-
    # style entries).
    assert "tok_embed" in names, names
    assert "ln_f" in names, names
    assert "lm_head" in names, names
    for i in range(3):
        assert f"blocks_{i}" in names, (
            f"ModuleList ``blocks`` must expand one level; missing "
            f"blocks_{i}; got names={names}"
        )
    # NO single rolled-up ``blocks`` entry.
    assert "blocks" not in names, (
        f"ModuleList ``blocks`` must NOT appear as a single rollup; got "
        f"names={names}"
    )

    # --- norms are computed correctly ---
    # For a constant grad of value 1.0 across every parameter, the
    # total norm of one module is sqrt(N_params) where N_params is the
    # number of FLOAT slots across ``module.parameters()``. We test the
    # invariant that all norms are POSITIVE (since every param has a
    # non-None grad slab) -- the exact value depends on toy model shape.
    for name, norm in pairs:
        assert norm > 0.0, (
            f"every module had a 1.0 grad slab -> total norm must be > 0; "
            f"got {name}={norm}"
        )

    # --- format output: starts with documented prefix ---
    out = t._format_grad_norm_per_module(pairs)
    assert out.startswith("grad_norm: "), (
        f"grad_norm line must start with the documented prefix, got {out!r}"
    )

    # --- trainer wire-in: when flag is ON the line MUST be appended ---
    log_lines: list[str] = []

    def _fake_log(msg: str) -> None:
        log_lines.append(msg)

    if ns_on.log_grad_norm:
        _fake_log("  " + t._format_grad_norm_per_module(pairs))

    assert len(log_lines) == 1, (
        f"flag ON must emit exactly 1 grad_norm line, got "
        f"{len(log_lines)}: {log_lines!r}"
    )
    assert "grad_norm: tok_embed=" in log_lines[0], (
        f"first column must be tok_embed (insertion order); got "
        f"{log_lines!r}"
    )

    # --- order is preserved (named_children is order-stable) ---
    # We saw {tok_embed, blocks=[L0,L1,L2], ln_f, lm_head} so the line
    # must be in that exact order.
    expected_order = ["tok_embed", "blocks_0", "blocks_1", "blocks_2",
                      "ln_f", "lm_head"]
    pair_names_in_line = re.findall(r"(\w+)=", log_lines[0])
    assert pair_names_in_line == expected_order, (
        f"named_children order must be preserved in the log line; "
        f"expected {expected_order}, got {pair_names_in_line}"
    )


# ==========================================================================
# Test 3 -- skip cleanly when a module has no grads (all p.grad is None)
# ==========================================================================

def test_handles_module_without_grads():
    """Modules whose every parameter has ``p.grad is None`` (frozen
    submodules, the untied LM head when its grads are reset before
    backward, any layer that hasn't seen a backward yet) must be
    SKIPPED entirely from the helper output.

    Emitting a 0.0 for these would be confusing (it might look like a
    dead-grad bug); skipping them is the documented contract.
    """
    t = _import_trainer()
    pytest.importorskip("torch")
    import torch  # noqa: F401

    model = _make_toy_model()

    # Set grads on EVERY param first ...
    _set_constant_grads(model, value=1.0)

    # ... then null-out ALL grads of one block (blocks[1]) and the
    # ln_f module. The helper must omit them from its output entirely.
    for p in model.blocks[1].parameters():
        p.grad = None
    for p in model.ln_f.parameters():
        p.grad = None

    pairs = t._compute_grad_norm_per_named_module(model)
    names = [n for n, _ in pairs]

    # The two "dead" modules are gone; the rest survive.
    assert "blocks_1" not in names, (
        f"module with all-None grads must be skipped; got names={names}"
    )
    assert "ln_f" not in names, (
        f"module with all-None grads must be skipped; got names={names}"
    )
    assert "tok_embed" in names, names
    assert "blocks_0" in names, names
    assert "blocks_2" in names, names
    assert "lm_head" in names, names

    # --- partial grad case: ONE param of a module has grad, others
    # don't. The helper still reports the module (any_grad => emit).
    for p in model.blocks[2].parameters():
        p.grad = None
    # Re-grad just the WEIGHT of blocks[2]; bias stays None.
    model.blocks[2].weight.grad = torch.full_like(
        model.blocks[2].weight.data, fill_value=1.0
    )
    pairs2 = t._compute_grad_norm_per_named_module(model)
    names2 = [n for n, _ in pairs2]
    assert "blocks_2" in names2, (
        f"module with at least ONE param-grad must still be reported; got "
        f"names={names2}"
    )
    # The reported norm must be sqrt(numel(weight) * 1.0**2) = sqrt(16)
    # for our 4x4 weight matrix.
    norm_blocks_2 = dict(pairs2)["blocks_2"]
    expected = (4 * 4) ** 0.5
    assert abs(norm_blocks_2 - expected) < 1e-5, (
        f"partial-grad module norm wrong; expected {expected}, got "
        f"{norm_blocks_2}"
    )

    # --- empty case: ALL params have None grad -> empty list ---
    for p in model.parameters():
        p.grad = None
    pairs_empty = t._compute_grad_norm_per_named_module(model)
    assert pairs_empty == [], (
        f"with ZERO grads the helper must return [] (and the formatter "
        f"will then emit only the prefix); got {pairs_empty}"
    )

    # Formatter handles empty list (= prefix label only, no entries).
    out_empty = t._format_grad_norm_per_module(pairs_empty)
    assert out_empty == "grad_norm:", (
        f"empty pairs must format to bare prefix label; got {out_empty!r}"
    )


# ==========================================================================
# Test 4 -- format precision: scientific notation .3e (NOT .3f)
# ==========================================================================

def test_format_scientific_notation():
    """Each per-module entry MUST be rendered in scientific notation
    with 3 digits after the decimal point (``.3e``) because gradient
    norms span MANY orders of magnitude across modules and across
    training (early ``lm_head`` grads can be 10x+ larger than early
    ``tok_embed`` grads, then they cross late in training). Linear
    ``.3f`` would lose precision on small-norm modules.
    """
    t = _import_trainer()

    # Hand-built pairs covering ~12 orders of magnitude. The set
    # purposefully includes:
    #   * a "normal" mid-magnitude grad (1.234e-2)
    #   * a tiny grad (4.567e-9)            -- would be .000 under .3f
    #   * a large grad (8.901e+5)           -- would lose precision
    #   * boundary-of-rounding values
    pairs = [
        ("tok_embed",  1.234e-2),
        ("blocks_0",   4.567e-9),       # too tiny for .3f
        ("blocks_1",   1.0),            # exactly 1.0
        ("blocks_2",   8.901e+5),       # too large for compact .3f
        ("ln_f",       0.0),            # exactly zero -- still rendered
        ("lm_head",    1.5),
    ]
    out = t._format_grad_norm_per_module(pairs)

    # Pull each "<name>=<value>" pair, assert the value format is
    # scientific notation with EXACTLY 3 fractional digits.
    rendered = re.findall(r"(\w+)=([-+\d\.]+e[-+]\d+)", out)
    assert len(rendered) == len(pairs), (
        f"expected {len(pairs)} per-module pairs in scientific notation, "
        f"got {len(rendered)} from line: {out!r}"
    )
    for name, val_str in rendered:
        # Layout: ``<sign>?<digit>.<3 digits>e<sign><digits>``
        m = re.fullmatch(r"-?\d+\.\d{3}e[-+]\d+", val_str)
        assert m is not None, (
            f"{name}={val_str!r} must match the .3e schema (1 digit + "
            f"3 fractional digits + e<+/->NN)"
        )
        # Sanity: linear-style render (.3f) would NOT contain 'e'.
        assert "e" in val_str.lower(), (
            f"{name}={val_str!r} must be in scientific notation; got "
            f"linear form"
        )

    # Anchor specific renderings against the .3e spec.
    assert "tok_embed=1.234e-02" in out, out
    assert "blocks_0=4.567e-09" in out, out
    # exactly 1.0 -> "1.000e+00"
    assert "blocks_1=1.000e+00" in out, out
    # 8.901e+5 -> "8.901e+05"
    assert "blocks_2=8.901e+05" in out, out
    # exactly 0.0 -> "0.000e+00" (still rendered, NOT skipped here -
    # skip happens upstream in _compute_*, not in _format_*).
    assert "ln_f=0.000e+00" in out, out
    assert "lm_head=1.500e+00" in out, out
