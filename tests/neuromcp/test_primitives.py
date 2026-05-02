"""Tests: 24 primitives + sandbox dispatch."""
from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name: str, relpath: str):
    """Load a single neuromcp module without triggering the full
    ``synapforge.__init__`` chain (which imports torch).

    Tests in this file are intentionally torch-free so they pass on a
    fresh Windows checkout without a CUDA install.
    """
    if name in sys.modules:
        return sys.modules[name]
    full = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load primitives so other modules can find it as a submodule.
_load("synapforge", "synapforge/__init__.py") if False else None
prim = _load(
    "synapforge.neuromcp.primitives",
    "synapforge/neuromcp/primitives.py",
)
sandbox = _load(
    "synapforge.neuromcp.sandbox",
    "synapforge/neuromcp/sandbox.py",
)
oa = _load(
    "synapforge.neuromcp.os_actuator",
    "synapforge/neuromcp/os_actuator.py",
)


def test_24_primitives_present():
    assert prim.NUM_PRIMITIVES == 24
    assert len(prim.PRIMITIVES) == 24
    # ids are 0..23, contiguous
    for i, p in enumerate(prim.PRIMITIVES):
        assert p.id == i, f"primitive {i}: id={p.id}"
    # No duplicate names
    nm = [p.name for p in prim.PRIMITIVES]
    assert len(nm) == len(set(nm)), "duplicate primitive names"


def test_param_slot_resolution():
    # click_at uses x,y
    p0 = prim.by_name("click_at")
    assert p0.param_slots == ("x", "y")
    assert prim.slot_indices(0) == (0, 1)
    # drag uses x,y,x2,y2
    drag = prim.by_name("drag")
    assert drag.param_slots == ("x", "y", "x2", "y2")
    assert prim.slot_indices(drag.id) == (0, 1, 2, 3)
    # type_text -> token_id (slot 4)
    tt = prim.by_name("type_text")
    assert tt.param_slots == ("token_id",)
    assert prim.slot_indices(tt.id) == (4,)


def test_sandbox_guard_subset():
    guard_ids = prim.sandbox_guarded_ids()
    assert 22 in guard_ids and 23 in guard_ids, (
        "file_delete + exec_shell must be sandbox-guarded"
    )
    assert all(prim.by_id(i).sandbox_guard for i in guard_ids)
    # Non-guarded primitives must outnumber guarded ones (safety prior).
    assert len(guard_ids) < prim.NUM_PRIMITIVES


def test_each_primitive_executes_in_sandbox():
    """Run all 24 primitives through the sandbox actuator without raising."""
    actuator = oa.OSActuator(backend="sandbox")
    # Pre-focus a text input so type_text has a target.
    actuator.execute(0, [0.05, 0.27, 0, 0, 0, 0, 0, 0])  # click on input
    failures = []
    for p in prim.PRIMITIVES:
        params = [0.5] * 8
        # press_key uses keysym; pin to a real key index
        if p.name in ("press_key", "key_chord", "key_down", "key_up"):
            params[5] = 0  # 'a'
        # type_text needs a real token_id
        if p.name == "type_text":
            params[4] = 1  # ' '
        result = actuator.execute(p.id, params)
        if not result.success:
            failures.append((p.id, p.name, result.error_msg))
    assert not failures, f"failed primitives: {failures}"


def test_sandbox_guarded_primitives_dont_touch_real_os():
    actuator = oa.OSActuator(
        backend="win32",
        allow_real_os=False,
    )
    # file_delete and exec_shell with allow_real_os=False -> sandbox no-op.
    out = actuator.execute(22, [0, 0, 0, 0, 1, 0, 0, 0])
    assert out.success
    assert "noop" in out.text.lower()
    out2 = actuator.execute(23, [0, 0, 0, 0, 1, 0, 0, 0])
    assert out2.success
    assert "noop" in out2.text.lower()
