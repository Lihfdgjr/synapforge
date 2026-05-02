"""Tests: each backend (sandbox / win32 / mcp_control) dispatches correctly."""
from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    full = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prim = _load("synapforge.neuromcp.primitives", "synapforge/neuromcp/primitives.py")
sandbox = _load("synapforge.neuromcp.sandbox", "synapforge/neuromcp/sandbox.py")
oa = _load("synapforge.neuromcp.os_actuator", "synapforge/neuromcp/os_actuator.py")


def test_sandbox_backend_executes_each_category():
    actuator = oa.OSActuator(backend="sandbox")
    # pointer
    out = actuator.execute(0, [0.5, 0.5] + [0.0] * 6)
    assert out.success and out.primitive_id == 0
    # keyboard
    out = actuator.execute(9, [0, 0, 0, 0, 0, 0, 0, 0])
    assert out.success
    # screen
    out = actuator.execute(16, [0.0] * 8)
    assert out.success and out.primitive_id == 16
    # system (sandbox-guarded -> sandbox no-op)
    out = actuator.execute(22, [0, 0, 0, 0, 1, 0, 0, 0])
    assert out.success


def test_win32_backend_falls_back_to_sandbox_when_disallowed():
    actuator = oa.OSActuator(backend="win32", allow_real_os=False)
    out = actuator.execute(0, [0.5, 0.5] + [0.0] * 6)
    # Even when called as backend=win32, with allow_real_os=False, the
    # dispatch must complete (sandbox-fallback path).
    assert out.success is True


def test_mcp_control_backend_dispatches_named_tools():
    """McpControlBackend's dispatch callable should receive the tool name
    and args we expect for each primitive."""
    captured = []

    def fake_dispatch(tool_name, args):
        captured.append((tool_name, dict(args)))
        return {"ok": True, "text": "fake"}

    actuator = oa.OSActuator(backend="mcp_control", mcp_dispatch=fake_dispatch)
    actuator.execute(0, [0.5, 0.5] + [0.0] * 6)        # click_at
    actuator.execute(7, [0, 0, 0, 0, 0, 0, 0.3, -0.2])  # scroll
    actuator.execute(8, [0, 0, 0, 0, 1, 0, 0, 0])       # type_text
    actuator.execute(16, [0.0] * 8)                     # screenshot

    tool_names = [t for t, _ in captured]
    assert "mcp__mcp-control_click_at" in tool_names
    assert "mcp__mcp-control_scroll_mouse" in tool_names
    assert "mcp__mcp-control_type_text" in tool_names
    assert "mcp__mcp-control_get_screenshot" in tool_names


def test_unknown_primitive_id_returns_failure():
    actuator = oa.OSActuator(backend="sandbox")
    out = actuator.execute(999, [0.0] * 8)
    assert out.success is False
    assert "unknown" in out.error_msg.lower()


def test_real_os_flag_required_for_win32():
    """Without allow_real_os, Win32Backend MUST sandbox-fallback even
    for a non-guarded primitive like click_at."""
    actuator = oa.OSActuator(backend="win32", allow_real_os=False)
    out = actuator.execute(0, [0.5, 0.5] + [0.0] * 6)
    assert out.success is True
    # Sandbox-fallback should be tagged in the text.
    assert "sandbox" in out.text.lower() or out.text == ""


def test_observation_is_serialisable():
    actuator = oa.OSActuator(backend="sandbox")
    out = actuator.execute(16, [0.0] * 8)
    d = out.as_dict()
    assert isinstance(d, dict)
    assert d["primitive_id"] == 16
    assert isinstance(d["params_used"], list)
