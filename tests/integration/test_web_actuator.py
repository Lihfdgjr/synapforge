# MIT-licensed — sf.action.web_actuator P18 integration tests.
#
# These tests must run on a torch-installed but Playwright-not-installed
# machine. We stub playwright via `sys.modules["playwright.sync_api"]` so
# the WebActuator import path inside the module never tries to import a
# real Page class.
"""Integration tests for synapforge.action.web_actuator (P18)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Stub playwright before importing the actuator so the lazy import doesn't
# fail on torch-only machines. WebActuator's import block already wraps
# the import in try/except, but stubbing here makes the test setup
# robust even if a future refactor switches to a hard import.
sys.modules.setdefault("playwright", MagicMock())
sys.modules.setdefault("playwright.sync_api", MagicMock())

torch = pytest.importorskip("torch")
torch_nn = pytest.importorskip("torch.nn")

from synapforge.action.web_actuator import (  # noqa: E402
    ACTION_CLICK,
    ACTION_NAMES,
    ACTION_NAVIGATE,
    ACTION_NOOP,
    ACTION_SCROLL,
    ACTION_TYPE,
    NUM_ACTION_TYPES,
    WebActuator,
)


def _mock_page(snapshot: dict | None = None) -> MagicMock:
    page = MagicMock()
    page.url = "https://example.com/"
    page.viewport_size = {"width": 1280, "height": 720}
    page.accessibility = MagicMock()
    page.accessibility.snapshot.return_value = snapshot or {
        "role": "WebArea",
        "name": "fixture",
        "children": [
            {"role": "button", "name": "alpha"},
            {"role": "button", "name": "beta"},
            {"role": "textbox", "name": "search"},
            {"role": "link", "name": "home"},
        ],
    }
    return page


def _hidden(action_dim: int, *, type_id: int) -> torch.Tensor:
    """Build a hidden vector that argmax-selects ``type_id``."""
    h = torch.zeros(action_dim)
    h[type_id] = 10.0  # dominate the type-slot logits
    return h


def test_encode_dom_returns_tensor_with_correct_shape() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    actuator = WebActuator(page, head, action_dim=64)
    v = actuator.encode_dom()
    assert v.shape == (64,), f"expected (64,), got {tuple(v.shape)}"
    assert v.dtype == torch.float32
    # First 8 slots are interpretable; rest is zero-padded.
    assert torch.all(v[8:] == 0)
    # Snapshot was queried.
    page.accessibility.snapshot.assert_called()


def test_step_dispatches_click_when_argmax_is_click() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    # Force action_head output[1] (= ACTION_CLICK) to dominate via identity slice.
    actuator = WebActuator(page, head, action_dim=64)
    h = _hidden(64, type_id=ACTION_CLICK)
    # Use an identity-ish head so logits[1] > all others.
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[ACTION_CLICK, ACTION_CLICK] = 1.0
    rec = actuator.step(h)
    assert rec["action"] == "click"
    assert rec["result"] == "ok"
    page.mouse.click.assert_called_once()


def test_step_dispatches_scroll_when_argmax_is_scroll() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[ACTION_SCROLL, ACTION_SCROLL] = 1.0
    actuator = WebActuator(page, head, action_dim=64)
    rec = actuator.step(_hidden(64, type_id=ACTION_SCROLL))
    assert rec["action"] == "scroll"
    page.mouse.wheel.assert_called_once()


def test_step_dispatches_type_when_argmax_is_type() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[ACTION_TYPE, ACTION_TYPE] = 1.0
    actuator = WebActuator(page, head, action_dim=64)
    rec = actuator.step(_hidden(64, type_id=ACTION_TYPE))
    assert rec["action"] == "type"
    page.keyboard.type.assert_called_once()
    assert "text" in rec


def test_step_dispatches_navigate_when_argmax_is_navigate() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[ACTION_NAVIGATE, ACTION_NAVIGATE] = 1.0
    actuator = WebActuator(page, head, action_dim=64)
    rec = actuator.step(_hidden(64, type_id=ACTION_NAVIGATE))
    assert rec["action"] == "navigate"
    page.goto.assert_called_once()
    assert "url" in rec


def test_step_noop_does_nothing_observable() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[ACTION_NOOP, ACTION_NOOP] = 1.0
    actuator = WebActuator(page, head, action_dim=64)
    rec = actuator.step(_hidden(64, type_id=ACTION_NOOP))
    assert rec["action"] == "noop"
    page.mouse.click.assert_not_called()
    page.mouse.wheel.assert_not_called()
    page.keyboard.type.assert_not_called()
    page.goto.assert_not_called()


def test_trace_accumulates_results_across_sequence() -> None:
    page = _mock_page()
    head = torch_nn.Linear(64, 64)
    actuator = WebActuator(page, head, action_dim=64)
    seq = torch.stack([
        _hidden(64, type_id=ACTION_NOOP),
        _hidden(64, type_id=ACTION_CLICK),
        _hidden(64, type_id=ACTION_SCROLL),
    ])
    # Identity head so that argmax over first 5 logits = type_id.
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        for k in range(NUM_ACTION_TYPES):
            head.weight[k, k] = 1.0
    results = actuator.trace(seq)
    assert len(results) == 3
    assert [r["action"] for r in results] == ["noop", "click", "scroll"]
    # New DOM hash present on every step.
    assert all("new_dom_hash" in r for r in results)
    assert all(isinstance(r["new_dom_hash"], str) for r in results)


def test_action_dim_too_small_raises() -> None:
    page = _mock_page()
    head = torch_nn.Linear(8, 8)
    with pytest.raises(ValueError):
        WebActuator(page, head, action_dim=8)


def test_action_names_match_constants() -> None:
    # Sanity: ACTION_NAMES must agree with the integer constants.
    assert ACTION_NAMES[ACTION_NOOP] == "noop"
    assert ACTION_NAMES[ACTION_CLICK] == "click"
    assert ACTION_NAMES[ACTION_SCROLL] == "scroll"
    assert ACTION_NAMES[ACTION_TYPE] == "type"
    assert ACTION_NAMES[ACTION_NAVIGATE] == "navigate"
    assert len(ACTION_NAMES) == NUM_ACTION_TYPES
