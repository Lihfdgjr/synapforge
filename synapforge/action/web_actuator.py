"""sf.action.web_actuator — DOM-only neural Computer-Use MVP (P18).

Resolves docs/MASTER_PLAN.md §6 P18 + §12.

User directive: *"让ai使用神经元直接操控computer上网自动学习"*.

This is the **MVP** scope: DOM-only, no vision pipeline, Playwright headless,
no login flows / no CAPTCHA / no multi-tab.

Pipeline:

    Playwright Page
          │  page.accessibility.snapshot()
          ▼
    encode_dom()  --> torch.Tensor of shape (action_dim,)
          │
          ▼
    ActionHead(hidden) --> action logits  (action_dim ≥ 5 + nav slots)
          │  argmax over the first 5 slots:
          │      0 = noop  | 1 = click  | 2 = scroll
          │      3 = type  | 4 = navigate
          ▼
    Playwright Page method dispatched directly (no JSON tool tokens).

The WebActuator is intentionally tiny. It exists so that:

- `python -c "from synapforge.action.web_actuator import WebActuator"`
  works on a torch-installed but **Playwright-less** machine.
- `scripts/web_actuator_smoke.sh` can run 50 random ActionHead steps
  against a static local HTML page and assert ≥ 1 successful click.
- A unit test (``tests/integration/test_web_actuator.py``) stubs
  Playwright via ``unittest.mock.MagicMock`` and verifies the dispatch
  table without ever booting a browser.

Anything more ambitious (multi-tab, login flows, vision-based DOM, etc.)
is explicitly OUT of MVP scope (MASTER_PLAN.md §12).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# torch is a hard dep of synapforge; still wrap so a future torch-less
# tooling box can `import` this file for static analysis.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard dep
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Optional dependency — Playwright.  Don't fail the whole module if missing.
# Matches the pattern used by sf.action.web_env.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from playwright.sync_api import Page  # noqa: F401

try:
    from playwright.sync_api import Page as _PlaywrightPage  # type: ignore
    _HAS_PLAYWRIGHT = True
except Exception:  # pragma: no cover - import-only branch
    _PlaywrightPage = None  # type: ignore[assignment]
    _HAS_PLAYWRIGHT = False


# ---------------------------------------------------------------------------
# Action vocabulary — keep tiny, MASTER_PLAN.md §12 says
#   {click(x,y), scroll(dy), type(text), navigate(url), noop}.
# ---------------------------------------------------------------------------

ACTION_NOOP = 0
ACTION_CLICK = 1
ACTION_SCROLL = 2
ACTION_TYPE = 3
ACTION_NAVIGATE = 4
NUM_ACTION_TYPES = 5

ACTION_NAMES: tuple[str, ...] = ("noop", "click", "scroll", "type", "navigate")

# Tiny URL/text codebooks. Keeps action_dim bounded while still letting the
# neural head pick a parameterised action without emitting tokens.
DEFAULT_URL_CODEBOOK: tuple[str, ...] = (
    "about:blank",
    "https://www.bing.com",
    "https://en.wikipedia.org/wiki/Neural_network",
    "https://arxiv.org/abs/2006.04439",
)

DEFAULT_TYPE_CODEBOOK: tuple[str, ...] = (
    "neural network",
    "liquid neural network",
    "spiking neuron",
    "STDP plasticity",
)


# ---------------------------------------------------------------------------
# DOM feature extraction
# ---------------------------------------------------------------------------


_INTERESTING_ROLES: tuple[str, ...] = (
    "button",
    "link",
    "textbox",
    "checkbox",
    "combobox",
    "menuitem",
    "tab",
    "searchbox",
)


def _walk_accessibility(node: Any | None) -> list[dict[str, Any]]:
    """Flatten a Playwright accessibility tree into a list of dicts."""
    if not node:
        return []
    out: list[dict[str, Any]] = [node]
    for child in node.get("children", []) or []:
        out.extend(_walk_accessibility(child))
    return out


def _hash_repr(s: str) -> str:
    return hashlib.blake2s(s.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# WebActuator
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """One step return from WebActuator.step()."""
    action: str
    result: str
    new_dom_hash: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "result": self.result,
            "new_dom_hash": self.new_dom_hash,
            **self.detail,
        }


class WebActuator:
    """DOM-only neural Computer-Use MVP.

    Args:
        page: a Playwright sync ``Page`` (or any duck-typed mock with the
            same methods). Required at construction.
        action_head: a ``torch.nn.Module`` whose forward maps a 1-D tensor
            of shape ``(action_dim,)`` to a 1-D tensor of length at least
            ``NUM_ACTION_TYPES``.  In tests this is just an ``nn.Linear``.
        action_dim: feature dimensionality used for ``encode_dom`` *and*
            the input to ``action_head``.  Keep ≥ 16 so the
            ``url_idx`` / ``type_idx`` slots have room.
        url_codebook: candidate URLs the neural head can navigate to.
        type_codebook: candidate strings the neural head can type.
    """

    def __init__(
        self,
        page: Any,
        action_head: Any,
        action_dim: int = 64,
        url_codebook: tuple[str, ...] = DEFAULT_URL_CODEBOOK,
        type_codebook: tuple[str, ...] = DEFAULT_TYPE_CODEBOOK,
    ) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("WebActuator requires torch")
        if action_dim < 16:
            raise ValueError(f"action_dim must be >= 16 for codebook slots, got {action_dim}")
        self.page = page
        self.action_head = action_head
        self.action_dim = int(action_dim)
        self.url_codebook = tuple(url_codebook)
        self.type_codebook = tuple(type_codebook)
        self._last_dom_hash: str = ""

    # ---------------------------------------------------------------- DOM
    def encode_dom(self) -> "torch.Tensor":
        """Walk page.accessibility.snapshot() and emit a fixed-shape vector.

        The first 8 slots are interpretable, the rest are zero-padded so
        callers can swap in a real backbone later. Fingerprint hash is
        cached on the actuator so ``step()`` can detect "did the page
        change?".
        """
        try:
            snap = self.page.accessibility.snapshot()
        except Exception:
            snap = None
        nodes = _walk_accessibility(snap)
        n_total = len(nodes)
        n_clickable = sum(1 for n in nodes if (n.get("role") or "") in {"button", "link"})
        n_input = sum(1 for n in nodes if (n.get("role") or "") in {"textbox", "searchbox", "combobox"})
        n_text = sum(1 for n in nodes if (n.get("role") or "") in {"text", "heading", "paragraph"})
        n_named = sum(1 for n in nodes if n.get("name"))

        try:
            box = self.page.viewport_size or {"width": 1280, "height": 720}
        except Exception:
            box = {"width": 1280, "height": 720}
        w, h = float(box.get("width", 1280)), float(box.get("height", 720))

        # Cache fingerprint for change-detection in step().
        try:
            url = str(self.page.url) if hasattr(self.page, "url") else ""
        except Exception:
            url = ""
        self._last_dom_hash = _hash_repr(f"{url}|{n_total}|{n_clickable}|{n_input}|{n_text}")

        v = torch.zeros(self.action_dim, dtype=torch.float32)
        v[0] = float(min(n_total, 1024)) / 1024.0
        v[1] = float(min(n_clickable, 256)) / 256.0
        v[2] = float(min(n_input, 64)) / 64.0
        v[3] = float(min(n_text, 256)) / 256.0
        v[4] = float(min(n_named, 256)) / 256.0
        v[5] = w / 4096.0
        v[6] = h / 4096.0
        v[7] = float(int(self._last_dom_hash[:4], 16)) / float(0xFFFF)
        return v

    # ---------------------------------------------------------------- step
    def step(self, hidden: "torch.Tensor") -> dict[str, Any]:
        """Run ActionHead, dispatch on argmax, return a step record dict."""
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("WebActuator.step requires torch")
        with torch.no_grad():
            logits = self.action_head(hidden)
            if logits.dim() > 1:
                logits = logits.reshape(-1)
            type_logits = logits[:NUM_ACTION_TYPES]
            action_id = int(type_logits.argmax().item())
            action_id = max(0, min(action_id, NUM_ACTION_TYPES - 1))

            # Decode parameters from remaining logits.
            xy_x = float(torch.sigmoid(logits[NUM_ACTION_TYPES + 0]).item()) if logits.numel() > NUM_ACTION_TYPES + 0 else 0.5
            xy_y = float(torch.sigmoid(logits[NUM_ACTION_TYPES + 1]).item()) if logits.numel() > NUM_ACTION_TYPES + 1 else 0.5
            scroll_dy = float(torch.tanh(logits[NUM_ACTION_TYPES + 2]).item()) if logits.numel() > NUM_ACTION_TYPES + 2 else 0.0

            # The codebook indices live in the tail so they don't clash
            # with the 0..4 type slots even in tiny action_dim runs.
            tail = logits[NUM_ACTION_TYPES + 3:]
            n_url = max(1, len(self.url_codebook))
            n_typ = max(1, len(self.type_codebook))
            if tail.numel() >= n_url + n_typ:
                url_idx = int(tail[:n_url].argmax().item()) % n_url
                typ_idx = int(tail[n_url:n_url + n_typ].argmax().item()) % n_typ
            else:
                url_idx = 0
                typ_idx = 0

        action_name = ACTION_NAMES[action_id]
        detail: dict[str, Any] = {}
        result = "ok"

        try:
            if action_id == ACTION_NOOP:
                detail["reason"] = "noop"
            elif action_id == ACTION_CLICK:
                box = (self.page.viewport_size or {"width": 1280, "height": 720})
                px = int(xy_x * float(box.get("width", 1280)))
                py = int(xy_y * float(box.get("height", 720)))
                self.page.mouse.click(px, py)
                detail["x"] = px
                detail["y"] = py
            elif action_id == ACTION_SCROLL:
                dy = int(scroll_dy * 600)
                self.page.mouse.wheel(0, dy)
                detail["dy"] = dy
            elif action_id == ACTION_TYPE:
                text = self.type_codebook[typ_idx] if self.type_codebook else ""
                self.page.keyboard.type(text)
                detail["text"] = text
            elif action_id == ACTION_NAVIGATE:
                url = self.url_codebook[url_idx] if self.url_codebook else "about:blank"
                self.page.goto(url, wait_until="domcontentloaded")
                detail["url"] = url
        except Exception as e:
            result = "error"
            detail["error"] = f"{type(e).__name__}: {e}"

        # Re-snapshot DOM for change-detection. encode_dom updates the
        # fingerprint as a side effect.
        try:
            self.encode_dom()
        except Exception as e:  # pragma: no cover - observation failure
            detail["observe_error"] = f"{type(e).__name__}: {e}"

        return {
            "action": action_name,
            "result": result,
            "new_dom_hash": self._last_dom_hash,
            **detail,
        }

    # --------------------------------------------------------------- trace
    def trace(self, hidden_seq: "torch.Tensor") -> list[dict[str, Any]]:
        """Run a sequence of hidden vectors and return per-step results.

        ``hidden_seq`` is shape ``(N, action_dim)``. Returns a list of N
        step-result dicts. Mirrors the contract used by trainers that
        unroll an ActionHead in eval mode.
        """
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("WebActuator.trace requires torch")
        if hidden_seq.dim() == 1:
            hidden_seq = hidden_seq.unsqueeze(0)
        results: list[dict[str, Any]] = []
        for i in range(hidden_seq.shape[0]):
            results.append(self.step(hidden_seq[i]))
        return results


__all__ = [
    "WebActuator",
    "StepResult",
    "ACTION_NAMES",
    "ACTION_NOOP",
    "ACTION_CLICK",
    "ACTION_SCROLL",
    "ACTION_TYPE",
    "ACTION_NAVIGATE",
    "NUM_ACTION_TYPES",
    "DEFAULT_URL_CODEBOOK",
    "DEFAULT_TYPE_CODEBOOK",
]
