"""sf.action.web_env — real Playwright browser env + mock fallback.

Closes the loop:
    pixel obs (3,64,64) -> backbone -> hidden -> ActionHead -> action dict
    -> WebBrowserEnv.step() -> next obs, reward, done

Reward shaping is *only* novelty + curiosity proxies, no language reward.
The reason: the ActionHead emits a structured action vector directly from
the hidden state — there are NO `<tool_call>` JSON tokens in this loop, so
the reward must work in pixel/page space, not in text space.

Two modes:

  WebBrowserEnv(headless=True, real=True)
      Drives a real headless Playwright Chromium.
      reset(url) -> ScreenObservation (3,64,64) downsampled from page.
      step(action) -> (next_obs, reward, done, info)

  WebBrowserEnv(real=False)
      Mock env over a hardcoded site graph (search-result -> article -> done).
      Same API.  Smoke-runnable without `playwright install chromium`.

Action contract matches sf.action.head.ActionHead.to_dict() exactly:
    {"type": "click"|"scroll"|"key"|"type"|"wait"|"done"|...,
     "x": float|None, "y": float|None,           # normalised [0,1]^2
     "scroll_dx": float|None, "scroll_dy": float|None,  # [-1,1]^2
     "key": str|None,
     "text_trigger": bool}

The TYPE-codebook (10 short queries) lives outside this module — callers
inject `text_for_id(text_id)` so the agent can later grow it via
DynamicActionCodebook without changing this env's surface.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

# -- optional Playwright import; never required for smoke ---------------------
try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PLAYWRIGHT = True
except Exception:  # pragma: no cover - import-only branch
    sync_playwright = None  # type: ignore
    _HAS_PLAYWRIGHT = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False


# Default text codebook — kept tiny so DynamicActionCodebook controls growth.
DEFAULT_TEXT_CODEBOOK: tuple[str, ...] = (
    "neural network",
    "神经形态计算",
    "liquid neural network",
    "spiking neuron",
    "STDP plasticity",
    "transformer attention",
    "computer use agent",
    "browser automation",
    "arxiv liquid time",
    "synapforge",
)


# ---------------------------------------------------------------------------
# Reward primitives — no LM, no JSON.
# ---------------------------------------------------------------------------


def _hash_bytes(b: bytes) -> str:
    return hashlib.blake2s(b, digest_size=8).hexdigest()


def _tensor_hash(t: torch.Tensor) -> str:
    """Page-fingerprint hash for a downsampled obs tensor.

    Quantises to 4-bit per channel to be invariant to JPEG noise.
    """
    q = (t.detach().clamp(0.0, 1.0) * 15.0).to(torch.uint8)
    return _hash_bytes(q.contiguous().cpu().numpy().tobytes())


@dataclass
class StepReward:
    """Accumulator for the four-term shaping signal."""

    page_changed: float = 0.0     # +1.0 first time we see this hash
    page_repeat: float = 0.0      # -0.1 when we revisit a recently-seen one
    progress_text: float = 0.0    # +1.0 when target regex matches new page
    progress_url: float = 0.0     # +0.5 when URL matches a checkpoint
    intrinsic: float = 0.0        # external curiosity / FE / STDP novelty bonus

    def total(self) -> float:
        return float(
            self.page_changed
            + self.page_repeat
            + self.progress_text
            + self.progress_url
            + self.intrinsic
        )


# ---------------------------------------------------------------------------
# Mock site graph — used when real=False.
# ---------------------------------------------------------------------------


_MOCK_GRAPH = {
    "about:blank": {
        "title": "blank",
        "text": "",
        "links": {"https://www.bing.com": (0.5, 0.1)},
    },
    "https://www.bing.com": {
        "title": "bing",
        "text": "Search the web. type query then press enter.",
        "links": {
            "https://www.bing.com/search?q=neural+network": (0.5, 0.5),
            "https://www.bing.com/search?q=neuromorphic": (0.6, 0.5),
        },
    },
    "https://www.bing.com/search?q=neural+network": {
        "title": "results: neural network",
        "text": "neural network — wikipedia. liquid neural networks ramin hasani.",
        "links": {
            "https://en.wikipedia.org/wiki/Neural_network": (0.3, 0.3),
            "https://arxiv.org/abs/2006.04439": (0.3, 0.6),
        },
    },
    "https://en.wikipedia.org/wiki/Neural_network": {
        "title": "Neural network — Wikipedia",
        "text": (
            "A neural network is a network or circuit of biological neurons, or, "
            "in a modern sense, an artificial neural network composed of "
            "artificial neurons or nodes."
        ),
        "links": {"https://www.bing.com": (0.05, 0.05)},
    },
    "https://arxiv.org/abs/2006.04439": {
        "title": "Liquid Time-constant Networks (Hasani et al)",
        "text": (
            "Continuous time recurrent neural networks with non-linear dynamic "
            "synapses. Liquid Time-constant Networks. STDP. Spiking. Plasticity."
        ),
        "links": {"https://www.bing.com": (0.05, 0.05)},
    },
}


def _mock_render(url: str, scroll_y: float, size: int = 64) -> torch.Tensor:
    """Render a deterministic 64x64 RGB tensor for a given (url, scroll)."""
    page = _MOCK_GRAPH.get(url, {"title": "404", "text": "", "links": {}})
    seed = int(_hash_bytes((url + f"@{scroll_y:.2f}").encode()), 16) & 0xFFFFFFFF
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, size, size, generator=g) * 0.2  # noise floor
    # Stripe pattern keyed by title length so different pages look different.
    n = max(1, len(page["title"]) % 8 + 1)
    for k in range(n):
        row = int((k / n) * size)
        img[:, row : row + 2, :] = 0.6 + 0.04 * k
    # Vertical band representing scroll position.
    sy = max(0.0, min(0.99, scroll_y))
    img[1, int(sy * (size - 2)) : int(sy * (size - 2)) + 2, :] = 0.95
    return img


# ---------------------------------------------------------------------------
# WebBrowserEnv
# ---------------------------------------------------------------------------


@dataclass
class WebEnvConfig:
    img_size: int = 64
    high_size: int = 256
    headless: bool = True
    real: bool = True                  # if False -> mock graph
    page_history: int = 64             # last-N hash window for repeat penalty
    target_text_regex: str | None = None
    target_url_substr: str | None = None
    nav_timeout_ms: int = 6000
    text_codebook: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TEXT_CODEBOOK)
    user_agent: str = "Mozilla/5.0 SynapForge/1.0 NeuralAction"


class WebBrowserEnv:
    """Browser environment that returns pixel observations only.

    Lifecycle:
        env = WebBrowserEnv(WebEnvConfig(real=False))
        obs = env.reset("https://www.bing.com")
        for _ in range(32):
            action = head.to_dict(actor(hidden))
            obs, reward, done, info = env.step(action)
            if done: break
        env.close()
    """

    def __init__(self, cfg: WebEnvConfig | None = None) -> None:
        self.cfg = cfg or WebEnvConfig()
        if self.cfg.real and not _HAS_PLAYWRIGHT:
            print(
                "[WebBrowserEnv] WARN: playwright unavailable; falling back to mock env."
            )
            self.cfg = WebEnvConfig(**{**self.cfg.__dict__, "real": False})

        # state
        self._url: str = "about:blank"
        self._scroll: float = 0.0
        self._steps: int = 0
        self._seen_hashes: list[str] = []
        self._step_history: list[dict] = []
        # real-mode handles
        self._pw = None
        self._browser = None
        self._page = None
        if self.cfg.real:
            self._init_real()

    # -------------------------------------------------- real-mode boot
    def _init_real(self) -> None:  # pragma: no cover - requires browser
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.cfg.headless)
        ctx = self._browser.new_context(user_agent=self.cfg.user_agent)
        self._page = ctx.new_page()
        self._page.set_default_timeout(self.cfg.nav_timeout_ms)

    # -------------------------------------------------- public API
    def reset(self, url: str) -> torch.Tensor:
        self._url = url
        self._scroll = 0.0
        self._steps = 0
        self._seen_hashes = []
        self._step_history = []
        if self.cfg.real:  # pragma: no cover - browser path
            self._page.goto(url, wait_until="domcontentloaded")
        obs = self._observe()
        self._seen_hashes.append(_tensor_hash(obs))
        return obs

    def step(self, action: dict) -> tuple[torch.Tensor, float, bool, dict]:
        self._steps += 1
        a_type = str(action.get("type", "wait"))
        info: dict = {"action": a_type}

        # 1. dispatch action
        try:
            self._dispatch(action, info)
        except Exception as e:
            info["dispatch_error"] = str(e)

        # 2. observe
        next_obs = self._observe()
        next_hash = _tensor_hash(next_obs)
        info["page_hash"] = next_hash

        # 3. reward
        rw = StepReward()
        if next_hash not in self._seen_hashes:
            rw.page_changed = 1.0
        else:
            # repeat penalty decays with how recent the repeat is.
            recent_idx = self._seen_hashes[::-1].index(next_hash)
            rw.page_repeat = -max(0.05, 0.5 / (1 + recent_idx))
        # progress
        page_text = self._page_text()
        if self.cfg.target_text_regex:
            import re

            if re.search(self.cfg.target_text_regex, page_text or "", re.IGNORECASE):
                rw.progress_text = 1.0
        if self.cfg.target_url_substr and self.cfg.target_url_substr in self._url:
            rw.progress_url = 0.5

        # 4. bookkeeping
        self._seen_hashes.append(next_hash)
        if len(self._seen_hashes) > self.cfg.page_history:
            self._seen_hashes = self._seen_hashes[-self.cfg.page_history :]
        info["url"] = self._url
        info["scroll"] = self._scroll
        info["reward_breakdown"] = rw.__dict__.copy()

        # 5. done
        done = (a_type == "done") or (
            rw.progress_text > 0 and rw.progress_url > 0
        )

        return next_obs, rw.total(), bool(done), info

    def attach_intrinsic(self, info: dict, bonus: float) -> None:
        """External curiosity hook — caller adds e.g. delta_F, STDP novelty."""
        info["intrinsic"] = float(bonus)

    def close(self) -> None:  # pragma: no cover - manual teardown
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._page = self._browser = self._pw = None

    # -------------------------------------------------- internals
    def _dispatch(self, action: dict, info: dict) -> None:
        a_type = str(action.get("type", "wait"))
        cfg = self.cfg

        if a_type in {"click", "double_click", "right_click"}:
            x = float(action.get("x") or 0.5)
            y = float(action.get("y") or 0.5)
            self._do_click(x, y, a_type, info)
        elif a_type == "scroll":
            dy = float(action.get("scroll_dy") or 0.0)
            self._do_scroll(dy)
        elif a_type == "type":
            tid = int(action.get("text_id") or 0)
            text = self._lookup_text(tid)
            self._do_type(text, info)
        elif a_type == "key":
            self._do_key(action.get("key") or "enter")
        elif a_type == "wait":
            time.sleep(0.02)
        elif a_type == "done":
            pass

    def _lookup_text(self, tid: int) -> str:
        cb = self.cfg.text_codebook
        return cb[tid % len(cb)] if cb else ""

    def _do_click(self, x: float, y: float, kind: str, info: dict) -> None:
        if self.cfg.real:  # pragma: no cover
            box = self._page.viewport_size or {"width": 1280, "height": 720}
            px, py = int(x * box["width"]), int(y * box["height"])
            if kind == "double_click":
                self._page.mouse.dblclick(px, py)
            elif kind == "right_click":
                self._page.mouse.click(px, py, button="right")
            else:
                self._page.mouse.click(px, py)
            time.sleep(0.05)
            self._url = self._page.url
            return
        # mock branch — closest link wins
        links = _MOCK_GRAPH.get(self._url, {}).get("links", {})
        if not links:
            return
        best, dist = None, 1e9
        for url, (lx, ly) in links.items():
            d = (lx - x) ** 2 + (ly - y) ** 2
            if d < dist:
                best, dist = url, d
        if best is not None:
            self._url = best
            self._scroll = 0.0
            info["mock_navigated_to"] = best

    def _do_scroll(self, dy: float) -> None:
        if self.cfg.real:  # pragma: no cover
            self._page.mouse.wheel(0, int(dy * 600))
            time.sleep(0.02)
            return
        self._scroll = max(0.0, min(1.0, self._scroll + dy * 0.25))

    def _do_type(self, text: str, info: dict) -> None:
        info["typed"] = text
        if self.cfg.real:  # pragma: no cover
            try:
                self._page.keyboard.type(text)
            except Exception as e:
                info["type_error"] = str(e)
            return
        # mock: a "type" on a search page becomes a search-result URL.
        if "bing.com" in self._url:
            from urllib.parse import quote_plus

            self._url = f"https://www.bing.com/search?q={quote_plus(text.replace(' ', '+'))}"

    def _do_key(self, key: str) -> None:
        if self.cfg.real:  # pragma: no cover
            try:
                self._page.keyboard.press(key)
            except Exception:
                pass
            return
        if key in {"enter", "return"}:
            # consume queued query (already navigated by _do_type)
            pass
        elif key == "backspace":
            self._scroll = max(0.0, self._scroll - 0.1)

    def _page_text(self) -> str:
        if self.cfg.real:  # pragma: no cover
            try:
                return self._page.inner_text("body")[:2000]
            except Exception:
                return ""
        return _MOCK_GRAPH.get(self._url, {}).get("text", "")

    def _observe(self) -> torch.Tensor:
        size = self.cfg.img_size
        if self.cfg.real:  # pragma: no cover
            return self._observe_real(size)
        return _mock_render(self._url, self._scroll, size=size)

    def _observe_real(self, size: int) -> torch.Tensor:  # pragma: no cover
        png = self._page.screenshot(type="jpeg", quality=70)
        if not _HAS_PIL:
            # Fallback hash of bytes -> deterministic noise tensor.
            h = int(_hash_bytes(png), 16) & 0xFFFFFFFF
            g = torch.Generator().manual_seed(h)
            return torch.rand(3, size, size, generator=g)
        from io import BytesIO

        img = Image.open(BytesIO(png)).convert("RGB").resize((size, size))
        b = img.tobytes()
        t = torch.frombuffer(b, dtype=torch.uint8).clone()
        return t.view(size, size, 3).permute(2, 0, 1).float() / 255.0


__all__ = [
    "WebBrowserEnv",
    "WebEnvConfig",
    "StepReward",
    "DEFAULT_TEXT_CODEBOOK",
]
