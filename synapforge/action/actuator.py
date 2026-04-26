"""sf.action.actuator — execute neural action vectors on the host OS.

Closes the loop between the neural ActionHead output and the underlying
operating system (Windows / Mac / Linux).  The actuator consumes the dict
emitted by ActionHead.to_dict() and dispatches to pyautogui (or to a
safe-mode print logger when running headless / in tests).

Two classes:

    OSActuator         from_action_dict(d) -> None
                       Platform branches:
                         Windows: pyautogui (preferred), fallback ctypes
                         Mac:     pyautogui
                         Linux:   pyautogui (X11)
                       safe_mode=True logs and never moves the mouse —
                       essential for rental CI (no display, no input device).

    ScreenObservation  capture(region=None) -> torch.Tensor (3, H, W) in [0,1]
                       Backends:
                         Windows: PIL.ImageGrab
                         Mac/Linux: mss (lightweight) -> PIL fallback

pyautogui may fail to import on a headless rental — we wrap that import
inside a try/except so OSActuator(safe_mode=True) still works without it.
"""
from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Optional dependency — pyautogui.  Don't fail the whole module if missing.
# ---------------------------------------------------------------------------
try:
    import pyautogui  # type: ignore
    _HAS_PYAUTOGUI = True
except Exception:
    pyautogui = None  # type: ignore
    _HAS_PYAUTOGUI = False

try:
    from PIL import Image, ImageGrab  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    ImageGrab = None  # type: ignore
    _HAS_PIL = False

try:
    import mss  # type: ignore
    _HAS_MSS = True
except Exception:
    mss = None  # type: ignore
    _HAS_MSS = False


# ---------------------------------------------------------------------------
# OSActuator
# ---------------------------------------------------------------------------


class OSActuator:
    """Execute action dicts on the host OS.

    Args:
        safe_mode: If True, log actions instead of dispatching them.  The
            module is then importable on a headless rental even when
            pyautogui is missing.  Recommended for tests and remote runs.
        screen_size: Optional (W, H) for unscaling normalised xy.  If None,
            queried from pyautogui at first call.

    The expected action dict shape (matches ActionHead.to_dict):

        {
            "type": str,                        # one of ACTION_TYPES
            "x": float | None,                  # [0,1] normalized
            "y": float | None,
            "scroll_dx": float | None,          # [-1,1]
            "scroll_dy": float | None,
            "key": str | None,                  # from KEY_VOCAB
            "text_trigger": bool,
        }
    """

    def __init__(
        self,
        safe_mode: bool = False,
        screen_size: tuple[int, int] | None = None,
        on_action=None,
    ) -> None:
        self.safe_mode = bool(safe_mode)
        self._on_action = on_action
        self._screen_size = screen_size
        self.platform = platform.system()
        self._pa_disabled_warned = False
        if not self.safe_mode and not _HAS_PYAUTOGUI:
            print(
                "[OSActuator] WARN: pyautogui unavailable; forcing safe_mode=True",
                file=sys.stderr,
            )
            self.safe_mode = True

    # -------------------------------------------------- helpers
    @property
    def screen_size(self) -> tuple[int, int]:
        if self._screen_size is not None:
            return self._screen_size
        if _HAS_PYAUTOGUI:
            try:
                w, h = pyautogui.size()  # type: ignore[union-attr]
                self._screen_size = (int(w), int(h))
                return self._screen_size
            except Exception:
                pass
        # default fallback
        self._screen_size = (1920, 1080)
        return self._screen_size

    def _denorm_xy(self, x: float, y: float) -> tuple[int, int]:
        W, H = self.screen_size
        return int(round(x * W)), int(round(y * H))

    # -------------------------------------------------- main API
    def from_action_dict(self, action: dict) -> None:
        """Execute a single action dict."""
        if self._on_action is not None:
            try:
                self._on_action(action)
            except Exception as e:  # pragma: no cover - user callback
                print(f"[OSActuator] on_action callback raised: {e}", file=sys.stderr)
        if self.safe_mode:
            print(f"[OSActuator safe_mode] {action}")
            return
        a_type = action.get("type", "wait")
        if a_type in {"click", "double_click", "right_click", "drag"}:
            self._do_xy(a_type, action)
        elif a_type == "scroll":
            self._do_scroll(action)
        elif a_type == "key":
            self._do_key(action)
        elif a_type == "type":
            # Text content is generated by the backbone LM head; the
            # actuator only signals readiness via on_action callback.
            pass
        elif a_type == "wait":
            time.sleep(0.05)
        elif a_type == "done":
            pass

    # -------------------------------------------------- per-type dispatchers
    def _do_xy(self, kind: str, action: dict) -> None:
        x = action.get("x", 0.5)
        y = action.get("y", 0.5)
        if x is None or y is None:
            return
        px, py = self._denorm_xy(float(x), float(y))
        if pyautogui is None:
            return
        try:
            if kind == "click":
                pyautogui.click(px, py)
            elif kind == "double_click":
                pyautogui.doubleClick(px, py)
            elif kind == "right_click":
                pyautogui.rightClick(px, py)
            elif kind == "drag":
                pyautogui.moveTo(px, py)
                pyautogui.dragTo(px, py, button="left")
        except Exception as e:
            if not self._pa_disabled_warned:
                print(f"[OSActuator] pyautogui error ({kind}): {e}", file=sys.stderr)
                self._pa_disabled_warned = True

    def _do_scroll(self, action: dict) -> None:
        dx = action.get("scroll_dx") or 0.0
        dy = action.get("scroll_dy") or 0.0
        if pyautogui is None:
            return
        try:
            # pyautogui.scroll is vertical; hscroll is horizontal (Mac/Linux).
            pyautogui.scroll(int(dy * 200))
            if hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(int(dx * 200))
        except Exception as e:
            if not self._pa_disabled_warned:
                print(f"[OSActuator] pyautogui scroll error: {e}", file=sys.stderr)
                self._pa_disabled_warned = True

    def _do_key(self, action: dict) -> None:
        key = action.get("key")
        if not key or pyautogui is None:
            return
        try:
            if "+" in key:  # chord, e.g. ctrl+s
                parts = [p.strip() for p in key.split("+")]
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(key)
        except Exception as e:
            if not self._pa_disabled_warned:
                print(f"[OSActuator] pyautogui key error: {e}", file=sys.stderr)
                self._pa_disabled_warned = True


# ---------------------------------------------------------------------------
# ScreenObservation
# ---------------------------------------------------------------------------


@dataclass
class ScreenObservation:
    """Capture screen pixels into a torch.Tensor.

    Returns shape (3, H, W) in [0, 1] float32.  region=(left, top, right,
    bottom) crops; None captures the whole screen.

    Backends preference: PIL.ImageGrab (Windows native), then mss
    (cross-platform) — matches synapforge's "no required deps" rule.
    """

    region: tuple[int, int, int, int] | None = None

    def capture(self, region: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        region = region or self.region
        if _HAS_PIL and platform.system() == "Windows" and ImageGrab is not None:
            img = ImageGrab.grab(bbox=region)  # type: ignore[arg-type]
            return self._image_to_tensor(img)
        if _HAS_MSS and mss is not None:
            with mss.mss() as sct:
                if region is None:
                    monitor = sct.monitors[0]  # all monitors merged
                    bbox = monitor
                else:
                    l, t, r, b = region
                    bbox = {"left": l, "top": t, "width": r - l, "height": b - t}
                raw = sct.grab(bbox)
                if Image is not None:
                    img = Image.frombytes("RGB", raw.size, raw.rgb)
                    return self._image_to_tensor(img)
                # Fallback: build from raw bytes
                arr = torch.frombuffer(raw.rgb, dtype=torch.uint8).clone()
                arr = arr.view(raw.height, raw.width, 3).permute(2, 0, 1).float() / 255.0
                return arr
        if _HAS_PIL and ImageGrab is not None:
            img = ImageGrab.grab(bbox=region)  # type: ignore[arg-type]
            return self._image_to_tensor(img)
        # No backend available — return a 64x64 zero placeholder to keep
        # the agent shape-stable in safe_mode tests.
        return torch.zeros(3, 64, 64, dtype=torch.float32)

    @staticmethod
    def _image_to_tensor(img) -> torch.Tensor:
        if Image is None:
            raise RuntimeError("PIL not available")
        if img.mode != "RGB":
            img = img.convert("RGB")
        # PIL -> bytes -> torch (avoid numpy dependency).
        b = img.tobytes()
        t = torch.frombuffer(b, dtype=torch.uint8).clone()
        t = t.view(img.height, img.width, 3).permute(2, 0, 1).float() / 255.0
        return t


__all__ = ["OSActuator", "ScreenObservation"]
