"""synapforge.neuromcp.os_actuator -- Layer-5 OS dispatch.

Closes the loop from neural primitive_id + 8-slot params to a *real*
OS-side effect.  Backends:

    sandbox     :: virtual desktop, no real OS calls (default for training)
    win32       :: pyautogui / win32api on Windows
    mcp_control :: dispatch to mcp__mcp-control_* tool surface

The contract is identical for all three -- the agent emits a
(primitive_id, params) pair and gets back an ObservationDict:

    {
        "success"        : bool,
        "screenshot_bytes": Optional[bytes],   # PNG bytes of the new frame
        "text"           : str,                # any text result (clipboard, etc.)
        "error_msg"      : str,                # empty when success=True
        "primitive_id"   : int,
        "params_used"    : list[float],
        "ts_ms"          : float,              # actuator-side timestamp
    }

torch is **not** imported -- this file is a pure dispatcher that calls
into pyautogui / mss / mcp_control.  The neural side is in
``action_head.py``; here we stay torch-free so a sandbox-only training
run never has to load CUDA.
"""
from __future__ import annotations

import io
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .primitives import PRIMITIVES, by_id, sandbox_guarded_ids


# ---------------------------------------------------------------------------
# ObservationDict -- canonical actuator output type.
# ---------------------------------------------------------------------------


@dataclass
class ObservationDict:
    success: bool
    screenshot_bytes: Optional[bytes] = None
    text: str = ""
    error_msg: str = ""
    primitive_id: int = -1
    params_used: List[float] = field(default_factory=list)
    ts_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "screenshot_bytes": self.screenshot_bytes,
            "text": str(self.text),
            "error_msg": str(self.error_msg),
            "primitive_id": int(self.primitive_id),
            "params_used": list(self.params_used),
            "ts_ms": float(self.ts_ms),
        }


# ---------------------------------------------------------------------------
# Optional deps -- guarded.  All three backends fall back gracefully when
# their preferred lib is missing.
# ---------------------------------------------------------------------------

try:
    import pyautogui  # type: ignore
    _HAS_PYAUTOGUI = True
except Exception:
    pyautogui = None  # type: ignore
    _HAS_PYAUTOGUI = False

try:
    from PIL import Image, ImageDraw, ImageGrab  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageGrab = None  # type: ignore
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Backend interface.
# ---------------------------------------------------------------------------


class _BackendBase:
    """Common interface every actuator backend implements."""

    name: str = "base"

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> ObservationDict:
        raise NotImplementedError

    def move(self, x: int, y: int, dwell_ms: int = 0) -> ObservationDict:
        raise NotImplementedError

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ObservationDict:
        raise NotImplementedError

    def scroll(self, dx: float, dy: float) -> ObservationDict:
        raise NotImplementedError

    def type_text(self, text: str) -> ObservationDict:
        raise NotImplementedError

    def press_key(self, key: str, chord: bool = False, hold: bool = False,
                  release: bool = False) -> ObservationDict:
        raise NotImplementedError

    def screenshot(self) -> ObservationDict:
        raise NotImplementedError

    def wait(self, ms: int) -> ObservationDict:
        raise NotImplementedError

    def window(self, action: str, title_hash: int = 0) -> ObservationDict:
        raise NotImplementedError

    def get_screen_size(self) -> ObservationDict:
        raise NotImplementedError

    def file_delete(self, path: str) -> ObservationDict:
        raise NotImplementedError

    def exec_shell(self, cmd: str) -> ObservationDict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sandbox backend -- virtual desktop, no real OS contact.
# ---------------------------------------------------------------------------


class SandboxBackend(_BackendBase):
    """In-process virtual desktop.  Used for training so a misclick by a
    PLIF-dead model can never touch the real filesystem."""

    name = "sandbox"

    def __init__(self, screen_size: Tuple[int, int] = (1024, 768)) -> None:
        from .sandbox import VirtualDesktop  # local import to avoid cycle
        self.screen_size = screen_size
        self.desktop = VirtualDesktop(screen_size)

    # -- helpers ----------------------------------------------------------
    def _ok(self, primitive_id: int, params: List[float], text: str = "") -> ObservationDict:
        return ObservationDict(
            success=True,
            screenshot_bytes=self.desktop.snapshot_png(),
            text=text,
            error_msg="",
            primitive_id=primitive_id,
            params_used=list(params),
        )

    # -- pointer ----------------------------------------------------------
    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> ObservationDict:
        hit = self.desktop.click(x, y, button=button, count=count)
        return self._ok(0 if button == "left" else (2 if button == "right" else 3),
                        [x, y], text=f"hit={hit}")

    def move(self, x: int, y: int, dwell_ms: int = 0) -> ObservationDict:
        self.desktop.move_cursor(x, y)
        return self._ok(5 if dwell_ms == 0 else 6, [x, y])

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ObservationDict:
        self.desktop.drag(x1, y1, x2, y2)
        return self._ok(4, [x1, y1, x2, y2])

    def scroll(self, dx: float, dy: float) -> ObservationDict:
        self.desktop.scroll(dx, dy)
        return self._ok(7, [dx, dy])

    # -- keyboard ---------------------------------------------------------
    def type_text(self, text: str) -> ObservationDict:
        self.desktop.type_text(text)
        return self._ok(8, [], text=f"typed={text!r}")

    def press_key(self, key: str, chord: bool = False, hold: bool = False,
                  release: bool = False) -> ObservationDict:
        self.desktop.press_key(key, chord=chord, hold=hold, release=release)
        if hold:
            pid = 11
        elif release:
            pid = 12
        elif chord:
            pid = 10
        else:
            pid = 9
        return self._ok(pid, [], text=f"key={key}")

    # -- screen -----------------------------------------------------------
    def screenshot(self) -> ObservationDict:
        return self._ok(16, [])

    def wait(self, ms: int) -> ObservationDict:
        # Don't actually sleep -- training would stall.  Just record.
        self.desktop.tick(ms)
        return self._ok(17, [ms / 1000.0])

    def window(self, action: str, title_hash: int = 0) -> ObservationDict:
        self.desktop.window_action(action, title_hash)
        if action == "focus":
            pid = 18
        elif action == "minimize":
            pid = 19
        else:
            pid = 20
        return self._ok(pid, [title_hash])

    def get_screen_size(self) -> ObservationDict:
        w, h = self.screen_size
        return ObservationDict(
            success=True,
            screenshot_bytes=None,
            text=f"{w}x{h}",
            error_msg="",
            primitive_id=21,
            params_used=[],
        )

    # -- system (sandbox-guarded primitives stay safe) -------------------
    def file_delete(self, path: str) -> ObservationDict:
        # Sandbox: just log.  No real fs touch.
        return ObservationDict(
            success=True, error_msg="", primitive_id=22,
            text=f"sandbox-delete-noop: {path!r}", params_used=[],
        )

    def exec_shell(self, cmd: str) -> ObservationDict:
        return ObservationDict(
            success=True, error_msg="", primitive_id=23,
            text=f"sandbox-exec-noop: {cmd!r}", params_used=[],
        )


# ---------------------------------------------------------------------------
# Win32 backend -- pyautogui-based.
# ---------------------------------------------------------------------------


class Win32Backend(_BackendBase):
    """Real-OS Windows backend via pyautogui.

    Falls back to a SandboxBackend transparently when pyautogui is
    missing OR when the caller passes ``allow_real_os=False`` (default).
    """

    name = "win32"

    def __init__(self, allow_real_os: bool = False) -> None:
        self.allow_real_os = bool(allow_real_os and _HAS_PYAUTOGUI)
        # Always keep a sandbox available so unsupported primitives
        # (e.g. exec_shell) on this backend can still respond meaningfully.
        self._fallback = SandboxBackend()
        if not self.allow_real_os:
            warnings.warn(
                "Win32Backend running in sandbox-fallback mode "
                "(allow_real_os=False or pyautogui missing).",
                RuntimeWarning, stacklevel=2,
            )

    def _safe(self, fn: Callable[[], None], primitive_id: int,
              params: List[float], text: str = "") -> ObservationDict:
        if not self.allow_real_os:
            return self._fallback._ok(primitive_id, params, text="sandbox-fallback")
        try:
            fn()
            return ObservationDict(
                success=True, screenshot_bytes=None, text=text,
                error_msg="", primitive_id=primitive_id,
                params_used=list(params),
            )
        except Exception as exc:
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=primitive_id, params_used=list(params),
            )

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            if count == 2:
                pyautogui.doubleClick(x, y)
            elif button == "right":
                pyautogui.rightClick(x, y)
            elif button == "middle":
                pyautogui.middleClick(x, y)
            else:
                pyautogui.click(x, y)
        pid = 0 if button == "left" and count == 1 else (
            1 if count == 2 else (2 if button == "right" else 3))
        return self._safe(_do, pid, [x, y])

    def move(self, x: int, y: int, dwell_ms: int = 0) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            pyautogui.moveTo(x, y, duration=dwell_ms / 1000.0)
        return self._safe(_do, 5 if dwell_ms == 0 else 6, [x, y])

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, button="left")
        return self._safe(_do, 4, [x1, y1, x2, y2])

    def scroll(self, dx: float, dy: float) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            pyautogui.scroll(int(dy * 200))
            if hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(int(dx * 200))
        return self._safe(_do, 7, [dx, dy])

    def type_text(self, text: str) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            pyautogui.typewrite(text, interval=0.0)
        return self._safe(_do, 8, [], text=text)

    def press_key(self, key: str, chord: bool = False, hold: bool = False,
                  release: bool = False) -> ObservationDict:
        def _do() -> None:
            if pyautogui is None:
                return
            if hold:
                pyautogui.keyDown(key)
            elif release:
                pyautogui.keyUp(key)
            elif chord and "+" in key:
                parts = [p.strip() for p in key.split("+")]
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(key)
        if hold:
            pid = 11
        elif release:
            pid = 12
        elif chord:
            pid = 10
        else:
            pid = 9
        return self._safe(_do, pid, [], text=key)

    def screenshot(self) -> ObservationDict:
        if not self.allow_real_os or not _HAS_PIL or ImageGrab is None:
            return self._fallback.screenshot()
        try:
            img = ImageGrab.grab()  # type: ignore[arg-type]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return ObservationDict(
                success=True, screenshot_bytes=buf.getvalue(),
                primitive_id=16, params_used=[],
            )
        except Exception as exc:
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=16, params_used=[],
            )

    def wait(self, ms: int) -> ObservationDict:
        if not self.allow_real_os:
            return self._fallback.wait(ms)
        try:
            time.sleep(min(1.0, ms / 1000.0))
            return ObservationDict(
                success=True, primitive_id=17, params_used=[ms / 1000.0],
            )
        except Exception as exc:
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=17, params_used=[ms / 1000.0],
            )

    def window(self, action: str, title_hash: int = 0) -> ObservationDict:
        # pyautogui has no native window manager API; fall back.
        return self._fallback.window(action, title_hash)

    def get_screen_size(self) -> ObservationDict:
        if not self.allow_real_os or pyautogui is None:
            return self._fallback.get_screen_size()
        try:
            w, h = pyautogui.size()  # type: ignore[union-attr]
            return ObservationDict(
                success=True, text=f"{w}x{h}",
                primitive_id=21, params_used=[],
            )
        except Exception as exc:
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=21, params_used=[],
            )

    def file_delete(self, path: str) -> ObservationDict:
        # NEVER actually rm a file from the agent.  Always sandbox.
        return self._fallback.file_delete(path)

    def exec_shell(self, cmd: str) -> ObservationDict:
        # NEVER actually exec from the agent.  Always sandbox.
        return self._fallback.exec_shell(cmd)


# ---------------------------------------------------------------------------
# MCP control backend -- dispatch to the mcp__mcp-control_* tool surface.
# ---------------------------------------------------------------------------


class McpControlBackend(_BackendBase):
    """Backend that emits ``mcp__mcp-control_*`` tool-call descriptors.

    The actual MCP tool dispatch is the *caller's* job (the runtime
    that hosts this agent).  We only build the descriptors and hand them
    over via a configurable callable.

    By default the callable is a no-op that just returns success=True --
    that lets the trainer / unit tests exercise this backend without
    needing a real MCP runtime.
    """

    name = "mcp_control"

    def __init__(self, dispatch: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        # ``dispatch(tool_name, args) -> mcp_response_dict``.
        self.dispatch = dispatch or (lambda tool, args: {"ok": True})
        self._fallback = SandboxBackend()

    def _call(self, tool: str, args: Dict[str, Any], primitive_id: int,
              params_used: List[float]) -> ObservationDict:
        try:
            resp = self.dispatch(tool, args) or {}
            ok = bool(resp.get("ok", True))
            text = str(resp.get("text", ""))
            return ObservationDict(
                success=ok,
                screenshot_bytes=resp.get("screenshot_bytes"),
                text=text,
                error_msg=str(resp.get("error", "")) if not ok else "",
                primitive_id=primitive_id,
                params_used=params_used,
            )
        except Exception as exc:
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=primitive_id, params_used=params_used,
            )

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> ObservationDict:
        if count == 2:
            return self._call("mcp__mcp-control_double_click", {"x": x, "y": y}, 1, [x, y])
        return self._call("mcp__mcp-control_click_at",
                          {"x": x, "y": y, "button": button}, 0, [x, y])

    def move(self, x: int, y: int, dwell_ms: int = 0) -> ObservationDict:
        return self._call("mcp__mcp-control_move_mouse",
                          {"x": x, "y": y}, 5, [x, y])

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ObservationDict:
        return self._call("mcp__mcp-control_drag_mouse",
                          {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                          4, [x1, y1, x2, y2])

    def scroll(self, dx: float, dy: float) -> ObservationDict:
        return self._call("mcp__mcp-control_scroll_mouse",
                          {"dx": dx, "dy": dy}, 7, [dx, dy])

    def type_text(self, text: str) -> ObservationDict:
        return self._call("mcp__mcp-control_type_text",
                          {"text": text}, 8, [])

    def press_key(self, key: str, chord: bool = False, hold: bool = False,
                  release: bool = False) -> ObservationDict:
        if hold:
            return self._call("mcp__mcp-control_hold_key",
                              {"key": key}, 11, [])
        if chord:
            return self._call("mcp__mcp-control_press_key_combination",
                              {"keys": key}, 10, [])
        return self._call("mcp__mcp-control_press_key",
                          {"key": key}, 9, [])

    def screenshot(self) -> ObservationDict:
        return self._call("mcp__mcp-control_get_screenshot", {}, 16, [])

    def wait(self, ms: int) -> ObservationDict:
        return self._fallback.wait(ms)

    def window(self, action: str, title_hash: int = 0) -> ObservationDict:
        tool = {
            "focus": "mcp__mcp-control_focus_window",
            "minimize": "mcp__mcp-control_minimize_window",
            "active": "mcp__mcp-control_get_active_window",
        }.get(action, "mcp__mcp-control_get_active_window")
        pid = {"focus": 18, "minimize": 19, "active": 20}.get(action, 20)
        return self._call(tool, {"title_hash": title_hash}, pid, [title_hash])

    def get_screen_size(self) -> ObservationDict:
        return self._call("mcp__mcp-control_get_screen_size", {}, 21, [])

    def file_delete(self, path: str) -> ObservationDict:
        return self._fallback.file_delete(path)

    def exec_shell(self, cmd: str) -> ObservationDict:
        return self._fallback.exec_shell(cmd)


# ---------------------------------------------------------------------------
# OSActuator -- dispatcher facade.
# ---------------------------------------------------------------------------


class OSActuator:
    """High-level dispatcher.  Reads (primitive_id, params) and routes
    to the chosen backend's typed methods.

    Args
    ----
    backend : "sandbox" | "win32" | "mcp_control"
    allow_real_os : bool
        Only honoured by win32; default False forces sandbox-fallback so
        a sandbox-only training run never touches the real OS even
        when backend="win32" is passed in error.
    screen_size : tuple[int, int]
        Used by sandbox + (when allow_real_os=False) win32.
    text_vocab : list[str] | None
        Optional id->str map for the type_text primitive's token_id slot.
        Defaults to single ASCII characters (' ' .. '~').
    key_vocab : list[str] | None
        Optional id->str map for the press_key family's keysym slot.
        Defaults to a 32-key superset of common keys (a-z + digits + nav).
    mcp_dispatch : Callable[[str, dict], dict] | None
        Required for backend="mcp_control"; default no-op returns ok=True.
    """

    BACKENDS = ("sandbox", "win32", "mcp_control")

    def __init__(
        self,
        backend: str = "sandbox",
        allow_real_os: bool = False,
        screen_size: Tuple[int, int] = (1024, 768),
        text_vocab: Optional[List[str]] = None,
        key_vocab: Optional[List[str]] = None,
        mcp_dispatch: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; valid: {self.BACKENDS}")
        self.backend_name = backend
        self.screen_size = screen_size
        self.allow_real_os = bool(allow_real_os)
        self.text_vocab = text_vocab or [chr(c) for c in range(0x20, 0x7F)]
        self.key_vocab = key_vocab or _DEFAULT_KEY_VOCAB
        if backend == "sandbox":
            self.backend: _BackendBase = SandboxBackend(screen_size=screen_size)
        elif backend == "win32":
            self.backend = Win32Backend(allow_real_os=allow_real_os)
        else:
            self.backend = McpControlBackend(dispatch=mcp_dispatch)

    # -- core ------------------------------------------------------------
    def execute(self, primitive_id: int, params) -> ObservationDict:
        """Dispatch (primitive_id, params) on the chosen backend.

        ``params`` is a length-8 sequence of floats in [0,1] (xy/dx/dy)
        or normalised ints (token_id/keysym).  The dispatcher reads only
        the slots the primitive actually uses.
        """
        try:
            primitive = by_id(int(primitive_id))
        except KeyError:
            return ObservationDict(
                success=False, error_msg=f"unknown primitive_id={primitive_id}",
                primitive_id=int(primitive_id),
                params_used=list(params or []),
            )
        # Sandbox guard: the brief says sandbox-by-default, real-OS opt-in.
        if primitive.sandbox_guard and not self.allow_real_os:
            # Force fallback to sandbox for guarded primitives.
            sandbox = SandboxBackend(screen_size=self.screen_size)
            return self._dispatch(sandbox, primitive, params)
        return self._dispatch(self.backend, primitive, params)

    def _dispatch(self, backend: _BackendBase, primitive,
                  params) -> ObservationDict:
        params = list(params or [])
        # Pad to 8 so slot reads are safe.
        while len(params) < 8:
            params.append(0.0)
        x = self._denorm_x(params[0])
        y = self._denorm_y(params[1])
        x2 = self._denorm_x(params[2])
        y2 = self._denorm_y(params[3])
        token_id = int(round(float(params[4])))
        keysym = int(round(float(params[5])))
        dx = float(params[6])
        dy = float(params[7])
        try:
            if primitive.id == 0:
                return backend.click(x, y, button="left", count=1)
            if primitive.id == 1:
                return backend.click(x, y, button="left", count=2)
            if primitive.id == 2:
                return backend.click(x, y, button="right", count=1)
            if primitive.id == 3:
                return backend.click(x, y, button="middle", count=1)
            if primitive.id == 4:
                return backend.drag(x, y, x2, y2)
            if primitive.id == 5:
                return backend.move(x, y, dwell_ms=0)
            if primitive.id == 6:
                return backend.move(x, y, dwell_ms=100)
            if primitive.id == 7:
                return backend.scroll(dx, dy)
            if primitive.id == 8:
                text = self._decode_text(token_id)
                return backend.type_text(text)
            if primitive.id == 9:
                return backend.press_key(self._decode_key(keysym),
                                         chord=False, hold=False, release=False)
            if primitive.id == 10:
                return backend.press_key(self._decode_key(keysym, allow_chord=True),
                                         chord=True)
            if primitive.id == 11:
                return backend.press_key(self._decode_key(keysym), hold=True)
            if primitive.id == 12:
                return backend.press_key(self._decode_key(keysym), release=True)
            if primitive.id == 13:
                return backend.press_key("ctrl+a", chord=True)
            if primitive.id == 14:
                return backend.press_key("ctrl+c", chord=True)
            if primitive.id == 15:
                return backend.press_key("ctrl+v", chord=True)
            if primitive.id == 16:
                return backend.screenshot()
            if primitive.id == 17:
                ms = int(min(1.0, abs(dx)) * 1000)
                return backend.wait(ms)
            if primitive.id == 18:
                return backend.window("focus", title_hash=keysym)
            if primitive.id == 19:
                return backend.window("minimize", title_hash=keysym)
            if primitive.id == 20:
                return backend.window("active", title_hash=keysym)
            if primitive.id == 21:
                return backend.get_screen_size()
            if primitive.id == 22:
                # path lookup -- vocab entry beyond ASCII range
                return backend.file_delete(self._decode_text(token_id))
            if primitive.id == 23:
                return backend.exec_shell(self._decode_text(token_id))
        except Exception as exc:  # pragma: no cover - defensive
            return ObservationDict(
                success=False, error_msg=str(exc),
                primitive_id=primitive.id, params_used=params,
            )
        return ObservationDict(
            success=False,
            error_msg=f"primitive {primitive.id} unimplemented",
            primitive_id=primitive.id, params_used=params,
        )

    # -- vocab / coord helpers ------------------------------------------
    def _denorm_x(self, x: float) -> int:
        w = max(1, int(self.screen_size[0]))
        return int(max(0, min(w - 1, round(float(x) * (w - 1)))))

    def _denorm_y(self, y: float) -> int:
        h = max(1, int(self.screen_size[1]))
        return int(max(0, min(h - 1, round(float(y) * (h - 1)))))

    def _decode_text(self, token_id: int) -> str:
        if not self.text_vocab:
            return ""
        idx = int(token_id) % len(self.text_vocab)
        return self.text_vocab[idx]

    def _decode_key(self, keysym: int, allow_chord: bool = False) -> str:
        if not self.key_vocab:
            return ""
        idx = int(keysym) % len(self.key_vocab)
        key = self.key_vocab[idx]
        if not allow_chord and "+" in key:
            # Strip chord parts to get the principal key.
            return key.split("+")[-1]
        return key


# ---------------------------------------------------------------------------
# Default key vocabulary.  Mirrors the chord-supporting subset of
# ``synapforge.action.head.KEY_VOCAB`` but keeps it short for the
# 8-bit-ish keysym slot.
# ---------------------------------------------------------------------------

_DEFAULT_KEY_VOCAB: List[str] = [
    *"abcdefghijklmnopqrstuvwxyz",          # 0..25
    "enter", "tab", "space", "backspace",   # 26..29
    "delete", "esc", "ctrl+c", "ctrl+v",    # 30..33
    "ctrl+a", "ctrl+s", "alt+tab", "ctrl+z",
    "up", "down", "left", "right",
]


__all__ = [
    "ObservationDict",
    "OSActuator",
    "SandboxBackend",
    "Win32Backend",
    "McpControlBackend",
]
