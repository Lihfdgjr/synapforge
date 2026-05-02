"""synapforge.neuromcp.sandbox -- virtual desktop for safe training.

Why this exists
---------------
Per memory ``feedback_neural_action_no_token_no_mcp`` and the brief's
HARD CONSTRAINTS, the model defaults to sandbox-only execution during
training.  A PLIF-dead model drilling random clicks must NEVER touch
the real filesystem -- that is what this module enforces at the
infrastructure layer.

Components
----------

    VirtualDesktop      -- 1024x768 fake screen with simulated buttons
                           and a tiny text-input widget.  Click / drag /
                           type / scroll all mutate an in-memory state.
    Button              -- (x, y, w, h, label, on_click_handler)
    TextInput           -- (x, y, w, h, value)
    snapshot_png        -- render to PNG bytes via PIL when available;
                           falls back to a 32-byte signature otherwise.

The virtual desktop is intentionally minimal: it exists to give the
neural action loop *some* state to predict, not to be a full UI sim.

torch is **not** imported here.  PIL is optional (snapshot becomes a
deterministic byte stub if PIL is missing).
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


@dataclass
class Button:
    x: int
    y: int
    w: int
    h: int
    label: str
    on_click: Optional[Callable[["VirtualDesktop"], None]] = None
    pressed: bool = False

    def hit(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h


@dataclass
class TextInput:
    x: int
    y: int
    w: int
    h: int
    value: str = ""
    focused: bool = False

    def hit(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h


# ---------------------------------------------------------------------------
# Virtual desktop
# ---------------------------------------------------------------------------


@dataclass
class VirtualDesktop:
    """In-process fake desktop.

    Layout (1024x768 default):
        - ``Button(x=50, y=50, "OK")``
        - ``Button(x=200, y=50, "Cancel")``
        - ``Button(x=50, y=120, "Submit")``
        - ``TextInput(x=50, y=200, w=400, h=30)``

    State is fully reproducible: ``screen_size``, ``cursor``, ``buttons``,
    ``text_inputs``, ``focused_input``, ``clipboard``, and a
    ``frame_counter`` so two snapshots of the same state hash equal.
    """

    screen_size: Tuple[int, int] = (1024, 768)
    cursor: Tuple[int, int] = (0, 0)
    buttons: List[Button] = field(default_factory=list)
    text_inputs: List[TextInput] = field(default_factory=list)
    focused_input: Optional[int] = None
    clipboard: str = ""
    last_event: str = ""
    frame_counter: int = 0
    held_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.buttons:
            self.buttons = [
                Button(50, 50, 100, 40, "OK"),
                Button(200, 50, 100, 40, "Cancel"),
                Button(50, 120, 120, 40, "Submit"),
                Button(200, 120, 120, 40, "Open Chrome"),
            ]
        if not self.text_inputs:
            self.text_inputs = [TextInput(50, 200, 400, 30, value="")]

    # -- snapshot --------------------------------------------------------
    def snapshot_png(self) -> bytes:
        """Render desktop state to PNG bytes.

        When PIL is available the snapshot is a real 1024x768 RGB PNG
        (cheap to generate, < 4 KB on average).  When PIL is missing we
        return a deterministic byte signature so unit tests can still
        compare snapshots without reaching for numpy.
        """
        self.frame_counter += 1
        if _HAS_PIL and Image is not None and ImageDraw is not None:
            w, h = self.screen_size
            img = Image.new("RGB", (w, h), color=(240, 240, 245))
            draw = ImageDraw.Draw(img)
            for b in self.buttons:
                fill = (180, 180, 190) if not b.pressed else (140, 140, 150)
                draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h],
                               outline=(40, 40, 60), fill=fill)
                # cheap centred text
                draw.text((b.x + 8, b.y + 8), b.label, fill=(0, 0, 0))
            for ti in self.text_inputs:
                draw.rectangle([ti.x, ti.y, ti.x + ti.w, ti.y + ti.h],
                               outline=(40, 40, 60), fill=(255, 255, 255))
                draw.text((ti.x + 4, ti.y + 6), ti.value[:60], fill=(0, 0, 0))
            # cursor cross
            cx, cy = self.cursor
            draw.line([cx - 5, cy, cx + 5, cy], fill=(255, 0, 0))
            draw.line([cx, cy - 5, cx, cy + 5], fill=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        # PIL-less: stable byte signature so snapshot != snapshot' when
        # state changes.
        sig = struct.pack(
            ">IIii",
            self.frame_counter,
            sum(int(b.pressed) << i for i, b in enumerate(self.buttons[:32])),
            self.cursor[0], self.cursor[1],
        )
        sig += b"|" + b"|".join(ti.value.encode("utf-8", "ignore")
                                for ti in self.text_inputs)
        sig += b"|cb=" + self.clipboard.encode("utf-8", "ignore")
        return b"VDSK" + sig

    # -- pointer ---------------------------------------------------------
    def move_cursor(self, x: int, y: int) -> None:
        w, h = self.screen_size
        self.cursor = (max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y))))

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> bool:
        self.move_cursor(x, y)
        # Defocus all text inputs first.
        for ti in self.text_inputs:
            ti.focused = False
        self.focused_input = None
        for b in self.buttons:
            if b.hit(x, y):
                b.pressed = True
                if b.on_click is not None:
                    try:
                        b.on_click(self)
                    except Exception:
                        pass
                self.last_event = f"click({b.label!r})"
                return True
        for i, ti in enumerate(self.text_inputs):
            if ti.hit(x, y):
                ti.focused = True
                self.focused_input = i
                self.last_event = f"focus_input({i})"
                return True
        self.last_event = f"click_miss({x},{y})"
        return False

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.move_cursor(x2, y2)
        self.last_event = f"drag({x1},{y1}->{x2},{y2})"

    def scroll(self, dx: float, dy: float) -> None:
        self.last_event = f"scroll({dx:+.2f},{dy:+.2f})"

    # -- keyboard --------------------------------------------------------
    def type_text(self, text: str) -> None:
        if self.focused_input is None:
            self.last_event = f"type_unfocused({text!r})"
            return
        ti = self.text_inputs[self.focused_input]
        ti.value += str(text)
        self.last_event = f"type_into({self.focused_input}, {text!r})"

    def press_key(self, key: str, chord: bool = False, hold: bool = False,
                  release: bool = False) -> None:
        # Special: enter 'submits' a focused text input via clearing it.
        if hold:
            if key not in self.held_keys:
                self.held_keys.append(key)
            self.last_event = f"key_down({key})"
            return
        if release:
            if key in self.held_keys:
                self.held_keys.remove(key)
            self.last_event = f"key_up({key})"
            return
        if chord and key.startswith("ctrl+"):
            sub = key.split("+")[-1]
            if sub == "c" and self.focused_input is not None:
                self.clipboard = self.text_inputs[self.focused_input].value
                self.last_event = "copy"
            elif sub == "v" and self.focused_input is not None:
                self.text_inputs[self.focused_input].value += self.clipboard
                self.last_event = "paste"
            elif sub == "a" and self.focused_input is not None:
                self.last_event = "select_all"
            else:
                self.last_event = f"chord({key})"
            return
        if key == "enter" and self.focused_input is not None:
            ti = self.text_inputs[self.focused_input]
            self.last_event = f"submit({ti.value!r})"
            return
        if key == "backspace" and self.focused_input is not None:
            ti = self.text_inputs[self.focused_input]
            ti.value = ti.value[:-1]
            self.last_event = "backspace"
            return
        self.last_event = f"key({key})"

    def tick(self, ms: int) -> None:
        self.last_event = f"wait({ms}ms)"

    def window_action(self, action: str, title_hash: int) -> None:
        self.last_event = f"window({action},#{title_hash})"

    # -- task helpers (used by tests / curiosity reward) ----------------
    def reset(self) -> None:
        for b in self.buttons:
            b.pressed = False
        for ti in self.text_inputs:
            ti.value = ""
            ti.focused = False
        self.focused_input = None
        self.clipboard = ""
        self.last_event = "reset"
        self.held_keys.clear()

    def state_signature(self) -> Dict[str, object]:
        return {
            "cursor": self.cursor,
            "pressed": [b.pressed for b in self.buttons],
            "values": [ti.value for ti in self.text_inputs],
            "focused": self.focused_input,
            "clipboard": self.clipboard,
            "last_event": self.last_event,
        }


__all__ = ["VirtualDesktop", "Button", "TextInput"]
