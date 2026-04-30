"""
TurnTakingDetector — text equivalent of voice activity detection.

Decides whether the user's current input is a complete turn or just a
mid-thought pause. Drives the kernel's "should I respond now or wait" gate.

Heuristics (any "complete" signal AND > min_chars):
  1. Trailing punctuation: . ! ? 。 ! ? + newline
  2. >= 1.2 seconds since last keystroke (typing pause)
  3. Question word at start ("what" / "where" / "怎么" / "什么")
     AND ends in any terminal mark
  4. Code block close (``` matched)
  5. JSON/dict closed brace match
  6. Explicit submit signal (Enter without shift, send button)

"Definitely incomplete" signals override completion:
  - Trailing comma + newline (continuation)
  - Open paren / bracket / quote not yet closed
  - Trailing dash or "..."
  - Last char is whitespace and last 2s have keystrokes (still typing)

The detector is **not** stateless. It tracks recent keystroke timestamps,
quote/bracket depth, code block state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TurnState(Enum):
    EMPTY = "empty"
    TYPING = "typing"
    PAUSE = "pause"
    COMPLETE = "complete"
    EXPLICIT_SUBMIT = "explicit_submit"


@dataclass
class _TypingTrace:
    last_keystroke_ts: float = 0.0
    chars_since_last_complete: int = 0
    paren_depth: int = 0
    bracket_depth: int = 0
    brace_depth: int = 0
    in_double_quote: bool = False
    in_single_quote: bool = False
    in_code_block: bool = False
    last_chars: str = ""


class TurnTakingDetector:
    def __init__(
        self,
        pause_threshold_s: float = 1.2,
        min_chars_for_complete: int = 2,
        long_pause_s: float = 4.0,
    ) -> None:
        self.pause_threshold_s = pause_threshold_s
        self.min_chars = min_chars_for_complete
        self.long_pause_s = long_pause_s
        self.trace = _TypingTrace()

    def reset(self) -> None:
        self.trace = _TypingTrace()

    def on_char(self, c: str) -> None:
        """Update internal state with one user character."""
        self.trace.last_keystroke_ts = time.time()
        self.trace.chars_since_last_complete += 1

        if self.trace.in_code_block:
            self.trace.last_chars += c
            if self.trace.last_chars.endswith("```"):
                self.trace.in_code_block = False
                self.trace.last_chars = ""
            return

        if c == "(":
            self.trace.paren_depth += 1
        elif c == ")":
            self.trace.paren_depth = max(0, self.trace.paren_depth - 1)
        elif c == "[":
            self.trace.bracket_depth += 1
        elif c == "]":
            self.trace.bracket_depth = max(0, self.trace.bracket_depth - 1)
        elif c == "{":
            self.trace.brace_depth += 1
        elif c == "}":
            self.trace.brace_depth = max(0, self.trace.brace_depth - 1)
        elif c == '"':
            self.trace.in_double_quote = not self.trace.in_double_quote
        elif c == "'":
            self.trace.in_single_quote = not self.trace.in_single_quote

        self.trace.last_chars = (self.trace.last_chars + c)[-3:]
        if self.trace.last_chars == "```":
            self.trace.in_code_block = True

    def on_string(self, s: str) -> None:
        for c in s:
            self.on_char(c)

    def on_explicit_submit(self) -> None:
        self.trace.last_keystroke_ts = 0.0

    def state(self, current_text: str) -> TurnState:
        """Read current state without mutating."""
        if not current_text:
            return TurnState.EMPTY
        if len(current_text) < self.min_chars:
            return TurnState.TYPING

        if self._has_open_structure():
            return TurnState.TYPING

        last = current_text[-1]

        if last in {",", "-"} or current_text.endswith("..."):
            return TurnState.TYPING

        terminal = last in {".", "!", "?", "。", "！", "？", "\n"}

        if self.trace.last_keystroke_ts == 0.0:
            return TurnState.EXPLICIT_SUBMIT

        if terminal:
            return TurnState.COMPLETE

        elapsed = time.time() - self.trace.last_keystroke_ts
        if elapsed >= self.long_pause_s:
            return TurnState.COMPLETE
        if elapsed >= self.pause_threshold_s:
            return TurnState.PAUSE

        return TurnState.TYPING

    def _has_open_structure(self) -> bool:
        return (
            self.trace.paren_depth > 0
            or self.trace.bracket_depth > 0
            or self.trace.brace_depth > 0
            or self.trace.in_double_quote
            or self.trace.in_single_quote
            or self.trace.in_code_block
        )

    def time_since_last_keystroke_s(self) -> float:
        if self.trace.last_keystroke_ts == 0.0:
            return 0.0
        return time.time() - self.trace.last_keystroke_ts

    def stats(self) -> dict:
        return {
            "chars_since_last": self.trace.chars_since_last_complete,
            "paren_depth": self.trace.paren_depth,
            "bracket_depth": self.trace.bracket_depth,
            "brace_depth": self.trace.brace_depth,
            "in_quote": self.trace.in_double_quote or self.trace.in_single_quote,
            "in_code_block": self.trace.in_code_block,
            "idle_s": self.time_since_last_keystroke_s(),
        }
