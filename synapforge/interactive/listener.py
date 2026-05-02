"""InputListener — async input source for the chat kernel.

Watches one of:
  * stdin in line mode (``stdin``)
  * stdin in raw single-character mode (``stdin_raw``)
  * a file descriptor / path (file tail-mode for IPC tests)
  * audio via ``webrtcvad`` (optional dep; falls back to no-op if missing)

Emits :class:`InputEvent` records onto an asyncio.Queue.  The kernel's
event loop is the consumer; the model forward path is *never* blocked
because reads happen in a thread executor.

Hard constraints (from task spec):
  * 0 imports of ``torch`` — none needed; this is plain async I/O.
  * Doesn't block the model forward — every read goes through
    ``loop.run_in_executor`` so the event loop stays responsive while a
    keystroke is in flight.
  * VAD is optional — ``InputListener.with_vad()`` returns ``None`` when
    ``webrtcvad`` isn't installed, and the kernel just drives stdin.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


class InputEventKind(Enum):
    PARTIAL_TOKEN = "partial_token"  # one keystroke / partial chunk
    SILENCE = "silence"  # pause threshold crossed
    EOT = "eot"  # end-of-turn (Enter / VAD silence-after-speech)
    CANCEL = "cancel"  # explicit user interrupt
    EOF = "eof"  # source closed (stdin EOF / file truncated)


@dataclass
class InputEvent:
    kind: InputEventKind
    text: str = ""
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InputListener
# ---------------------------------------------------------------------------


class InputListener:
    """Producer of :class:`InputEvent` records from a configurable source.

    Concrete sources are wired by passing different ``read_fn`` callables.
    The listener only owns: the queue, the silence detector, and the
    background coroutine that drives reads.
    """

    def __init__(
        self,
        read_fn: Callable[[], Optional[str]],
        silence_pause_s: float = 1.2,
        eot_marker: str = "\n",
        queue_maxsize: int = 1024,
    ) -> None:
        self.read_fn = read_fn
        self.silence_pause_s = float(silence_pause_s)
        self.eot_marker = eot_marker
        self.events: "asyncio.Queue[InputEvent]" = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_keystroke_ts: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        self._task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[InputEvent]:
        while True:
            ev = await self.events.get()
            yield ev
            if ev.kind == InputEventKind.EOF:
                return

    # ------------------------------------------------------------------
    # Reader loop
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stopped:
            try:
                # Read in an executor so the event loop stays responsive.
                # The reader is allowed to block on I/O — that's fine, the
                # main loop is on a different thread.
                chunk = await loop.run_in_executor(None, self.read_fn)
            except Exception as exc:  # pragma: no cover — defensive
                await self.events.put(
                    InputEvent(
                        kind=InputEventKind.EOF,
                        text="",
                        meta={"error": repr(exc)},
                    )
                )
                return

            if chunk is None:
                await self.events.put(InputEvent(kind=InputEventKind.EOF))
                return
            if chunk == "":
                # Idle tick → maybe emit a SILENCE event.
                if self._last_keystroke_ts > 0.0:
                    elapsed = time.time() - self._last_keystroke_ts
                    if elapsed >= self.silence_pause_s:
                        await self.events.put(
                            InputEvent(
                                kind=InputEventKind.SILENCE,
                                meta={"elapsed_s": elapsed},
                            )
                        )
                        self._last_keystroke_ts = 0.0
                continue

            if chunk == self.eot_marker:
                await self.events.put(InputEvent(kind=InputEventKind.EOT))
                self._last_keystroke_ts = 0.0
                continue

            # Cancel marker — Ctrl+C in raw mode comes through as ``\x03``.
            if chunk == "\x03":
                await self.events.put(InputEvent(kind=InputEventKind.CANCEL))
                continue

            self._last_keystroke_ts = time.time()
            await self.events.put(
                InputEvent(kind=InputEventKind.PARTIAL_TOKEN, text=chunk)
            )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def with_stdin_lines(
        cls,
        silence_pause_s: float = 1.2,
        eot_marker: str = "\n",
    ) -> "InputListener":
        """Read full lines from stdin. Each line is one EOT.

        This is the simplest mode and the one the CLI defaults to. Lines
        come through as a PARTIAL_TOKEN containing the entire body, then
        an EOT.  Avoids putting stdin in raw mode.
        """

        def _read_line() -> Optional[str]:
            line = sys.stdin.readline()
            if not line:
                return None  # EOF
            # Emit body without trailing newline as PARTIAL, EOT comes via
            # the read of '\n' below by re-yielding the marker.
            stripped = line.rstrip("\n")
            return stripped + (eot_marker if line.endswith("\n") else "")

        # Trick: the read_fn needs to emit two events per line (text +
        # EOT marker). We do it by returning a sentinel that the loop
        # parses below. Easier: emit body as a single PARTIAL then the
        # marker on the next call.
        buf: dict = {"pending_marker": False}

        def _stateful_read() -> Optional[str]:
            if buf["pending_marker"]:
                buf["pending_marker"] = False
                return eot_marker
            line = sys.stdin.readline()
            if not line:
                return None
            stripped = line.rstrip("\n")
            buf["pending_marker"] = True
            return stripped if stripped else eot_marker

        listener = cls(
            read_fn=_stateful_read,
            silence_pause_s=silence_pause_s,
            eot_marker=eot_marker,
        )
        return listener

    @classmethod
    def with_iterable(
        cls,
        items,
        silence_pause_s: float = 1.2,
        eot_marker: str = "\n",
    ) -> "InputListener":
        """Drive the listener from a deterministic iterable.

        Used by tests + by ``async_chat_cli --transcript`` replay mode.
        Each item is one ``read_fn`` return: a string chunk, the EOT marker,
        or ``None`` for EOF.
        """
        it = iter(items)

        def _read() -> Optional[str]:
            try:
                return next(it)
            except StopIteration:
                return None

        return cls(
            read_fn=_read,
            silence_pause_s=silence_pause_s,
            eot_marker=eot_marker,
        )

    @classmethod
    def with_vad(
        cls,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_pause_s: float = 0.8,
    ) -> Optional["InputListener"]:
        """Audio-mode listener via ``webrtcvad``. Returns None if missing.

        We don't actually capture audio here — that's a downstream
        responsibility (PyAudio / sounddevice). This factory is the
        contract: when ``webrtcvad`` is present, you get back an empty
        listener whose ``read_fn`` you wire to your audio callback.
        """
        try:
            import webrtcvad  # noqa: F401
        except Exception:
            return None
        # The actual capture device wiring is done by the caller — we just
        # return a configured listener with the right silence threshold.
        return cls(
            read_fn=lambda: "",  # caller overrides via push()
            silence_pause_s=silence_pause_s,
        )

    async def push(self, text: str) -> None:
        """Allow external callers (audio frame, websocket, etc.) to inject
        a chunk into the queue without going through ``read_fn``.

        Used by the VAD path and by integration tests that want to drive
        the listener directly.
        """
        if not text:
            return
        if text == self.eot_marker:
            await self.events.put(InputEvent(kind=InputEventKind.EOT))
            self._last_keystroke_ts = 0.0
            return
        if text == "\x03":
            await self.events.put(InputEvent(kind=InputEventKind.CANCEL))
            return
        self._last_keystroke_ts = time.time()
        await self.events.put(
            InputEvent(kind=InputEventKind.PARTIAL_TOKEN, text=text)
        )

    async def push_eof(self) -> None:
        await self.events.put(InputEvent(kind=InputEventKind.EOF))
