"""``synapforge-chat --async --proactive`` — interactive chat CLI.

Wires :mod:`synapforge.interactive` to a terminal:

  * stdin lines are user EOTs (one line per turn).
  * Model output streams to stdout one chunk at a time.
  * ``Ctrl+C`` cancels the in-flight model response.
  * ``--proactive`` enables the :class:`ProactiveTrigger` loop.
  * ``--interrupt-aggressive`` drops the interrupt urgency threshold
    from 0.95 to 0.7.
  * ``--transcript path.jsonl`` replays a deterministic input transcript
    instead of stdin (used by tests + the demo recipe).

This CLI is additive — it does NOT replace :mod:`synapforge.demo.chat_demo`
(which stays as the synchronous fallback / investor pitch path).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional


from synapforge.interactive import (
    AsyncChatSession,
    ChatChunk,
    InputListener,
    ProactiveTrigger,
    StreamingGen,
    TurnTakingPolicy,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synapforge-chat",
        description=(
            "Async chat CLI — streaming, listening, interruption, "
            "proactive speaking."
        ),
    )
    p.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        default=True,
        help="Run in async kernel mode (default).",
    )
    p.add_argument(
        "--sync",
        dest="async_mode",
        action="store_false",
        help=(
            "Forward to synapforge.demo.chat_demo (the synchronous fallback). "
            "Use this when you want the legacy 5 EN + 5 ZH demo path."
        ),
    )
    p.add_argument(
        "--proactive",
        action="store_true",
        default=False,
        help="Enable the proactive trigger loop (default: off).",
    )
    p.add_argument(
        "--interrupt-aggressive",
        dest="interrupt_aggressive",
        action="store_true",
        default=False,
        help=(
            "Drop the interrupt urgency threshold from 0.95 to 0.7 — model "
            "will interrupt more aggressively. Use with care."
        ),
    )
    p.add_argument(
        "--ckpt",
        default=os.environ.get("SYNAPFORGE_CKPT"),
        help="Path to a SynapForge model ckpt (.pt). Required for live mode.",
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.environ.get("SYNAPFORGE_TOKENIZER", "Qwen/Qwen2.5-0.5B"),
        help="HF tokenizer path or repo id.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cuda / cpu / auto (default: auto).",
    )
    p.add_argument(
        "--max-new",
        type=int,
        default=200,
        help="Max tokens per turn (default 200).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default 0.7).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--target-tps",
        type=float,
        default=80.0,
        help="Target tokens-per-second pace (default 80; 0 = unlimited).",
    )
    p.add_argument(
        "--politeness-pause-ms",
        type=int,
        default=300,
    )
    p.add_argument(
        "--interrupt-threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--proactive-silence-window",
        type=float,
        default=5.0,
        help="Seconds of silence before silence-relevance trigger fires.",
    )
    p.add_argument(
        "--save",
        default=None,
        help=(
            "Path to write the session transcript JSON on shutdown. "
            "Includes user/model messages + trigger audit log."
        ),
    )
    p.add_argument(
        "--transcript",
        default=None,
        help=(
            "JSONL file of deterministic input chunks to drive the listener "
            "(used by tests + the demo recipe). Each line is a string; "
            'a literal "<EOT>" line marks turn end, "<EOF>" marks shutdown.'
        ),
    )
    p.add_argument(
        "--log-triggers",
        action="store_true",
        default=False,
        help="Print proactive trigger telemetry as triggers fire.",
    )
    return p


# ---------------------------------------------------------------------------
# Listener wiring
# ---------------------------------------------------------------------------


def _read_transcript(path: str) -> Iterator[str]:
    """Yield chunks from a JSONL transcript.

    Each line is either a JSON-encoded string or one of the markers
    ``<EOT>`` / ``<EOF>``.  Markers come through verbatim so the
    listener applies its standard handling.

    Both raw markers (``<EOT>``) and JSON-encoded markers (``"<EOT>"``)
    are accepted so the file is human-editable.
    """
    p = Path(path)
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Try JSON first so quoted strings get properly decoded; fall back
        # to the raw line for unquoted markers.
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            obj = line
        text = str(obj)
        if text == "<EOF>":
            yield None  # type: ignore[misc]
            return
        if text == "<EOT>":
            yield "\n"
            continue
        yield text
    yield None  # type: ignore[misc]


def _build_listener(args: argparse.Namespace) -> InputListener:
    if args.transcript:
        items = list(_read_transcript(args.transcript))
        return InputListener.with_iterable(items)
    return InputListener.with_stdin_lines()


# ---------------------------------------------------------------------------
# Generator wiring (lazy torch / model import)
# ---------------------------------------------------------------------------


def _build_generator(args: argparse.Namespace) -> StreamingGen:
    """Lazy-load the model + tokenizer and wrap with StreamingGen.

    Falls back to a stub generator if no ckpt is provided OR torch / model
    cannot be loaded.  The stub emits canned echo responses so the CLI
    is still usable for pipeline smoke tests on CPU-less boxes.
    """
    if args.ckpt and Path(args.ckpt).is_file():
        try:
            from synapforge.demo.chat_demo import (
                _resolve_device,
                _try_load_live,
            )

            device = _resolve_device(args.device)
            live = _try_load_live(
                args.ckpt,
                args.tokenizer_path,
                device=device,
                verbose=False,
            )
            if live is not None:
                model, tokenizer, _ = live
                return StreamingGen(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    target_tokens_per_sec=args.target_tps,
                )
        except Exception as exc:
            print(
                f"[async-chat] live model load failed: {exc!r}; "
                f"using stub.",
                file=sys.stderr,
            )

    # Stub path.
    return _StubStreamingGen(target_tokens_per_sec=args.target_tps)


class _StubStreamingGen(StreamingGen):
    """A torch-free stub generator that emits a canned response per turn.

    Bypasses ``_run`` entirely so this works without torch — useful for
    pipeline smoke tests, CI, and the ``--transcript`` demo recipe.
    The kernel doesn't care: a StreamingGen that produces a
    GenerationHandle with a working output_queue is all the kernel needs.
    """

    _CANNED = [
        "Hi there! ",
        "Sure, what about it? ",
        "Got it. ",
        "Tell me more. ",
        "Understood. ",
    ]

    def __init__(self, target_tokens_per_sec: float = 80.0):
        self.target_tokens_per_sec = float(target_tokens_per_sec)
        self.eos_token_id = None
        self._idx = 0
        # Keep field names in sync with StreamingGen so any downstream
        # introspection works.
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.check_cancel_every = 1

    async def start(
        self,
        prompt_text: str,
        request_id: str = "",
        max_new: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        import asyncio
        import time

        from synapforge.interactive.streaming import (
            CancellationToken,
            GenerationHandle,
            _CANCEL,
            _DONE,
        )

        token = CancellationToken()
        queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=128)
        handle = GenerationHandle(
            request_id=request_id or f"stub_{self._idx}",
            token=token,
            output_queue=queue,
        )
        text = self._CANNED[self._idx % len(self._CANNED)]
        self._idx += 1

        async def _emit():
            min_step_s = (
                1.0 / self.target_tokens_per_sec
                if self.target_tokens_per_sec > 0
                else 0.0
            )
            try:
                for ch in text:
                    if handle.token.cancelled:
                        await handle.output_queue.put(_CANCEL)
                        return
                    await handle.output_queue.put(ch)
                    handle.n_tokens_emitted += 1
                    if min_step_s > 0:
                        await asyncio.sleep(min_step_s)
                    else:
                        await asyncio.sleep(0)
                await handle.output_queue.put(_DONE)
            except Exception as exc:  # pragma: no cover
                await handle.output_queue.put(f"<<<__ERROR__>>>{exc!r}")

        asyncio.create_task(_emit())
        return handle


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


async def _run_async(args: argparse.Namespace) -> int:
    listener = _build_listener(args)
    generator = _build_generator(args)

    policy = TurnTakingPolicy(
        politeness_pause_ms=args.politeness_pause_ms,
        interrupt_threshold=args.interrupt_threshold,
    )
    if args.interrupt_aggressive:
        policy.set_aggressive()

    trigger: Optional[ProactiveTrigger] = None
    if args.proactive:
        trigger = ProactiveTrigger(
            silence_threshold_s=args.proactive_silence_window,
        )

    log_triggers = args.log_triggers

    async def _on_chunk(chunk: ChatChunk) -> None:
        if chunk.proactive and chunk.text.startswith("[Proactive]"):
            sys.stdout.write("\n")
        if chunk.interrupt_marker:
            sys.stdout.write(f"\n{chunk.interrupt_marker} ")
        sys.stdout.write(chunk.text)
        sys.stdout.flush()

    session = AsyncChatSession(
        generator=generator,
        listener=listener,
        proactive_trigger=trigger,
        turn_policy=policy,
        on_chunk=_on_chunk,
    )

    if log_triggers and trigger is not None:
        async def _drain_triggers():
            while True:
                await asyncio.sleep(2.0)
                if not session.trigger_log:
                    continue
                last = session.trigger_log[-1]
                print(
                    f"\n[model-proactive] curiosity="
                    f"{last.get('confidence', 0.0):.2f} "
                    f"goal={last.get('trigger', '?')} → "
                    f"{last.get('decision', '?')}",
                    file=sys.stderr,
                )

        asyncio.create_task(_drain_triggers())

    try:
        await session.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await session.stop()
        print("\n[async-chat] interrupted.", file=sys.stderr)
    finally:
        if args.save:
            try:
                session.save_session(args.save)
                print(
                    f"[async-chat] session saved to {args.save}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"[async-chat] save failed: {exc!r}",
                    file=sys.stderr,
                )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.async_mode:
        # User passed --sync; forward to the legacy synchronous demo.
        from synapforge.demo.chat_demo import main as sync_main

        return sync_main([])
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
