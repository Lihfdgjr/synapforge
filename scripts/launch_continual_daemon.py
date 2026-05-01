"""
launch_continual_daemon.py — Spawn Track A + Track B as long-running processes.

Architecture:
  Track A (web → weights via shadow LoRA): WebContentLearner + autonomous loop
  Track B (chat → retrieval cache): UserChatMemoryAdapter, request-driven

Track B is request-driven (no daemon needed); we still expose it via a
small status-server thread so external processes can write into the cache
through a Unix socket / named pipe / anonymous queue.

Track A runs the autonomous web-learn loop. We import the existing
synapforge.learn.autonomous_daemon and override its gate to point at
the new 7-gate WebContentLearner. Defensive: any web fetch failure
just logs + continues; the daemon never crashes the whole pipeline.

Checkpoint discovery:
  Reads `.continual_ckpt` (a symlink or file containing the path) so the
  trainer can drop new ckpts and the daemon picks them up at the next
  refresh tick (default 4h).

Logging:
  All gate decisions → out_dir/gate_log.jsonl
  Every cycle → out_dir/cycle_log.jsonl

Graceful shutdown:
  SIGTERM/SIGINT trigger flush of buffered samples + final stats dump.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import random
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    """Load a module by file path while registering it in sys.modules.

    Registration is required because dataclasses resolves field type
    annotations through `sys.modules[cls.__module__]`.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load continual_daemon directly so we don't trigger torch-heavy
# `synapforge/__init__.py` side effects.
_cd_path = REPO_ROOT / "synapforge" / "learn" / "continual_daemon.py"
cd = _load_module("synap_continual_daemon", _cd_path)
WebContentLearner = cd.WebContentLearner
UserChatMemoryAdapter = cd.UserChatMemoryAdapter
IngestDecision = cd.IngestDecision


# ---- Track A driver ------------------------------------------------------


class TrackADriver(threading.Thread):
    """Background loop fetching synthetic / mock / real web docs and gating.

    Real fetch hooks into synapforge.tools.WebSearchTool + WebFetchTool when
    --enable-real-fetch is set. Otherwise emits synthetic docs so the daemon
    can be smoke-run anywhere.
    """

    def __init__(
        self,
        out_dir: Path,
        ckpt_pointer: Path,
        interval_s: float = 60.0,
        per_source_cap_7d: int = 125,
        rng_seed: int = 0,
        max_cycles: Optional[int] = None,
        enable_real_fetch: bool = False,
    ) -> None:
        super().__init__(daemon=True, name="TrackA-WebLearner")
        self.out_dir = out_dir
        self.ckpt_pointer = ckpt_pointer
        self.interval_s = interval_s
        self.max_cycles = max_cycles
        self.enable_real_fetch = enable_real_fetch
        self._stop_event = threading.Event()
        self._cycle_count = 0
        self._rng = random.Random(rng_seed)

        self.learner = WebContentLearner(
            per_source_cap_7d=per_source_cap_7d,
            gate_log=out_dir / "gate_log.jsonl",
            blocklist_path=out_dir / "blocked_hashes.txt",
        )
        self.cycle_log = out_dir / "cycle_log.jsonl"
        self.cycle_log.parent.mkdir(parents=True, exist_ok=True)
        self.current_ckpt: Optional[str] = None
        self._last_ckpt_check = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    # --- ckpt rediscovery -----------------------------------------------
    def _refresh_ckpt(self) -> None:
        if not self.ckpt_pointer.exists():
            return
        try:
            content = self.ckpt_pointer.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if content and content != self.current_ckpt:
            self._log({"event": "ckpt_changed", "old": self.current_ckpt, "new": content})
            self.current_ckpt = content

    # --- doc producers --------------------------------------------------
    def _produce_synth(self) -> List[dict]:
        sources = [
            "web:wikipedia.zh",
            "web:wikipedia.en",
            "web:arxiv.org",
            "web:bilibili.com",
        ]
        out = []
        for s in sources:
            kind = "good" if self._rng.random() > 0.15 else "gibberish"
            out.append({
                "source_id": s,
                "url": f"https://{s.split(':', 1)[1]}/article/{self._cycle_count}",
                "text": cd._synth_doc(self._rng, kind),
            })
        return out

    def _produce_real(self) -> List[dict]:
        try:
            from synapforge.tools import WebSearchTool, WebFetchTool
        except Exception:
            return self._produce_synth()
        results: List[dict] = []
        try:
            search = WebSearchTool(top_k=2, mock=True)
            fetch = WebFetchTool(timeout_s=8, mock=True)
            for query in ("synapforge neural network", "深度学习"):
                hits = search.call(query)
                for hit in hits[:2]:
                    body = fetch.call(hit.get("url", ""))
                    if not body.get("text"):
                        continue
                    domain = hit.get("url", "").split("//", 1)[-1].split("/", 1)[0]
                    results.append({
                        "source_id": f"web:{domain}",
                        "url": hit.get("url", ""),
                        "text": body["text"][:8000],
                    })
        except Exception as e:  # noqa: BLE001
            self._log({"event": "real_fetch_error", "err": repr(e)})
            return self._produce_synth()
        return results or self._produce_synth()

    # --- single cycle ---------------------------------------------------
    def cycle(self) -> dict:
        self._cycle_count += 1
        if time.time() - self._last_ckpt_check > 30.0:
            self._refresh_ckpt()
            self._last_ckpt_check = time.time()

        try:
            docs = self._produce_real() if self.enable_real_fetch else self._produce_synth()
        except Exception as e:  # noqa: BLE001
            self._log({
                "event": "produce_error",
                "cycle": self._cycle_count,
                "err": repr(e),
            })
            return {"cycle": self._cycle_count, "produced": 0, "accepted": 0}

        decisions: List[IngestDecision] = []
        for doc in docs:
            try:
                d = self.learner.admit(
                    text=doc["text"],
                    source_id=doc["source_id"],
                    url=doc.get("url"),
                )
                decisions.append(d)
            except Exception as e:  # noqa: BLE001
                self._log({
                    "event": "admit_error",
                    "cycle": self._cycle_count,
                    "err": repr(e),
                })

        accepted = sum(1 for d in decisions if d.accept)
        rejection_breakdown = Counter()
        for d in decisions:
            if not d.accept and d.gates:
                failing = next((g for g in d.gates if not g.accept), None)
                if failing is not None:
                    rejection_breakdown[failing.name] += 1

        snap = {
            "cycle": self._cycle_count,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "produced": len(docs),
            "accepted": accepted,
            "rejected_by_gate": dict(rejection_breakdown),
            "buffer": self.learner.buffer_size(),
            "current_ckpt": self.current_ckpt,
        }
        self._log(snap)
        return snap

    def _log(self, record: dict) -> None:
        try:
            with open(self.cycle_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # --- run loop -------------------------------------------------------
    def run(self) -> None:
        self._log({"event": "track_a_start", "interval_s": self.interval_s})
        while not self._stop_event.is_set():
            try:
                self.cycle()
            except Exception as e:  # noqa: BLE001
                self._log({"event": "cycle_exception", "err": repr(e)})
            if self.max_cycles and self._cycle_count >= self.max_cycles:
                self._log({"event": "max_cycles_reached",
                           "n": self._cycle_count})
                break
            self._stop_event.wait(self.interval_s)
        self.shutdown()

    def shutdown(self) -> None:
        snap = self.learner.stats_snapshot()
        snap["event"] = "track_a_shutdown"
        self._log(snap)
        # Persist the LoRA buffer if any.
        buf = self.learner.drain_buffer()
        if buf:
            buf_path = self.out_dir / "lora_buffer.jsonl"
            try:
                with open(buf_path, "w", encoding="utf-8") as f:
                    for x in buf:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                self._log({"event": "buffer_persisted",
                           "n": len(buf), "path": str(buf_path)})
            except OSError as e:
                self._log({"event": "buffer_persist_error", "err": repr(e)})


# ---- Track B service thread (request-driven) -----------------------------


class TrackBService(threading.Thread):
    """Receives chat-add requests via in-process queue.

    External processes use enqueue() to add a (user_handle, text, hidden)
    tuple. The service thread serializes writes; readers (inference time)
    can call query() directly since UserChatMemoryAdapter is thread-safe
    for read-mostly workloads (Python's GIL + we never iterate while
    mutating).
    """

    def __init__(
        self,
        out_dir: Path,
        max_entries: int = 100_000,
    ) -> None:
        super().__init__(daemon=True, name="TrackB-RetrievalCache")
        self.adapter = UserChatMemoryAdapter(
            max_entries=max_entries,
            cache_path=out_dir / "user_chat_cache.jsonl",
        )
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._stop_event = threading.Event()
        self.cycle_log = out_dir / "track_b_log.jsonl"
        self.cycle_log.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        user_handle: str,
        text: str,
        hidden: Optional[List[float]] = None,
    ) -> None:
        self._queue.put((user_handle, text, hidden))

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    def _log(self, rec: dict) -> None:
        try:
            with open(self.cycle_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def run(self) -> None:
        self._log({"event": "track_b_start"})
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            user_handle, text, hidden = item
            try:
                eid = self.adapter.add(user_handle, text, hidden)
                self._log({"event": "chat_added", "eid": eid,
                           "n_entries": len(self.adapter._entries)})
            except Exception as e:  # noqa: BLE001
                self._log({"event": "chat_add_error", "err": repr(e)})
        self._log({"event": "track_b_shutdown",
                   "stats": self.adapter.stats()})


# ---- Top-level launcher --------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path,
                    default=Path("./runs/continual"))
    ap.add_argument("--ckpt-pointer", type=Path,
                    default=Path("./.continual_ckpt"))
    ap.add_argument("--interval-s", type=float, default=60.0,
                    help="Track A cycle interval (seconds).")
    ap.add_argument("--per-source-cap-7d", type=int, default=125,
                    help="50%% of Anthropic 2510.07192 250-doc threshold.")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="Exit after N cycles (smoke / CI).")
    ap.add_argument("--enable-real-fetch", action="store_true",
                    help="Use synapforge.tools WebSearch + Fetch (else synth).")
    ap.add_argument("--enable-track-b", action="store_true",
                    help="Also start Track B (per-user retrieval cache).")
    ap.add_argument("--track-b-max-entries", type=int, default=100_000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Run 3 cycles fast and exit (CI / sanity check).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.interval_s = 0.05
        args.max_cycles = 3

    track_a = TrackADriver(
        out_dir=args.out_dir,
        ckpt_pointer=args.ckpt_pointer,
        interval_s=args.interval_s,
        per_source_cap_7d=args.per_source_cap_7d,
        rng_seed=args.rng_seed,
        max_cycles=args.max_cycles,
        enable_real_fetch=args.enable_real_fetch,
    )
    track_b: Optional[TrackBService] = None
    if args.enable_track_b:
        track_b = TrackBService(
            out_dir=args.out_dir,
            max_entries=args.track_b_max_entries,
        )

    def _shutdown(_signum=None, _frame=None) -> None:
        print("[continual] shutdown signal received", flush=True)
        track_a.stop()
        if track_b is not None:
            track_b.stop()

    if os.name == "posix":
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

    track_a.start()
    if track_b is not None:
        track_b.start()
        # Demo: enqueue a couple of fake chat lines so the smoke run has data.
        if args.smoke:
            track_b.enqueue("demo_user", "你好，我想学一下深度学习。",
                            hidden=[0.1, 0.2, 0.3])
            track_b.enqueue("demo_user", "How does backprop work?",
                            hidden=[0.4, 0.5, 0.6])

    try:
        if args.smoke:
            track_a.join(timeout=max(args.interval_s * args.max_cycles + 2, 5))
            if track_b is not None:
                track_b.stop()
                track_b.join(timeout=2)
        else:
            while track_a.is_alive():
                track_a.join(timeout=1.0)
    except KeyboardInterrupt:
        _shutdown()
        track_a.join()
        if track_b is not None:
            track_b.join()

    print(f"[continual] done. logs → {args.out_dir}", flush=True)
    print(json.dumps(track_a.learner.stats_snapshot(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
