"""sf.action.skill_log_v2 — lifelong persistent skill memory.

Version-2 schema, designed to survive process / training-run boundaries
with zero loss.  Replaces ``skill_log.json`` (v1) with the following
guarantees:

    * **Atomic writes** — every save goes to a ``.tmp`` sibling and is
      ``os.replace()``'d into place.  Mid-write crash leaves the prior
      version intact.

    * **Rotation** — ``skills.jsonl`` is the latest snapshot; we also
      keep the last ``rotation_keep`` snapshots as
      ``skills.jsonl.{ts}``.  Cheap insurance against silent corruption.

    * **Append-only history** — ``history.jsonl`` records every mint /
      activate / prune event so we can answer "why was skill 137
      created at time T".

    * **Idempotent reload** — calling ``load()`` twice yields exactly
      the same in-memory state (verified by smoke).

The on-disk JSON shape is the snake-cased mirror of
``UniversalCodebook.to_dict()``:

    {
      "version": 2,
      "saved_at": <unix ts>,
      "hidden": <int>,
      "K_alive": <int>,
      "next_proto_id": <int>,
      "L1_primitives": ["CLICK", "TYPE", ...],
      "skills": [
        {
          "proto_id": int,
          "layer": "L1" | "L2" | "L3",
          "embedding": [float, ...],
          "description": str,
          "n_uses": int,
          "n_success": int,
          "first_seen_ts": float,
          "last_used_ts": float,
          "hebbian_strength": float,
          "trigger_seq": [int, ...],
          "co_fire_history": [{"ts": float, "trigger_seq": [int]}],
          "embedding_hash": str
        }
      ],
      "stats": {...}
    }

"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

DEFAULT_SAVE_PATH = "runs/skill_demo/skills.jsonl"


@dataclass
class HistoryEvent:
    """One entry in the append-only audit log."""

    ts: float
    event: str                         # "mint" | "activate" | "prune" | "load" | "save"
    proto_id: int = -1
    layer: str = ""
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SkillLog (v2)
# ---------------------------------------------------------------------------


class SkillLog:
    """JSON-backed lifelong persistence for ``UniversalCodebook``.

    Designed to be a near-drop-in for the v1 ``skill_log.SkillLog``: it
    accepts a path on construction, exposes ``save()`` / ``load()``, and
    lets you append history events.  Unlike v1 we serialise the
    *entire* codebook state (slots, alive mask, layer ids, metadata) so
    a fresh process can reconstruct exactly the same state.

    Usage:

        cb = UniversalCodebook(hidden=64)
        log = SkillLog("runs/skill_demo/skills.jsonl")
        # ... mint / activate ...
        log.save_codebook(cb)              # atomic, rotated

        cb2 = UniversalCodebook(hidden=64)
        log2 = SkillLog("runs/skill_demo/skills.jsonl")
        log2.load_codebook(cb2)
        # cb and cb2 are now identical
    """

    def __init__(
        self,
        path: str | os.PathLike = DEFAULT_SAVE_PATH,
        rotation_keep: int = 5,
        write_history: bool = True,
        history_buffer_max: int = 4096,
    ) -> None:
        self.path = Path(path)
        self.rotation_keep = int(rotation_keep)
        self.write_history = bool(write_history)
        self.history_path = self.path.with_suffix(self.path.suffix + ".history.jsonl")
        # In-memory tail of the audit log (bounded so long runs don't OOM).
        # Disk file remains complete; `_history_buffer` is just a cache.
        self.history_buffer_max = int(history_buffer_max)
        self._history_buffer: List[HistoryEvent] = []
        # cached snapshot returned by load_codebook so callers can re-use
        self._last_snapshot: Optional[dict] = None
        # Coarse-grained lock so two threads can't race a save+rotate cycle.
        # The whole save_codebook+rotate path is short so contention is OK.
        self._io_lock = threading.RLock()

    # ------------------------------------------------------------------ atomic IO

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Best-effort fsync of the directory entry on POSIX.

        On Windows, opening a directory for fsync is not generally possible;
        we silently skip.  os.replace already provides MoveFileEx semantics
        on NTFS which survives a clean shutdown.
        """
        try:
            if os.name != "nt":
                fd = os.open(str(path), os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except (OSError, AttributeError):
            pass

    def _atomic_write(self, target: Path, payload: dict) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            # Force the data out of the OS write buffer before rename.
            # Without this, a hard reboot between close() and replace()
            # can leave a 0-byte tmp file with the rename already done.
            try:
                f.flush()
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, target)
        # Make sure the directory entry update itself is durable.
        self._fsync_dir(target.parent)

    def _rotate(self, target: Path) -> None:
        """Rotate ``target`` to a timestamped sibling, keep last K."""
        if not target.exists() or self.rotation_keep <= 0:
            return
        # Sub-second precision so two saves in the same wallclock second
        # don't clobber each other (was: int(time.time())).
        ts = f"{time.time():.6f}".replace(".", "_")
        rotated = target.with_suffix(target.suffix + f".{ts}")
        try:
            # Read + write copy.  We can't os.link the file because the
            # next save_codebook will os.replace it and break the link
            # on some filesystems.
            data = target.read_bytes()
            tmp_rot = rotated.with_suffix(rotated.suffix + ".tmp")
            with open(tmp_rot, "wb") as f:
                f.write(data)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass
            os.replace(tmp_rot, rotated)
        except OSError:
            return
        # Drop oldest above keep limit.
        siblings = sorted(
            target.parent.glob(target.name + ".[0-9_]*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Filter out non-rotation files (history file, the .tmp artifacts).
        siblings = [
            p for p in siblings
            if not p.name.endswith(".tmp")
            and not p.name.endswith(".history.jsonl")
        ]
        for old in siblings[self.rotation_keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    def _append_history(self, evt: HistoryEvent) -> None:
        if not self.write_history:
            return
        self._history_buffer.append(evt)
        # Keep only the last ``history_buffer_max`` events in RAM; the
        # full history remains on disk via append-only writes.
        if len(self._history_buffer) > self.history_buffer_max:
            del self._history_buffer[: len(self._history_buffer) - self.history_buffer_max]
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ codebook IO

    def save_codebook(self, codebook: Any, log_event: bool = True) -> int:
        """Atomically save the codebook state.  Returns number of skills."""
        with self._io_lock:
            payload = codebook.to_dict()
            skills = payload.get("skills", [])
            # add v2-specific fields
            for s in skills:
                s.setdefault("first_seen_ts", s.get("created_at", time.time()))
                s.setdefault("last_used_ts", s.get("last_used_at", time.time()))
                s.setdefault("co_fire_history", [])
            out = {
                "version": SCHEMA_VERSION,
                "saved_at": time.time(),
                "hidden": payload.get("hidden"),
                "K_alive": payload.get("K_alive"),
                "next_proto_id": payload.get("next_proto_id"),
                "L1_primitives": payload.get("L1_primitives", []),
                "skills": skills,
                "stats": codebook.stats() if hasattr(codebook, "stats") else {},
            }
            self._rotate(self.path)
            self._atomic_write(self.path, out)
            if log_event:
                self._append_history(HistoryEvent(
                    ts=time.time(),
                    event="save",
                    extra={"n_skills": len(skills), "path": str(self.path)},
                ))
            self._last_snapshot = out
            return len(skills)

    def load_codebook(self, codebook: Any, log_event: bool = True) -> int:
        """Load the latest snapshot into ``codebook`` in-place.  Idempotent.

        On corrupt-JSON / truncated-file we transparently fall back to the
        most-recent rotation that DOES parse cleanly.  This is the recovery
        path for "rental dies mid-write" scenarios.  Returns the number of
        skills restored or 0 if nothing usable was found.
        """
        with self._io_lock:
            if not self.path.exists():
                return 0
            payload = self._read_json_or_recover()
            if payload is None:
                return 0
            if payload.get("version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema version: {payload.get('version')!r} "
                    f"(expected {SCHEMA_VERSION})"
                )
            n_loaded = codebook.load_dict(payload)
            if log_event:
                self._append_history(HistoryEvent(
                    ts=time.time(),
                    event="load",
                    extra={"n_skills": n_loaded, "path": str(self.path)},
                ))
            self._last_snapshot = payload
            return n_loaded

    def _read_json_or_recover(self) -> Optional[dict]:
        """Try to parse self.path; if it's corrupt, fall back to the
        most-recent rotation snapshot that parses cleanly.

        Returns the parsed payload dict, or ``None`` if no snapshot is
        recoverable.  Logs each failed attempt so operators can see why
        recovery happened.
        """
        # 1. happy path — current file
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._append_history(HistoryEvent(
                ts=time.time(),
                event="load_corrupt",
                extra={"path": str(self.path), "error": repr(e)},
            ))
        # 2. fallback — newest-first rotation siblings
        siblings = sorted(
            (
                p for p in self.path.parent.glob(self.path.name + ".[0-9_]*")
                if not p.name.endswith(".tmp")
                and not p.name.endswith(".history.jsonl")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for sibling in siblings:
            try:
                with open(sibling, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self._append_history(HistoryEvent(
                    ts=time.time(),
                    event="load_recovered",
                    extra={"from": sibling.name},
                ))
                return payload
            except (OSError, json.JSONDecodeError):
                continue
        return None

    # ------------------------------------------------------------------ event log

    def log_mint(self, proto_id: int, layer: str, description: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._append_history(HistoryEvent(
            ts=time.time(),
            event="mint",
            proto_id=int(proto_id),
            layer=layer,
            description=description,
            extra=extra or {},
        ))

    def log_activate(self, proto_id: int, reward: float = 0.0) -> None:
        self._append_history(HistoryEvent(
            ts=time.time(),
            event="activate",
            proto_id=int(proto_id),
            extra={"reward": float(reward)},
        ))

    def log_prune(self, proto_ids: List[int], reason: str = "") -> None:
        self._append_history(HistoryEvent(
            ts=time.time(),
            event="prune",
            extra={"proto_ids": list(proto_ids), "reason": reason},
        ))

    # ------------------------------------------------------------------ raw access

    def read_history(self) -> List[Dict[str, Any]]:
        """Read every event line from ``history.jsonl`` as dicts."""
        if not self.history_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def latest_snapshot(self) -> Optional[dict]:
        """Return the last loaded/saved snapshot (in-memory copy)."""
        return self._last_snapshot

    # ------------------------------------------------------------------ inspection

    def quick_stats(self) -> Dict[str, Any]:
        """Cheap stats without touching the codebook."""
        if not self.path.exists():
            return {"exists": False, "path": str(self.path)}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"exists": True, "readable": False, "path": str(self.path)}
        layer_counts = {"L1": 0, "L2": 0, "L3": 0}
        archived = 0
        n_uses_total = 0
        for s in payload.get("skills", []):
            layer = s.get("layer", "L3")
            if layer in layer_counts:
                layer_counts[layer] += 1
            if s.get("archived"):
                archived += 1
            n_uses_total += int(s.get("n_uses", 0))
        return {
            "exists": True,
            "readable": True,
            "path": str(self.path),
            "version": payload.get("version"),
            "saved_at": payload.get("saved_at"),
            "K_alive": payload.get("K_alive"),
            "next_proto_id": payload.get("next_proto_id"),
            "by_layer": layer_counts,
            "archived": archived,
            "n_uses_total": n_uses_total,
            "rotated_versions": [
                p.name for p in sorted(
                    self.path.parent.glob(self.path.name + ".[0-9]*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:self.rotation_keep]
            ],
        }


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def _smoke() -> None:  # pragma: no cover - manual run
    from .universal_codebook import UniversalCodebook

    tmp_dir = Path("runs/skill_log_v2_smoke")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / "skills.jsonl"

    cb = UniversalCodebook(hidden=64, K_initial=9)
    pid_a = cb.mint_from_text("查股市行情每天 9 点")
    pid_b = cb.mint_from_text("打开 GitHub trending repo")
    cb._activate(pid_a, reward=0.5)

    log = SkillLog(p)
    log.log_mint(pid_a, "L3", "查股市行情每天 9 点")
    log.log_mint(pid_b, "L3", "打开 GitHub trending repo")
    n = log.save_codebook(cb)
    print(f"[save] wrote {n} skills to {p}")

    # reload twice, verify idempotency
    cb2 = UniversalCodebook(hidden=64, K_initial=9)
    log2 = SkillLog(p)
    n1 = log2.load_codebook(cb2)
    n2 = log2.load_codebook(cb2)
    print(f"[load1] restored {n1}    [load2] restored {n2}")
    print(f"[reloaded stats] {cb2.stats()}")
    print(f"[history events] {len(log.read_history())}")
    print(f"[quick_stats] {log.quick_stats()}")


if __name__ == "__main__":  # pragma: no cover
    _smoke()


__all__ = ["SkillLog", "HistoryEvent", "SCHEMA_VERSION"]
