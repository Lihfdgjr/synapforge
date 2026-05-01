"""
RetrievalMemory: Track B per-user cache with **multi-user namespace isolation**.

Industry pattern (Claude Memory, ChatGPT Memory, character.ai):
  - User chat NEVER updates model weights
  - Each user gets isolated retrieval cache
  - At inference: retrieve top-K relevant memories, prepend to context
  - User can delete their memory at any time (compliance + safety)

Storage layout (zero-VRAM, CPU/disk only):
  ~/.synapforge/memory/<user_id>/
        log.jsonl       conversation history (append-only)
        index.bin       lightweight HNSW-style hashed dense cache
                        (we ship a numpy/dict fallback so the test
                        suite works without faiss / hnswlib)
        prefs.json      learned preferences {key: value}

Backward compatibility:
  - Legacy callers used `add(user_hash, text, sample_id)` with cache files
    `<user_hash>.jsonl` in `cache_dir`. We preserve that shape via the
    `_legacy` namespace whose log file is the original `<user_hash>.jsonl`
    so existing run logs and continual_daemon never need to migrate.
  - New API uses `mem = RetrievalMemory(user_id="alice")` plus per-user
    `add(text, ...)`, `query(text, ...)`, `purge(user_id=...)`.

Why this is the right pattern (per agent synthesis 2026-04-30):
  - 4/5 adversarial attack classes are white-box-bypassable on weights
  - Frozen-weight retrieval sidesteps gradient attacks entirely
  - Compliance-friendly (user can delete)
  - Cheap (no continual training cost per user)
  - Multi-user safe: filesystem-perm isolation, namespace per user,
    `purge("alice")` cannot collide with bob's data

Threat model:
  - User A may NEVER access user B's memory. Enforced at the API surface
    (every read/write requires a `user_id`) and at the filesystem layer
    (each user has its own directory; OS perms can be set per directory).
  - The default `user_id="default"` is for single-user / unit-test mode.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Per-user namespace
# ---------------------------------------------------------------------------

class PerUserMemory:
    """One namespace = one user's memory.

    Holds:
      - Conversation log (jsonl, recency-weighted)
      - Learned preference store (prefs.json)
      - Tiny dense index header (index.bin, optional, just a marker today —
        the lookup falls back to lexical+recency, but the file is reserved
        so a real HNSW/FAISS index can drop in without any API changes).

    All on CPU/disk. Zero VRAM impact.
    """

    def __init__(
        self,
        user_id: str,
        root_dir: Path,
        recency_days: int = 30,
        max_entries: int = 1000,
        legacy_log_path: Optional[Path] = None,
    ) -> None:
        self.user_id = user_id
        self.recency_days = recency_days
        self.max_entries = max_entries

        # When legacy_log_path is set we use the historical
        # `<cache_dir>/<user_hash>.jsonl` location for backward compat.
        if legacy_log_path is not None:
            self.dir = legacy_log_path.parent
            self.log_path = legacy_log_path
            self.prefs_path = legacy_log_path.with_suffix(".prefs.json")
            self.index_path = legacy_log_path.with_suffix(".index.bin")
        else:
            self.dir = root_dir / user_id
            self.dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.dir / "log.jsonl"
            self.prefs_path = self.dir / "prefs.json"
            self.index_path = self.dir / "index.bin"

        # Touch index.bin so the storage layout is observable to tooling
        # (and future HNSW writers). Keep it 0 bytes if no embedder ran.
        if not self.index_path.exists():
            try:
                self.index_path.touch()
            except OSError:
                pass

    # ------- writes -------

    def add(self, text: str, sample_id: int = 0, meta: Optional[dict] = None) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text,
            "sample_id": sample_id,
        }
        if meta:
            entry["meta"] = meta
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def set_pref(self, key: str, value) -> None:
        prefs = self._load_prefs()
        prefs[key] = value
        with open(self.prefs_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)

    def _load_prefs(self) -> dict:
        if not self.prefs_path.exists():
            return {}
        try:
            with open(self.prefs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def get_pref(self, key: str, default=None):
        return self._load_prefs().get(key, default)

    def all_prefs(self) -> dict:
        return self._load_prefs()

    # ------- reads -------

    def _iter_entries(self):
        if not self.log_path.exists():
            return
        cutoff_s = time.time() - self.recency_days * 86400
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    ts_s = time.mktime(time.strptime(e["ts"], "%Y-%m-%dT%H:%M:%S"))
                except (ValueError, KeyError):
                    continue
                if ts_s < cutoff_s:
                    continue
                yield e, ts_s

    def query(self, query_text: str, top_k: int = 4) -> List[dict]:
        """Recency-weighted lexical-overlap query.

        A real impl uses sentence-embedding cosine; we keep the lexical fallback
        so the test suite (and CPU-only investor demo) work with zero deps.
        """
        memories = list(self._iter_entries())
        if not memories:
            return []
        q_words = set(query_text.lower().split())
        scored = []
        now = time.time()
        for e, ts_s in memories:
            text = e["text"]
            t_words = set(text.lower().split())
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            recency = max(0.0, 1.0 - (now - ts_s) / (self.recency_days * 86400))
            score = 0.7 * overlap + 0.3 * recency
            scored.append((score, e))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in scored[:top_k]]

    def lookup(self, query_text: str, top_k: int = 4) -> List[dict]:
        """Alias for `query` — matches the API in the spec."""
        return self.query(query_text, top_k=top_k)

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # ------- destructive -------

    def purge(self) -> bool:
        """Delete all data for this user.

        Returns True iff actual user *data* (log entries or preferences)
        existed before deletion. The empty index.bin marker file is
        ignored so that calling purge() twice in a row reports
        ``True`` then ``False``.
        """
        had_data = self.log_path.exists() or self.prefs_path.exists()
        for p in (self.log_path, self.prefs_path, self.index_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        # Remove the per-user directory if it's our own (not legacy flat layout)
        if self.dir.exists() and self.dir.name == self.user_id:
            try:
                # Only rmdir if empty — never recursive-delete user data we
                # didn't expect.
                if not any(self.dir.iterdir()):
                    self.dir.rmdir()
            except OSError:
                pass
        return had_data


# ---------------------------------------------------------------------------
# RetrievalMemory: top-level multi-user store
# ---------------------------------------------------------------------------

DEFAULT_USER_ID = "default"


def _default_root_dir() -> Path:
    """Storage root: $SYNAPFORGE_MEMORY_HOME or ~/.synapforge/memory."""
    env = os.environ.get("SYNAPFORGE_MEMORY_HOME")
    if env:
        return Path(env)
    return Path.home() / ".synapforge" / "memory"


class RetrievalMemory:
    """Multi-user retrieval cache.

    Two construction modes:

      1. Single-user (default-namespace) mode:
            mem = RetrievalMemory(user_id="alice")
            mem.add("我喜欢猫")
            mem.query("你记得我喜欢什么吗")

      2. Multi-tenant mode (one instance per process serving many users):
            mem = RetrievalMemory()           # user_id="default"
            mem.add("hi", user_id="alice")
            mem.query("hi", user_id="bob")

    Either mode keeps user data in `<root_dir>/<user_id>/{log.jsonl,
    prefs.json, index.bin}` so users CANNOT see each other.

    Backward compatibility:
      - The legacy positional arg `cache_dir` still works. When supplied,
        we keep the historical flat layout `<cache_dir>/<user_hash>.jsonl`
        for the bound user_hash and only nest fresh users.
      - The legacy 3-arg form `mem.add(user_hash, text, sample_id)` is
        still accepted (string-typed first arg = user_hash).
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        recency_days: int = 30,
        max_per_user: int = 1000,
        user_id: str = DEFAULT_USER_ID,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        # Resolve root_dir. If `cache_dir` was passed (legacy positional arg),
        # we treat it as the root directory of the per-user namespaces.
        if root_dir is not None:
            self.root_dir = Path(root_dir)
        elif cache_dir is not None:
            self.root_dir = Path(cache_dir)
        else:
            self.root_dir = _default_root_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.recency_days = recency_days
        self.max_per_user = max_per_user
        self.default_user_id = user_id

        # Per-user namespace cache (one PerUserMemory per user_id).
        self._users: Dict[str, PerUserMemory] = {}

        # Eagerly construct the bound user so single-user code paths
        # have a working namespace immediately.
        self._get_or_create(user_id)

    # ------------------------------------------------------------------
    # Backward-compat property for legacy callers that referenced
    # `mem.cache_dir`. New code should use `mem.root_dir`.
    # ------------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:  # noqa: D401 — legacy alias
        return self.root_dir

    # ------------------------------------------------------------------
    # Namespace resolution
    # ------------------------------------------------------------------

    def _resolve_uid(self, user_id: Optional[str]) -> str:
        if user_id is None:
            return self.default_user_id
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")
        if "/" in user_id or "\\" in user_id or user_id in ("..", "."):
            raise ValueError(f"user_id must not contain path separators: {user_id!r}")
        return user_id

    def _get_or_create(self, user_id: str) -> PerUserMemory:
        uid = self._resolve_uid(user_id)
        if uid not in self._users:
            # Detect legacy flat-layout file: cache_dir/<uid>.jsonl
            legacy = self.root_dir / f"{uid}.jsonl"
            if legacy.exists() and not (self.root_dir / uid).exists():
                self._users[uid] = PerUserMemory(
                    user_id=uid,
                    root_dir=self.root_dir,
                    recency_days=self.recency_days,
                    max_entries=self.max_per_user,
                    legacy_log_path=legacy,
                )
            else:
                self._users[uid] = PerUserMemory(
                    user_id=uid,
                    root_dir=self.root_dir,
                    recency_days=self.recency_days,
                    max_entries=self.max_per_user,
                )
        return self._users[uid]

    def for_user(self, user_id: str) -> PerUserMemory:
        """Return (or create) the namespace object for `user_id`."""
        return self._get_or_create(user_id)

    # ------------------------------------------------------------------
    # New API (user_id-scoped)
    # ------------------------------------------------------------------

    def add(
        self,
        *args,
        user_id: Optional[str] = None,
        text: Optional[str] = None,
        sample_id: int = 0,
        meta: Optional[dict] = None,
    ) -> None:
        """Append a memory entry.

        Three call patterns supported:

          mem.add("text")                         # bound default user
          mem.add("text", user_id="alice")        # explicit namespace
          mem.add("alice", "text", 17)            # LEGACY 3-arg form
                                                  # (first positional == user_hash)
        """
        # Legacy positional form: add(user_hash, text, sample_id)
        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], str):
            uid, text_arg, sid = args
            self._get_or_create(uid).add(text_arg, sample_id=int(sid), meta=meta)
            return

        # Legacy 2-arg: add(user_hash, text)
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            uid, text_arg = args
            self._get_or_create(uid).add(text_arg, sample_id=sample_id, meta=meta)
            return

        # New API: add("text", user_id=..., sample_id=...)
        if len(args) == 1 and isinstance(args[0], str):
            text_arg = args[0]
        elif text is not None:
            text_arg = text
        else:
            raise TypeError("add() requires at least a text string")
        self._get_or_create(self._resolve_uid(user_id)).add(
            text_arg, sample_id=sample_id, meta=meta
        )

    def query(
        self,
        *args,
        user_id: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 4,
    ) -> List[dict]:
        """Retrieve top-k memories from a user's namespace.

        Call patterns:

          mem.query("hello")                       # bound default user
          mem.query("hello", user_id="alice")      # explicit namespace
          mem.query("alice", "hello", 4)           # LEGACY 3-arg form
        """
        # Legacy: query(user_hash, query_text, top_k)
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            uid = args[0]
            q = args[1]
            k = int(args[2]) if len(args) >= 3 else top_k
            return self._get_or_create(uid).query(q, top_k=k)

        if len(args) == 1 and isinstance(args[0], str):
            q = args[0]
        elif query_text is not None:
            q = query_text
        else:
            raise TypeError("query() requires a query string")

        return self._get_or_create(self._resolve_uid(user_id)).query(q, top_k=top_k)

    def lookup(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        top_k: int = 4,
    ) -> List[dict]:
        """Alias for `query` matching the API spec."""
        return self.query(query_text, user_id=user_id, top_k=top_k)

    def purge(self, user_id: Optional[str] = None) -> bool:
        """Delete a single user's memory (default = the bound user)."""
        uid = self._resolve_uid(user_id)
        ns = self._get_or_create(uid)
        ok = ns.purge()
        # Drop from cache so next call rebuilds a clean namespace
        self._users.pop(uid, None)
        return ok

    # Backward-compat alias used by continual_daemon and legacy tests.
    def delete_user(self, user_hash: str) -> bool:
        return self.purge(user_id=user_hash)

    # ------------------------------------------------------------------
    # Preferences (learned facts) — first-class per-user store
    # ------------------------------------------------------------------

    def set_pref(self, key: str, value, user_id: Optional[str] = None) -> None:
        self._get_or_create(self._resolve_uid(user_id)).set_pref(key, value)

    def get_pref(self, key: str, default=None, user_id: Optional[str] = None):
        return self._get_or_create(self._resolve_uid(user_id)).get_pref(key, default)

    def prefs(self, user_id: Optional[str] = None) -> dict:
        return self._get_or_create(self._resolve_uid(user_id)).all_prefs()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_users(self) -> List[str]:
        """Discover all users with persisted memory under root_dir."""
        users = set()
        if not self.root_dir.exists():
            return []
        for p in self.root_dir.iterdir():
            if p.is_dir():
                users.add(p.name)
            elif p.suffix == ".jsonl":  # legacy flat layout
                users.add(p.stem)
        return sorted(users)

    def stats(self) -> dict:
        out = {"users": 0, "total_memories": 0, "root_dir": str(self.root_dir)}
        seen = set()
        # Per-user dirs
        for p in self.root_dir.glob("*/log.jsonl"):
            uid = p.parent.name
            if uid in seen:
                continue
            seen.add(uid)
            out["users"] += 1
            try:
                out["total_memories"] += sum(1 for _ in open(p, "r", encoding="utf-8"))
            except OSError:
                pass
        # Legacy flat-layout files
        for p in self.root_dir.glob("*.jsonl"):
            uid = p.stem
            if uid in seen:
                continue
            seen.add(uid)
            out["users"] += 1
            try:
                out["total_memories"] += sum(1 for _ in open(p, "r", encoding="utf-8"))
            except OSError:
                pass
        return out
