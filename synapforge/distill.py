"""sf.distill — knowledge distillation for synapforge.

Three pieces:

  * TeacherCache       SQLite-backed cache for teacher logits on disk
  * DistillationLoss   KL divergence with temperature + optional CE blend
  * DistillTrainer     wraps an optimiser to step on (CE, KD) jointly

This is the *local-tensor* form of distillation — the teacher is a function
``(input) -> logits``. For OpenAI-API teachers see the optional
``mscfc.distill`` ensemble; that path requires real API keys.

bf16-friendly: KL is always computed in fp32. Cache stores tensors as
contiguous float32 numpy bytes.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. TeacherCache
# ---------------------------------------------------------------------------


_DEFAULT_CACHE_PATH = os.environ.get(
    "SYNAPFORGE_TEACHER_CACHE_PATH",
    str(Path.home() / ".cache" / "synapforge" / "teacher_cache.sqlite"),
)
_DEFAULT_MAX_ROWS = int(os.environ.get("SYNAPFORGE_TEACHER_CACHE_MAX_ROWS", "50000"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS teacher_logits (
    key       TEXT PRIMARY KEY,
    teacher   TEXT NOT NULL,
    ts        REAL NOT NULL,
    blob      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ts ON teacher_logits(ts);
"""


class TeacherCache:
    """Thread-safe SQLite cache for cached teacher logits.

    Key = (teacher_name, sha256(input_bytes)). Value = pickled tensor (CPU).
    On a miss the caller computes logits and ``put()`` them. Errors fall
    through to "miss" so a corrupted DB never breaks training.
    """

    def __init__(
        self,
        path: str | None = None,
        max_rows: int = _DEFAULT_MAX_ROWS,
    ) -> None:
        self.path = path or _DEFAULT_CACHE_PATH
        self.max_rows = int(max_rows)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.errors = 0
        self._conn: sqlite3.Connection | None = None
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self.path,
                check_same_thread=False,
                isolation_level=None,
                timeout=5.0,
            )
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.Error:
                pass
        except Exception:
            self.errors += 1
            self._conn = None

    @staticmethod
    def _make_key(teacher: str, x: torch.Tensor) -> str:
        b = x.detach().cpu().contiguous().float().numpy().tobytes()
        h = hashlib.sha256(b).hexdigest()[:32]
        return f"{teacher}|{tuple(x.shape)}|{h}"

    def get(self, teacher: str, x: torch.Tensor) -> torch.Tensor | None:
        if self._conn is None:
            return None
        key = self._make_key(teacher, x)
        try:
            with self.lock:
                row = self._conn.execute(
                    "SELECT blob FROM teacher_logits WHERE key=?", (key,)
                ).fetchone()
        except Exception:
            self.errors += 1
            return None
        if row is None:
            self.misses += 1
            return None
        try:
            t = pickle.loads(row[0])
        except Exception:
            self.errors += 1
            return None
        self.hits += 1
        return t

    def put(self, teacher: str, x: torch.Tensor, logits: torch.Tensor) -> None:
        if self._conn is None:
            return
        key = self._make_key(teacher, x)
        try:
            blob = pickle.dumps(logits.detach().cpu(), protocol=4)
        except Exception:
            self.errors += 1
            return
        try:
            with self.lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO teacher_logits(key, teacher, ts, blob) "
                    "VALUES (?,?,?,?)",
                    (key, teacher, time.time(), blob),
                )
                self.stores += 1
                if self.stores % 500 == 0:
                    self._evict_locked()
        except Exception:
            self.errors += 1

    def _evict_locked(self) -> None:
        if self._conn is None:
            return
        try:
            cur = self._conn.execute("SELECT COUNT(*) FROM teacher_logits")
            (n,) = cur.fetchone()
            if n <= self.max_rows:
                return
            excess = n - self.max_rows
            self._conn.execute(
                "DELETE FROM teacher_logits WHERE key IN "
                "(SELECT key FROM teacher_logits ORDER BY ts ASC LIMIT ?)",
                (excess,),
            )
        except Exception:
            self.errors += 1

    def stats(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "errors": self.errors,
            "enabled": self._conn is not None,
        }


# ---------------------------------------------------------------------------
# 2. DistillationLoss
# ---------------------------------------------------------------------------


class DistillationLoss(nn.Module):
    """Hinton-style KL distillation with temperature.

        L = alpha * KL(softmax(student/T) || softmax(teacher/T)) * T^2
            + (1 - alpha) * CE(student, hard_target)

    If ``hard_target`` is None, only the KL term is used.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s = student_logits.float()
        t = teacher_logits.float()
        if s.shape != t.shape:
            vmin = min(s.shape[-1], t.shape[-1])
            s = s[..., :vmin]
            t = t[..., :vmin]
        T = self.temperature
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        if hard_target is None:
            return kl
        ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), hard_target.reshape(-1))
        return self.alpha * kl + (1.0 - self.alpha) * ce


# ---------------------------------------------------------------------------
# 3. DistillTrainer
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    teacher_name: str = "default"
    temperature: float = 4.0
    alpha: float = 0.7
    use_cache: bool = True
    cache_path: str | None = None


class DistillTrainer:
    """Wraps an outer optimiser; runs CE+KD step.

    Caller passes ``teacher`` as a callable ``(x) -> logits`` and the
    optimiser. ``train_step(x, y)`` does:

        with cache: t_logits = teacher(x)  # cached
        s_logits = student(x)
        loss = DistillationLoss()(s_logits, t_logits, y)
        loss.backward(); opt.step()
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: Callable[[torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        cfg: DistillConfig | None = None,
        defense: object | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.cfg = cfg or DistillConfig()
        self.cache = (
            TeacherCache(path=self.cfg.cache_path) if self.cfg.use_cache else None
        )
        self.loss_fn = DistillationLoss(self.cfg.temperature, self.cfg.alpha)
        self.defense = defense

    def _teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache is not None:
            cached = self.cache.get(self.cfg.teacher_name, x)
            if cached is not None:
                return cached.to(x.device)
        with torch.no_grad():
            try:
                t = self.teacher(x)
            except TypeError:
                t = self.teacher(tokens=x)  # type: ignore[arg-type]
            if isinstance(t, tuple):
                t = t[0]
        if self.cache is not None:
            self.cache.put(self.cfg.teacher_name, x, t)
        return t

    def _student_logits(self, x: torch.Tensor) -> torch.Tensor:
        try:
            s = self.student(tokens=x)
        except TypeError:
            s = self.student(x)
        if isinstance(s, tuple):
            s = s[0]
        return s

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        t_logits = self._teacher_logits(x)
        s_logits = self._student_logits(x)
        loss = self.loss_fn(s_logits, t_logits, hard_target=y)
        if self.defense is not None:
            self.defense.pre_step()
        loss.backward()
        if self.defense is not None:
            self.defense.after_grads()
        self.optimizer.step()
        if self.defense is not None:
            if self.defense.regressed_and_rollback():
                pass
            self.defense.tick()
        return float(loss.detach())


__all__ = [
    "TeacherCache",
    "DistillationLoss",
    "DistillConfig",
    "DistillTrainer",
]
