"""Web daemon -> parquet sink for trainer ingestion.

The continual-learning daemon (synapforge.learn.continual_daemon /
scripts/launch_continual_daemon.py) admits filtered web docs through
the 7-gate pipeline and buffers them in memory. This module persists
that buffer to a *rolling* parquet shard at:

    <out_dir>/web_<NNNN>.parquet

with rotation when EITHER

  * the file is older than ``rotate_seconds`` (default 3600 = 1 hour), OR
  * the file exceeds ``rotate_max_rows`` rows (default 50_000), OR
  * the file exceeds ``rotate_max_bytes`` bytes (default 200 MiB)

whichever comes first. Each row carries:

  * ``text`` (str)         -- the gated content
  * ``source_id`` (str)    -- e.g. ``web:wikipedia.zh``
  * ``url`` (str)          -- nullable; URL of fetch
  * ``ingest_ts`` (float)  -- unix seconds at admit-time
  * ``quality_score`` (float)  -- min(g.score for g in gates) ∈ [0, 1]
  * ``gate_scores`` (str)  -- JSON dict of per-gate {name -> score}

The trainer's ``--data-files`` accepts a glob like
``/workspace/data/web_self_learn/web_*.parquet:0.10`` so the rolling
shards plug into ``ParquetTokenStream(files_with_weights=...)`` at low
weight. Pollution defence (per ``feedback_continual_vs_poison_balance.md``):

  * Per-source 7-day cap of 125 docs (50% of Anthropic 250 threshold)
    is already enforced by ``WebContentLearner``.
  * The 7 gate scores are persisted; the trainer can additionally drop
    rows with ``quality_score < 0.5`` via a thin filter.
  * The shard glob is at LOW weight (0.10) so even if the gate fails
    open the trainer's quality is dominated by the curated 90%.

Public API:

    >>> sink = WebDaemonSink(out_dir="/workspace/data/web_self_learn")
    >>> sink.append(text="...", source_id="web:wikipedia.zh", url="...",
    ...             quality_score=0.85,
    ...             gate_scores={"G1_source_trust": 0.6, ...})
    >>> sink.flush()  # explicit; also called on close()
    >>> sink.close()

Smoke-runnable on Windows-dev (no torch needed; pyarrow only).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as _exc:  # pragma: no cover - env without pyarrow
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _PYARROW_IMPORT_EXC: Optional[BaseException] = _exc
else:
    _PYARROW_IMPORT_EXC = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# pyarrow schema kept module-level so the trainer can mirror it when
# reading the rolling shards. Columns intentionally include the full
# gate-score breakdown so a downstream pollution audit can trace
# "this row got admitted because gate X scored 0.92".
_SCHEMA: Optional["pa.Schema"]
if pa is not None:
    _SCHEMA = pa.schema([
        ("text", pa.string()),
        ("source_id", pa.string()),
        ("url", pa.string()),
        ("ingest_ts", pa.float64()),
        ("quality_score", pa.float32()),
        ("gate_scores", pa.string()),  # JSON-encoded {name: score}
    ])
else:  # pragma: no cover - env without pyarrow
    _SCHEMA = None


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


@dataclass
class WebDaemonSinkConfig:
    out_dir: Path
    rotate_seconds: float = 3600.0    # 1 hour default
    rotate_max_rows: int = 50_000
    rotate_max_bytes: int = 200 * 1024 * 1024  # 200 MiB
    write_batch_size: int = 256        # rows per parquet RowGroup write
    file_prefix: str = "web_"
    file_pad: int = 4                  # web_0001.parquet
    quality_score_floor: float = 0.0   # rows below this never written


class WebDaemonSink:
    """Append + rotate parquet shard sink.

    Thread-safe (the daemon runs gating + admit on a single thread; this
    sink is touched from at most one writer thread at a time, but we
    still hold a lock around the buffer mutations + flush so the
    rotation reset is atomic).

    The daemon owns the lifecycle:

        sink = WebDaemonSink("/workspace/data/web_self_learn")
        try:
            for decision in stream_of_decisions:
                if decision.accept:
                    sink.append(text=..., source_id=..., url=...,
                                quality_score=..., gate_scores=...)
        finally:
            sink.close()
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        rotate_seconds: float = 3600.0,
        rotate_max_rows: int = 50_000,
        rotate_max_bytes: int = 200 * 1024 * 1024,
        write_batch_size: int = 256,
        file_prefix: str = "web_",
        file_pad: int = 4,
        quality_score_floor: float = 0.0,
    ) -> None:
        if pa is None or pq is None:
            raise ImportError(
                "WebDaemonSink requires pyarrow. Install via "
                "`pip install pyarrow`."
            ) from _PYARROW_IMPORT_EXC
        self.cfg = WebDaemonSinkConfig(
            out_dir=Path(out_dir),
            rotate_seconds=float(rotate_seconds),
            rotate_max_rows=int(rotate_max_rows),
            rotate_max_bytes=int(rotate_max_bytes),
            write_batch_size=int(write_batch_size),
            file_prefix=str(file_prefix),
            file_pad=int(file_pad),
            quality_score_floor=float(quality_score_floor),
        )
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional["pq.ParquetWriter"] = None
        self._writer_path: Optional[Path] = None
        self._writer_open_at: float = 0.0
        self._writer_rows: int = 0
        self._next_index: int = self._scan_next_index()

    # ------------------------------------------------------------------
    # File-naming + rotation helpers
    # ------------------------------------------------------------------

    def _scan_next_index(self) -> int:
        """Walk the out_dir and return the smallest unused shard index.

        Rotation across daemon restarts: if web_0001.parquet ... web_0017
        already exist, restart writes web_0018 (never overwrites).
        """
        existing = sorted(
            p for p in self.cfg.out_dir.glob(
                f"{self.cfg.file_prefix}*.parquet"
            )
        )
        if not existing:
            return 1
        max_idx = 0
        for p in existing:
            stem = p.stem
            try:
                idx = int(stem[len(self.cfg.file_prefix):])
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
        return max_idx + 1

    def _next_path(self) -> Path:
        idx = self._next_index
        self._next_index += 1
        return self.cfg.out_dir / (
            f"{self.cfg.file_prefix}"
            f"{idx:0{self.cfg.file_pad}d}.parquet"
        )

    def _should_rotate(self) -> bool:
        if self._writer is None or self._writer_path is None:
            return False
        if (time.time() - self._writer_open_at) >= self.cfg.rotate_seconds:
            return True
        if self._writer_rows >= self.cfg.rotate_max_rows:
            return True
        try:
            sz = self._writer_path.stat().st_size
        except OSError:
            sz = 0
        if sz >= self.cfg.rotate_max_bytes:
            return True
        return False

    # ------------------------------------------------------------------
    # Open / close / flush
    # ------------------------------------------------------------------

    def _open_writer(self) -> None:
        path = self._next_path()
        # write_batch_size >= 1 so the first append always materialises.
        self._writer = pq.ParquetWriter(
            str(path),
            _SCHEMA,
            compression="snappy",
        )
        self._writer_path = path
        self._writer_open_at = time.time()
        self._writer_rows = 0

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._writer = None
        self._writer_path = None
        self._writer_open_at = 0.0
        self._writer_rows = 0

    def _flush_buffer(self) -> int:
        """Write the in-memory buffer as one RowGroup.

        Returns
        -------
        int
            Number of rows actually written (may be 0 if buffer was empty
            or quality-floor filter dropped all rows).
        """
        if not self._buffer:
            return 0
        if self._writer is None:
            self._open_writer()
        # Build a column-oriented dict for pyarrow.
        rows = self._buffer
        cols = {
            "text": pa.array([r["text"] for r in rows], type=pa.string()),
            "source_id": pa.array(
                [r.get("source_id", "") for r in rows], type=pa.string(),
            ),
            "url": pa.array(
                [r.get("url", "") or "" for r in rows], type=pa.string(),
            ),
            "ingest_ts": pa.array(
                [float(r.get("ingest_ts", time.time())) for r in rows],
                type=pa.float64(),
            ),
            "quality_score": pa.array(
                [float(r.get("quality_score", 0.0)) for r in rows],
                type=pa.float32(),
            ),
            "gate_scores": pa.array(
                [json.dumps(r.get("gate_scores", {}), ensure_ascii=False)
                 for r in rows],
                type=pa.string(),
            ),
        }
        table = pa.Table.from_pydict(cols, schema=_SCHEMA)
        assert self._writer is not None
        self._writer.write_table(table)
        n = len(rows)
        self._writer_rows += n
        self._buffer = []
        return n

    def flush(self) -> int:
        """Force a flush of any pending rows. Safe to call repeatedly."""
        with self._lock:
            return self._flush_buffer()

    def close(self) -> int:
        """Flush + close the open writer. Returns total rows in last shard."""
        with self._lock:
            n = self._flush_buffer()
            self._close_writer()
            return n

    # ------------------------------------------------------------------
    # Public append
    # ------------------------------------------------------------------

    def append(
        self,
        text: str,
        source_id: str,
        url: Optional[str] = None,
        *,
        quality_score: float = 1.0,
        gate_scores: Optional[Dict[str, float]] = None,
        ingest_ts: Optional[float] = None,
    ) -> bool:
        """Append a single row. Returns True if accepted, False if filtered.

        The quality-floor filter is the LAST safety net: if the daemon's
        per-row quality_score is below ``quality_score_floor`` we drop
        the row before it ever hits parquet. Default floor 0.0 = pass-
        through (the daemon's 7 gates already decided).
        """
        if not text:
            return False
        if quality_score < self.cfg.quality_score_floor:
            return False
        row = {
            "text": str(text),
            "source_id": str(source_id) if source_id else "",
            "url": "" if url is None else str(url),
            "ingest_ts": float(ingest_ts) if ingest_ts is not None
                         else time.time(),
            "quality_score": float(quality_score),
            "gate_scores": dict(gate_scores or {}),
        }
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.cfg.write_batch_size:
                self._flush_buffer()
            if self._should_rotate():
                # close current shard, the next flush opens a new one
                self._flush_buffer()
                self._close_writer()
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "out_dir": str(self.cfg.out_dir),
            "next_index": self._next_index,
            "open_writer_path": (
                str(self._writer_path) if self._writer_path else None
            ),
            "open_writer_rows": self._writer_rows,
            "open_writer_age_s": (
                time.time() - self._writer_open_at
                if self._writer_open_at else 0.0
            ),
            "buffer_rows": len(self._buffer),
            "rotate_seconds": self.cfg.rotate_seconds,
            "rotate_max_rows": self.cfg.rotate_max_rows,
            "rotate_max_bytes": self.cfg.rotate_max_bytes,
            "quality_score_floor": self.cfg.quality_score_floor,
        }

    def shard_glob(self) -> str:
        """Return the canonical glob pattern matching this sink's shards.

        Useful for the trainer / launcher: pass this to the
        ``--data-files`` argument with a small weight (e.g. 0.10).
        """
        return str(
            self.cfg.out_dir / f"{self.cfg.file_prefix}*.parquet"
        )


# ---------------------------------------------------------------------------
# Daemon glue: turn an IngestDecision -> sink.append() one-liner
# ---------------------------------------------------------------------------


def append_decision(
    sink: "WebDaemonSink",
    decision: Any,                 # synapforge.learn.continual_daemon.IngestDecision
    text: str,
    source_id: str,
    url: Optional[str] = None,
) -> bool:
    """Glue: turn an admitted ``IngestDecision`` into a sink row.

    Translates the per-gate scores into the sink's columns. The
    ``quality_score`` is the MIN of all gate scores (most pessimistic
    gate dominates), capturing the "weakest link" semantics of an
    AND-gated admission.
    """
    if not getattr(decision, "accept", False):
        return False
    gates = getattr(decision, "gates", []) or []
    gate_scores: Dict[str, float] = {}
    for g in gates:
        name = getattr(g, "name", None) or g.__class__.__name__
        score = float(getattr(g, "score", 0.0))
        gate_scores[name] = score
    quality_score = min(gate_scores.values()) if gate_scores else 1.0
    return sink.append(
        text=text,
        source_id=source_id,
        url=url,
        quality_score=quality_score,
        gate_scores=gate_scores,
    )


__all__ = [
    "WebDaemonSink",
    "WebDaemonSinkConfig",
    "append_decision",
]
