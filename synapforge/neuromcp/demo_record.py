"""synapforge.neuromcp.demo_record -- record + replay human demos.

Why
---
Per the brief, the imitation seed is a non-negotiable training input:
neuron seed = imitation learning *before* exploration.  The model
learns the primitive vocabulary by watching humans before it ever runs
in a sandbox env on its own.

Recording
---------

    rec = DemoRecorder(actuator)
    rec.record_event(primitive_id=0, params=[0.25, 0.25, 0, 0, 0, 0, 0, 0])
    rec.record_event(primitive_id=8, params=[0, 0, 0, 0, 65, 0, 0, 0])
    rec.save("demo.parquet")        # writes parquet via pyarrow
                                     # falls back to JSONL when pyarrow missing

Each row is:
    timestamp_ms : int
    primitive_id : int
    params       : list[float]   length 8
    obs_before   : bytes         PNG snapshot before the action
    obs_after    : bytes         PNG snapshot after the action
    success      : bool

Replay
------

    rep = DemoReplayer.from_file("demo.parquet")
    for primitive_id, params in rep.iter_actions():
        env.step(primitive_id, params)

torch is **not** imported.  pyarrow + pandas are optional; we degrade
to JSONL when they're missing.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .os_actuator import ObservationDict, OSActuator


try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_PYARROW = True
except Exception:
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAS_PYARROW = False


# ---------------------------------------------------------------------------
# DemoEvent -- one recorded action.
# ---------------------------------------------------------------------------


@dataclass
class DemoEvent:
    timestamp_ms: int
    primitive_id: int
    params: List[float]
    obs_before: bytes
    obs_after: bytes
    success: bool

    def to_row(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "primitive_id": int(self.primitive_id),
            "params": list(self.params),
            "obs_before": bytes(self.obs_before or b""),
            "obs_after": bytes(self.obs_after or b""),
            "success": bool(self.success),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "DemoEvent":
        return cls(
            timestamp_ms=int(row["timestamp_ms"]),
            primitive_id=int(row["primitive_id"]),
            params=list(row["params"]),
            obs_before=bytes(row.get("obs_before") or b""),
            obs_after=bytes(row.get("obs_after") or b""),
            success=bool(row["success"]),
        )


# ---------------------------------------------------------------------------
# DemoRecorder
# ---------------------------------------------------------------------------


@dataclass
class DemoRecorder:
    """Captures (action, observation) pairs and persists to disk.

    Args
    ----
    actuator : OSActuator | None
        When passed, ``record_event`` will execute the action through
        the actuator and capture the resulting observation.  When None
        (just-the-trace mode), the caller supplies pre/post observation
        bytes manually.
    """

    actuator: Optional[OSActuator] = None
    events: List[DemoEvent] = field(default_factory=list)

    def record_event(self,
                     primitive_id: int,
                     params: List[float],
                     obs_before: Optional[bytes] = None,
                     obs_after: Optional[bytes] = None,
                     success: Optional[bool] = None) -> DemoEvent:
        # If we have an actuator + no explicit observations, do the dispatch.
        if self.actuator is not None and obs_after is None:
            if obs_before is None:
                screen = self.actuator.execute(16, [0.0] * 8)  # screenshot prim
                obs_before = screen.screenshot_bytes or b""
            res = self.actuator.execute(int(primitive_id), params)
            obs_after_b = res.screenshot_bytes or b""
            success_b = bool(res.success) if success is None else bool(success)
        else:
            obs_before = obs_before or b""
            obs_after_b = obs_after or b""
            success_b = bool(success) if success is not None else True

        ev = DemoEvent(
            timestamp_ms=int(time.time() * 1000.0),
            primitive_id=int(primitive_id),
            params=list(params),
            obs_before=obs_before or b"",
            obs_after=obs_after_b or b"",
            success=success_b,
        )
        self.events.append(ev)
        return ev

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> str:
        """Save to parquet (preferred) or JSONL fallback.

        Returns the actual file path written -- may differ from ``path``
        when we fall back to JSONL (.jsonl extension swapped in).
        """
        rows = [ev.to_row() for ev in self.events]
        if _HAS_PYARROW and path.endswith(".parquet") and pa is not None and pq is not None:
            table = pa.table({
                "timestamp_ms": [r["timestamp_ms"] for r in rows],
                "primitive_id": [r["primitive_id"] for r in rows],
                "params": [r["params"] for r in rows],
                "obs_before": [r["obs_before"] for r in rows],
                "obs_after": [r["obs_after"] for r in rows],
                "success": [r["success"] for r in rows],
            })
            pq.write_table(table, path)
            return path
        # Fallback: JSONL with hex-encoded blobs.
        if path.endswith(".parquet"):
            path = path[:-len(".parquet")] + ".jsonl"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                obj = {
                    "timestamp_ms": r["timestamp_ms"],
                    "primitive_id": r["primitive_id"],
                    "params": r["params"],
                    "obs_before_hex": r["obs_before"].hex(),
                    "obs_after_hex": r["obs_after"].hex(),
                    "success": r["success"],
                }
                fh.write(json.dumps(obj))
                fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str) -> "DemoRecorder":
        """Load events from a parquet or JSONL file."""
        rec = cls()
        if path.endswith(".parquet") and _HAS_PYARROW and pa is not None and pq is not None:
            table = pq.read_table(path)
            df = table.to_pylist()
            rec.events = [DemoEvent.from_row(r) for r in df]
        else:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rec.events.append(DemoEvent(
                        timestamp_ms=int(obj["timestamp_ms"]),
                        primitive_id=int(obj["primitive_id"]),
                        params=list(obj["params"]),
                        obs_before=bytes.fromhex(obj.get("obs_before_hex", "")),
                        obs_after=bytes.fromhex(obj.get("obs_after_hex", "")),
                        success=bool(obj["success"]),
                    ))
        return rec


# ---------------------------------------------------------------------------
# DemoReplayer
# ---------------------------------------------------------------------------


@dataclass
class DemoReplayer:
    events: List[DemoEvent] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "DemoReplayer":
        rec = DemoRecorder.load(path)
        return cls(events=rec.events)

    def iter_actions(self) -> Iterable[Tuple[int, List[float]]]:
        for ev in self.events:
            yield ev.primitive_id, ev.params

    def replay(self, env) -> List[ObservationDict]:
        """Replay demo events through a ClosedLoopEnv."""
        results: List[ObservationDict] = []
        env.reset()
        for primitive_id, params in self.iter_actions():
            res = env.step(primitive_id, params, confidence=1.0)
            results.append(res.obs)
        return results

    def __len__(self) -> int:
        return len(self.events)


__all__ = ["DemoEvent", "DemoRecorder", "DemoReplayer"]
