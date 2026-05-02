"""synapforge.native.data -- torch-free data loading for native synapforge.

A drop-in replacement for the ``synapforge.data.ParquetTokenStream`` family
that emits **numpy** arrays (and optionally **cupy** arrays for async
host-to-device copy) instead of torch tensors. Designed to feed the native
trainer and the investor demo without dragging the torch import surface.

Modules
-------
parquet_stream
    ``NativeParquetStream`` -- pyarrow-driven row iterator with a streaming
    Fisher-Yates shuffle, multi-thread prefetch, and optional CUDA pinned
    memory via cupy.cuda.runtime.hostAlloc.
jsonl_stream
    ``NativeJsonlStream`` -- same emit shape, but reads one-JSON-per-line
    files (KD-distill, alpaca-zh, instruction SFT seeds).
mixed_stream
    ``WeightedMixedStream`` -- round-robin mixture of N sub-streams given
    ``[(source, weight), ...]``. Replaces the ``--data-files PATH:W,...``
    parsing path natively.
tokenizer
    ``NativeTokenizer`` -- Qwen-2.5-compatible BPE wrapper. Uses
    ``transformers.AutoTokenizer`` when available; otherwise loads a
    ``tokenizer.json`` directly via ``json`` and runs a minimal byte-level
    BPE.
bench
    Throughput parity bench between the native loader and the legacy
    torch ``ParquetTokenStream`` on the same parquet shard.

Hard constraints
----------------
* Zero ``import torch`` in this subpackage.
* The HF ``transformers`` package may be imported by ``tokenizer.py`` only
  when present; the JSON-fallback BPE has zero third-party deps beyond
  ``regex`` (and degrades to ``re`` if ``regex`` is missing).

Public API
----------
>>> from synapforge.native.data import (
...     NativeParquetStream,
...     NativeJsonlStream,
...     WeightedMixedStream,
...     NativeTokenizer,
... )
"""

from __future__ import annotations

from synapforge.native.data.jsonl_stream import NativeJsonlStream
from synapforge.native.data.mixed_stream import WeightedMixedStream
from synapforge.native.data.parquet_stream import NativeParquetStream
from synapforge.native.data.tokenizer import NativeTokenizer

__all__ = [
    "NativeJsonlStream",
    "NativeParquetStream",
    "NativeTokenizer",
    "WeightedMixedStream",
]
