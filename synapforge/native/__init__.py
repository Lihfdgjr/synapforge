"""synapforge.native -- pure-Python / numpy / cupy production primitives.

The ``synapforge.native`` namespace hosts torch-free production primitives
that the LNN+SNN backbone uses at runtime. The hard rule for every file
under this package is **no ``import torch``**. Everything is built on
``numpy`` (always available), ``cupy`` (optional, GPU acceleration), and
``triton`` (optional, fused kernels).

Subpackages
-----------
* :mod:`synapforge.native.dispatch` -- heterogeneous CPU+GPU async pipeline
* :mod:`synapforge.native.vjp`      -- VJP catalog for custom backward
* :mod:`synapforge.native.modal`    -- multimodal byte-patch packing layer

Why these are torch-free
------------------------
Torch is fine for prototyping (and the model code in ``synapforge.cells``,
``synapforge.modal``, ``synapforge.bio`` etc. uses it). But the runtime
hot path -- specifically the parts that get called millions of times per
training step -- benefits from being torch-free for three reasons:

1. **Static memory layout**. ``numpy``/``cupy`` arrays do not carry an
   autograd tape, version counters, or device guards. The pack/unpack
   operations on a flat token tensor are zero-overhead.
2. **Fused execution.** Triton kernels written against raw ``cupy``
   pointers integrate one layer below the torch dispatcher.
3. **CPU/GPU parity.** The same ``numpy``-based code path runs on a
   CPU-only host (e.g. a developer laptop), with a one-line swap to
   ``cupy`` on the A800.
"""
from __future__ import annotations

__all__ = []
