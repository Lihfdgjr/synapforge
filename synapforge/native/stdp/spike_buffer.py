"""SpikeRingBuffer — last-K timestep window for pre+post spikes.

STDP needs spike pairs that fall within a coincidence window Delta_t.
Rather than storing the full spike history we keep a ring buffer of
the last ``K`` timesteps for each neuron, plus the running pre-trace
and post-trace (exponentially decayed spike rate). The Triton kernel
in :mod:`stdp_optimizer` consumes both the trace (continuous) and the
binary windowed spike (discrete).

Hard constraint: NO ``import torch`` here. The buffer holds raw memory
described by ``(ptr, shape, stride, dtype)``. Producers (the SNN
forward pass) hand us a CPU/CUDA pointer; the Triton kernel reads it
back. We use numpy for tests and CPU fallback only.

Layout
------
For a layer with ``in_dim`` (pre) and ``out_dim`` (post) neurons we
keep:

    pre_window:   [K, in_dim]   uint8  (1 if neuron j fired at t-k)
    post_window:  [K, out_dim]  uint8
    pre_trace:    [in_dim]      float32  (low-pass of pre spikes)
    post_trace:   [out_dim]     float32

The ring's write head ``cursor`` is advanced once per call to
:meth:`push`, modulo ``K``. The producer is responsible for calling
:meth:`push` inside its forward pass — the buffer DOES NOT clone the
input tensor; producers must pass detached arrays.

Cost model
----------
* :meth:`push`:                O(in_dim + out_dim)
* :meth:`update_traces`:       O(in_dim + out_dim) (decay + add)
* :meth:`pair_outer`:          O(active_pre * active_post) (sparse)

So when spike density is rho the per-step plasticity cost is
``rho^2 * in_dim * out_dim`` rather than the dense ``in_dim * out_dim``
weight update. At 10% density that's a 100x reduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpikeRingBuffer:
    """Per-layer ring buffer of last K spike timesteps + EMA traces.

    The buffer can be backed by either numpy (CPU tests / fallback) or
    a producer-supplied raw pointer (Triton path). We default to numpy
    for portability; switch to ``backend="cuda"`` and supply external
    arrays via :meth:`bind_external` to use CUDA.

    Attributes
    ----------
    in_dim : int        pre-synaptic dimension (number of input neurons)
    out_dim : int       post-synaptic dimension
    window : int        K, number of past timesteps held
    decay_pre : float   exp(-1/tau_pre) — multiplier on pre_trace per push
    decay_post : float  exp(-1/tau_post)
    cursor : int        next write index, in [0, window)
    """

    in_dim: int
    out_dim: int
    window: int = 20
    decay_pre: float = 0.9512  # exp(-1/20)
    decay_post: float = 0.9512
    cursor: int = field(default=0, init=False)
    pre_window: np.ndarray = field(init=False)
    post_window: np.ndarray = field(init=False)
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.in_dim <= 0 or self.out_dim <= 0:
            raise ValueError(
                f"in_dim/out_dim must be positive; got "
                f"in_dim={self.in_dim} out_dim={self.out_dim}"
            )
        if self.window <= 0:
            raise ValueError(f"window must be positive; got {self.window}")
        self.pre_window = np.zeros((self.window, self.in_dim), dtype=np.uint8)
        self.post_window = np.zeros((self.window, self.out_dim), dtype=np.uint8)
        self.pre_trace = np.zeros(self.in_dim, dtype=np.float32)
        self.post_trace = np.zeros(self.out_dim, dtype=np.float32)

    # ------------------------------------------------------------------ push
    def push(
        self,
        pre_spike: np.ndarray,
        post_spike: np.ndarray,
    ) -> None:
        """Advance the ring by one timestep.

        ``pre_spike`` and ``post_spike`` are 1D binary arrays of shape
        ``(in_dim,)`` and ``(out_dim,)``. They must be 0/1 valued (we
        do not enforce; producer responsibility) and dtype-castable to
        uint8.
        """
        if pre_spike.shape != (self.in_dim,):
            raise ValueError(
                f"pre_spike shape {pre_spike.shape} != ({self.in_dim},)"
            )
        if post_spike.shape != (self.out_dim,):
            raise ValueError(
                f"post_spike shape {post_spike.shape} != ({self.out_dim},)"
            )
        # Window write
        self.pre_window[self.cursor] = pre_spike.astype(np.uint8)
        self.post_window[self.cursor] = post_spike.astype(np.uint8)
        self.cursor = (self.cursor + 1) % self.window
        # Trace decay-and-add
        self.pre_trace *= self.decay_pre
        self.pre_trace += pre_spike.astype(np.float32)
        self.post_trace *= self.decay_post
        self.post_trace += post_spike.astype(np.float32)

    # ---------------------------------------------------------------- traces
    def get_traces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current ``(pre_trace, post_trace)`` views.

        These are continuously valued exponentially-decayed spike rates
        used inside the STDP kernel.
        """
        return self.pre_trace, self.post_trace

    def latest_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recent ``(pre_spike, post_spike)`` we saw.

        ``cursor`` already advanced past the last push so we read at
        ``cursor - 1`` (mod ``window``).
        """
        idx = (self.cursor - 1) % self.window
        return self.pre_window[idx], self.post_window[idx]

    # ----------------------------------------------------------- pair_outer
    def pair_outer(
        self,
        a_plus: float = 0.02,
        a_minus: float = 0.02,
    ) -> np.ndarray:
        """Compute the dense LTP-LTD outer product for a single step.

        This is the pure-numpy fallback (used by the optimizer when the
        Triton kernel is not available). The math matches
        :class:`synapforge.bio.stdp_fast.STDPFastWeight`:

            dW = a_plus * outer(post_spike, pre_trace)
               - a_minus * outer(post_trace, pre_spike)

        Cost is O(in_dim * out_dim) when dense; the kernel path
        downstream uses spike sparsity to bring it to O(active^2).
        """
        pre_spike, post_spike = self.latest_spikes()
        pre_spike_f = pre_spike.astype(np.float32)
        post_spike_f = post_spike.astype(np.float32)
        ltp = a_plus * np.outer(post_spike_f, self.pre_trace)
        ltd = a_minus * np.outer(self.post_trace, pre_spike_f)
        return (ltp - ltd).astype(np.float32)

    # -------------------------------------------------------------- summary
    def active_spike_count(self) -> int:
        """Total #pre + #post that fired in the latest pushed step."""
        pre_spike, post_spike = self.latest_spikes()
        return int(pre_spike.sum() + post_spike.sum())

    def reset(self) -> None:
        """Zero out all windows, traces, and the cursor."""
        self.pre_window.fill(0)
        self.post_window.fill(0)
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.cursor = 0


__all__ = ["SpikeRingBuffer"]
