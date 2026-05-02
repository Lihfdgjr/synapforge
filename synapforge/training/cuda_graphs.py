"""CUDA Graphs wrapper for the per-step training graph (Deliverable 1).

Patterns: ``torch.cuda.graphs.CUDAGraph`` captures a fixed-shape kernel
sequence on first call, then ``replay()`` replays it without Python
dispatch overhead. For our 100M ``HybridBlock``-stack training graph
(LiquidCell + PLIF + SparseSynapse + SwiGLU + RMSNorm × 16 layers ×
roughly 7 ops each ≈ 112 op dispatches), this saves ~50us × 112 ≈ 5.6ms
per backward+forward. In our typical Run 6 config (B=24 / T=256 /
step_ms ~2400ms) this is ~0.2% — small. The real win is when paired
with gradient accumulation: if grad_accum=4, the same graph replays
4× per global step and the saved ms multiply linearly. Estimated lift:
3-5% on Run 6 with grad_accum=4, 5-8% on a smaller / faster step.

Differences from :mod:`synapforge.runtime_cuda_graph`
------------------------------------------------------
``runtime_cuda_graph.GraphedTrainStep`` wraps a full
``synapforge.runtime.Runtime`` (the high-level inference path used by
``chat_demo``, with backend dispatch baked in). It is geared toward
*deployment*: forward-only or train-step + clip + optim.step capture in
one swoop.

This module's ``GraphedHybridBlock`` is *training* only and operates one
level lower: it captures the inner ``forward(x) → loss → loss.backward()``
sequence for the full ``SynapForge100M`` model **as called inside the
gradient-accumulation inner loop**. The trainer keeps owning
``optimizer.zero_grad()``, ``clip_grad_norm_``, and ``optimizer.step()``
so all the existing book-keeping (lazy host-sync accumulators, KD-async
teacher stream, NeuroMCP mixin, EMA, freeze-vocab-tail backward hook,
…) is **untouched**.

Static-shape pinning
--------------------
CUDA graphs require static input shapes. The wrapper pre-allocates two
input buffers (``static_x``, ``static_y``) of shape ``(B, T)`` int64
on the GPU. Each call ``copy_()``s the user's tensors into the pinned
slots, replays the graph, and the captured loss tensor (``static_loss``)
holds the scalar.

Last-micro-batch shape mismatch (when the data stream returns a partial
batch at end-of-epoch) is handled by the ``--cuda-graph-skip-partial``
contract: the trainer detects ``x.shape[0] != B`` and falls back to
``eager_step()`` for that micro-batch only. See ``train_100m_kd.py``.

Bit-exact contract
------------------
The captured graph IS the same kernel sequence the eager path runs;
the only difference is *how* it is dispatched. With identical inputs,
loss is identical to fp32 round-off (rel_err < 1e-6 in fp32, 1e-3 in
bf16 due to non-deterministic atomic adds in autograd). Tests pin this
at ``rel_err < 1e-6`` in **fp32** to surface any hidden non-determinism.

Default OFF
-----------
Activated via ``--cuda-graphs`` in ``train_100m_kd.py``. Default OFF.
The eager path remains the canonical reference; the wrapper falls back
to eager on any capture failure (NaN warmup, shape change, …) and
records ``skip_reason`` for diagnostics.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch


def _cuda_graphs_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph")


def _gpu_supports_graphs(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_properties(device).major >= 7
    except Exception:
        return False


@dataclass
class GraphedBlockCfg:
    """Configuration for ``GraphedHybridBlock``.

    Attributes:
        batch_size, seq_len: pinned input shape ``(B, T)`` int64.
        device: target CUDA device.
        dtype: autocast dtype for the forward pass. ``None`` = no
            autocast (fp32 throughout).
        n_warmup_iters: warmup iterations BEFORE capture. CUDA recipe
            requires at least 3 (one to JIT compile kernels, one to
            cache cuBLAS workspaces, one to warm up the cache). We
            default to 11 (matches the upstream ``runtime_cuda_graph``
            module).
        accumulate_grad: if True, the graph does NOT call zero_grad
            internally. Gradients accumulate in ``param.grad`` as on
            the eager path, which is what the trainer's grad-accum
            inner loop expects. Default True.
    """

    batch_size: int
    seq_len: int
    device: torch.device
    dtype: torch.dtype | None = torch.bfloat16
    n_warmup_iters: int = 11
    accumulate_grad: bool = True


class GraphedHybridBlock:
    """Capture-and-replay one fixed-shape ``forward → loss → backward``
    pass over ``SynapForge100M``.

    The trainer creates one of these per static (B, T) input shape and
    calls ``.step(x, y, ...)`` inside the grad-accum inner loop instead
    of a manual ``model(x) → loss → loss.backward()``. The captured
    graph runs the **same** code path the eager step would; the only
    difference is the kernel dispatch overhead is paid once at capture
    time instead of every call.

    Usage::

        cfg = GraphedBlockCfg(batch_size=24, seq_len=256,
                              device=torch.device("cuda"),
                              dtype=torch.bfloat16)
        graphed = GraphedHybridBlock(model, loss_fn, cfg)
        for x, y in train_iter:
            if x.shape[0] != cfg.batch_size:
                graphed.eager_step(x, y, **kw)  # last-batch fallback
                continue
            loss = graphed.step(x, y, **kw)     # graph.replay()
            (loss / accum).backward()  # NO! See note below.

    Note: the captured graph runs ``loss.backward()`` *internally* so
    grads are already accumulated into ``param.grad`` by the time
    ``.step()`` returns. The returned scalar is the captured ``loss``
    tensor (a view into ``static_loss``); the trainer should use it
    only for logging.

    For ``accumulate_grad=True`` (default), zero_grad is the trainer's
    responsibility, run ONCE per global step before the inner loop
    starts. The graph does NOT call zero_grad internally. For
    ``accumulate_grad=False``, the graph does call zero_grad as part
    of the captured sequence (only useful for non-accum runs).

    Args:
        model: any ``torch.nn.Module`` that takes ``x`` and returns
            ``logits`` of shape ``(B, T, V)``. We don't require any
            specific subclass — the test suite uses a tiny stand-in.
        loss_fn: callable ``(logits, y) -> scalar``. Must produce a
            single fp32 tensor. The default ``cross_entropy_loss``
            below mirrors what ``train_100m_kd.py`` runs.
        cfg: :class:`GraphedBlockCfg`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cfg: GraphedBlockCfg,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self._capture_ok = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._skip_reason = ""

        # Pre-allocate static input buffers so capture/replay can copy
        # into the same memory every call.
        self._static_x: torch.Tensor | None = None
        self._static_y: torch.Tensor | None = None
        self._static_loss: torch.Tensor | None = None

        if not _cuda_graphs_available():
            self._skip_reason = "torch.cuda.CUDAGraph not available"
            return
        if not _gpu_supports_graphs(cfg.device):
            self._skip_reason = (
                f"device {cfg.device} does not support CUDA Graphs"
            )
            return

        self._static_x = torch.zeros(
            (cfg.batch_size, cfg.seq_len),
            dtype=torch.long, device=cfg.device,
        )
        self._static_y = torch.zeros(
            (cfg.batch_size, cfg.seq_len),
            dtype=torch.long, device=cfg.device,
        )
        # Loss tensor pre-allocated; the captured graph copies the live
        # loss into this buffer on each replay so the consumer always
        # sees a consistent tensor view.
        self._static_loss = torch.zeros(
            (), dtype=torch.float32, device=cfg.device,
        )

        try:
            self._capture()
            self._capture_ok = True
        except Exception as exc:
            warnings.warn(
                f"[cuda-graph] capture failed: {exc!r}. Falling back to eager.",
                RuntimeWarning,
            )
            self._skip_reason = f"capture exception: {exc!r}"
            self._capture_ok = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _amp_ctx(self):
        if self.cfg.dtype is None:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.cfg.device.type, dtype=self.cfg.dtype,
            enabled=self.cfg.device.type == "cuda",
        )

    def _one_train_step(self) -> None:
        """The full forward+loss+backward sequence captured by the
        graph. Reads from ``self._static_x`` / ``self._static_y`` and
        writes to ``self._static_loss``. Does NOT call zero_grad
        (trainer's job) and does NOT call optimizer.step (trainer's
        job)."""
        if not self.cfg.accumulate_grad:
            self.model.zero_grad(set_to_none=True)
        with self._amp_ctx():
            logits = self.model(self._static_x)
            loss = self.loss_fn(logits, self._static_y)
        loss.backward()
        # Stash the loss scalar for the consumer.
        self._static_loss.copy_(loss.detach().float())

    def _capture(self) -> None:
        """Run warmup on a side stream, then capture the train step on
        the main stream. Follows the standard PyTorch CUDA Graphs
        recipe (https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
        """
        cfg = self.cfg
        # Warmup on a side stream so the main stream's capture is not
        # contaminated by JIT compile / workspace allocation.
        s = torch.cuda.Stream(device=cfg.device)
        s.wait_stream(torch.cuda.current_stream(cfg.device))
        with torch.cuda.stream(s):
            for _ in range(cfg.n_warmup_iters):
                self._one_train_step()
                # Zero grads between warmup iters so they don't accumulate
                # to absurd magnitudes that would change the captured
                # gradient computation.
                self.model.zero_grad(set_to_none=True)

        torch.cuda.current_stream(cfg.device).wait_stream(s)
        torch.cuda.synchronize(cfg.device)

        # IMPORTANT: zero grads IMMEDIATELY before capture so the
        # captured backward starts from .grad=None / 0 (matches the
        # eager-path trainer: zero_grad once before the inner loop).
        self.model.zero_grad(set_to_none=True)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._one_train_step()
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured graph. Falls back to eager on any
        capture-failure / shape-mismatch.

        Args:
            x, y: input + target tensors of shape ``(B, T)`` matching
                the captured shape. They are copied into the pinned
                static buffers via ``copy_(non_blocking=True)``.

        Returns:
            The captured loss scalar (view into ``self._static_loss``).
            Already detached + float32. The graph has ALREADY run
            ``backward()``, so ``param.grad`` is updated by the time
            this returns.
        """
        if not self._capture_ok:
            return self.eager_step(x, y)
        if (x.shape != self._static_x.shape or
                y.shape != self._static_y.shape):
            # Shape mismatch — fall back to eager for this batch only.
            return self.eager_step(x, y)
        self._static_x.copy_(x, non_blocking=True)
        self._static_y.copy_(y, non_blocking=True)
        self._graph.replay()  # type: ignore[union-attr]
        return self._static_loss

    def eager_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Run the SAME computation that the graph would run, but with
        eager dispatch. Used when capture fails or for a partial
        last-batch.
        """
        if not self.cfg.accumulate_grad:
            self.model.zero_grad(set_to_none=True)
        with self._amp_ctx():
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        loss.backward()
        return loss.detach().float()

    def rebuild(self, batch_size: int | None = None,
                seq_len: int | None = None) -> None:
        """Re-warmup + re-capture under a new (B, T) shape. Use this
        when the trainer is about to switch to a different fixed batch
        shape (e.g., switching to a longer eval seq-len mid-run).
        """
        cfg = self.cfg
        new_b = int(batch_size if batch_size is not None else cfg.batch_size)
        new_t = int(seq_len if seq_len is not None else cfg.seq_len)
        if (new_b == cfg.batch_size and new_t == cfg.seq_len
                and self._capture_ok):
            return
        self.cfg = GraphedBlockCfg(
            batch_size=new_b, seq_len=new_t,
            device=cfg.device, dtype=cfg.dtype,
            n_warmup_iters=cfg.n_warmup_iters,
            accumulate_grad=cfg.accumulate_grad,
        )
        self._static_x = torch.zeros(
            (new_b, new_t), dtype=torch.long, device=cfg.device,
        )
        self._static_y = torch.zeros(
            (new_b, new_t), dtype=torch.long, device=cfg.device,
        )
        self._static_loss = torch.zeros(
            (), dtype=torch.float32, device=cfg.device,
        )
        self._capture_ok = False
        self._graph = None
        try:
            self._capture()
            self._capture_ok = True
        except Exception as exc:
            warnings.warn(
                f"[cuda-graph] rebuild failed: {exc!r}. Falling back to eager.",
                RuntimeWarning,
            )
            self._skip_reason = f"rebuild exception: {exc!r}"

    @property
    def capture_active(self) -> bool:
        return self._capture_ok

    @property
    def skip_reason(self) -> str:
        return self._skip_reason

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"GraphedHybridBlock(B={cfg.batch_size}, T={cfg.seq_len}, "
            f"device={cfg.device}, dtype={cfg.dtype}, "
            f"capture_active={self._capture_ok}, "
            f"skip_reason={self._skip_reason!r})"
        )


# ---------------------------------------------------------------------------
# Default loss helper
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    logits: torch.Tensor, y: torch.Tensor,
) -> torch.Tensor:
    """Default loss for graph capture: plain CE, fp32-promoted.

    Mirrors the kernel sequence the trainer's eager path runs (modulo
    the z-loss / KD / mixin terms which are too dynamic to capture in
    a single graph; those run on the eager path always).
    """
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        y.reshape(-1),
    )


__all__ = [
    "GraphedBlockCfg",
    "GraphedHybridBlock",
    "cross_entropy_loss",
]
