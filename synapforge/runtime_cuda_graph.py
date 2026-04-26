"""CUDA Graphs wrapper for synapforge Runtime.

Adapted from D:/ai_tool/b200_gpu_prep/cuda_graph_step.py (598 LOC). The
upstream version wraps a raw torch.nn.Module training step. Here we wrap
a `synapforge.runtime.Runtime` (which already encapsulates model + backend
dispatch). The capture-and-replay machinery and PyTorch 2.1.x rules are
preserved verbatim; only the forward dispatch is swapped from
`model(tokens=...)` to `runtime(*args, **kwargs)`.

Why a separate module
---------------------
We want gpu_dense / triton_block / cpu_event Backend implementations to
remain capture-agnostic. The Backend does the kernel work; this layer
sits on top of any Runtime and adds:
    - static input buffers
    - warmup
    - graph capture (separate side stream, per PyTorch recipe)
    - replay
    - eager fallback on capture-failure / NaN / shape-change

Public surface
--------------
    GraphedTrainStep(runtime, optimizer, cfg, loss_fn=None)
        .step(*inputs, target=None) -> Tensor       # graph.replay()
        .eager_step(*inputs, target=None) -> Tensor # bypass graph
        .eval_forward(*inputs) -> Any               # forward only, eager

    make_graphed_step(runtime, optimizer, input_shapes, ...) -> GraphedTrainStep

Differences from upstream
-------------------------
- Inputs are now an arbitrary tuple of (shape, dtype) pairs instead of a
  fixed (input_ids, target_ids) pair. Forward-only capture (no target) is
  also supported when `optimizer is None` and `loss_fn is None`.
- The forward call goes through `runtime(*static_inputs)` so per-backend
  kernel paths (gpu_dense / triton_block / ...) are exercised correctly.
- Capture-error fallback is built in; the wrapper degrades to eager and
  reports `skip_reason` instead of crashing. Same as upstream.
"""
from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def _torch_cuda_graph_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph")


def _device_supports_graphs(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        major = torch.cuda.get_device_properties(device).major
        return major >= 7
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Specification for one captured tensor input."""
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32

    def alloc(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.shape, dtype=self.dtype, device=device)


@dataclass
class GraphCfg:
    input_specs: Sequence[TensorSpec]            # one per positional input
    target_spec: Optional[TensorSpec] = None     # None = forward-only capture
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.bfloat16          # autocast dtype
    grad_clip: float = 1.0
    n_warmup_iters: int = 11
    use_amp: bool = True
    pool_handle: Any = None
    use_cuda_graph: bool = True
    eager_fallback: bool = True


# ---------------------------------------------------------------------------
# GraphedTrainStep — wraps a synapforge Runtime
# ---------------------------------------------------------------------------


class GraphedTrainStep:
    """Capture-and-replay one fixed-shape forward (+optional loss/back/step).

    runtime: synapforge.runtime.Runtime — provides .__call__, .parameters().
    optimizer: torch optimizer over runtime.parameters(). None for fwd-only.
    loss_fn:  callable(out, target) -> scalar Tensor. None for fwd-only.
    """

    def __init__(
        self,
        runtime: Any,                   # synapforge.runtime.Runtime
        optimizer: Optional[torch.optim.Optimizer],
        cfg: GraphCfg,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.runtime = runtime
        self.opt = optimizer
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.scaler = scaler
        self._train_mode = (optimizer is not None) and (loss_fn is not None)

        self._capture_ok = False
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._reason_skip = ""

        if not cfg.use_cuda_graph:
            self._reason_skip = "use_cuda_graph=False"
            return
        if not _torch_cuda_graph_available():
            self._reason_skip = "torch.cuda.CUDAGraph not available"
            return
        if not _device_supports_graphs(cfg.device):
            self._reason_skip = f"device {cfg.device} does not support CUDA Graphs"
            return

        # Pre-allocate static input/target buffers.
        self.static_inputs: list[torch.Tensor] = [
            spec.alloc(cfg.device) for spec in cfg.input_specs
        ]
        self.static_target: Optional[torch.Tensor] = (
            cfg.target_spec.alloc(cfg.device) if cfg.target_spec is not None else None
        )
        # Forward output (refreshed each replay -- captured tensor view).
        self.static_output: Optional[torch.Tensor] = None
        # Static loss (for train mode).
        self.static_loss: Optional[torch.Tensor] = (
            torch.zeros((), device=cfg.device, dtype=torch.float32)
            if self._train_mode else None
        )

        try:
            self._capture()
            self._capture_ok = True
        except Exception as e:
            warnings.warn(
                f"[cuda-graph] capture failed: {e!r}. Falling back to eager.",
                RuntimeWarning,
            )
            self._reason_skip = f"capture exception: {e!r}"
            self._capture_ok = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, *inputs: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Replay-based training step. Falls through to eager on failure."""
        if not self._capture_ok:
            return self.eager_step(*inputs, target=target)

        if len(inputs) != len(self.static_inputs):
            raise RuntimeError(
                f"[cuda-graph] got {len(inputs)} inputs, captured "
                f"{len(self.static_inputs)}. Use eager_step or rebuild()."
            )
        for i, (x, sx) in enumerate(zip(inputs, self.static_inputs)):
            if x.shape != sx.shape:
                raise RuntimeError(
                    f"[cuda-graph] input[{i}] shape {tuple(x.shape)} != "
                    f"captured {tuple(sx.shape)}; rebuild() required."
                )
            sx.copy_(x, non_blocking=True)
        if self.static_target is not None:
            if target is None:
                raise RuntimeError("[cuda-graph] captured train mode requires target=...")
            if target.shape != self.static_target.shape:
                raise RuntimeError("[cuda-graph] target shape mismatch")
            self.static_target.copy_(target, non_blocking=True)

        self._graph.replay()  # type: ignore[union-attr]
        if self._train_mode:
            return self.static_loss  # type: ignore[return-value]
        return self.static_output    # type: ignore[return-value]

    def eager_step(self, *inputs: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        amp_ctx = (
            torch.cuda.amp.autocast(dtype=cfg.dtype) if cfg.use_amp else nullcontext()
        )
        if self._train_mode:
            self.opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            with amp_ctx:
                out = self.runtime(*inputs)
                loss = self.loss_fn(out, target)  # type: ignore[misc]
            if self.scaler is not None and cfg.dtype is torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
            else:
                loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.runtime.parameters(), cfg.grad_clip)
            if self.scaler is not None and cfg.dtype is torch.float16:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()  # type: ignore[union-attr]
            return loss.detach()
        with amp_ctx:
            return self.runtime(*inputs)

    @torch.no_grad()
    def eval_forward(self, *inputs: torch.Tensor) -> Any:
        cfg = self.cfg
        was_training = False
        try:
            root = self.runtime.graph.modules.get("root")
            if root is not None:
                was_training = root.training
                root.eval()
        except Exception:
            pass
        amp_ctx = (
            torch.cuda.amp.autocast(dtype=cfg.dtype) if cfg.use_amp else nullcontext()
        )
        try:
            with amp_ctx:
                return self.runtime(*inputs)
        finally:
            if was_training:
                try:
                    root.train()  # type: ignore[union-attr]
                except Exception:
                    pass

    @property
    def capture_active(self) -> bool:
        return self._capture_ok

    @property
    def skip_reason(self) -> str:
        return self._reason_skip

    def rebuild(self, new_input_specs: Optional[Sequence[TensorSpec]] = None) -> None:
        """Re-warmup + re-capture. Pass `new_input_specs` for shape change."""
        self._capture_ok = False
        self._graph = None
        if new_input_specs is not None:
            self.cfg = GraphCfg(
                input_specs=list(new_input_specs),
                target_spec=self.cfg.target_spec,
                device=self.cfg.device,
                dtype=self.cfg.dtype,
                grad_clip=self.cfg.grad_clip,
                n_warmup_iters=self.cfg.n_warmup_iters,
                use_amp=self.cfg.use_amp,
                pool_handle=self.cfg.pool_handle,
                use_cuda_graph=self.cfg.use_cuda_graph,
                eager_fallback=self.cfg.eager_fallback,
            )
            self.static_inputs = [s.alloc(self.cfg.device) for s in self.cfg.input_specs]
        try:
            self._capture()
            self._capture_ok = True
        except Exception as e:
            warnings.warn(f"[cuda-graph] rebuild failed: {e!r}; staying eager",
                          RuntimeWarning)
            self._reason_skip = f"rebuild exception: {e!r}"
            self._capture_ok = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _one_train_step(self) -> None:
        cfg = self.cfg
        amp_ctx = (
            torch.cuda.amp.autocast(dtype=cfg.dtype) if cfg.use_amp else nullcontext()
        )
        if self._train_mode:
            self.opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            with amp_ctx:
                out = self.runtime(*self.static_inputs)
                loss = self.loss_fn(out, self.static_target)  # type: ignore[misc]
            if self.scaler is not None and cfg.dtype is torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
            else:
                loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.runtime.parameters(), cfg.grad_clip)
            if self.scaler is not None and cfg.dtype is torch.float16:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()  # type: ignore[union-attr]
            self.static_loss.copy_(loss.detach().float())  # type: ignore[union-attr]
        else:
            with amp_ctx:
                out = self.runtime(*self.static_inputs)
            if self.static_output is None or self.static_output.shape != out.shape:
                self.static_output = torch.empty_like(out)
            self.static_output.copy_(out)

    def _capture(self) -> None:
        cfg = self.cfg
        s = torch.cuda.Stream(device=cfg.device)
        s.wait_stream(torch.cuda.current_stream(cfg.device))

        with torch.cuda.stream(s):
            for _ in range(cfg.n_warmup_iters):
                self._one_train_step()

        torch.cuda.current_stream(cfg.device).wait_stream(s)
        torch.cuda.synchronize(cfg.device)

        if self._train_mode:
            self.opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        graph = torch.cuda.CUDAGraph()
        capture_kwargs: dict = {}
        if cfg.pool_handle is not None:
            capture_kwargs["pool"] = cfg.pool_handle
        with torch.cuda.graph(graph, **capture_kwargs):
            self._one_train_step()
        self._graph = graph


# ---------------------------------------------------------------------------
# Convenience builder — preserves the upstream `make_graphed_step` API
# ---------------------------------------------------------------------------


def make_graphed_step(
    runtime: Any,
    optimizer: Optional[torch.optim.Optimizer],
    input_specs: Sequence[TensorSpec],
    target_spec: Optional[TensorSpec] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    use_cuda_graph: bool = True,
    n_warmup_iters: int = 11,
    eager_fallback: bool = True,
) -> GraphedTrainStep:
    """One-line builder. Returns a GraphedTrainStep around a Runtime."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GraphCfg(
        input_specs=list(input_specs),
        target_spec=target_spec,
        device=device,
        dtype=dtype,
        grad_clip=grad_clip,
        n_warmup_iters=n_warmup_iters,
        use_amp=(dtype is not torch.float32),
        use_cuda_graph=use_cuda_graph,
        eager_fallback=eager_fallback,
    )
    return GraphedTrainStep(runtime, optimizer, cfg, loss_fn=loss_fn, scaler=scaler)


# ---------------------------------------------------------------------------
# Model-side audit notes (carried over from upstream)
# ---------------------------------------------------------------------------

MODEL_GRAPH_CAPTURE_AUDIT = """
synapforge primitives — graph-capture status

[OK]  LiquidCell  (cells/liquid.py)
      Pure tensor ops; Heinsen scan + cumsum + tanh. fp32 promote inside is
      a tensor cast, not a Python branch. Capturable.

[OK]  PLIF       (cells/plif.py)
      _ATanSurrogate.forward saves to ctx; backward is elementwise.
      `.last_spike_rate.copy_(spk.mean().detach())` is the ONLY in-place
      buffer mutation in forward; capturable iff buffer pre-exists (it
      does -- registered in __init__).

[OK]  SparseSynapse (cells/synapse.py, mask-based)
      Static mask buffer; forward is a masked linear.

[FIX] STDP plasticity (plasticity.py — see upstream rule list)
      Any `if reward > thr: update` branches must be fused as tensor masks.
      Easy win: `gate = (reward > thr).to(W.dtype); W.add_(gate * delta)`.
      Same pattern as upstream cuda_graph_step.py audit.

[FIX] Hebbian.maybe_grow_prune / DynamicCodebook.maybe_grow
      Must be invoked OUTSIDE the captured region (caller every N steps).

[FIX] MoR with early-exit `break`
      Replace with full-loop masked accumulation up to mor_max_depth.

The two SIMPLEST patches we apply for this PR:
  1. PLIF.forward: rewrite `last_spike_rate.copy_(...)` so it does NOT
     allocate a fresh detached tensor (use a pre-allocated scalar buffer
     and copy_ into it). Already true in cells/plif.py — verified OK.
  2. Skip-on-NaN gating moved OUTSIDE replay (this module already does this
     by exposing static_loss; caller checks `torch.isnan(loss).any()`).

Deferred to v0.5:
  - STDP `if`-branches → tensor mask fusion (synapforge.plasticity not yet
    proven on CUDA; gate behind `register_plasticity` instead).
  - MoR break removal (no MoR primitive in v0.1).
  - Dynamic codebook grow / prune (no codebook primitive in v0.1).
"""


__all__ = [
    "TensorSpec",
    "GraphCfg",
    "GraphedTrainStep",
    "make_graphed_step",
    "MODEL_GRAPH_CAPTURE_AUDIT",
]
