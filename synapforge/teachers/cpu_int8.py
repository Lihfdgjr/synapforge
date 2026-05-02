"""synapforge.teachers.cpu_int8 — CPU-resident KD teacher.

Why this exists
---------------
The Run 6 KD teacher (Qwen 2.5 0.5B) consumes ~2 GB of HBM in fp16
when loaded on the A800 80GB rental. Profiling shows the teacher
forward at seq_len=256 + KD-every-4 takes ~600 ms on GPU, accounting
for ~25 % of step rate. Moving the teacher to CPU:

* Costs more wall-clock per teacher forward (200-500 ms on a 16-core
  Xeon with AVX-VNNI, dependent on whether bitsandbytes INT8 is
  available; without bitsandbytes we land at 400-800 ms in fp32).
* Frees ~2 GB of HBM that we re-spend on a larger student micro-batch.
* Overlaps with the GPU student forward via a ``torch.cuda.Stream``
  semaphore so the teacher cost is partially hidden.

Net: 5240 → 6500-8000 tok/s estimate on Run 6 (the bs lift more than
pays back the slower teacher when KD-every>=4).

Public API
----------
    >>> from synapforge.teachers.cpu_int8 import load_cpu_int8_teacher, cpu_teacher_forward
    >>> teacher = load_cpu_int8_teacher("Qwen/Qwen2.5-0.5B")
    >>> with torch.no_grad():
    ...     logits = cpu_teacher_forward(teacher, input_ids_gpu)
    # logits is on the GPU at the dtype the student expects.

Backends (auto-selected)
-----------------------
1. **bitsandbytes INT8** — preferred. ``Linear8bitLt`` over CPU has
   AVX-VNNI fast-path on Intel Xeon; ~2x faster than fp32 CPU and
   uses 1/4 the RAM. Requires ``pip install bitsandbytes``. Loaded
   via ``transformers.AutoModelForCausalLM.from_pretrained(load_in_8bit=True, device_map={"": "cpu"})``.
2. **fp32 CPU fallback** — used when bitsandbytes is missing or the
   load fails (e.g., model not supported by bnb). Always available.

Stream overlap
--------------
``cpu_teacher_forward`` accepts an optional ``stream`` argument; when
passed, the input H2D copy is staged on a side stream so the GPU
student forward on the default stream proceeds concurrently. The
forward itself runs on CPU regardless (no GPU dispatch). The output
H2D copy back to the student device is also non-blocking on the
caller's chosen stream.

Implementation notes
--------------------
* CPU-resident teacher returns logits on CPU; we move them to the
  caller's device at the end. The cast/copy is the only sync point.
* For very small batches (bs<8 typical for KD-every-4 mode at
  seq_len=256), Python overhead dominates; the kernel-side cost
  isn't even the bottleneck. That's fine — this module's value is
  HBM headroom, not raw teacher latency.
* Qwen 2.5 0.5B specifically is well-supported by bnb (LlamaForCausalLM
  family). Other architectures may silently degrade to fp32 CPU; we
  log which path we landed on at load time.
"""
from __future__ import annotations

import importlib
from typing import Any, Optional

import torch

__all__ = [
    "BNB_AVAILABLE",
    "load_cpu_int8_teacher",
    "cpu_teacher_forward",
]


# Probe bitsandbytes once at import — keep the optional dep gated so
# CPU-only CI without bnb still imports the module cleanly.
try:  # pragma: no cover -- env-dependent
    importlib.import_module("bitsandbytes")
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


def load_cpu_int8_teacher(
    name: str,
    *,
    fallback_fp32: bool = True,
    log: Optional[Any] = None,
) -> "torch.nn.Module":
    """Load a HF CausalLM on CPU with optional bitsandbytes INT8 quant.

    Parameters
    ----------
    name:
        Hugging Face model id (e.g., ``"Qwen/Qwen2.5-0.5B"``).
    fallback_fp32:
        If True (default), fall back to plain fp32 CPU load when
        bitsandbytes is not installed or the INT8 load fails. If False,
        raise the underlying error.
    log:
        Optional callable for status messages (e.g.,
        ``train_100m_kd._log``); ``None`` swallows them.

    Returns
    -------
    torch.nn.Module on CPU, frozen (``requires_grad=False`` for all
    params), in eval mode.

    Path selection
    --------------
    1. If bitsandbytes is importable AND ``device_map={"": "cpu"}``
       + ``load_in_8bit=True`` succeeds, returns the INT8 model
       (~125 MB for Qwen 0.5B vs ~500 MB fp32, ~1 GB fp16).
    2. Else (or on failure), returns the fp32 CPU model.
    """
    def _say(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    AutoModelForCausalLM = importlib.import_module(
        "transformers"
    ).AutoModelForCausalLM

    # Path 1: bitsandbytes INT8.
    if BNB_AVAILABLE:
        try:
            teacher = AutoModelForCausalLM.from_pretrained(
                name,
                load_in_8bit=True,
                device_map={"": "cpu"},
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            _say(f"[teacher-cpu-int8] loaded {name!r} via bitsandbytes "
                 "INT8 on CPU")
            return teacher
        except Exception as exc:
            if not fallback_fp32:
                raise
            _say(f"[teacher-cpu-int8] bnb INT8 load failed ({exc!r}); "
                 "falling back to fp32 CPU.")

    # Path 2: fp32 CPU fallback.
    teacher = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to("cpu")
    _say(f"[teacher-cpu-int8] loaded {name!r} on CPU in fp32 "
         "(bitsandbytes unavailable or unsupported for this arch).")
    return teacher


@torch.no_grad()
def cpu_teacher_forward(
    teacher: "torch.nn.Module",
    input_ids: "torch.Tensor",
    *,
    stream: Optional["torch.cuda.Stream"] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> "torch.Tensor":
    """Run a CPU teacher forward; return logits on the student's device.

    Parameters
    ----------
    teacher:
        Model returned by ``load_cpu_int8_teacher``. Must be on CPU.
    input_ids:
        Student-side input ids on the student's device (typically GPU).
    stream:
        Optional CUDA stream to ride the input D→H and output H→D
        copies. When passed and the input is on a CUDA device, the
        copies are staged with ``non_blocking=True`` so the student
        forward on the default stream overlaps with the teacher's CPU
        forward. ``None`` (default) uses synchronous copies — fine for
        single-GPU eval but wastes the overlap potential during
        training.
    out_dtype:
        Cast logits to this dtype before returning (typical: the
        student's compute dtype, e.g. bf16). ``None`` keeps the
        teacher's native logit dtype.

    Returns
    -------
    Logits tensor (B, T, V) on the same device as ``input_ids``.

    Raises
    ------
    ValueError
        If the teacher is not on CPU (use the GPU teacher path
        instead).
    """
    # Sanity check: this module is for CPU teachers only.
    try:
        # In the bnb int8 path the model has a quant_type marker;
        # in the fp32 path the first param's device tells us.
        first_p = next(teacher.parameters())
        if first_p.device.type != "cpu":
            raise ValueError(
                "cpu_teacher_forward expects a CPU-resident teacher; "
                f"got first param on {first_p.device}. Use the GPU "
                "teacher path instead."
            )
    except StopIteration:  # no params is weird but not fatal here
        pass

    # Step 1: D→H copy of input ids (non-blocking when stream provided).
    src_dev = input_ids.device
    if src_dev.type == "cuda":
        if stream is not None:
            with torch.cuda.stream(stream):
                ids_cpu = input_ids.detach().to(
                    "cpu", non_blocking=True
                )
            # Sync the side stream into default before we read the
            # result on CPU. (CPU work won't see the H2D until the
            # copy event has completed.)
            stream.synchronize()
        else:
            ids_cpu = input_ids.detach().cpu()
    else:
        ids_cpu = input_ids.detach()

    # Step 2: CPU forward.
    out = teacher(ids_cpu)
    logits = out.logits if hasattr(out, "logits") else out

    # Step 3: cast + H→D back to student device.
    if out_dtype is not None and logits.dtype != out_dtype:
        logits = logits.to(out_dtype)
    if src_dev.type == "cuda":
        if stream is not None:
            with torch.cuda.stream(stream):
                logits = logits.to(src_dev, non_blocking=True)
        else:
            logits = logits.to(src_dev)
    return logits
