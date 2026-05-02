"""Smoke tests for ``synapforge.teachers.cpu_int8``.

The full Qwen 2.5 0.5B load requires a network round-trip + ~500 MB
of disk / RAM, which is too heavy for CPU CI. We split the test into
two tiers:

* **Tier 1 (always run)** — module-import + signature smoke. Asserts
  that the public API surface is callable and the BNB_AVAILABLE flag
  imports without error. No model is loaded.

* **Tier 2 (gated, RUN_HEAVY_TEACHER_TESTS=1)** — actually loads
  Qwen 2.5 0.5B (or a smaller backstop in CI) and runs one forward.
  Asserts the output shape and dtype the trainer expects.

Why split: the deliverable contract is "load Qwen 0.5B on CPU, smoke
1 forward, assert output shape and dtype". On the rental that's the
real test; on CPU CI we don't want to download 500 MB every commit.
"""
from __future__ import annotations

import os

import pytest
import torch

from synapforge.teachers import cpu_int8


# ---------------------------------------------------------------------------
# Tier 1 — always run on every CI machine
# ---------------------------------------------------------------------------


def test_module_imports():
    """The public API surface imports without error."""
    assert callable(cpu_int8.load_cpu_int8_teacher)
    assert callable(cpu_int8.cpu_teacher_forward)
    # BNB_AVAILABLE is a bool indicating whether bitsandbytes is
    # importable — we don't care which value, just that the probe
    # didn't throw.
    assert isinstance(cpu_int8.BNB_AVAILABLE, bool)


def test_cpu_forward_rejects_gpu_teacher():
    """cpu_teacher_forward must reject a teacher on the wrong device.

    Builds a tiny stand-in nn.Module on the device-under-test and
    verifies that placing it on a non-CPU device causes
    ``cpu_teacher_forward`` to raise. We can't actually exercise this
    on CPU-only CI (there's no other device to put the teacher on),
    so the assertion only fires on CUDA hosts. On CPU-only the test
    runs the same harness with a CPU teacher and verifies the happy
    path.
    """
    # Build a tiny LM with the same shape contract as the real teacher
    # (forward(ids) -> logits (B, T, V)).
    class _TinyLM(torch.nn.Module):
        def __init__(self, vocab: int, hidden: int) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, hidden)
            self.head = torch.nn.Linear(hidden, vocab, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.emb(ids))

    teacher = _TinyLM(vocab=32, hidden=8)
    teacher.eval()

    if torch.cuda.is_available():
        teacher.to("cuda:0")
        ids = torch.zeros(1, 4, dtype=torch.long, device="cuda:0")
        # Mismatched device should raise ValueError.
        with pytest.raises(ValueError, match="CPU-resident"):
            cpu_int8.cpu_teacher_forward(teacher, ids)
    else:
        # Happy path on CPU CI.
        teacher.to("cpu")
        ids = torch.zeros(1, 4, dtype=torch.long)
        logits = cpu_int8.cpu_teacher_forward(teacher, ids)
        assert logits.shape == (1, 4, 32)
        assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)


def test_cpu_forward_returns_correct_shape_and_dtype():
    """Smoke 1 forward through ``cpu_teacher_forward`` on a tiny LM.

    Asserts the (B, T, V) shape contract the trainer relies on and
    that ``out_dtype`` casts work as advertised. This is the
    equivalent of "load Qwen 0.5B + 1 forward + assert shape/dtype"
    for Tier 1 (no network).
    """
    class _TinyLM(torch.nn.Module):
        def __init__(self, vocab: int, hidden: int) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, hidden)
            self.head = torch.nn.Linear(hidden, vocab, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.emb(ids))

    teacher = _TinyLM(vocab=64, hidden=16).cpu()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Caller-side input: simulate "student lives on CPU" for tier-1.
    ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = cpu_int8.cpu_teacher_forward(teacher, ids)
    assert logits.shape == (2, 8, 64), (
        f"unexpected output shape {logits.shape}, expected (2, 8, 64)"
    )

    # Cast contract — request bf16 on a CPU forward should land at bf16.
    logits_bf16 = cpu_int8.cpu_teacher_forward(
        teacher, ids, out_dtype=torch.bfloat16
    )
    assert logits_bf16.dtype == torch.bfloat16, (
        f"out_dtype={torch.bfloat16} requested but got {logits_bf16.dtype}"
    )


# ---------------------------------------------------------------------------
# Tier 2 — heavyweight, gated on RUN_HEAVY_TEACHER_TESTS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TEACHER_TESTS", "0") != "1",
    reason=(
        "Heavyweight Qwen 2.5 0.5B load test; run with "
        "RUN_HEAVY_TEACHER_TESTS=1 to enable. CPU CI default-skips."
    ),
)
def test_qwen_0_5b_load_and_forward():  # pragma: no cover -- heavy
    """Real Qwen 2.5 0.5B load on CPU + 1 forward + shape/dtype check.

    This is the deliverable's stated contract test, gated behind an
    env flag so we don't download 500 MB on every CI run. On the
    rental (where bitsandbytes is installed), this exercises the
    INT8 path. On a clean Windows host, the bnb load typically fails
    and we land on the fp32 fallback — both are valid.
    """
    teacher = cpu_int8.load_cpu_int8_teacher("Qwen/Qwen2.5-0.5B")
    # Frozen
    assert all(not p.requires_grad for p in teacher.parameters())
    # On CPU
    first_p = next(teacher.parameters())
    assert first_p.device.type == "cpu"

    # 1 forward at small seq_len.
    ids = torch.zeros(1, 8, dtype=torch.long)
    logits = cpu_int8.cpu_teacher_forward(teacher, ids)
    # Qwen 2.5 0.5B vocab is 151643.
    assert logits.dim() == 3
    assert logits.shape[0] == 1 and logits.shape[1] == 8
    assert logits.shape[2] in (151643, 151936)
    # Dtype: fp32 in fp32 path, fp16/bf16 in bnb int8 (linear outputs
    # cast back to compute dtype). All valid.
    assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
