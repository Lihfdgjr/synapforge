"""MASTER_PLAN.md §6 P2 + P13 -- KD chunk auto-tune from VRAM headroom.

Validates ``train_100m_kd._kd_chunk_size`` returns sensible chunk sizes
across the three regimes that matter for the 100M trainer:

* abundant VRAM (A800-80GB cold start) -> cap at ``batch_size``
* tight VRAM (post-OOM 5GB free)        -> chunk drops below 16
* near-empty VRAM (200MB free)           -> chunk == 1 (floor)
* CUDA unavailable (CPU smoke)           -> falls back to ``bs // 4``

The whole module is mocked at ``torch.cuda.mem_get_info`` so it runs on
this Windows dev box without GPU.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_module():
    """Import ``train_100m_kd`` lazily; skip cleanly when torch is absent."""
    pytest.importorskip("torch")
    # train_100m_kd lives at repo root; importable because of sys.path above.
    if "train_100m_kd" in sys.modules:
        return importlib.reload(sys.modules["train_100m_kd"])
    return importlib.import_module("train_100m_kd")


def test_abundant_vram_returns_full_batch():
    """80GB free with bs=128, seq=256, vocab=151936 -> chunk == bs (128)."""
    mod = _import_module()
    with patch.object(mod.torch.cuda, "is_available", return_value=True), \
         patch.object(mod.torch.cuda, "mem_get_info",
                      return_value=(int(80e9), int(80e9))):
        chunk = mod._kd_chunk_size(128, 256, 151936)
    # 80GB * 0.5 / (256 * 151936 * 4) ~= 257 rows, capped at batch_size=128.
    assert chunk >= 64, f"abundant VRAM should return >=64; got {chunk}"
    assert chunk == 128, f"should cap at batch_size; got {chunk}"


def test_tight_vram_drops_chunk_to_safe_size():
    """5GB free should shrink chunk to <=16 (math: ~16 with 0.5 headroom)."""
    mod = _import_module()
    with patch.object(mod.torch.cuda, "is_available", return_value=True), \
         patch.object(mod.torch.cuda, "mem_get_info",
                      return_value=(int(5e9), int(80e9))):
        chunk = mod._kd_chunk_size(128, 256, 151936)
    # 5GB * 0.5 / (256 * 151936 * 4) ~= 16.06 rows.
    assert 1 <= chunk <= 16, f"tight VRAM should give <=16; got {chunk}"


def test_near_empty_vram_floors_at_one():
    """200MB free -> per-row budget exhausted, chunk floors at 1."""
    mod = _import_module()
    with patch.object(mod.torch.cuda, "is_available", return_value=True), \
         patch.object(mod.torch.cuda, "mem_get_info",
                      return_value=(int(200e6), int(80e9))):
        chunk = mod._kd_chunk_size(128, 256, 151936)
    # 200MB * 0.5 / (256 * 151936 * 4) ~= 0.64 rows -> floor at 1.
    assert chunk == 1, f"near-empty VRAM should floor at 1; got {chunk}"


def test_cpu_fallback_uses_bs_div_4():
    """CUDA unavailable -> deterministic bs//4 fallback (preserves prior behavior)."""
    mod = _import_module()
    with patch.object(mod.torch.cuda, "is_available", return_value=False):
        chunk = mod._kd_chunk_size(128, 256, 151936)
    assert chunk == 32, f"CPU fallback should be bs//4 (32); got {chunk}"

    # bs=1 path: floor at 1, never 0.
    with patch.object(mod.torch.cuda, "is_available", return_value=False):
        chunk_bs1 = mod._kd_chunk_size(1, 256, 151936)
    assert chunk_bs1 == 1, f"CPU fallback floor=1 even for bs=1; got {chunk_bs1}"


def test_headroom_parameter_scales_chunk():
    """headroom=1.0 should give ~2x the chunk of headroom=0.5 (linear)."""
    mod = _import_module()
    with patch.object(mod.torch.cuda, "is_available", return_value=True), \
         patch.object(mod.torch.cuda, "mem_get_info",
                      return_value=(int(5e9), int(80e9))):
        chunk_50 = mod._kd_chunk_size(128, 256, 151936, headroom=0.5)
        chunk_100 = mod._kd_chunk_size(128, 256, 151936, headroom=1.0)
    # Doubled headroom -> ~doubled chunk (allow off-by-one for int floor).
    assert chunk_100 >= chunk_50, \
        f"larger headroom should give >= chunk: 0.5={chunk_50} 1.0={chunk_100}"
    assert chunk_100 in (chunk_50 * 2, chunk_50 * 2 - 1, chunk_50 * 2 + 1), \
        f"headroom should scale linearly: 0.5={chunk_50} 1.0={chunk_100}"


def test_kd_loss_passes_chunk_override():
    """``--kd-chunk N`` path: explicit override wins over auto-tune."""
    mod = _import_module()
    torch = mod.torch
    F = mod.F  # noqa: F841 -- used implicitly via mod._kd_loss

    # Tiny tensors; chunk_override=2 means we'll loop bs/2 = 2 iters at bs=4.
    student = torch.randn(4, 3, 16)
    teacher = torch.randn(4, 3, 16)
    loss = mod._kd_loss(student, teacher, T=2.0, chunk_override=2)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0, "KD loss must be scalar"
    assert torch.isfinite(loss).item(), "KD loss must be finite"

    # chunk_override=0 hits the auto-tune path; mock to CPU fallback.
    with patch.object(torch.cuda, "is_available", return_value=False):
        loss_auto = mod._kd_loss(student, teacher, T=2.0, chunk_override=0)
    assert torch.isfinite(loss_auto).item()
