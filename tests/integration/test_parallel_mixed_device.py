"""Smoke test for ``synapforge.parallel.place_mixed_device`` (P14).

Layer 2 of ``parallel.py`` is documented as a 3-layer feature but, until
this test, had **zero coverage** and only one caller
(``examples/mixed_device_training.py``). Master plan §10 admitted
"mixed-device exists but not validated".

We're keeping the code (potentially useful for VRAM-tight inference) but
ship a real smoke test so the docstring's "~40% of params off the GPU"
claim is verified, and the function is proven not to crash on a
CPU-only Windows dev box (the most common pitch-laptop config).

Two paths exercised:

1. **CPU-only path** — pass ``gpu="cpu"``. The function should still produce a
   ``MixedPlacement`` where ``cpu_param_count > 0``, the named module's
   tensors live on cpu, and the placement counts are honest.
2. **GPU path** — gated behind ``pytest.mark.gpu``. When CUDA is
   available, the backbone params land on ``cuda:0`` while the named
   CPU module stays on cpu.

The toy model is intentionally tiny (~2M params) and uses plain
``nn.Embedding`` + ``nn.Linear`` so the test stays hermetic — no
SynapForge100M / Triton / model_100m import cost.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def torch_or_skip():
    return pytest.importorskip("torch")


def _build_toy_model(torch):
    """2-layer toy: large embed_tokens + small backbone.

    Shapes pinned to make assertions deterministic:
      embed_tokens: vocab=1000, d=1024 -> 1,024,000 params
      backbone:     Linear(1024, 16) + ReLU + Linear(16, 16) -> ~16.6K params
      lm_head:      Linear(1024, 1000) -> 1,025,000 params

    Total ~2.07M; embed alone is ~49.5% of params (well above the
    docstring's ~40% claim, so the test verifies that bound).
    """
    import torch.nn as nn

    class TinyChat(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1000, 1024)
            self.backbone = nn.Sequential(
                nn.Linear(1024, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
            )
            self.lm_head = nn.Linear(1024, 1000)

        def forward(self, ids):  # pragma: no cover -- not exercised here
            return self.lm_head(self.backbone(self.embed_tokens(ids)))

    return TinyChat()


def test_place_mixed_device_cpu_only_path_does_not_crash(torch_or_skip):
    """CPU-only fallback: gpu='cpu' must produce a valid MixedPlacement.

    On a CPU-only box (no CUDA), the example calls a different branch,
    but the function itself must still operate correctly when asked to
    target the cpu device — degrading to "everything on CPU" rather
    than raising.
    """
    torch = torch_or_skip
    from synapforge.parallel import MixedPlacement, place_mixed_device

    model = _build_toy_model(torch)

    placement = place_mixed_device(
        model, gpu="cpu", cpu_module_names=("embed_tokens",)
    )

    # Type contract.
    assert isinstance(placement, MixedPlacement)

    # cpu_param_count must reflect the embed_tokens module (1024 * 1000 = 1.024M).
    assert placement.cpu_param_count > 0, "embed_tokens should be counted on CPU"
    assert placement.cpu_param_count == 1024 * 1000, (
        f"expected exactly {1024 * 1000} CPU params (embed_tokens), "
        f"got {placement.cpu_param_count}"
    )

    # The targeted module name should be in cpu_modules.
    assert "embed_tokens" in placement.cpu_modules

    # All embed_tokens tensors should live on cpu.
    assert model.embed_tokens.weight.device.type == "cpu"

    # gpu_param_count covers backbone + lm_head; with gpu='cpu' they also
    # land on cpu, but the bookkeeping must still be honest.
    assert placement.gpu_param_count > 0, (
        "backbone + lm_head should be counted in gpu_param_count even when "
        "gpu='cpu' (it's the placement target, not the device name)"
    )

    # Total parameter accounting: cpu_count + gpu_count == total params.
    total_params = sum(p.numel() for p in model.parameters())
    assert placement.cpu_param_count + placement.gpu_param_count == total_params, (
        f"placement counts ({placement.cpu_param_count} + "
        f"{placement.gpu_param_count}) do not sum to total ({total_params})"
    )


def test_place_mixed_device_docstring_40pct_claim_holds(torch_or_skip):
    """Docstring says place_mixed_device 'moves ~40% of params off the GPU'.

    For our toy model with embed_tokens (1.024M) on CPU and total 2.07M,
    the CPU fraction must exceed 0.30 (we use a generous lower bound;
    actual fraction is ~0.496 for this geometry, ~0.42 for the original
    375M shape).
    """
    torch = torch_or_skip
    from synapforge.parallel import place_mixed_device

    model = _build_toy_model(torch)

    placement = place_mixed_device(
        model, gpu="cpu", cpu_module_names=("embed_tokens",)
    )

    total = placement.cpu_param_count + placement.gpu_param_count
    cpu_fraction = placement.cpu_param_count / total

    assert cpu_fraction > 0.30, (
        f"docstring claims ~40% off-GPU; toy model only achieves "
        f"{cpu_fraction:.3f} which would invalidate the claim. Either "
        f"adjust the docstring or the test geometry."
    )


def test_place_mixed_device_returns_correct_default_module_names(torch_or_skip):
    """Default cpu_module_names match the docstring claim about LM heads."""
    torch = torch_or_skip
    from synapforge.parallel import place_mixed_device

    model = _build_toy_model(torch)

    # Use defaults (embed_tokens, lm_head, lm_logits) — both embed_tokens
    # and lm_head from our toy match.
    placement = place_mixed_device(model, gpu="cpu")

    assert "embed_tokens" in placement.cpu_modules
    assert "lm_head" in placement.cpu_modules

    # With both heavy modules on CPU, fraction is even higher.
    total = placement.cpu_param_count + placement.gpu_param_count
    cpu_fraction = placement.cpu_param_count / total
    assert cpu_fraction > 0.95, (
        f"with both embed_tokens AND lm_head on CPU, CPU fraction should "
        f"be >0.95 (toy backbone is tiny); got {cpu_fraction:.3f}"
    )


@pytest.mark.gpu
def test_place_mixed_device_gpu_path(torch_or_skip):
    """When CUDA is available, backbone moves to cuda:0; embed stays on CPU."""
    torch = torch_or_skip
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from synapforge.parallel import place_mixed_device

    model = _build_toy_model(torch)

    placement = place_mixed_device(
        model, gpu="cuda:0", cpu_module_names=("embed_tokens",)
    )

    # embed_tokens stays on cpu.
    assert model.embed_tokens.weight.device.type == "cpu"

    # backbone parameters land on cuda:0.
    backbone_devices = {p.device.type for p in model.backbone.parameters()}
    assert backbone_devices == {"cuda"}, (
        f"backbone should be on cuda; got devices {backbone_devices}"
    )

    # lm_head also on cuda (not in cpu_module_names).
    assert model.lm_head.weight.device.type == "cuda"

    # Bookkeeping is honest.
    assert placement.cpu_param_count == 1024 * 1000
    assert placement.gpu_param_count > 0
