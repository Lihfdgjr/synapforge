"""sf.quantize — BitNet b1.58 ternary quantization-aware training (QAT).

This module brings BitNet-style ternary weight QAT (Wang et al. 2024,
"BitNet: Scaling 1-bit Transformers for LLMs"; Ma et al. 2024,
"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits") into
synapforge. The win is at *deployment* time:

  * Weights stored as {-1, 0, +1} -> log2(3) ~= 1.58 bits per weight.
  * On commodity CPUs, ternary matmul reduces to integer add/subtract;
    bitnet.cpp reports 2-6x speedup for batch=1 inference and similar
    energy savings, with no quality loss after a short fine-tune.

We do NOT implement the int8 deployment kernel here -- that's the job of
bitnet.cpp / a future synapforge runtime backend. This file is the QAT
front-end: keep the model in fp32 during training but constrain the
forward pass to use ternarized weights, so the model adapts to the
discretization. Backward uses the straight-through estimator (STE).

Convention notes for synapforge users:

  * Embeddings (`emb`) and language-model heads (`lm_head`) are NOT
    quantized -- they're the bottleneck for token quality and benefit
    from full precision.
  * Plasticity / fast-weight buffers (`hebb_*`, `fast_*`, `stdp_*`) are
    NOT quantized -- they need fine gradient resolution for online updates.
  * The per-tensor scale `gamma = mean(|w|)` is tracked via EMA on the
    first 1000 steps then frozen, so the forward is deterministic and the
    backward STE pass-through stays consistent.

Public API
----------
    import synapforge as sf
    from synapforge.quantize import (
        TernaryLinear, convert_model_to_ternary, quantize_ternary,
    )

    n_replaced = convert_model_to_ternary(model, exclude=("emb", "lm_head"))
    print(f"replaced {n_replaced} nn.Linear -> TernaryLinear")
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

#: Names whose substring appearance in a module's qualified name causes
#: `convert_model_to_ternary` to skip it. Embeddings and LM heads are
#: always excluded by default; plasticity/fast-weight buffers are excluded
#: because they're plain `nn.Parameter`/buffers, not `nn.Linear`, so they
#: never get visited anyway.
DEFAULT_EXCLUDE: tuple[str, ...] = ("emb", "embedding", "lm_head", "head")

#: Number of forward passes during which gamma is updated via EMA. After
#: this many calls, gamma is frozen and treated as a constant. This keeps
#: the QAT pass deterministic for the bulk of training and avoids drift.
DEFAULT_GAMMA_WARMUP_STEPS: int = 1000

#: EMA momentum applied to gamma during warmup. gamma <- m * gamma + (1-m) * batch_gamma.
DEFAULT_GAMMA_EMA: float = 0.99

#: Numerical safety: gamma is clamped to >= EPS to avoid div-by-zero.
EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Core: ternary quantizer with STE
# ---------------------------------------------------------------------------


class TernaryQuantizer(torch.autograd.Function):
    """Ternary {-gamma, 0, +gamma} quantizer with straight-through gradient.

    Forward:
        w_n = w / max(gamma, EPS)
        w_q = round(w_n).clamp(-1, 1)            # in {-1, 0, +1}
        return w_q * gamma                       # dequantized for fp32 forward

    Backward (STE):
        grad_w = grad_output                     # gradient passes straight through
        grad_gamma = None                        # gamma is a buffer, not learned
                                                 # (it's tracked via EMA outside autograd)

    The forward is *not* hard-saturating: with the recommended gamma =
    mean(|w|), almost no weight has |w / gamma| > 1.5, so clamping mainly
    serves as a safety net. The {-1, 0, +1} bucketing is the actual
    discretization step.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # gamma is a 0-d tensor (per-tensor scale). Clamp for safety.
        scale = gamma.clamp(min=EPS)
        w_n = w / scale
        w_q = torch.round(w_n).clamp_(-1.0, 1.0)
        return w_q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: gradient w.r.t. weight is the upstream gradient.
        # gamma has no autograd path (it's an EMA buffer).
        return grad_output, None


def quantize_ternary(w: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize `w` to ternary using gamma (or compute gamma = mean(|w|) if None).

    This is the functional entry point. Most users go through TernaryLinear
    which handles gamma EMA bookkeeping.
    """
    if gamma is None:
        gamma = w.detach().abs().mean()
    return TernaryQuantizer.apply(w, gamma)


# ---------------------------------------------------------------------------
# TernaryLinear: drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------


class TernaryLinear(nn.Module):
    """nn.Linear-compatible layer that ternarizes its weight on every forward.

    Behavior
    --------
    * `weight` is a normal fp32 Parameter (so optimizer + gradient flow work).
    * On forward, weight is passed through `TernaryQuantizer` to emit
      effective ternary weights of value {-gamma, 0, +gamma}.
    * `gamma` is a registered buffer. During the first
      `gamma_warmup_steps` forward passes (only when training), it's
      updated via EMA. After warmup, it's frozen.

    Serialization
    -------------
    state_dict serializes the fp32 weight + gamma + bias + step counter.
    Re-loading restores the QAT-trained state. Conversion to packed
    int2/int8 weights for deployment is done at export time, not here.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gamma_warmup_steps: int = DEFAULT_GAMMA_WARMUP_STEPS,
        gamma_ema: float = DEFAULT_GAMMA_EMA,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.gamma_warmup_steps = int(gamma_warmup_steps)
        self.gamma_ema = float(gamma_ema)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # gamma is a buffer (not learned, tracked via EMA on the first
        # gamma_warmup_steps calls). Stored as a 0-d tensor so it goes
        # along with .to(device).
        self.register_buffer("gamma", torch.zeros((), dtype=torch.float32))
        # Step counter (integer buffer) so we know when to freeze gamma
        # and so it serializes with state_dict.
        self.register_buffer("ternary_step", torch.zeros((), dtype=torch.long))
        # First-call flag: the very first forward should *initialize* gamma
        # rather than EMA-blend it (otherwise we mix with a 0 init).
        self.register_buffer("ternary_initialized", torch.zeros((), dtype=torch.bool))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def _update_gamma(self) -> None:
        """Update the per-tensor scale `gamma` via EMA. Called only in training."""
        # Avoid recompute per-step after warmup.
        step = int(self.ternary_step.item())
        if step >= self.gamma_warmup_steps:
            return
        batch_gamma = self.weight.detach().abs().mean()
        if not bool(self.ternary_initialized.item()):
            # First call: just snapshot the current weight scale.
            self.gamma.copy_(batch_gamma)
            self.ternary_initialized.fill_(True)
        else:
            new = self.gamma_ema * self.gamma + (1.0 - self.gamma_ema) * batch_gamma
            self.gamma.copy_(new)
        self.ternary_step.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_gamma()
        else:
            # Eval-time safety: if the module was never trained, init gamma now.
            if not bool(self.ternary_initialized.item()):
                with torch.no_grad():
                    self.gamma.copy_(self.weight.detach().abs().mean())
                    self.ternary_initialized.fill_(True)
        # Use the *frozen* gamma for forward (or the warmup-blended one).
        # Cast gamma to weight dtype so mixed-precision works.
        gamma = self.gamma.to(self.weight.dtype)
        w_q = TernaryQuantizer.apply(self.weight, gamma)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ternary=True"
        )

    @torch.no_grad()
    def quantized_weight(self) -> torch.Tensor:
        """Return the *current* ternary weight as a fp32 tensor in {-gamma, 0, +gamma}.

        Useful for tests / inspection / packed-int export.
        """
        gamma = self.gamma.clamp(min=EPS).to(self.weight.dtype)
        w_n = self.weight.detach() / gamma
        w_q = torch.round(w_n).clamp_(-1.0, 1.0)
        return w_q * gamma

    @torch.no_grad()
    def ternary_codes(self) -> torch.Tensor:
        """Return the ternary code tensor as int8 in {-1, 0, +1} (no scale).

        This is what bitnet.cpp / a future packed kernel would consume.
        Combine with `self.gamma` for dequantization.
        """
        gamma = self.gamma.clamp(min=EPS)
        w_n = self.weight.detach() / gamma
        return torch.round(w_n).clamp_(-1.0, 1.0).to(torch.int8)


# ---------------------------------------------------------------------------
# Whole-model conversion utility
# ---------------------------------------------------------------------------


def _should_exclude(qualname: str, exclude: Iterable[str]) -> bool:
    """Return True if any token in `exclude` appears as a substring in `qualname`."""
    return any(token and token in qualname for token in exclude)


def convert_model_to_ternary(
    model: nn.Module,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    gamma_warmup_steps: int = DEFAULT_GAMMA_WARMUP_STEPS,
) -> int:
    """Walk `model`, replace every `nn.Linear` with `TernaryLinear`.

    Parameters
    ----------
    model
        Any nn.Module / sf.Module subtree. Mutated in place.
    exclude
        Iterable of substring tokens. If any appears in a submodule's
        qualified name (e.g. "transformer.lm_head.weight"), that
        submodule's children are NOT converted. Defaults to embeddings
        and LM heads.
    gamma_warmup_steps
        Forwarded to each new `TernaryLinear`.

    Returns
    -------
    Number of `nn.Linear` layers replaced.

    Notes
    -----
    * Plasticity / Hebbian / STDP weights live in `nn.Parameter` or buffers
      attached directly to a module, NOT inside `nn.Linear`, so they're
      never visited and never quantized -- which is the correct behavior.
    * The replacement copies the weight and bias data so the post-conversion
      model starts numerically equivalent (modulo discretization noise).
      That's why a short post-conversion fine-tune (1-5% of original steps)
      typically recovers full accuracy.
    """
    exclude = tuple(exclude)
    count = 0
    # iterate over (parent_module, child_name, child) so we can rebind.
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualname = f"{parent_name}.{child_name}" if parent_name else child_name
            if _should_exclude(qualname, exclude):
                continue
            tern = TernaryLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                gamma_warmup_steps=gamma_warmup_steps,
            )
            tern.weight.data.copy_(child.weight.data)
            if child.bias is not None and tern.bias is not None:
                tern.bias.data.copy_(child.bias.data)
            # Move to the same device/dtype as the original child.
            tern = tern.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, tern)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Helpers for tests / inspection
# ---------------------------------------------------------------------------


@torch.no_grad()
def freeze_gamma(model: nn.Module) -> int:
    """Force every TernaryLinear in `model` past warmup so gamma is frozen.

    Returns the number of layers frozen. Useful before evaluation or before
    exporting the model -- guarantees no further EMA updates.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.ternary_step.fill_(m.gamma_warmup_steps)
            m.ternary_initialized.fill_(True)
            n += 1
    return n


@torch.no_grad()
def count_ternary_params(model: nn.Module) -> tuple[int, int]:
    """Return (n_ternary_params, n_total_params) over all parameters.

    `n_ternary_params` counts elements that *would* be stored as ternary
    codes at deployment (i.e. weights inside TernaryLinear, excluding bias).
    """
    n_tern = 0
    n_total = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            n_tern += int(m.weight.numel())
    for p in model.parameters():
        n_total += int(p.numel())
    return n_tern, n_total


__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_GAMMA_WARMUP_STEPS",
    "DEFAULT_GAMMA_EMA",
    "TernaryQuantizer",
    "TernaryLinear",
    "quantize_ternary",
    "convert_model_to_ternary",
    "freeze_gamma",
    "count_ternary_params",
]
