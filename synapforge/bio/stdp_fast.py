"""STDPFastWeight — spike-timing-dependent plasticity fast-weight memory.

Standard Hebbian fast weights (Schmidhuber 1992; Ba et al. 2016) update
``W = lambda * W + xx^T``.  STDP adds **timing**: LTP when pre fires
*before* post (within ~20ms window), LTD otherwise.

Implementation
--------------
Per channel we keep two exponentially-decaying traces:

    pre_trace[t]  = decay_plus  * pre_trace[t-1]  + pre[t]
    post_trace[t] = decay_minus * post_trace[t-1] + post[t]

and update the fast-weight matrix:

    dW = a_plus * outer(post, pre_trace) - a_minus * outer(post_trace, pre)
    W  = clamp(W + dW, -1, 1)

Buffer-safety
-------------
The forward path reads ``self.W`` then mutates it in-place.  Direct
in-place mutation breaks autograd version checks, so we **emit a
detached clone** to downstream consumers and only mutate the buffer
inside the no-grad block (memory rule "torch_buffer_inplace_break_grad").

Plasticity hook
---------------
The buffer update is wrapped as a ``register_plasticity`` rule on the
parent module so distributed plasticity sync (``PlasticBufferSync``)
sees and synchronises ``W`` across replicas.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..module import Module


class STDPFastWeight(Module):
    """Spike-timing-dependent fast-weight memory.

    Args:
        hidden_size: feature dimension.
        a_plus:      LTP rate (pre-before-post weight increment).
        a_minus:     LTD rate (post-before-pre weight decrement).
        tau_plus:    LTP trace timescale in steps (decay = exp(-1/tau)).
        tau_minus:   LTD trace timescale in steps.
        clip:        hard bound on |W| after each update.

    Forward:
        x:     [B, D]  input.
        spike: optional [B, D] binary mask gating the post-trace update.

    Returns:
        out: [B, D]   q @ W^T, where W is the *current* fast-weight
                      matrix (detached so downstream callers don't
                      backprop through the running buffer).
    """

    def __init__(
        self,
        hidden_size: int,
        a_plus: float = 0.02,
        a_minus: float = 0.02,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        clip: float = 1.0,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if tau_plus <= 0 or tau_minus <= 0:
            raise ValueError("tau_plus and tau_minus must be positive")
        self.hidden_size = int(hidden_size)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.decay_plus = float(math.exp(-1.0 / tau_plus))
        self.decay_minus = float(math.exp(-1.0 / tau_minus))
        self.clip = float(clip)
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("W", torch.zeros(hidden_size, hidden_size))
        self.register_buffer("pre_trace", torch.zeros(hidden_size))
        self.register_buffer("post_trace", torch.zeros(hidden_size))
        # Register self as a plasticity rule so PlasticBufferSync sees W.
        self.register_plasticity("stdp_W", _stdp_no_op_rule)

    @torch.no_grad()
    def reset(self) -> None:
        """Zero the fast-weight matrix and traces."""
        self.W.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def forward(
        self,
        x: torch.Tensor,
        spike: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, D].  spike: optional [B, D] binary mask."""
        if x.dim() != 2 or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"x must be [B, {self.hidden_size}], got {tuple(x.shape)}"
            )
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # Snapshot W (detached) so downstream gradient does NOT touch
        # the running buffer.  This is the critical "buffer_inplace"
        # safety pattern from synapforge memory.
        W_snap = self.W.detach().clone()
        out = q @ W_snap.T

        # Plasticity update (no-grad).  Update buffers in-place after
        # computing readout so autograd version checks see consistent W.
        if self.training:
            with torch.no_grad():
                pre = k.detach().mean(dim=0)   # [D]
                post = v.detach().mean(dim=0)  # [D]
                if spike is not None:
                    if spike.shape != x.shape:
                        raise ValueError(
                            f"spike {tuple(spike.shape)} != x {tuple(x.shape)}"
                        )
                    post = post * spike.detach().mean(dim=0)
                self.pre_trace.mul_(self.decay_plus).add_(pre)
                self.post_trace.mul_(self.decay_minus).add_(post)
                ltp = self.a_plus * torch.outer(post, self.pre_trace)
                ltd = self.a_minus * torch.outer(self.post_trace, pre)
                self.W.add_(ltp - ltd)
                self.W.clamp_(-self.clip, self.clip)
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"a_plus={self.a_plus}, a_minus={self.a_minus}, "
            f"decay_plus={self.decay_plus:.3f}, decay_minus={self.decay_minus:.3f}"
        )


def _stdp_no_op_rule(module, inputs, outputs):
    """Plasticity-rule placeholder: update happens *inside* forward.

    The framework's ``plasticity_step`` will call this after each
    forward, but the actual buffer update has already been performed
    in the forward pass (so we can return the snapshot view).  This
    rule exists only to register ``W`` as a plasticity-tracked buffer
    so distributed sync picks it up.
    """
    del module, inputs, outputs


__all__ = ["STDPFastWeight"]
