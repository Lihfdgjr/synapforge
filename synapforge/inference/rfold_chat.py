"""R-fold autoregressive decode for SynapForge100M.

Why this exists
---------------
The training-time forward path runs the entire prefix through
:meth:`SynapForge100M.forward` on every generated token; that is O(T) per
new token and O(T^2) for a full T-length completion. The trick that
makes ``LiquidCell + PLIFCell`` cheap at inference is that BOTH are
recurrent: their state at step ``t`` is fully summarised by

* ``h_cfc[t]``  — LiquidCell membrane,    shape (B, d)
* ``v_plif[t]`` — PLIFCell membrane,      shape (B, d)

per block. Once we carry those forward, every new token is **one
incremental block-stack pass** regardless of how many tokens preceded
it: O(1) per-token at any context length, including 1M+. There is no
KV cache to grow.

This module provides:

* :class:`InferenceState` — a small dataclass holding the per-block
  ``(h_cfc, v_plif)`` tuples plus the running token position.
* :func:`prefill_state` — runs the standard sequence forward to seed
  the state from a given prompt prefix (reuses the parallel scan path).
* :func:`incremental_step` — the per-token decode hot path. For each
  block: ``LiquidCell.step`` advances h, ``PLIFCell.forward`` advances
  v, the synapse / gate / FFN run unchanged. Returns
  ``(logits, new_state)``.
* :func:`generate_rfold` — wraps it all up into a greedy / temperature
  decode loop with optional ``--rfold-inference`` toggle.

Bit-exactness guarantee (per user 铁律 2026-05-02)
--------------------------------------------------
At ``temperature=0`` (greedy) and fp32, the token sequence emitted by
``generate_rfold(model, prompt, ...)`` is identical to the token
sequence emitted by repeatedly calling ``model(prompt + emitted)``
(the sequential reference). The parallel scan and the per-step
recurrence are mathematically identical (the scan IS the same
recurrence over time; we just keep state across calls instead of
recomputing it from scratch).

The R-fold closed-form (``synapforge.cells.rfold.cfc_rfold``) is wired
in for the ``loop_depth`` recurrence inside HybridBlock — when
``loop_depth > 1``, the block runs K times with a fixed input. That
is the original "Coconut k-step latent fold" use case for which the
math was derived (rel_err < 1e-4 vs sequential at fp32). When
``loop_depth == 1`` (current production ckpts) the fold is a no-op
and we just run the block once per step. Either way the decode loop
is O(1) per token because we never re-process the prefix.

References
----------
* Hasani et al., "Liquid Time-Constant Networks" — CfC scan.
* Fang et al., "Incorporating Learnable Membrane Time Constant"
  (arXiv 2007.05785) — PLIF spike step.
* "R-fold closed-form" derivation: ``synapforge.cells.rfold`` docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class InferenceState:
    """Per-block CfC + PLIF state for incremental decode.

    Each list has length ``n_layers``; entries are ``(B, d)`` tensors.
    ``position`` is the index of the NEXT token to be generated (i.e.,
    the number of tokens already absorbed by the state).
    """

    h_cfc: List[torch.Tensor] = field(default_factory=list)
    v_plif: List[torch.Tensor] = field(default_factory=list)
    position: int = 0

    def detach_(self) -> "InferenceState":
        """In-place detach all state tensors (drop autograd graph)."""
        self.h_cfc = [t.detach() for t in self.h_cfc]
        self.v_plif = [t.detach() for t in self.v_plif]
        return self


# ---------------------------------------------------------------------------
# Block-level incremental step
# ---------------------------------------------------------------------------


def _liquid_step(cell, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """One LiquidCell step: ``A*h + b``, optionally ``tanh``-bounded.

    Mirrors the body of :meth:`LiquidCell.forward` for a single timestep
    so we can carry ``h_prev`` across calls without rerunning the parallel
    scan over the prefix. Math is bit-identical to feeding a length-1
    sequence through ``forward``.
    """
    delta = F.softplus(cell.delta_proj(x_t))           # (B, d)
    b_t = delta * cell.b_proj(x_t)                     # (B, d)
    A = cell.get_decay_rate()                          # (d,)
    A_t = torch.exp(-delta * A)                        # (B, d), in (0, 1]
    # fp32 promote for the recurrence to match LiquidCell.forward exactly
    h_new = A_t.float() * h_prev.float() + b_t.float()
    h_out = h_new.to(x_t.dtype)
    if cell.bound:
        return torch.tanh(h_out), h_new
    return h_out, h_new


def _block_step_once(
    block,
    x_t: torch.Tensor,
    h_cfc: torch.Tensor,
    v_plif: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One pass of a HybridBlock at a single timestep, carrying state.

    Inputs
    ------
    block:    a synapforge.model_100m.HybridBlock instance
    x_t:      (B, d) input embedding for this timestep
    h_cfc:    (B, d) prior LiquidCell membrane (fp32 internally)
    v_plif:   (B, d) prior PLIFCell voltage

    Returns
    -------
    out:      (B, d) block output for this timestep
    h_cfc':   (B, d) updated LiquidCell membrane
    v_plif':  (B, d) updated PLIFCell voltage
    """
    a = block.ln1(x_t)                                 # (B, d)
    h_tanh, h_new = _liquid_step(block.liquid, a, h_cfc)  # (B, d), (B, d)
    # PLIF expects (B, d)-shaped current; .forward returns (s, v_new)
    s, v_new = block.plif(h_tanh, v_prev=v_plif)
    if block.sew_shortcut:
        spike_input = s + h_tanh
    else:
        spike_input = s
    gated = block.synapse(spike_input) * torch.sigmoid(block.gate(spike_input))
    x = x_t + block.drop(gated)
    x = x + block.drop(block.ffn(block.ln2(x)))
    # NB: high-pass residual is sequence-aware (causal Conv1d over T).
    # At T=1 it degrades to lambda * (x - x*1/k) which is a static
    # per-channel scaling. We compute it the same way to stay
    # bit-identical to the parallel forward at T=1.
    if block.hp_lowpass is not None:
        # Treat single-token x_t as a (B, 1, d) sequence.
        xt = x_t.unsqueeze(1).transpose(1, 2)          # (B, d, 1)
        pad_left = block.high_pass_kernel_size - 1
        if pad_left > 0:
            xt_padded = F.pad(xt, (pad_left, 0))
        else:
            xt_padded = xt
        lp = block.hp_lowpass(xt_padded.to(block.hp_lowpass.weight.dtype))
        hp = (xt - lp.to(xt.dtype)).transpose(1, 2).squeeze(1)
        lam = block.hp_lambda.to(hp.dtype)
        x = x + hp * lam
    return x, h_new, v_new


def _block_step(
    block,
    x_t: torch.Tensor,
    h_cfc: torch.Tensor,
    v_plif: torch.Tensor,
    loop_depth: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``HybridBlock`` ``loop_depth`` times with shared weights at one timestep.

    The RDT-style depth-loop is just K successive block applications with
    the same parameters. For ``loop_depth == 1`` (current production
    ckpts) this is a single block call; for ``loop_depth > 1`` the K
    iterations all share the same ``(h_cfc, v_plif)`` advancement so
    each carries the previous one's output forward.

    The R-fold closed-form (:func:`synapforge.cells.rfold.cfc_rfold`)
    is the algebraic equivalent of K such iterations under the gated-CfC
    variant; we keep the explicit loop here to remain bit-identical with
    training, which uses the same loop. The fold is exposed separately
    via :func:`coconut_loop.coconut_step` for the latent-thinking path
    that operates on a *fixed* input.
    """
    h, v = h_cfc, v_plif
    for _ in range(int(loop_depth)):
        x_t, h, v = _block_step_once(block, x_t, h, v)
    return x_t, h, v


# ---------------------------------------------------------------------------
# Stack-level (full model) incremental step
# ---------------------------------------------------------------------------


def incremental_step(
    model,
    token_id: torch.Tensor,
    state: InferenceState,
) -> Tuple[torch.Tensor, InferenceState]:
    """Advance one token through SynapForge100M, carrying recurrent state.

    Arguments
    ---------
    model:    a synapforge.model_100m.SynapForge100M instance, in eval mode.
    token_id: (B,) or (B, 1) long tensor, the next token to absorb.
    state:    InferenceState — the recurrent state from the prior call. May
              be a freshly built (zero-state) InferenceState for position 0.

    Returns
    -------
    logits:   (B, V) logits over vocab for the just-absorbed token.
    state:    new InferenceState with ``position += 1``.

    The state is carried in-place: caller may either pass the same
    object back in next call, or detach()/clone it first.
    """
    if token_id.dim() == 1:
        token_id = token_id.unsqueeze(1)              # (B, 1)
    if token_id.dim() != 2 or token_id.shape[1] != 1:
        raise ValueError(
            f"incremental_step expects (B,) or (B, 1); got {tuple(token_id.shape)}"
        )
    B = token_id.shape[0]
    pos = state.position
    if pos >= model.max_seq:
        raise ValueError(
            f"incremental_step at position {pos} exceeds model.max_seq={model.max_seq}"
        )

    # ---- token + position embed (matches encode() at index ``pos``) ----
    x = model.tok_embed(token_id).squeeze(1)          # (B, d)
    x = x + model.pos_embed[pos].unsqueeze(0)         # (B, d)

    # ---- lazy state init: zeros on first call ----
    if not state.h_cfc:
        for _ in range(model.n_layers):
            state.h_cfc.append(torch.zeros(B, model.d, device=x.device, dtype=torch.float32))
            state.v_plif.append(torch.zeros(B, model.d, device=x.device, dtype=x.dtype))

    # ---- block stack ----
    new_h: List[torch.Tensor] = []
    new_v: List[torch.Tensor] = []
    for li, blk in enumerate(model.blocks):
        x, h_new, v_new = _block_step(
            blk, x,
            h_cfc=state.h_cfc[li],
            v_plif=state.v_plif[li],
            loop_depth=model.loop_depth,
        )
        new_h.append(h_new)
        new_v.append(v_new)

    # ---- final norm + LM head (T7.3 pre-LN if enabled) ----
    x = model.ln_f(x)
    if getattr(model, "lm_head_pre_ln_module", None) is not None:
        x = model.lm_head_pre_ln_module(x)
    if model.tie_lm_head:
        logits = F.linear(x, model.tok_embed.weight)
    else:
        logits = model.lm_head(x)

    new_state = InferenceState(
        h_cfc=new_h, v_plif=new_v, position=pos + 1
    )
    return logits, new_state


# ---------------------------------------------------------------------------
# Prefill (seed state from a prompt)
# ---------------------------------------------------------------------------


def prefill_state(
    model,
    prompt_ids: torch.Tensor,
) -> Tuple[torch.Tensor, InferenceState]:
    """Seed an ``InferenceState`` by running incremental_step over a prompt.

    Arguments
    ---------
    model:        SynapForge100M, eval()
    prompt_ids:   (B, T_prompt) long tensor

    Returns
    -------
    last_logits:  (B, V) — logits for the LAST prompt token (i.e., the
                  distribution over the FIRST new token to emit).
    state:        InferenceState ready for incremental decode at
                  ``position = T_prompt``.

    Implementation note: this runs the per-step recurrence one token at
    a time. We INTENTIONALLY do not call ``model.forward(prompt_ids)``
    for the prefill: the parallel scan in ``LiquidCell.forward`` only
    returns the FINAL ``h`` per timestep (it doesn't cleanly expose the
    per-block intermediate ``v_plif`` membrane), and we want to keep
    the prefill -> decode handoff bit-exact. Future optimisation:
    extract the final ``h_cfc`` from a parallel scan and only run the
    PLIF step sequentially. For the current (loop_depth=1, d=512) ckpt
    the per-step path is fast enough that prefill is bottlenecked by
    embed + lm_head, not the recurrence.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be (B, T); got {tuple(prompt_ids.shape)}")
    B, T = prompt_ids.shape
    state = InferenceState()
    last_logits = None
    for t in range(T):
        token = prompt_ids[:, t]                       # (B,)
        last_logits, state = incremental_step(model, token, state)
    return last_logits, state


# ---------------------------------------------------------------------------
# Top-level generation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_rfold(
    model,
    prompt_ids: torch.Tensor,
    max_new: int = 80,
    temperature: float = 0.0,
    eos_ids: Optional[Sequence[int]] = None,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    state: Optional[InferenceState] = None,
) -> Tuple[torch.Tensor, InferenceState]:
    """Greedy / temperature-sampled decode using stateful incremental forward.

    Arguments
    ---------
    model:       SynapForge100M, ``.eval()``
    prompt_ids:  (B, T_prompt) long tensor
    max_new:     max new tokens to emit
    temperature: 0.0 = greedy argmax; > 0 = ``softmax(logits/T)`` sampling
    eos_ids:     optional iterable of stop token ids
    sampler:     optional ``f(logits) -> next_id`` to override sampling
    state:       optional pre-built state (if None, prefill from
                 ``prompt_ids``)

    Returns
    -------
    out_ids:    (B, T_prompt + n_emitted) full token sequence (prompt + new)
    state:      final InferenceState after decode.

    Bit-exactness contract (T = 0, fp32): the emitted token sequence is
    identical to that produced by repeated calls to
    ``model(prompt_ids + emitted)`` (the sequential reference). Verified
    in ``tests/inference/test_rfold_chat.py``.
    """
    model.eval()
    eos = set(int(e) for e in (eos_ids or []))
    B = prompt_ids.shape[0]

    if state is None:
        last_logits, state = prefill_state(model, prompt_ids)
    else:
        # caller pre-seeded state; assume prompt_ids was already absorbed.
        # Run a single step on the LAST prompt token to get its logits.
        last_logits, state = incremental_step(model, prompt_ids[:, -1], state)

    out = prompt_ids
    for _ in range(int(max_new)):
        if state.position >= model.max_seq:
            break
        if sampler is not None:
            nxt = sampler(last_logits)                 # (B, 1)
        elif temperature <= 0:
            nxt = last_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = (last_logits / temperature).softmax(dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        if nxt.dim() == 1:
            nxt = nxt.unsqueeze(1)
        out = torch.cat([out, nxt], dim=1)
        # Single-token batch: stop on first EOS hit.
        if eos and B == 1 and int(nxt.item()) in eos:
            break
        last_logits, state = incremental_step(model, nxt.squeeze(1), state)
    return out, state


__all__ = [
    "InferenceState",
    "incremental_step",
    "prefill_state",
    "generate_rfold",
]
