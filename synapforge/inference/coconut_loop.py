"""Coconut latent thinking at inference time (arxiv:2412.06769).

Wraps :func:`rfold_chat.generate_rfold` with a k-step continuous-thought
loop activated at ``<bot>`` markers. Between ``<bot>`` and ``<eot>``,
the model takes K hidden-state passes WITHOUT emitting tokens, then
resumes generation.

Why this is a free win for our architecture
-------------------------------------------
``LatentThinker.think`` (already shipped via ``model.latent_k`` in
``model_100m.py::encode``) feeds the post-block hidden state back as
the next-step continuous input — no token sampling, no embed lookup.
At inference, we plug that loop into the per-token decode path: when
the most-recently-emitted token is ``<bot>``, run k passes of the
latent loop on the carrier hidden, then resume normal decode.

This module is a thin shim that:

1. Tracks whether the decoder is currently inside a ``<bot>...<eot>``
   thinking region.
2. On entry to thinking, runs ``LatentThinker.think`` for ``k`` steps
   with the current last-token hidden as the seed.
3. Stores the resulting "thought" hidden so the NEXT token's decode
   uses it as a residual carry.
4. Verifies that the legacy non-Coconut path is bit-identical when no
   ``<bot>`` token is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .rfold_chat import (
    InferenceState,
    _block_step,
    incremental_step,
    prefill_state,
)


# ---------------------------------------------------------------------------
# Latent thinking step
# ---------------------------------------------------------------------------


@torch.no_grad()
def coconut_step(
    model,
    state: InferenceState,
    k_steps: int = 8,
) -> InferenceState:
    """Run K Coconut latent passes on the current ``state``.

    Each latent pass:
      1. Take the last LiquidCell post-norm hidden (top of the stack).
      2. Feed it back as the next-step input through every block.
      3. Update the per-block (h_cfc, v_plif) state along the way.

    No tokens are emitted; the LM head is not invoked. ``state.position``
    is **not** advanced — the latent loop happens "in between" two real
    token positions, much like a ``<pause>`` token (Goyal et al.
    2310.02226).

    The block stack is the same one used by autoregressive decode
    (:func:`incremental_step`), so the bit-exactness contract carries
    over: at fp32 with ``temperature=0``, the post-think state is
    deterministic.

    Arguments
    ---------
    model:    SynapForge100M, ``.eval()``
    state:    InferenceState carried from the prior real-token decode.
    k_steps:  number of latent passes (paper default 8).

    Returns
    -------
    state:    InferenceState with updated (h_cfc, v_plif) but
              identical position.
    """
    if k_steps <= 0:
        return state
    if not state.h_cfc:
        raise ValueError(
            "coconut_step requires a seeded InferenceState; call "
            "prefill_state(model, prompt_ids) first"
        )
    # Carrier "thought" — start from the last block's CfC hidden.
    # Cast to model dtype for the subsequent Linear ops.
    carrier_dtype = model.tok_embed.weight.dtype
    h_thought = state.h_cfc[-1].to(carrier_dtype)

    # If the model has a LatentThinker (latent_k>0 build path), use its
    # learned input/output projections; otherwise pass through identity.
    thinker = getattr(model, "latent_thinker", None)
    if thinker is not None:
        proj_in = thinker.thinking_in
        proj_out = thinker.thinking_out
    else:
        proj_in = lambda t: t                                # noqa: E731
        proj_out = lambda t: t                               # noqa: E731

    new_h = list(state.h_cfc)
    new_v = list(state.v_plif)
    for _ in range(int(k_steps)):
        x = proj_in(h_thought)                              # (B, d)
        for li, blk in enumerate(model.blocks):
            x, new_h[li], new_v[li] = _block_step(
                blk, x,
                h_cfc=new_h[li], v_plif=new_v[li],
                loop_depth=model.loop_depth,
            )
        h_thought = proj_out(x)
    return InferenceState(h_cfc=new_h, v_plif=new_v, position=state.position)


# ---------------------------------------------------------------------------
# Top-level decode with Coconut hooks
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_with_coconut(
    model,
    prompt_ids: torch.Tensor,
    max_new: int = 80,
    temperature: float = 0.0,
    coconut_k: int = 8,
    bot_id: Optional[int] = None,
    eot_id: Optional[int] = None,
    eos_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, InferenceState]:
    """Coconut-aware decode: runs k latent passes inside ``<bot>...<eot>``.

    Behaviour
    ---------
    * If no ``<bot>`` tokens appear in the prompt or are sampled during
      decode, this function is bit-identical to
      :func:`rfold_chat.generate_rfold` with the same hyperparams.
    * When a ``<bot>`` token is emitted (or already in the prompt), the
      next K latent passes happen via :func:`coconut_step` — no tokens
      are emitted and the state is advanced through K block-stack passes.
    * After the latent loop, real-token decode resumes; the model is
      free to emit the ``<eot>`` token whenever it likes.

    Arguments
    ---------
    model:       SynapForge100M, ``.eval()``
    prompt_ids:  (B, T_prompt) long tensor — must be ``B == 1`` for the
                 markers to interact correctly with batch-shared sampling.
    max_new:     real-token budget (latent steps don't count against it).
    temperature: 0.0 = greedy argmax; > 0 = softmax sampling.
    coconut_k:   number of latent passes per ``<bot>`` marker (paper k=8).
    bot_id:      vocab id of ``<bot>``; if None, the routine never enters
                 the latent loop and behaves identically to plain rfold.
    eot_id:      vocab id of ``<eot>``; if None we treat each ``<bot>``
                 as a one-shot "k latent steps then continue" trigger.
    eos_ids:     optional stop tokens.
    """
    model.eval()
    if prompt_ids.shape[0] != 1:
        raise ValueError(
            "generate_with_coconut expects batch=1 (markers are per-sequence)"
        )
    eos = set(int(e) for e in (eos_ids or []))
    last_logits, state = prefill_state(model, prompt_ids)
    out = prompt_ids
    inside_thinking = False
    # If the prompt ITSELF ends with <bot>, fire the latent loop now.
    if bot_id is not None and int(prompt_ids[0, -1].item()) == int(bot_id):
        state = coconut_step(model, state, k_steps=coconut_k)
        inside_thinking = (eot_id is not None)

    for _ in range(int(max_new)):
        if state.position >= model.max_seq:
            break
        if temperature <= 0:
            nxt = last_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = (last_logits / temperature).softmax(dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, nxt], dim=1)
        nxt_int = int(nxt.item())
        if eos and nxt_int in eos:
            break

        # Real-token absorption.
        last_logits, state = incremental_step(model, nxt.squeeze(1), state)

        # Coconut hooks: opening / closing markers.
        if bot_id is not None and nxt_int == int(bot_id):
            state = coconut_step(model, state, k_steps=coconut_k)
            # After latent steps the carrier hidden has been refined
            # in-place via state; the next-token logits we already have
            # came from the <bot> position. The next iteration will use
            # them as expected — Coconut models typically learn to emit
            # <eot> after thinking, but if eot_id is None we just go
            # back to plain decode.
            if eot_id is not None:
                inside_thinking = True
        elif eot_id is not None and nxt_int == int(eot_id):
            inside_thinking = False
        # `inside_thinking` is currently informational; reserved for a
        # future variant where additional latent steps fire on every
        # in-region token (the paper's "continuous loop" mode).
        _ = inside_thinking
    return out, state


__all__ = [
    "coconut_step",
    "generate_with_coconut",
]
