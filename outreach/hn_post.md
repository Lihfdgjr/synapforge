# Hacker News submission

**Title:** Show HN: closed-form 2-3x speedup for k-step latent reasoning in CfC nets (math verified)

**URL:** https://github.com/Lihfdgjr/synapforge/blob/master/docs/RFOLD_PAPER.md

**Topic / category:** Show HN

---

## Body (paste into HN's "text" field if you submit as a Show HN with no link)

We've been working on small recurrent backbones for latent-thinking-style reasoning
(Coconut, arXiv:2412.06769) and ran into a question that turned out to have a clean
algebraic answer: when you loop a gated CfC cell R times on a *fixed* input slot, can
the loop collapse into a single algebraic step?

We wrote it up honestly, including the parts where the headline number is small and the
parts where the technique loses. Both the paper draft and the reference implementation
are linked above.

---

### What it is

`R-fold` is a closed-form replacement for the inner loop of Coconut-style latent
thinking on a gated CfC cell. The CfC step is

    pre   = W_in x + W_h h
    g     = sigmoid(W_gate pre)             <- data-dependent gate
    h'    = (1 - g*alpha) * h + g*alpha * tanh(pre)

We make two approximations: (1) freeze `g` at iteration anchor `h_0`, and (2) linearize
`tanh` at the same anchor. The recurrence collapses to `h_{t+1} = M h_t + b`. Then for
R steps:

    h_R = M^R h_0 + (I - M)^{-1} (I - M^R) b

`M^R` via repeated squaring (O(N^3 log R)). One LU solve for the geometric piece. We
force fp32 inside the fold (bf16 catastrophic at R > 16) and ridge `(I - M)` to make the
solve well-posed when the gate is near skip.

Reference implementation: `synapforge/cells/rfold.py`, ~225 LOC including a Neumann-series
CPU variant and chunked re-anchoring for large R. There's a `cfc_rfold_auto` dispatcher
that picks the right path by `(device, N, R)`.

---

### Why it matters

Coconut and Recurrent-Depth Transformers want to use a few extra latent steps as a
test-time-compute axis. The cost has been linear in `R`, which limits how many steps
you can afford per token. A `2-3x` reduction in the latent loop's wall-clock makes the
"a couple more thinking steps" knob nearly free. That's narrow but useful.

This is *not* a replacement for parallel scan in S4 / S5 / Mamba: those already exploit
linearity and don't need a fold. R-fold is for the gated, non-linear CfC family
specifically.

---

### Empirical numbers (honest)

All measurements on a consumer GPU (RTX 4070-class) with `B=8`. Sequential reference
is the same code path with the loop unrolled.

Math correctness:

| R | rel_err |
|---|---|
| 1 | 1.55e-6 (linearization exact at h_0) |
| 8 | 0.32% |
| 16 | ~0.9% (single fold) |

Speed:

| N | R | seq_ms | fold_ms | speedup |
|---|---|---|---|---|
| 64 | 4 | 0.61 | 0.34 | 1.79x |
| 64 | 8 | 1.04 | 0.41 | 2.54x |
| 64 | 16 | 1.91 | 0.64 | **2.99x** (peak) |
| 256 | 8 | 1.87 | 1.42 | 1.32x |
| 512 | 8 | 3.81 | 4.63 | 0.82x (loses) |

CPU: loses for N >= 64 because of the LAPACK solve fixed cost (~50ms). Don't fold on
CPU.

Chunk ablation (R=8): single-fold reconstruction error 9.2e-3, chunk=2 brings it to
9.2e-4 (10x reduction) at half the speedup. We recommend `chunk=2` at R=8 and
`chunk=8` at R=64.

A historical note: an earlier internal draft of this work claimed `167x at R=1024`. That
was an extrapolation from O(R) -> O(log R) complexity that ignored the cubic constant
of the matpow, the solve fixed cost, and the gate-drift collapse at large R. Once the
verifier was written, the real peak was 2.99x. We kept the work because 2.99x at the
sweet spot is genuine; we wrote it up because the math is clean and the implementation
is small.

---

### Try it

Reference implementation:
- `synapforge/cells/rfold.py` (the four functions `cfc_rfold`, `cfc_rfold_neumann`,
  `cfc_rfold_chunked`, `cfc_rfold_auto`)
- `scripts/verify_rfold.py` (asserts correctness + prints a CPU sweep)
- `synapforge/demo/rfold_bench.py` (CPU/GPU sweep used in the paper)

Reproduce every number in the paper:

    bash scripts/rfold_paper_repro.sh
    # writes paper_repro/{rfold_correctness.json, rfold_speed_cpu.json, rfold_speed_gpu.json}

Paper draft:
- `docs/RFOLD_PAPER.md` (~600 LOC, references LiquidS4 2401.13386, S5 2208.04933,
  Coconut 2412.06769, Liquid-S4 follow-up 2410.10841)

Limitations we know about and have not fixed:
- gate-drift error bound is empirical, not theoretical
- `lm_head` dominates inference at LLM scale -- system-level speedup is closer to
  1.1-1.3x once the projection is included
- repeated squaring is sequential in `log R`; a parallel-scan variant should be possible
  but we haven't built it
- forced fp32 inside the fold is a memory hit on bf16-only accelerators

Feedback welcome -- especially the "this is wrong / has been done / is uninteresting"
flavor. The repo issue tracker takes both.
