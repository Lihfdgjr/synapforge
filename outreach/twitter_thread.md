# R-fold Twitter / X thread (8 tweets)

Format note: each tweet is plain text, no images. Edit, don't auto-post. Char-count
each tweet before posting; X soft cap is 280 unless you want to pay-to-post longer.

---

**1/ (hook)**

R-fold: a closed-form algebraic collapse of the k-step latent-thinking loop in a gated
CfC cell.

Math verified, GPU measured, no hype:
- R=1 rel err 1.55e-6
- R=8 rel err 0.32%
- consumer GPU peak speedup 2.99x at N=64, R=16

paper draft + repro script below.

---

**2/ (motivation)**

Coconut-style latent thinking (arxiv 2412.06769) feeds the same latent slot back into a
recurrent block for k steps. It works -- but k steps is sequential, hurts throughput.

Question: can we collapse the loop into a single algebraic step on GPU?

For a gated CfC cell, yes.

---

**3/ (the math, part 1)**

Two clean approximations.

(a) Freeze the data-dependent gate at iteration anchor h_0:
    g = sigmoid(W_gate (W_in x + W_h h_0))

(b) Linearize tanh at the same anchor:
    tanh(pre_t) approximately tanh(pre_0) + sech^2(pre_0) * (pre_t - pre_0)

Together: h_{t+1} = M h_t + b. Affine.

---

**4/ (the math, part 2)**

Geometric series collapses R steps:

    h_R = M^R h_0 + (I - M)^{-1} (I - M^R) b

Repeated squaring gives M^R in O(N^3 log R). One linear solve for the geometric piece.

LiquidS4 (2401.13386) has a cousin algebra; S5 (2208.04933) too. The
Coconut-loop case is its own thing, and we name it.

---

**5/ (empirical, the honest table)**

GPU, N=64, B=8:

  R=4   speedup 1.79x
  R=8   speedup 2.54x
  R=16  speedup 2.99x  <- peak
  N=512 R=8 speedup 0.82x  <- loses

CPU: loses for N >= 64 (LAPACK solve fixed cost). Don't fold on CPU.

The infamous "167x at R=1024" extrapolation was wrong. Real number is 2.99x. We say so.

---

**6/ (when it helps / hurts)**

Helps:
- GPU with batched bmm
- small N (64-256)
- R = 4-16 (Coconut sweet spot)
- chunked re-anchoring at R >= 64

Hurts:
- CPU
- N >= 256
- Large vocab (lm_head dominates)
- R = 1 (nothing to fold)

---

**7/ (chunked re-anchoring)**

For R >= 64, single-fold gate drift kills retrieval (>8% at R=64).

Re-anchor every L steps. At R=8, chunk=2 shrinks reconstruction error 10x
(9.2e-3 -> 9.2e-4) at half the speedup.

Default rec: chunk=2 at R=8, chunk=8 at R=64.

---

**8/ (links)**

Paper draft (markdown, ~600 LOC, honest tone):
github.com/Lihfdgjr/synapforge/blob/master/docs/RFOLD_PAPER.md

Reference impl (~225 LOC, fp32-forced inside fold):
github.com/Lihfdgjr/synapforge/blob/master/synapforge/cells/rfold.py

Repro all numbers:
bash scripts/rfold_paper_repro.sh

Feedback welcome. Ruthless review especially welcome.
