# TOKEN_SOUP_DIAGNOSIS — Synap-1 Ultra Run 6

**Date:** 2026-05-02
**Branch:** `feature/token-soup-diagnosis`
**Status:** root-cause located; fix proposed for Run 7

## 1. Root cause (one sentence)

**The backbone (CfC + PLIF + SwiGLU) is producing near-zero hidden state and contributes essentially nothing to the logits; what little signal makes it to the LM head is the *per-row L2 norm of the tied tok_embed table*, which after spectral_norm trains directly on token unigram frequency — so the model emits global high-frequency tokens regardless of context.**

This is not a data problem, not a tokenizer problem, and only superficially a PLIF problem. The LM head and the embed table are the same matrix, the backbone is broken, and CE is being minimised by collapsing onto the unigram prior of the teacher's softmax (whose argmax over a typical fineweb prefix is `","`, `" the"`, `" of"`, `"."`).

## 2. Evidence

1. **Top-10 high-norm embed rows are exactly the failure modes.** Loaded `step_020000.pt`; `tok_embed.weight_orig.norm(dim=1).topk(10)` gives ids `{11, 279, 315, 13, 311, 323, 304, 264, 429, 624}` which decode to `{",", " the", " of", ".", " to", " and", " in", " a", " that", ".\n"}`. After spectral_norm normalises by `σ ≈ 17.26`, the top-1 row "," has effective norm 0.21 vs mean 0.032 — a **6.6× outlier** that the LM head sees on every forward.

2. **Student logit shape after a single prompt = stopword fan with ~uniform mass.** I ran the warmstart ckpt's forward pass on `"Once upon a time"` (CPU, fp32). Student top-10 by softmax probability is the same set of stopwords as the embed-norm top-10, with `top_1 = 4.6%`, `logits_max = 9.4`, `logits_min = -0.25`, `logits_mean = -0.004`. Compare teacher Qwen 2.5 0.5B on the same prompt: `top_1 = 49.3%`, top-1 token is also "," but the rest of the distribution is sharply context-conditioned (" there", " I", " when"). **Student has no context-conditioning whatsoever** — the ranking comes entirely from the per-row embed magnitude.

3. **PLIF spike rate = 0.000 across all 16 layers, all 21 000 steps.** Honest_eval logs every val pass: `spike: mean=0.000 range=[0.000, 0.000] dead=16/16 sat=0/16`. With `--sew-shortcut` ON the LiquidCell output `h` is added: `spike_input = s + h` where `s == 0`, so backbone propagation is `h → synapse → gate(σ) → +x`. But `LiquidCell` itself is stuck — see point 4.

4. **CE is flat at ~8.3 nats for 4500 steps under 0.1 LR.** `step 1: ce=8.289`; `step 4500: ce=9.022`; `step 4570: ce=8.257`. CE for uniform-over-live-vocab is `log(151643) = 11.93`, CE for "always emit ','" given typical text is `~8.5` (since "," is ~3% of fineweb tokens). The model has converged to **the unigram prior of its own embed table** and cannot improve because the gradient flowing back through `lm_head = tok_embed.weight_orig / σ` only reshapes per-row magnitudes — and the spectral_norm reparameterisation actively pushes those magnitudes back toward the trained singular value σ ≈ 17.

5. **z_loss = 130-140 (huge) every step.** `log Z = logsumexp(logits) ≈ 11.5-12`, so `z_loss = (logZ)^2 ≈ 130-145`. With `--z-loss-weight 1e-4` the contribution is 0.013 (pct_z ≈ 0.1%). z-loss is NOT dominating; it is correctly tiny. **(Hypothesis 10 from your list is therefore ruled out.)**

6. **Warmstart only restored 149/261 keys.** `[sf.hf.adv_warmstart] WarmstartReport(matched=149/261, src=437, missing=112, extra=288, shape_mismatch=0)`. The lite_mixin model that produced step_020000 had a *different* HybridBlock structure (probably MoR-style with `liquid.shared.block.A_log` etc; current Run 6 model uses `liquid.weight` etc.) so 112 of the 261 current parameters are **cold init** (random Normal std=0.02). The `tok_embed.weight_orig` did transfer, which is why the failure mode is preserved across the lite_mixin → Run 6 boundary — **the broken signal lives in the embed table, not the blocks.**

## 3. Why other candidates are NOT root cause

- **(1) Data quality.** fineweb_edu 2.15 GiB / 726 000 rows tokenised with Qwen 2.5 — that's at least 100M tokens, more than enough for a 535M model that can't learn anything *anyway*. The teacher emits sharp distributions on the same data (top-1 = 49%), proving the data carries signal. The student would be in the same hole on wikitext-103, alpaca, or any English corpus. **Misleading hypothesis — do not let it absorb attention.**

- **(3) KD sparse top-2048.** Teacher's `top-2048 mass = 98.96%` on the test prompt; KL approximation error < 1% — totally adequate. KD is logged at `kd ≈ 2.0-2.4` nats which is consistent with a teacher whose entropy is ~2.5 nats and a student whose distribution is wildly off. KD is the only loss term doing useful work right now.

- **(5) Tokenizer/padded-vocab leak.** Verified: `student_pad_mass[151643:] = 0.001` (0.1%) and `teacher_pad_mass = 1e-5`. The grad hook `_zero_tail_grad_hook` on rows ≥151643 is working. Padded tokens are NOT the failure mode.

- **(7) Optimizer stale.** Run 6 log: `warmstart: optim_state layout mismatch (ckpt=plasticity_adamw, running=torch_adamw); skipping load`. Optim moments cold-started; this is fine and ruled out.

- **(10) z-loss collapse.** z-loss is 0.1% of total loss. Ruled out by direct measurement.

## 4. Proposed fix for Run 7 (concrete + testable)

Build a **diagnostic ckpt** then a **clean restart**. In order:

**Step A — diagnostic (30 min, no GPU needed):** load `step_020000.pt` on CPU, swap `tok_embed.weight_orig` for the Qwen 2.5 0.5B `embed_tokens.weight` (rows 0..151643 verbatim, rest zero), then run honest_eval. Predicted result: still WORD_SALAD because the *backbone* is broken — Qwen embeds + broken backbone cannot produce coherent logits. This isolates the root cause to the backbone, not the head.

**Step B — primary fix:** disable `--lm-head-spectral-norm` AND `--tie-lm-head` for Run 7. The spectral_norm wrap on a tied 151936×1280 matrix is what couples the LM-head conditioning to the embed magnitudes. With separate `lm_head: Linear(1280, 151936, bias=False)` initialised cold (`nn.init.normal_(std=0.02)`), the gradient flows can shape lm_head independently from embed lookup. Cost: +194M params (1280 × 151936). At 535M total this is acceptable on A800-80GB.

**Step C — secondary fix:** start from RANDOM init, not from `step_020000.pt`. The warmstart is contaminated. Run 7 should be `--no-warmstart` for the first 1000 steps with KD weight bumped to 0.7 (Hinton recipe for cold-start KD). The dense-bypass should run for 4000 steps not 2000 — PLIF needs more time when the backbone is also reset.

**Step D — predictive checkpoint:** add a CI assert in `train_100m_kd.py` at step 500: `if logits.softmax().argmax(dim=-1).unique().numel() < 20: raise StuckException(...)` — fail fast on token soup so future runs cannot eat 21 000 steps of GPU time.

**Acceptance criteria for Run 7:** at step 1000, honest_eval verdict must be NOT TOKEN_SOUP and NOT WORD_SALAD (i.e., output diverse non-stopword tokens). At step 5000, val_ppl < 500. If neither holds, branch back to Step A.

## 5. Risk if my diagnosis is wrong

**Next-most-likely cause: PLIF is dead but the 535M backbone is also numerically near-zero because LiquidCell `A_log` warmstart was random-init.** The 112 missing keys include `blocks.X.liquid.shared.block.A_log` etc, so LiquidCell is cold. If untying lm_head doesn't fix it, the backbone still emits ~zero hidden state regardless of `tok_embed`'s shape.

**A/B test to disambiguate (15 min):** pre-fix-build, freeze `tok_embed.weight` AND `lm_head.weight` to the Qwen 2.5 0.5B embed (no training on either) and train Run 7 for 500 steps. If output is still token soup → backbone is broken (LiquidCell + PLIF need separate revival, possibly initialise from `synapforge.cells.liquid.LiquidCell.from_xavier` and dense-bypass-steps=10000). If output is coherent → my diagnosis is correct and the LM-head/embed coupling was the problem.

A second cause-fork is **the freeze_vocab_tail hook racing with spectral_norm**: the backward hook zeros gradients on rows ≥151643 *of `weight_orig`*, but spectral_norm computes `weight = weight_orig / σ` where `σ` depends on ALL rows including the padded tail. This means tail rows DO push σ around, which globally rescales the LM head every step. Test: disable `--freeze-vocab-tail` AND `--lm-head-spectral-norm` together.
