# PLIF Dead Diagnosis (Run 5 -- Synap-1 Ultra) -- 2026-05-02

## Symptom

After 10000+ training steps of Run 5 (synap-1 ultra, d=1280, n_layers=16,
loop_depth=2, ffn_ratio=3.0, batch=24, T2.5/T2.6 enabled):

```
spike: mean=0.000 range=[0.000, 0.000] dead=16/16 sat=0/16
[PLIF-REVIVE] homeostatic step rate=0.000 -> thr 0.0500 -> 0.0151
```

* `spike-target-loss-weight=0.05` has been firing for 10k+ steps with no
  effect on spike rate.
* Homeostasis pulled threshold from 0.05 -> 0.0151 (3x reduction).  Still
  zero spike.
* `T2.6 lm-head-spectral-norm` is on; T2.3 surrogate-anneal is off.
* This kills the paper's #1 claim ("LNN+SNN hybrid"): the SNN component
  contributes nothing.

## Hypotheses considered

### (a) Surrogate gradient too sharp (no signal at threshold-far input)

* ATan g'(x) = alpha / (2 * (1 + (pi/2 * alpha * x)^2)).
* alpha=2.0, x = mem - thr ~ -0.05 -> grad ~ 0.96.
* That's a HEALTHY gradient.  Not the cause at runtime.

### (b) Membrane potential drifts away from threshold and never recovers

* Tested at fresh init with d=1280, T=32, post-RMSNorm input -> liquid_out
  std=0.48, max|.|=0.99.  Spike rate at init = 0.39.  HEALTHY.
* So at init PLIF fires fine.  Death must develop *during* training.

### (c) Forward pass dense bypass (`if self.training` skips PLIF)

* Searched all `if self.training:` gates in synapforge/.  PLIF forward
  has no such gate.  `dual_path_gate.py` exists but is NOT used in
  HybridBlock.  Not the cause.

### (d) Threshold can't go below floor

* `clamp_threshold(min=0.005, max=0.5)` floor is 0.005, well above zero.
  Run 5 reports threshold dropped to 0.0151, not at floor.  Not the
  immediate cause.

### (e) Spike-target-loss not actually backpropagating

* Trainer reads `m.last_spike_rate()` which returns `_last_spike_rate_live`
  -- a graph-attached tensor.  Verified in `synapforge/surrogate.py:423`.
* Loss term `(low - rate).clamp(min=0).pow(2)` flows back through
  `s_t.float().mean()` -> `spike(v_t, threshold)` -> `_ATanSurrogate` ->
  `v - threshold`.  Threshold autograd is correct.
* Sign: dL/dthr = +2*(low-rate)*ATan_grad > 0 -> Adam decreases thr.
  Correct direction.
* So the gradient IS flowing.  Why doesn't it move thr below 0.0151?

### (f) ROOT CAUSE -- collapsed liquid output post-training

When PLIF dies once (spike s=0 for all channels), the gradient chain
through the spiking sub-block becomes:

```
gated = synapse(s) * sigmoid(gate(s))
      = synapse(0) * sigmoid(gate(0))
      = 0 * sigmoid(0)
      = 0
```

So `x = x_in + dropout(0) = x_in` -- the entire spike branch contributes
nothing to the LM logits.  The CE/KD gradient flowing back into `liquid`
is `d(loss)/d(liquid_out) = synapse.grad * (...) = 0`.  This means:

1. **liquid_out gradient -> 0** when s=0.  LiquidCell is no longer
   trained by the LM signal once spikes die.
2. **synapse.weight, gate.weight** are still under weight decay drift,
   so they shrink toward 0.  Once shrunk, even when PLIF revives,
   `synapse(s)` is tiny -> downstream still ignores it.
3. **Threshold gradient via spike-target-loss** is the ONLY push left.
   Weight 0.05 vs CE ~5.0 means it's 100x weaker.  Adam updates are
   bounded, so each step moves thr by ~0.05 * 0.001 (lr) * 0.96 (grad)
   = ~5e-5.  10000 steps -> 0.5 cumulative.  But Adam normalizes, and
   the grad direction is consistent, so thr does drop -- but not enough
   when liquid_out has already collapsed.

## What does Run 5 actually look like at step 10000?

Hypothesis: at step 10000, the liquid_out is no longer a `tanh(.)` with
|max| ~0.5.  Instead `delta_proj.weight` and `b_proj.weight` have shrunk
under weight decay (no LM gradient to oppose decay) so:

```
delta = softplus(W_delta x_t)  -- W_delta near 0 -> delta near softplus(0) = log 2 = 0.69
A_t   = exp(-delta * exp(A_log))  -- A_log fixed (Hasani init), exp(A_log) in [0.5, 2.0]
                                  -- A_t in [exp(-0.69*2), exp(-0.69*0.5)] = [0.25, 0.71]
b_t   = delta * (W_B x_t)  -- W_B near 0 -> b_t near 0
h_t   = A_t * h_{t-1} + 0  -- decays to 0 from any h0
out   = tanh(h_t) -> 0
```

So liquid_out DEGENERATES to 0.  Membrane integrates to 0.  PLIF stays
dead permanently.  This is a classic positive-feedback dead-end:

```
spike=0 -> synapse(0)=0 -> upstream CE grad has no path through plif
       -> liquid weights decay to 0 -> liquid_out -> 0
       -> mem -> 0 -> spike stays 0
```

## What unblocks it

Three orthogonal ways to break the feedback loop, each behind opt-in flag:

### Fix #1: Dense PLIF Bypass (warmup mode, then hand off)

For the first N steps, the `forward_seq` returns `tanh(mem)` (a dense
analog of "spike rate") instead of `(mem >= thr)`.  This:
- keeps the LM gradient flowing through liquid_out -> tanh(mem) ->
  synapse, so `liquid` and `synapse` train normally
- after N steps, switch to spike forward -- mem is already in the right
  range, surrogate gradient is well-defined
- inspired by `feedback_plif_dead_bootstrap.md` (SEW recipe)

### Fix #2: Increase spike-target-loss-weight 10x

Run 5 used 0.05; bump default to 0.5.  This puts the spike penalty in
the same order of magnitude as the CE loss.  Should pull thresholds
down faster.  Risk: if model finds it cheaper to drive `mem` very high
(saturate) than to fix collapsed liquid weights, we get sat=N/N.

### Fix #3: SEW-shortcut residual (arxiv 2102.04159)

Instead of `gated = synapse(s) * sigmoid(gate(s))`, use:
```
gated = (synapse(s) + h_pre_plif) * sigmoid(gate(s))
```
i.e. the synapse adds the spike contribution to the liquid output.
The dense liquid path is preserved, so liquid weights have a non-zero
LM gradient even when s=0.  SEW preserves spike binary-ness for
inference but lets the dense path carry the LM signal during training.

### Fix #4: Surrogate-width init at 5.0 (wide), anneal to 1.0

Wider surrogate (smaller alpha) gives gradient over a much wider
membrane range.  At alpha=0.5, the gradient is non-zero out to |x|=4.
This is the "warmup" surrogate trick.  Already discussed (feedback memory)
but Run 5 didn't enable it.  T2.3 -- not enabled.

## Decision

Ship #1 (dense bypass) + #2 (weight 10x) + #3 (SEW shortcut) all behind
opt-in flags (default OFF, Run 5 still bit-identical).  Add tests that
each fix independently lifts spike rate from 0 in a 50-step micro-train.

## Experiment plan

| Experiment | Flag | Spike rate at step 50 | Test |
|------------|------|-----------------------|------|
| Baseline (Run 5 settings) | (none) | 0 (collapsed) | smoke_baseline |
| Fix #1 dense bypass first 50 | --plif-dense-bypass-steps 30 | >0.1 | smoke_dense_bypass |
| Fix #2 weight 10x | --spike-target-loss-weight 0.5 | >0.05 | smoke_weight_10x |
| Fix #3 SEW residual | --plif-sew-shortcut | >0.1 (always, since dense path open) | smoke_sew |

## Verification

Each fix has a CPU smoke test under `tests/integration/test_plif_dead_fix.py`:
- builds a tiny SynapForge100M (d=64, n=2, T=8)
- collapses LiquidCell weights to 0 (simulating end-of-bad-Run-5)
- runs 50 steps of a fake CE+spike-target loss
- asserts spike rate > 0.05 at the end

## References

* Fang et al. (2021) -- PLIFCell + ATan surrogate
* Fang et al. (NeurIPS 2025, arXiv:2505.18608) -- SNN frequency limits
* arXiv:2102.04159 -- SEW (Spike-Element-Wise) shortcut
* Memory feedback_plif_dead_bootstrap.md -- SEW dense-bypass recipe
* synapforge/memory/dual_path_gate.py -- existing dual-path scaffolding
