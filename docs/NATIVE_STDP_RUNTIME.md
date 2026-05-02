# Native STDP Runtime

`synapforge/native/stdp/` — a local-Hebbian replacement for the
AdamW step on plasticity-tagged weights. Plasticity weights bypass
autograd entirely and update via STDP (Spike-Timing-Dependent
Plasticity), the biological learning rule:

```
Delta_w_ij(t) = a_plus  * post_spike[i,t] * pre_trace[j,t-1]   (LTP)
              - a_minus * post_trace[i,t-1] * pre_spike[j,t]   (LTD)

pre_trace[t]  = decay_pre  * pre_trace[t-1]  + pre_spike[t]
post_trace[t] = decay_post * post_trace[t-1] + post_spike[t]
```

No gradient. No optimizer state. No backward pass. Cost per step is
`O(active_pre * active_post)`, not `O(weights)` — at 10% spike
density that's a 100x reduction.

## Why STDP, not AdamW

AdamW for plasticity weights costs five things STDP doesn't pay:

| Cost                      | AdamW | STDP |
|--------------------------|-------|------|
| H2D grad copy             | yes   | no   |
| First-moment math (m)     | yes   | no   |
| Second-moment math (v)    | yes   | no   |
| Decoupled WD              | yes   | no   |
| D2H param copy (offload)  | yes   | no   |

For the 730M parameter model the AdamW step is the CPU bottleneck
(~140 ms/step on 16-core Xeon). The fraction of that time spent on
plasticity weights is proportional to `n_plasticity / n_total`. If
plasticity is ~5M of 730M (0.7%), the absolute step-time win is
small (~1 ms). The bigger wins are:

1. **Inference-time learning.** AdamW needs `loss.backward()` —
   inference has no loss to backward. STDP runs on the spikes
   themselves, so plasticity weights keep adapting at inference.
   This is the in-context-learning lever.

2. **Sparse-spike scaling.** When 90% of layers have zero co-firing
   in a step, STDP costs zero for those layers. AdamW always pays
   `O(weights)` even on a "do nothing" step.

3. **No autograd graph leakage.** The plasticity-aware AdamW wraps
   each param in a `MultiSourceParam` with id-based dict lookups; on
   the 730M model that's a dict probe per param per step. STDP
   skips the dict entirely.

## Honest perf numbers

`scripts/bench_stdp_vs_adamw.py` (CPU, Python 3.8 + numpy 1.24 +
torch 2.0):

| Setup                             | AdamW (us) | STDP (us) | Speedup |
|-----------------------------------|-----------:|----------:|--------:|
| 5M params, 256x256, 10% density   |    26,237  |   14,144  |   1.86x |
| 5M params, 256x256, 2% density    |    29,684  |   11,368  |   2.61x |
| 5M params, 2048x2048, 1% density  |    24,739  |    8,015  |   3.09x |

CPU bottleneck is Python per-layer overhead (76 layers x dict
lookup + `np.flatnonzero`). The Triton kernel
(`triton_kernel.py`) fuses LTP + LTD + clamp into two grid passes
on CUDA, removing the per-layer overhead. **We expect 50x+ on real
GPU hardware** but cannot exercise CUDA from this CPU-only test
environment.

## Wiring into the trainer

Default OFF; opt in with `--stdp-only-plasticity`:

```bash
python train_100m_kd.py \
    --stdp-only-plasticity \
    --stdp-alpha 0.02 \
    ...other-flags
```

When OFF the dispatcher is not constructed and the trainer's
optimizer codepath is bit-identical to current production. Quality
guard.

When ON, params are split:

* `_sf_grad_source` subset of `{stdp, hebb, synaptogenesis}` ->
  `STDPOnlyOptimizer` (no AdamW)
* everything else -> the existing AdamW (vanilla / fused /
  synapforge / cpu-offload, whichever was already chosen)

The dispatcher steps both halves; STDP runs after AdamW so the
local rule sees the freshly-updated BP weights — same
OBSERVE/DELTA/APPLY contract as `synapforge.plasticity`.

## Spike observation

The producer (the SNN forward pass) is responsible for
calling `dispatcher.observe_spike(layer_name, pre_spike,
post_spike)` after each forward. `pre_spike` and `post_spike` are
1D binary arrays of shape `(in_dim,)` and `(out_dim,)`.

Layers without registered observations remain at their initialised
weights — STDP only fires on observed pairs. To wire a layer in,
ensure (a) its weight is tagged `_sf_grad_source = ["stdp"]` (or
`["hebb"]`), and (b) its forward pass pushes spikes:

```python
class MyHebbianLayer(nn.Module):
    def __init__(self, in_d, out_d, layer_name):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(out_d, in_d))
        self.w._sf_grad_source = ["stdp"]  # tag for the dispatcher
        self.layer_name = layer_name

    def forward(self, x, spike_in, spike_out, dispatcher=None):
        out = (self.w * mask) @ x
        if dispatcher is not None:
            dispatcher.observe_spike(
                self.layer_name,
                spike_in.detach().cpu().numpy(),
                spike_out.detach().cpu().numpy(),
            )
        return out
```

For models that already use `synapforge.bio.STDPFastWeight` or
`synapforge.action.NeuroMCPHead`, the spike arrays are already
computed; we only need to push them into the dispatcher.

## Per-layer tau scaling

Different layers have different CfC tau. Plasticity should match:
fast-tau layers adapt fast, slow-tau layers slow. The dispatcher
accepts an optional `per_param_alpha` mapping computed by

```python
from synapforge.native.stdp import per_param_alpha

alphas = per_param_alpha(
    base_alpha=0.02,
    tau_per_param={id(p): tau for p, tau in tau_map.items()},
)
```

The result is `id(param) -> alpha` with geometric-mean rescaling so
no single layer dominates. Outside the `[1e-6, 1.0]` band alphas
are clipped.

## Limitations

* **STDP is local.** Deep credit assignment still needs backprop.
  We split the model so deep layers (BP-tagged) keep their full
  AdamW, only plasticity layers go local.

* **No second-order moments.** AdamW's m/v give per-coordinate LR
  adaptation; STDP relies on the trace and the per-layer alpha
  alone. For very-non-stationary plasticity weights this can be
  noisier; tune `a_plus / a_minus` asymmetry instead.

* **CPU bench is overhead-bound.** The numpy fast path tops out at
  ~3x on CPU because Python-loop-per-layer dominates. The Triton
  kernel removes this on GPU but we cannot demonstrate that on
  the CPU-only CI bench machine.

* **No grad accumulation.** Each `step()` consumes exactly one
  pushed observation. If you accumulate over multiple forwards,
  you must call `observe()` for each; the trace handles the
  temporal smoothing internally.

## File map

```
synapforge/native/stdp/
  __init__.py                    public API
  spike_buffer.py                SpikeRingBuffer (numpy ring + EMA)
  stdp_optimizer.py              STDPOnlyOptimizer + sparse step
  hybrid_optim_dispatch.py       routes plasticity to STDP, BP to AdamW
  per_param_lr.py                per-layer alpha = base * (tau/tau_avg)
  triton_kernel.py               fused LTP/LTD/clamp on CUDA

scripts/bench_stdp_vs_adamw.py   AdamW vs STDP microbench
docs/NATIVE_STDP_RUNTIME.md      this file
tests/native/stdp/               24 tests (numpy + torch paths)
```
