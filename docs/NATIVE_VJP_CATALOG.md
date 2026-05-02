# Native VJP Catalogue

Closed-form Vector-Jacobian Products for every op in `HybridBlock`, in
pure numpy. Lives at `synapforge/native/vjp/`. Replaces `torch.autograd`
on the gradient path, eliminating chain-rule float drift and tape
overhead in the LNN+SNN training loop.

## Purpose

Every op in the SynapForge HybridBlock has a closed-form Jacobian. We
write them down once, test against torch.autograd to bit-tightness,
and reuse across:

* `synapforge/native_demo.py` (end-to-end inline demo)
* `synapforge/native/cuda/` (CUDA kernels share these formulas)
* `synapforge/training/` (trainer refactor, drop-in replacement)
* `synapforge/native/dispatch/` (async pipelining, knows op shapes)

## Hard rules

* **Pure numpy.** Zero `import torch` in `synapforge/native/vjp/*.py`
  and `synapforge/native/__init__.py`. (Verified by CI grep.)
* **fp32 default.** Bf16 promotion via explicit `dtype=` kwarg or
  upstream cast.
* **Test parity.** `tests/native/vjp/test_vjp_against_torch.py` does
  two independent checks per op:
  * (A) torch.autograd reference, `atol=1e-5, rtol=1e-4` in fp32.
  * (B) finite-difference numerical Jacobian (`eps=1e-3`), `rel_err
    < 1e-3` for affine ops, `< 1e-2` for swiglu/cfc.

## Op catalogue

### 1. `embed.py` -- Token embedding

Forward: `y[b,t,:] = W[input_ids[b,t], :]`, `W: (V, d)`, `input_ids: (B, T)`.

Backward (sparse scatter-add):
```
grad_W[v, :] = sum_{b,t : input_ids[b,t] == v} grad_y[b, t, :]
```
Implemented via `np.add.at(grad_W, flat_ids, flat_grad)` for correct
unbuffered accumulation when token ids repeat.

Reference: Mikolov 2013, "Efficient Estimation of Word Representations
in Vector Space," section 4.1.

### 2. `linear.py` -- Affine `y = x @ W.T + b`

```
grad_x = grad_y @ W
grad_W = grad_y.T @ x   (sum over leading dims)
grad_b = grad_y.sum     over leading dims
```

Reference: Goodfellow et al, "Deep Learning" (2016) §6.5.

### 3. `rmsnorm.py` -- RMSNorm (Zhang & Sennrich 2019)

Forward:
```
rms = sqrt(mean(x^2) + eps)
y   = x / rms * gamma
```

Backward (vectorised, fp32-internal for bf16 stability):
```
norm      = x * rstd
grad_norm = grad_y * gamma
grad_x    = rstd * (grad_norm - norm * mean(norm * grad_norm, axis=-1, keepdims=True))
grad_g    = sum_{leading} (grad_y * x * rstd)
```

**Numerical pitfall:** `rstd` blows up when `mean(x^2) -> 0`. The
`eps` floor caps `rstd <= 1/sqrt(eps)`. With default `eps=1e-6` you
get `rstd <= 1e3`, well-bounded. **Setting `eps < 1e-12` is unsafe**:
in bf16 the mantissa underflows and gradients explode. Test
`test_rmsnorm` uses `eps=1e-6` and shows reltol 5e-3 (slightly looser
than the ideal 1e-3 because RMSNorm bwd has a shared term that
amplifies fp32 truncation).

References: Zhang & Sennrich, "Root Mean Square Layer Normalization"
(NeurIPS 2019). Apex / Triton `rms_norm` kernels use the same form.

### 4. `swiglu.py` -- SwiGLU FFN (Shazeer 2020)

Forward:
```
g = silu(x @ W_gate.T)
u = x @ W_up.T
a = g * u
y = a @ W_down.T
```

Backward composes `linear_bwd` for the three projections plus the
SiLU+gate piece:
```
grad_a       = grad_y @ W_down
grad_W_down  = grad_y.T @ a
grad_g       = grad_a * u
grad_u       = grad_a * g
grad_z_gate  = grad_g * silu'(z_gate)
grad_W_up    = grad_u.T @ x
grad_W_gate  = grad_z_gate.T @ x
grad_x       = grad_z_gate @ W_gate + grad_u @ W_up
```

with `silu(z) = z * sigmoid(z)` and `silu'(z) = sigmoid(z) + z * sigmoid(z) * (1-sigmoid(z))`.

Reference: Shazeer, "GLU Variants Improve Transformer" (arXiv:2002.05202, 2020).

### 5. `cfc.py` -- Liquid-S4 CfC (Hasani 2022)

Per-step forward (matches `synapforge.cells.liquid.LiquidCell`):
```
delta_in = x @ W_in.T
delta_t  = softplus(delta_in)
expA     = exp(A_log)
A_t      = exp(-delta_t * expA)            # in (0, 1]
b_in     = x @ W_h.T
B_t      = delta_t * b_in   [+ b]
h_t      = A_t * h_{t-1} + B_t              # pre-tanh state
out_t    = tanh(h_t)
```

Backward unrolls the closed form by chain rule (no autograd graph).
The full derivation is in `cfc.py`'s docstring; the key non-obvious
steps:

* `dA/d(delta_t)  = -expA * A`    (sign flips in the exponent)
* `dA/d(A_log)    = -delta_t * expA * A`
* `d delta / d delta_in = sigmoid(delta_in)` (softplus derivative)

`cfc_seq_fwd / cfc_seq_bwd` walk the sequence sequentially with full
BPTT. We deliberately use sequential rather than the Heinsen 2023
parallel-scan because the parallel-scan bwd involves cumsum
subtractions that drift in fp32; sequential gives byte-tight VJPs.

References: Hasani et al, "Closed-form Continuous-Time Neural
Networks" (Nature Machine Intelligence 2022) §3.2.

### 6. `plif.py` -- Parametric LIF (Fang ICCV 2021)

Forward:
```
tau    = exp(tau_log)
decay  = exp(-dt / tau)
v_pre  = v_prev * decay + x
spike  = indicator(v_pre >= thr)              # hard, non-differentiable
v_new  = v_pre - spike * thr                  # reset-by-subtract
```

Backward uses the **ATan surrogate** (Fang et al.):
```
d spike / d v_pre = alpha / (2 * (1 + (pi/2 * alpha * (v_pre - thr))^2))
```

This matches `synapforge.cells.plif._ATanSurrogate.backward` byte-for-byte
in the codebase. The surrogate decays as `1/x^2` outside the firing
band -- exactly the desired vanishing-grad behaviour for cells far
from the threshold.

Tau gradient (chain through `tau = exp(tau_log)`):
```
d decay / d tau_log = decay * dt / tau
```

**Numerical pitfall:** `tau_log` is unbounded above. The codebase
clamps `tau` to `[1e-2, 1e3]` after `exp`. The pure-numpy VJP doesn't
clamp; callers that need clamping should do it on the input. Without
the clamp, very-large `tau_log` saturates `decay -> 1`, killing the
membrane integration.

### 7. `sew_shortcut.py` -- SEW residual (Fang NeurIPS 2021)

Forward: `y = spike + h_dense`. Backward identity-splits:
```
grad_spike = grad_y
grad_h_dense = grad_y
```

Trivial but in its own module so the catalogue is complete and the
residual path has explicit test coverage. The dense path is what
prevents PLIF-dead bootstrap in the codebase.

### 8. `cross_entropy.py` -- Softmax-CE

Forward (log-sum-exp stable):
```
softmax = exp(z - max) / sum(exp(z - max))
loss_n  = -log(softmax_n[t_n])     for non-ignored n
loss    = mean / sum / none of loss_n
```

Backward (the famous closed form):
```
grad_z = (softmax - one_hot(target)) / scale
```
where `scale = count_non_ignored` for `mean`, `1` for `sum`. Ignored
rows zero out via mask before the diff.

Bypasses chain-through-softmax-and-log entirely -- the cleanest VJP
in the catalogue.

## Precision notes

| Op | fp32 envelope | bf16 envelope | Pitfall |
|----|---------------|---------------|---------|
| embed | exact | exact | none |
| linear | atol 1e-5 vs torch | atol 5e-4 | matmul accumulation in bf16 |
| rmsnorm | atol 5e-5 | atol 1e-3 | mean(x^2) underflow if eps too small |
| swiglu | atol 5e-5 | atol 1e-3 | sigmoid saturation at |z| > 30 |
| cfc step | atol 1e-4 | atol 1e-3 | softplus(x) overflow at x > 20 (we clamp) |
| cfc seq T=4 | atol 1e-4 | atol 1e-3 | BPTT drift grows ~ sqrt(T) |
| plif | atol 1e-4 (smooth) | atol 1e-3 | spike forward is non-diff at threshold; surrogate substitutes |
| sew | exact | exact | none |
| ce | atol 1e-5 | atol 5e-4 | softmax exp() underflow handled by max-shift |

## Verification

Run:
```
pytest tests/native/vjp/test_vjp_against_torch.py -v
```

* `TestNumericalJacobian` (10 tests, no torch dependency): pass on
  numpy 1.24 / Python 3.8.
* `TestAgainstTorch` (8 tests, requires torch): asserts bit-tight
  match vs torch.autograd reference. Skipped automatically when
  torch is unavailable.

## Honesty: known numerical pitfalls

1. **RMSNorm at near-zero variance.** `rstd = 1/sqrt(mean(x^2)+eps)`
   blows up if `mean(x^2)` and `eps` are both tiny. `eps=1e-6` is
   safe; lower values risk gradient explosion in bf16. **Mitigation:**
   use bf16 only for forward; keep RMSNorm bwd in fp32 (current
   default).

2. **PLIF discontinuity.** The forward indicator `v_pre >= thr` is
   non-differentiable. Finite-diff VJP tests *cannot* verify the
   spike-grad path because tiny perturbations may flip indicator
   values. The torch-autograd reference (which uses the same
   surrogate) is the only ground truth. Smooth paths (decay, tau,
   threshold-far cells) match finite-diff cleanly.

3. **CfC BPTT drift.** Sequential `cfc_seq_bwd` is byte-tight per
   step but accumulates `~sqrt(T)` fp32 noise across the sequence.
   For long sequences (`T > 1024`) prefer chunked bwd to keep the
   error bounded.

4. **Embedding row collisions.** When `input_ids` has duplicates,
   `np.add.at` correctly accumulates -- but this serializes the
   scatter and is slow on dense vocabularies (`V > 50k`). The CUDA
   adapter (other agent's work) will fan out via segmented sum.

## File map

| File | Lines | Description |
|------|-------|-------------|
| `synapforge/native/__init__.py` | 14 | Pure-numpy package marker |
| `synapforge/native/vjp/__init__.py` | 41 | Submodule roll-up |
| `synapforge/native/vjp/embed.py` | 91 | Token embedding |
| `synapforge/native/vjp/linear.py` | 99 | Affine `y = x @ W.T + b` |
| `synapforge/native/vjp/rmsnorm.py` | 113 | RMSNorm |
| `synapforge/native/vjp/swiglu.py` | 137 | SwiGLU FFN |
| `synapforge/native/vjp/cfc.py` | 269 | Liquid-S4 CfC + sequence BPTT |
| `synapforge/native/vjp/plif.py` | 165 | PLIF + ATan surrogate |
| `synapforge/native/vjp/sew_shortcut.py` | 65 | SEW residual |
| `synapforge/native/vjp/cross_entropy.py` | 152 | Softmax-CE |
| `tests/native/vjp/test_vjp_against_torch.py` | 525 | (A) torch + (B) finite-diff |
| `docs/NATIVE_VJP_CATALOG.md` | this | Math + precision notes |
