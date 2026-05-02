# SNN High-Frequency Limitation — NeurIPS 2025 Paper Notes

**Status**: Research note 2026-05-02. Anchors Synap-1 architecture decisions for v2+ (post-PLIF-revival).

## §1. Paper bibliography

- **Title**: *Spiking Neural Networks Need High-Frequency Information*
- **Authors**: Yuetong Fang, Deming Zhou, Ziqing Wang, Hongwei Ren, Zecui Zeng, Lusong Li, Shibo Zhou, Renjing Xu
- **Venue**: NeurIPS 2025 (Poster — Wed Dec 3 2025, San Diego)
- **arXiv ID**: `2505.18608`
- **Links**:
  - arXiv: https://arxiv.org/abs/2505.18608
  - OpenReview: https://openreview.net/forum?id=owNPAl7LNK
  - NeurIPS poster: https://neurips.cc/virtual/2025/poster/115997
  - Code: https://github.com/bic-L/MaxFormer
- **BibTeX**:
  ```bibtex
  @article{fang2025spiking,
    title  = {Spiking Transformers Need High Frequency Information},
    author = {Fang, Yuetong and Zhou, Deming and Wang, Ziqing and others},
    journal= {arXiv preprint arXiv:2505.18608},
    year   = {2025}
  }
  ```

## §2. Key claims

The paper reframes the long-standing SNN-vs-ANN performance gap. Conventional wisdom blamed **information loss from sparse binary spikes**. Fang et al. argue this is the wrong diagnosis:

1. **Spiking neurons are network-level low-pass filters.** The membrane integration `mem_t = mem_{t-1}·decay + I_t` plus the threshold reset together attenuate high-frequency components of the input current and preferentially propagate low-frequency content. The paper proves this in the frequency domain (FFT of spike trains shows energy concentrated in low-freq bands).
2. **LIF beats IF *because* leak-decay restores some high-freq response.** Pure IF (no leak, `decay = 1`) integrates indefinitely — equivalent to an even more aggressive low-pass. LIF with finite τ has a flatter frequency response. This explains the well-known empirical IF→LIF gap from a frequency perspective for the first time.
3. **The frequency imbalance is the *root cause* of degraded feature representation**, not binarization per se. Fix the frequencies and the gap closes — even with binary spikes preserved.
4. **Pooling choice matters more than people thought.** Avg-Pool is a low-pass filter; Max-Pool is closer to a high-pass / edge detector. Swapping Avg→Max in a Spiking Transformer alone closes a non-trivial chunk of the ANN gap.

## §3. Empirical evidence

- **CIFAR-100 (Spiking Transformer baseline)**: Avg-Pooling 76.73% → Max-Pooling **79.12%** (single-knob ablation isolating pool frequency response).
- **CIFAR-10 (Max-ResNet-18)**: **97.17%** top-1.
- **CIFAR-100 (Max-ResNet-18)**: **83.06%** top-1.
- **ImageNet (Max-Former, 63.99M params)**: **82.39%** top-1, **+7.58 pp over Spikformer** at fewer parameters (Spikformer 66.34M) and **~30% energy** of Spikformer.

**Max-Former architecture (proposed fix)**:
1. Add **Max-Pool in patch embedding** (high-pass injection at the input).
2. Replace early-stage self-attention with **depth-wise convolution** (local high-freq mixer; cheap; energy-efficient).

Code map (per repo): `mixer_hub.py` for token mixers, `embedding_hub.py` for patch embeddings.

## §4. Implication for Synap-1

Synap-1's SNN component is `synapforge/cells/plif.py` — a textbook PLIF with `decay = exp(-dt/τ)`, so it inherits the low-pass behavior the paper identifies. Concrete impact:

1. **PLIF dead 10/10 may be a frequency problem, not just a surrogate-gradient/init problem.** If the upstream LiquidCell (CfC, also smooth) feeds an already-low-pass current into PLIF, the spiking layer sees mostly DC content. Threshold = 0.3 with low-energy high-freq residue → low spike rate → flat surrogate gradient → no learning signal. This compounds the documented SEW-style bootstrap issue from `feedback_plif_dead_bootstrap.md`.
2. **CfC's continuous-time ODE is *also* a low-pass filter** (τ-decay structure identical to LIF). Synap-1 stacks **two** low-pass filters in series — the frequency squeeze is worse than a pure SNN. None of the existing band-mixing tricks (Hyena, FNet, WaveFormer) currently sit between CfC and PLIF.
3. **Tokenizer / patch embedding has no high-pass injector.** `byte_patch.py` does mean-pool over byte windows — Avg-Pool, exactly the operation the paper calls out as harmful.
4. **Multi-τ timescale band (`feedback_long_context_quality.md` LEM multi-τ) needs to span a *high* frequency band**, not just slow ones for long-context. Otherwise we trade off freq-response for context length.
5. **Long-context monotonic-quality goal collides with low-pass cascade.** As context grows, leaky integration accumulates more low-freq DC drift; high-freq content (rare tokens, code structure, math symbols) gets washed out faster than transformer attention loses it.

## §5. Action items

Three concrete trainer/model changes, prioritized by ROI for the demo + paper.

### A1. Max-Pool injection in byte-patch embedding *(Priority 1, ~1 day)*
- Edit `synapforge/modal/byte_patch.py`: parallel branch that does `MaxPool1d(kernel=patch_size)` alongside the current mean-pool, concat features. Cheap (no new params if we reuse the linear projection), directly mirrors Max-Former's patch-embed change.
- Validation: chat-sample qualitative + token-rarity probe (does the model now distinguish `def` from `decode`?). Expected: small WikiText ppl bump (1-3%), large code/math ppl drop (5-15%).

### A2. High-pass residual around the PLIF block *(Priority 1, ~2 days)*
- Wrap `HybridBlock` (CfC → PLIF) with a frequency-balancing residual: `out = PLIF(CfC(x)) + λ · (x - LowPass(x))`. The `(x - LowPass(x))` term is a high-pass filter that bypasses both low-pass stages. λ learnable per-channel, init 0.1.
- Implementation: 1-line change in `synapforge/blocks/hybrid_block.py`; `LowPass` = depth-wise conv kernel=3 with smoothing init.
- Validation: spike rate health monitor — should rise from ~0 to the [0.05, 0.20] band that `feedback_lnn_snn_hybrid_real.md` mandates. If yes, addresses PLIF-dead root cause directly via frequency, not init.

### A3. Frequency-aware τ initialization for PLIF *(Priority 2, ~half day)*
- Current `PLIF` init uses bimodal τ (DA-LIF). Extend to **tri-modal**: 30% short-τ (high-pass, τ≈0.5), 40% mid-τ (band-pass, τ≈2.0), 30% long-τ (low-pass, τ≈8.0). Forces the layer to span the full frequency range from initialization rather than relying on training to discover it.
- Diagnostic: log per-bucket spike rate. If short-τ neurons fire at 0.1+ while long-τ fire at 0.02, the heterogeneous frequency response is working.

### Paper hook
This is also a paper angle. We can claim **first joint frequency-domain analysis of an LNN+SNN cascade** — Fang et al. only studied pure SNNs. The double-low-pass cascade we identify in §4 is novel; A2's high-pass residual is a clean extension of Max-Former to recurrent hybrid models.

### Out of scope (intentionally)
- We do **not** swap CfC for a high-pass cell. CfC is a load-bearing pitch element. Fix the frequency at the *boundary* (input embed + PLIF residual), keep CfC core unchanged.
- We do **not** delete Avg-Pool entirely; we add Max-Pool *alongside* (concat). Avg still useful for slow context.

## §6. Implementation notes (auto-snn-freq-fixes — feature/snn-freq-fixes)

All three knobs landed on `feature/snn-freq-fixes` and default OFF so the
historical baseline is bit-identical until a flag is passed.

### A1 — `synapforge/modal/byte_patch.py`
A new reusable `BytePatch(in_feat, hidden, patch, pool)` primitive:
- `pool="avg"` (default): single `Linear(patch * in_feat -> hidden)` over
  the flattened window.  Bit-equivalent to the legacy reshape+Linear
  byte-patch projection.
- `pool="max"`: `amax` along the patch axis, `Linear(in_feat -> hidden)`.
- `pool="max+avg"`: concat avg + max, single `Linear(2 * in_feat -> hidden)`
  (the Max-Former 1x1 mix recipe).

Trainer flag: `--byte-patch-pool {avg|max|max+avg}`, default `avg`.

### A2 — High-pass residual on `HybridBlock`
`synapforge/model_100m.py::HybridBlock` gained two ctor kwargs:
- `high_pass_residual_weight: float = 0.0` (default OFF).
- `high_pass_kernel_size: int = 3`.

When the weight is non-zero, the block instantiates:
- `hp_lowpass`: depth-wise causal `Conv1d(d, d, kernel_size=k, groups=d)`,
  weight initialised to `1/k` (uniform-average smoother), bias zero.
- `hp_lambda`: per-channel learnable `nn.Parameter(d,)` initialised to the
  passed scalar (so a single global init scalar still lets each channel
  diverge during training).

Forward becomes:
```
x_in = x
... (legacy CfC -> PLIF -> Synapse -> FFN body) ...
if hp_lowpass is not None:
    x = x + lambda * (x_in - LowPass(x_in))
```

LowPass is causal (left-padded with `k-1` zeros) so the high-pass residual
is order-preserving and works at any sequence length.

Trainer flag: `--high-pass-residual-weight 0.0`, default OFF.

### A3 — Tri-modal tau init for PLIF
`synapforge/surrogate.py::PLIFCell` `tau_init` now accepts `"trimodal"`:
- 30% of channels initialised to `tau = 0.5` (short / high-pass band).
- 40% to `tau = 2.0` (mid / band-pass).
- 30% to `tau = 8.0` (long / low-pass band).

This forces the layer to span the full frequency range from the first
forward, instead of relying on training to discover heterogeneity.

Trainer flag: `--plif-tau-init {unimodal|bimodal|trimodal|log_uniform}`,
default `unimodal` (legacy `tau=2.5` uniform).

### Tests
`tests/integration/test_snn_frequency_fixes.py` — four CPU-only tests:
1. A1: each `pool` variant runs, output shape correct, no NaN.
2. A2: default OFF -> no extra modules; ON -> output differs.
3. A3: trimodal init produces exactly 3 tau bands with the documented
   30/40/30 split.
4. All-on smoke: `build_synapforge_100m(plif_tau_init="trimodal",
   high_pass_residual_weight=0.05)` forwards cleanly end-to-end.

### Branch
`feature/snn-freq-fixes`, commit
`auto-snn-freq-fixes: A1+A2+A3 from Fang et al. NeurIPS 2025 — Max-Pool +
high-pass residual + tri-modal tau init + tests`.
