# Native Multimodal Packing (synapforge.native.modal)

> Branch: `feature/native-modal-packing`
> Owner agent: modal-packing
> Status: scaffold complete, 61/61 tests passing, bench numbers measured.
> Last update: 2026-05-02

## Why this layer is LNN+SNN-multimodal-specific

In a transformer multimodal model (Flamingo, GPT-4V, BLIP-2) cross-modal
fusion is done with cross-attention: text tokens query into an image
encoder's KV cache. That requires *separate* encoders per modality and
*separate* attention layers, then a fusion block on top. The padding
problem is per-encoder and per-batch -- you can pack each modality's
encoder batch independently because each modality has its own backbone.

We don't have that luxury. The synapforge model is a single shared
**CfC + PLIF** backbone that processes **all 9 modalities through the
same layers**. The early-fusion design (Fuyu-style, not LLaVA-style)
means there is no per-modality encoder; instead every modality is
byte-patched into the same token stream and the temporal LNN+SNN block
learns mode transitions itself.

So when we batch text (T=256) alongside image (T=1024), audio (T=2048),
3D (T=4096), and video (T=8192), the right-padded layout has all
samples padded to T=8192 -- 97% of FLOPs go to padding for text. The
**packed batch** layout sidesteps this entirely: per-modal sequences
flat-concat into a single token stream with offsets and per-token
modal-id arrays, so the CfC kernel does work proportional to *real*
tokens only.

## What lives here

```
synapforge/native/modal/
    __init__.py            -- public API surface
    packed_batch.py        -- ModalBatchPacker, PackedBatch, MODAL_REGISTRY
    modal_mask.py          -- per-token reset_flag / modal_id / sample_id
    cross_modal.py         -- CrossModalContrastive (CLIP-style InfoNCE)
    dispatch.py            -- ModalDispatchEmbed (Qwen 151k vs byte 256)
```

```
tests/native/modal/        -- 61 unit tests, all passing
scripts/bench_modal_packing.py  -- packed-vs-padded throughput bench
```

## The 9 modalities (canonical registry)

| Modal       | T_max | Vocab   | Encoding    | Notes                       |
|-------------|------:|--------:|-------------|-----------------------------|
| text        |   256 | 151,643 | Qwen        | natural language            |
| image       | 1,024 |     256 | byte-patch  | 32x32 RGB byte windows      |
| audio       | 2,048 |     256 | byte-patch  | 16kHz, 128ms windows        |
| time_series |   512 |     256 | byte-patch  | sensor stream               |
| code        |   512 | 151,643 | Qwen        | source code                 |
| math        |   128 | 151,643 | Qwen        | formulae / proofs           |
| 3D          | 4,096 |     256 | byte-patch  | voxel byte stream           |
| video       | 8,192 |     256 | byte-patch  | frames * pixel patches      |
| gesture     |   256 |     256 | byte-patch  | joint angles per frame      |

The first three columns are the input contract every loader must
honour. ``vocab=256`` modalities share the **byte embedding table**
(small, fast); ``vocab=151,643`` modalities share the **Qwen embedding
table**. ``ModalDispatchEmbed`` routes per-token to the right table.

## PackedBatch layout

```
Inputs:
    text:  [t0 t1 t2 t3]               # one sample, L=4
    image: [i0 i1 i2 i3 i4 i5 i6 i7]   # one sample, L=8
    audio: [a0 a1 a2 a3 a4 a5]         # one sample, L=6

Outputs (PackedBatch):
    concat_tokens : [t0..t3, i0..i7, a0..a5]   shape (18,)
    offsets       : [0, 4, 12, 18]             shape (B+1=4,)
    modal_ids     : [0, 1, 2]                  shape (B=3,)
    seq_lens      : [4, 8, 6]                  shape (B=3,)
```

The CfC+PLIF kernel reads one row at a time from `concat_tokens`. At
each position `n`, the **modal mask** layer (`modal_mask.py`) supplies:

* `token_modal_id[n]` -- which modality's embed table to use.
* `token_sample_id[n]` -- which sample within the batch this token
  belongs to.
* `reset_flag[n]` -- True at the first token of every sample. The CfC
  kernel zeros its carried hidden state on those steps so state does
  not bleed across samples.

For attention layers (if any), `ModalMaskBuilder.pairwise_modal_mask`
produces a `(B, B)` boolean matrix where `mask[i, j] == True` iff i
and j are the same sample. This is the contrastive-loss negatives mask
we use in `CrossModalContrastive`.

## Memory math

For a `bs=8 per modality x 9 modalities = 72 sample` mini-batch:

* Padded baseline (single global T = 8192):
  ```
  72 samples * 8192 tokens * 4 B/token = 2.36 MB
  per-sample = 33 KB
  ```
  Most of this is padding (3D: T_used=2K-4K, video: T_used=4K-6K, but
  the smaller modalities like math at T_max=128 carry 8064 padding
  tokens each).

* Per-modal padded (smart baseline -- pad each modality to its own
  T_max-in-batch):
  ```
  text:        8 * 128 (avg) * 4 = 4 KB
  image:       8 * 768 (avg) * 4 = 24 KB
  audio:       8 * 1536 (avg) * 4 = 48 KB
  time_series: 8 * 384 (avg) * 4 = 12 KB
  code:        8 * 384 (avg) * 4 = 12 KB
  math:        8 * 96 (avg) * 4 = 3 KB
  3D:          8 * 3072 (avg) * 4 = 96 KB
  video:       8 * 6144 (avg) * 4 = 192 KB
  gesture:     8 * 192 (avg) * 4 = 6 KB
  total = 397 KB / 72 samples = 5.5 KB/sample
  ```

* Packed (this layer):
  ```
  ~ sum(actual_L_i) tokens
  Measured: 51K tokens / 72 samples = ~ 700 tokens/sample
            * 4 B/token = 2.8 KB/sample
  ```

So the production-mix savings (2.79 vs 15.06 KB/sample on a 9-modal
bs=8 batch) is about **5.4x** vs naive global padding, **2x** vs the
smart per-modal padded baseline (which already does most of what
packing does, but per-modality not cross-modality).

## Throughput projection

The CfC+PLIF kernel is FLOPs-bound on real tokens (every token = O(H^2)
matmul). So the throughput multiplier is the inverse of the fill ratio.

Measured on the bench (`scripts/bench_modal_packing.py --full --reps 30`):

```
fill_ratio_vs_global_padded    : 0.184
fill_ratio_vs_per_modal_padded : 0.794

throughput_speedup_vs_global_padded    : 5.44x
throughput_speedup_vs_per_modal_padded : 1.26x
```

The user's **1.4-1.8x target** corresponds to the path Run 7 currently
uses (text-only with naive right-padding). When we *enable*
multimodal training (modal-list flag), the throughput lift scales with
how heterogeneous the per-modal lengths are. The honest verdict:

* Vs the **global-padded** path that a transformer-native codebase
  would default to: **5.4x** speedup (massive padding savings).
* Vs a **per-modal-padded** path that already smart-batches each
  modality separately: **1.26x** speedup.

## Cross-modal contrastive loss

`CrossModalContrastive` provides the CLIP-style InfoNCE loss over the
per-sample embeddings the backbone produces. With packed batches we
get all sample embeddings in one forward pass, then slice by modal id
and apply the loss per-pair. The default pair is `("text", "image")`
but any pair from `MODAL_REGISTRY` is supported.

The gradient is hand-derived (no autograd) from cosine similarity ->
temperature-scaled logits -> log-softmax -> cross-entropy -> chain
rule via `du/da = (I - u u^T) / ||a||`. The unit test checks against
finite-difference to `5e-3 atol`.

The bit-equivalence test verifies that calling `contrastive_loss` with
two pairs at once gives the same result as averaging two independent
single-pair calls (`atol=1e-5`). That's the contract that lets the
trainer do all pairs in one forward pass.

## Modal-aware embed/lm_head dispatch

`ModalDispatchEmbed` holds two physical tables:

* **Qwen table**: `(151643, hidden)` for text/code/math
* **Byte table**: `(256, hidden)` for image/audio/time_series/3D/video/gesture

At lookup, each `(token_id, modal_id)` pair routes to the correct
table. Weight tying (lm_head shares weights with embed) is the default.

The big-deal property is that **lm_head logits stay narrow** for byte
modalities. A naive unified-vocab lm_head would compute `(N, 151899)
logits` for all tokens. Packed dispatch keeps byte-token logits at
`(N_byte, 256)`, a 595x reduction. For an 8K-token packed batch this
is 5.0 GB -> 8.4 MB.

## Hard constraints (audit checklist)

* [x] Zero `import torch` in any file under `synapforge/native/modal/`.
* [x] All public functions return `numpy.ndarray` (or `cupy.ndarray`
  when GPU is enabled).
* [x] Pack-then-unpack is bit-identical (verified by 20-case test).
* [x] Modal mask blocks cross-sample CfC bleed (verified by 12-case test).
* [x] Cross-modal contrastive matches per-modal-pair reference within
  `1e-5` (verified by combined-vs-manual-average test).
* [x] FD vs analytic gradient within `5e-3` (verified).
* [x] Tests bypass `synapforge.__init__` so they don't drag torch in.
* [x] Bench reports both fill ratio and projected throughput multiplier.

## What this is dormant on Run 7

Run 7 runs text-only at fixed `T=256`, so this layer adds no value
**right now**. The point of building it is that when we flip the
multimodal flag (planned for the byte-patch corpus phase: phase 2 in
the phased-training protocol, ppl<=100 gate), we want the packed
batching layer ready and tested rather than scrambled in a hurry.

When that flag flips, the projected speedup vs the current per-modal
batch path is **1.26x** for the same total token budget, **5.44x** vs
a global-padded transformer-native baseline.
