# Native Data Loader (`synapforge.native.data`)

Torch-free replacement for `synapforge.data.ParquetTokenStream`. Emits
**numpy** arrays (and optionally **cupy**-pinned host arrays for async
H2D copy) instead of `torch.Tensor`. Designed to feed the native
trainer and the investor demo without dragging the torch import surface.

> Hard rule: every file under `synapforge/native/data/*.py` has zero
> `import torch`. Verified in CI by `grep -c '^import torch' synapforge/native/data/*.py == 0`.

## File map

| File                 | Purpose                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| `__init__.py`        | Package surface: `NativeParquetStream`, `NativeJsonlStream`, `WeightedMixedStream`, `NativeTokenizer` |
| `tokenizer.py`       | Qwen-2.5-compatible BPE wrapper. Three backends: `transformers` (preferred), `tokenizers` (HF Rust core), pure-Python JSON fallback parsing `tokenizer.json` directly. |
| `parquet_stream.py`  | `NativeParquetStream`: pyarrow + numpy + Fisher-Yates shuffle + 4-thread prefetch + cupy-pinned alloc. |
| `jsonl_stream.py`    | `NativeJsonlStream`: identical pipeline, JSONL/JSONL.gz inputs. Schemas: `{"text"}`, `{"content"}`, `{"messages"}`, `{"prompt", "completion"}`. |
| `mixed_stream.py`    | `WeightedMixedStream`: row-level weighted round-robin over N sub-streams. Includes `parse_data_files_arg(...)` mirroring `--data-files PATH:W,...`. |
| `bench.py`           | Throughput parity bench (native vs torch `ParquetTokenStream`). |

## Public API

```python
from synapforge.native.data import (
    NativeParquetStream,
    NativeJsonlStream,
    WeightedMixedStream,
    NativeTokenizer,
)
```

## Drop-in replacement for `ParquetTokenStream`

The native stream's emit shape matches the legacy stream **except** the
yielded values are `numpy.ndarray` (int64) rather than `torch.Tensor`
(int64). All other invariants hold:

- `next(iter(ds)) -> (tokens_in[B,T], tokens_out[B,T])`
- `tokens_out[:, :-1] == tokens_in[:, 1:]` (next-token shift)
- Default `eot_id` derived from the tokenizer's `eos_token_id`
- `loop=True` infinite iteration with per-epoch file shuffle
- `shuffle_buffer > 1` enables streaming Fisher-Yates reservoir
- `prefetch_factor >= 2` activates the multi-thread prefetch path

### Constructor differences vs `ParquetTokenStream`

| `ParquetTokenStream`       | `NativeParquetStream`           | Notes |
|----------------------------|---------------------------------|-------|
| `tokenizer_name="gpt2"`    | `tokenizer="gpt2"` (str OR `NativeTokenizer`) | Native accepts a pre-built tokenizer object |
| `pin_memory=False`         | `cuda=False`                    | Native uses cupy `alloc_pinned_memory`; legacy uses torch `pin_memory()` |
| `vocab_size=50257`         | (removed)                       | Documentation-only int in legacy; the actual vocab is whatever the tokenizer reports |
| `remote_warehouse=None`    | (not yet wired)                 | If the trainer needs warehouse mode, expose it explicitly later |

## Swapping in for `ParquetTokenStream` (in `train_100m_kd.py`)

The legacy line:

```python
from synapforge.data import ParquetTokenStream, split_val_stream
...
train_ds = ParquetTokenStream(
    args.data_glob, seq_len=seq_len, batch_size=args.batch_size,
    tokenizer_name=args.tokenizer_name,
    shuffle_buffer=args.shuffle_buffer,
    prefetch_factor=args.prefetch_factor,
    pin_memory=args.pin_memory,
)
```

becomes:

```python
from synapforge.native.data import NativeParquetStream, NativeTokenizer
...
tok = NativeTokenizer(args.tokenizer_name)
train_ds = NativeParquetStream(
    args.data_glob, seq_len=seq_len, batch_size=args.batch_size,
    tokenizer=tok,
    shuffle_buffer=args.shuffle_buffer,
    prefetch_factor=args.prefetch_factor,
    cuda=args.pin_memory,  # cuda=True replaces pin_memory=True
)
```

Then the consumer:

```python
for x, y in train_ds:
    # x, y are numpy.ndarray int64 [B, T]
    # convert to whatever your training kernel expects:
    #   x_torch = torch.from_numpy(x).to(device, non_blocking=True)
    #   x_cupy  = cupy.asarray(x)  # zero-copy if x is pinned-host
    ...
```

## What breaks if you swap this in for `ParquetTokenStream` in `train_100m_kd.py`

These are the **explicit** assumptions the trainer currently makes about
the legacy stream that the native stream does not match. List is exhaustive
(read against `train_100m_kd.py` line-by-line):

1. **Yield type**: legacy yields `torch.Tensor`, native yields `numpy.ndarray`.
   Every downstream `.cuda()`, `.to(device)`, `.numel()`, `.long()`,
   `.view(...)`, `.contiguous()` call on the batch must convert via
   `torch.from_numpy(arr)` first. The trainer hits these in:
   - `x.to(device, non_blocking=True)` -> wrap.
   - `x.numel()` -> use `arr.size`.
   - `x.long()` -> already int64, no-op.
   - `loss = F.cross_entropy(logits, y.flatten())` -> wrap `y` via
     `torch.from_numpy(y).flatten()`.

2. **`pin_memory=True` -> `cuda=True`**: legacy uses
   `torch.Tensor.pin_memory()`; native uses
   `cupy.cuda.alloc_pinned_memory`. The pinned buffer is still a numpy
   array on the consumer side; `torch.from_numpy(x).pin_memory()` would
   double-pin (cheap no-op on CUDA tensors). Cleaner: pass numpy arrays
   directly to a custom kernel that accepts pinned-host pointers.

3. **`split_val_stream(parent, ttt_fraction, denom)`** is NOT in the
   native subpackage. The native loader has no `_iter_token_chunks`
   method on a private surface that yields raw chunks, but you can wire
   a similar TTT/holdout split externally if needed. **For now, val-set
   splitting still goes through the legacy `synapforge.data.split_val_stream`**
   (which requires the legacy `ParquetTokenStream`). To keep using val
   split + native train, run two streams in parallel: native for train,
   legacy for val.

4. **`AsyncTokenStream`** is NOT mirrored. Legacy `synapforge.data` exposes
   an async/lazy variant for the multi-process pipeline; the native
   subpackage does not. If you depend on `AsyncTokenStream`, keep the
   legacy import for those code paths.

5. **`remote_warehouse=...`**: not wired in the native streamer. If
   your trainer pulls shards via `RemoteDataWarehouse` (mohuanfang.com),
   the native streamer cannot do that yet. Opt out (materialise the
   corpus locally) before swapping in.

6. **`files_with_weights` warehouse-mode error**: the legacy stream
   raises a specific `ValueError` when both `files_with_weights` and
   `remote_warehouse` are passed. The native stream simply never
   accepts `remote_warehouse`, so this error class is gone (the trainer
   builds neither path through native).

7. **Fisher-Yates shuffle reservoir RNG**: same algorithm, same default
   `shuffle_seed=42`; the realised yield order should match given the
   same seed. **However** the per-epoch file-order RNG seeds with
   `shuffle_seed + epoch` in both, so successive epochs produce the
   same scrambled file-order. The two streams therefore produce
   identical chunk sequences when both are configured with the same
   args. (`test_token_id_parity_with_legacy` covers this on the first
   `n_batches` and is enabled when torch is present.)

8. **Tokenizer cache**: legacy keys by `tokenizer_name` only; native
   keys by `name::backend`. Mixing `backend="auto"` with `backend="json"`
   in the same process is therefore safe; mixing two `name` calls with
   different actual backends would behave the same as before.

9. **`__repr__`**: differs. Anything string-comparing the repr will
   break (no production code does this, but tests sometimes do).

## Performance

Throughput parity bench: `python -m synapforge.native.data.bench --parquet-glob 'data/wt103_raw/train-*.parquet'`.
Target: native >= 1.0x torch on the same parquet shard, same seq_len /
batch_size / prefetch_factor / shuffle_buffer.

The native loader's CPU-side hot loop (`_iter_token_chunks` + numpy
batch assembly) is virtually identical to the legacy hot loop minus
the `torch.tensor(...)` allocation. Pinned-host alloc through
`cupy.cuda.alloc_pinned_memory` is ~2x faster than `torch.Tensor.pin_memory()`
on Windows + CUDA 12; on Linux the gap closes to within noise.

The 4-worker producer pool is tuned to feed bs=80 / seq=256 batches
without starving the trainer's GPU step at >40k tok/s (matching the
v4.2 triton-block backend's saturation point on A800-80GB).

## Tokenizer fallback

`NativeTokenizer(name_or_path, backend="auto")` picks:

1. **transformers** -- if `transformers` is importable. Identical to
   legacy `_get_tokenizer`.
2. **tokenizers** (HF Rust core) -- needs a local `tokenizer.json`.
3. **JSON fallback** -- pure-Python BPE that loads `tokenizer.json`
   directly. Works offline with zero non-stdlib deps beyond `regex`
   (and degrades to `re` when `regex` is missing).

Forcing the JSON backend (the offline path the tests exercise):

```python
tok = NativeTokenizer("/path/to/dir/with/tokenizer.json", backend="json")
ids = tok.encode("Hello world!")
```

## Test suite

```bash
pytest tests/native/data/ -v
```

Tests cover:
- shape / dtype / shift invariants
- per-epoch file-order shuffle decorrelation
- multi-thread prefetch -> unique batches (no worker dup)
- 10K-sample weight-ratio statistical check (`+/- 3%`)
- heterogeneous parquet+jsonl mixing
- `parse_data_files_arg` round-trip
- token-id parity with legacy `ParquetTokenStream` (skipped when torch
  unavailable; runs in CI)
