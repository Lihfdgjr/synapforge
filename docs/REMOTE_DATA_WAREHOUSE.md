# Remote Data Warehouse (mohuanfang ⟷ rental)

**Status:** opt-in, lands 2026-05-01.
**Code:** `synapforge/data/remote_warehouse.py`, `synapforge/data/__init__.py`,
`scripts/setup_mohuanfang_warehouse.sh`.
**Target:** free up rental SSD (recover from the 99.9 % full-disk
incident on 2026-05-01) and unblock phase-2 multimodal pretrain
(50–200 GB) which will not fit on the rental's 100 GB local disk.

## TL;DR

| Tier      | Host          | Capacity | Purpose                              |
|-----------|---------------|----------|--------------------------------------|
| Warehouse | mohuanfang    | 1.5 TB   | Canonical corpus, all datasets       |
| Cache     | rental SSD    | 30 GB    | LRU window of recently-touched shards|
| Compute   | rental GPU    | A100/A800| Trainer — reads from local cache only|

The trainer never touches mohuanfang directly; it asks the
``RemoteDataWarehouse`` for a shard and gets back a path under
``/workspace/data_cache/<dataset>/``. First access pulls via ``rsync``;
subsequent accesses are local-SSD speed. When the cache crosses the
configured cap, the LRU evictor pops the oldest-atime shards.

## Architecture

```
   mohuanfang.com                          rental box (A100/A800)
   ─────────────                            ────────────────────
   /home/liu/synapforge_data/               /workspace/data_cache/   <-- LRU cache
   ├── fineweb_edu/*.parquet                ├── fineweb_edu/
   ├── alpaca-zh/*.parquet                  │   ├── train-0001.parquet  <- fetched
   ├── multimodal-image/*.parquet           │   └── train-0007.parquet  <- fetched
   └── multimodal-audio/*.parquet           ├── alpaca-zh/
                  ▲                          │   └── shuf-0003.parquet   <- fetched
                  │                          └── multimodal-image/  (cold; fetched on demand)
                  │
                  ssh / rsync (~50 MB/s)
                  │
   ┌──────────────┴───────────────┐
   │ RemoteDataWarehouse.get_shard │
   │   - LRU atime sidecar         │
   │   - file-lock per shard       │
   │   - retries with backoff      │
   └──────────────▲───────────────┘
                  │
   ┌──────────────┴───────────────┐
   │ ParquetTokenStream.__iter__   │
   │   - prefetch_factor=2         │
   │   - shuffle_buffer=10000      │
   └──────────────▲───────────────┘
                  │
                  ▼
              GPU step
```

## Bandwidth and step latency

* mohuanfang link: ~50 MB/s sustained over rsync (measured on the
  existing `sync_to_mohuanfang.sh` backup loop).
* Typical shard size: 200 MB (FineWeb chunked) → **4 s per fetch**.
* Trainer reads: bs=80, seq=256, ~80 MB/s of parquet bytes when no
  shuffle reservoir is hit.
* Dataloader runs with `prefetch_factor=2` (a daemon producer thread).
  At steady state, the CPU producer is one shard ahead of the GPU
  consumer; the 4 s rsync is amortised across ~25 batches (~5 s of GPU
  work at 21 k tok/s).

If the cache is warm enough that >80 % of accesses hit local SSD, the
rsync hit shows up only at shard boundaries (every ~2 GB of trained
tokens at the default 200 MB shard size). For typical training runs
that's **4 s every ~3 minutes** — invisible behind the prefetch queue.

## Capacity math (phase 2 multimodal)

* Phase-2 corpus: estimated 200 GB (image 60 GB + audio 50 GB +
  video 90 GB pre-tokenised).
* Mohuanfang: holds all 200 GB; **uses 13 % of free space**.
* Rental: holds 30 GB cache (15 % of corpus). Modal data is
  **shuffled across shards** so the LRU window covers a representative
  slice; one epoch through the cache touches ~150 GB of remote data
  via rotation.
* Steady-state local disk: ~30 GB warehouse cache + ~60 GB
  ckpts/logs/code = ~90 GB on the 100 GB rental SSD. **9 GB
  headroom**, well above the 5 GB threshold that triggered the
  2026-05-01 incident.

## Cache size choice (`--cache-max-gb 30.0`)

The default is sized for `bs=80 × seq=256 × loop=true` over a
200 MB-shard corpus:

* one epoch = ~50 shards = 10 GB of unique reads
* prefetch lookahead = 1 shard ahead at 200 MB
* shuffle reservoir at `shuffle_buffer=10000` rows holds ~3 shards' worth
  of rows scattered across as many parquet files
* margin for a second concurrent stream (val) = ~5 GB

→ 30 GB covers the working set with ~10 GB slack for shard rotation.
Smaller caches (e.g. 10 GB) work but increase rsync rate; the trainer
still progresses, just with slightly higher CPU load.

## Failure modes

| Failure                       | Behaviour                                          |
|-------------------------------|----------------------------------------------------|
| mohuanfang offline at startup | Falls back to whatever's in the local cache; raises if cache empty too. |
| mohuanfang offline mid-run    | Trainer keeps running on cached shards; new shard miss triggers retry-with-backoff (3 attempts), then raises. |
| Rental disk fills up          | `_evict_lru` runs after every fetch; can never grow past `max_cache_gb`. |
| Concurrent train + val        | File lock per shard prevents double-rsync; both streams share the cache. |
| LRU thrash (corpus > cache)   | Each shard rsync'd at most once per cache rotation. With 200 GB corpus / 30 GB cache, that's a ~6 % rsync rate ceiling, ~3 MB/s steady-state — 6 % of the link. |
| Pre-existing local data       | Setup script keeps the original corpus at `/workspace/data_archived` so a `mv` rolls back the symlink in seconds. |

## Operational runbook

### Bootstrap (run once, on rental)

```bash
bash scripts/setup_mohuanfang_warehouse.sh                # full setup
bash scripts/setup_mohuanfang_warehouse.sh --dry-run      # show plan
bash scripts/setup_mohuanfang_warehouse.sh --skip-upload  # corpus already remote
```

What it does:

1. Probes ssh to `liu@mohuanfang.com`.
2. Discovers each first-level subdir of `/workspace/data` as a dataset.
3. Creates `/home/liu/synapforge_data/<dataset>/` on mohuanfang and
   rsyncs each dataset up.
4. On rental: `mv /workspace/data /workspace/data_archived` then
   `mkdir /workspace/data_cache && ln -s data_cache data` so existing
   trainer paths keep resolving.

Rollback: `rm /workspace/data && mv /workspace/data_archived /workspace/data`.

### Launch trainer with warehouse on

```bash
.venv/bin/python train_100m_kd.py \
   --data-glob '/workspace/data/fineweb_edu/*.parquet' \
   --remote-warehouse-host 'liu@mohuanfang.com' \
   --remote-warehouse-base /home/liu/synapforge_data \
   --remote-warehouse-dataset fineweb_edu \
   --cache-max-gb 30.0 \
   ... (rest of usual flags)
```

`--data-glob` becomes a basename pattern when warehouse mode is on
(matched against the remote shard listing). Existing flags
(`--shuffle-buffer`, `--prefetch-factor`, `--pin-memory`) compose
unchanged.

### Monitor cache state

```bash
ls -lh /workspace/data_cache/<dataset>/                   # current resident set
cat /workspace/data_cache/.<dataset>.atime.json | jq      # LRU sidecar
```

### Disable warehouse temporarily

Just drop the `--remote-warehouse-host` flag. Trainer reverts to local
direct reads of `/workspace/data` (which, if you already pivoted, still
works because `/workspace/data` is a symlink to the cache; whatever's
already cached gets used and missing shards raise a `FileNotFoundError`
that the trainer log will surface).

## Why not sshfs?

We measured sshfs at 22-30 MB/s parquet-scan throughput vs ~700 MB/s
on the local SSD. At bs=80 / seq=256 / 21k tok/s the trainer reads
~80 MB/s of parquet bytes; sshfs is **3× too slow** and would
bottleneck the GPU. Local cache + on-demand fetch keeps the trainer
SSD-speed for cache hits and only pays the network cost on cache misses.

## Test surface

`tests/integration/test_remote_warehouse.py`:
- first-fetch-then-cached contract
- LRU evicts oldest-atime shard
- concurrent `get_shard` serialises (no double rsync)
- rsync retries with backoff
- `list_remote_shards` TTL cache
- `ParquetTokenStream(remote_warehouse=...)` integration

All tests are subprocess-mocked (no network, no rsync binary needed),
so they run on the Windows dev box and CI alike.
