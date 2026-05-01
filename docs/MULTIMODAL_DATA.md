# Multi-modal data sources -- 10 corpora, byte-patch contract, anti-fakery binding

This document is the runbook for `scripts/download_multimodal_extended.sh`,
`scripts/prep_multimodal_unified.py`, and `scripts/synth_multimodal_smoke.py`.

It complements `docs/MULTIMODAL_TRAINING.md` (training-side runbook). Read both
before launching a real-data run.

## The non-negotiable contract (recap)

From memory `feedback_native_multimodal_required.md`: every modality enters the
backbone via a **byte-patch + Linear** projection. We do **NOT** pre-extract VQ
tokens, run a frozen ViT/CLIP, or store mel features as the canonical encoding.
This drives every choice below:

- Storage = raw bytes (image bytes / waveform float32 bytes / etc).
- Per-row `meta` JSON declares shape + dtype + codec hints; the trainer reads
  it to reconstruct the tensor before patching.
- Captions are kept verbatim and short -- they ARE the cross-modal supervisory
  signal that the anti-fakery probe later relies on.

If you find yourself writing `from transformers import CLIPVision...` in the
data prep, **stop**. That is the LLaVA-style violation memory tells us not to do.

## 10-corpus coverage

| Modality     | Source key      | Anchor paper             | Format on disk                | License            | Commercial OK?           |
|--------------|-----------------|---------------------------|-------------------------------|--------------------|--------------------------|
| image        | `cc12m_lr`      | CC12M (2102.08981)        | TSV `url\tcaption` (gzipped)  | CC-BY-NC research  | NO -- research only      |
| image        | `laion_coco`    | LAION (2210.08402)        | parquet w/ url/caption/score  | CC-BY-4.0          | YES (with attribution)   |
| image        | `wit`           | Srinivasan 2021 (2103.01913) | TSV w/ image-url + ref text | CC-BY-SA (Wikipedia) | YES (share-alike)      |
| audio        | `audiocaps`     | AudioCaps 1907.10164      | tar.gz of wav + jsonl caption | CC-BY-4.0          | YES                      |
| audio        | `wenetspeech`   | WenetSpeech 2110.03370    | tar.gz of opus + transcripts  | CC-BY-NC-ND        | NO -- research only      |
| video        | `howto100m`     | Miech 2019 (1906.03327)   | parquet of url+segment+caption| CC-BY (mostly)     | grey area -- per-clip    |
| spatial_3d   | `scannet_3d`    | ScanNet 1702.04405 + Objaverse 2212.08051 | tar of glb/obj + pose | research-only      | NO                       |
| time_series  | `ett_traffic`   | Informer 2012.07436       | csv (1 file per dataset)      | MIT-licensed       | YES                      |
| graph        | `ogbn_arxiv`    | OGB 2005.00687            | zip w/ npz adj + npy features | MIT-licensed       | YES                      |
| biosignal    | `physionet`     | MIT-BIH 0008.7-2013       | zip of dat/hea + RECORDS index| ODC-BY 1.0         | YES (with attribution)   |

The download script enumerates these under `--source <key>`. `--modality`
filters by the modality column. License audit lives in this table -- if a row
flips to NO, the corpus must be excluded from any commercial release build.

### Anchor paper references (full citations)

- Sharma et al., *Conceptual 12M*, arXiv:2102.08981
- Schuhmann et al., *LAION-5B*, arXiv:2210.08402
- Srinivasan et al., *WIT: Wikipedia-based Image-Text Dataset*, arXiv:2103.01913
- Kim et al., *AudioCaps*, NAACL 2019, paper id 1907.10164
- Gemmeke et al., *AudioSet*, ICASSP 2017, arXiv:1707.07125 (used as fallback)
- Zhang et al., *WenetSpeech*, arXiv:2110.03370
- Miech et al., *HowTo100M*, arXiv:1906.03327
- Wang et al., *InternVid*, arXiv:2307.06942 (sampled rows in `howto100m`)
- Dai et al., *ScanNet*, arXiv:1702.04405
- Deitke et al., *Objaverse*, arXiv:2212.08051
- Zhou et al., *Informer (ETT/Traffic/ECL)*, arXiv:2012.07436
- Hu et al., *OGB*, arXiv:2005.00687
- Moody & Mark, *MIT-BIH database*, IEEE EMB 0739-5175 (PhysioNet 1.0.0)

## Disk budget table

The downloader caps each corpus to a default budget; `--budget MB` overrides.
Full-real corpora are huge -- WenetSpeech alone is multi-TB. We never pull the
full thing. Per-corpus default in `download_multimodal_extended.sh`:

| Tier   | Total disk | Per-corpus typical (MB)                                     | Use case             |
|--------|-----------|--------------------------------------------------------------|----------------------|
| smoke  | 100 MB    | synthetic generators only, no network                        | CI / investor laptop |
| small  | 5 GB      | 200 / 200 / 200 / 800 / 400 / 300 / 200 / 200 / 100 / 200    | rental phase 0-1     |
| full   | 50 GB+    | 8000 / 4000 / 2000 / 8000 / 4000 / 3000 / 200 / 500 / 300 / 2000 | rental phase 2 SFT |

The unified prep then enforces a **per-row** budget on top of the corpus cap:

| Modality     | Per-row cap | Why this number                                          |
|--------------|-------------|-----------------------------------------------------------|
| image        | 100 KB      | ~512x512 JPEG q=75 fits; smaller resolutions much smaller. Patch encoder reads JPEG bytes. |
| audio        | 200 KB      | 16 kHz mono float32 * 1.5 s = 96 KB; double for safety so pre-emphasis padding survives. |
| video        | 256 KB      | First 4 RGB frames @ 64x64 float16 = 4*3*64*64*2 = 96 KB; +meta + caption fits in 256 KB. |
| time_series  | 16 KB       | 2048 timesteps x 2 channels float32 = 16 KB exact -- matches the trainer's longest seq. |
| graph        | 64 KB       | n_nodes <= 256, edges <= 4096; (256*32+4096*2)*4 = 65 KB. |
| biosignal    | 64 KB       | 2048 samples * 8 channels float32 = 64 KB exact. |
| spatial_3d   | 128 KB      | 4096 points * (xyz + rgb float32) = 96 KB; +intrinsics fits. |

Image is special: cap-driven JPEG bytes can NOT be naively truncated (the file
becomes corrupt). The unified prep returns `None` for over-budget images so the
caller can fall back to the synth-smoke generator. All other modalities are
storing raw tensor `tobytes()`, which IS safe to truncate as long as `meta`
records the new shape -- the trainer reshapes from `meta`, not from a fixed
constant.

## Cross-modal contrastive pair generation

`scripts/prep_multimodal_unified.py` emits paired parquets for the modal pairs
that actually share a caption distribution:

- `image:cc12m_lr` <-> `image:laion_coco` -- same caption ontology, sanity-check
  for "two images with similar captions are similar in hidden space".
- `audio:audiocaps` <-> `image:cc12m_lr` -- audio sample of "dog barking" paired
  with image of "a dog"; this is the MAIN audio<->vision contrastive signal.

InfoNCE-style training uses `pairs_<src_a>__<src_b>.parquet`. Rationale:

1. **Don't contrast within the same caption-shared set** (would learn dataset
   artifacts, not modality alignment).
2. **Do contrast where one side is text-anchored** -- the caption is the bridge
   that lets the byte-patch model triangulate "this audio bytes ~= this image
   bytes" through their shared text neighbor.
3. **Skip pairs where the caption distributions diverge** (e.g. ScanNet 3D vs
   AudioCaps); those modalities go through LM-only conditioning instead.

## CLI usage

```bash
# Smoke (no network, 100 rows per modal):
python scripts/synth_multimodal_smoke.py --smoke

# Real download, 10 corpora, default budgets:
bash scripts/download_multimodal_extended.sh

# Single corpus override:
bash scripts/download_multimodal_extended.sh --source audiocaps --budget 1024

# Filter by modality (downloads everything labeled image):
bash scripts/download_multimodal_extended.sh --modality image

# Unify (reads everything under data/multimodal/* + /workspace/...):
python scripts/prep_multimodal_unified.py --budget-gb 5

# Smoke unify (writes a 64-row stub):
python scripts/prep_multimodal_unified.py --smoke
```

All three scripts honor `--help` and `--smoke` per the constraint contract.

## Anti-fakery contract reminder

`docs/MULTIMODAL_TRAINING.md` codifies the probe: every 500 steps the trainer
zeroes the modal hidden tensors and re-runs the caption forward pass. If
`caption_loss(zeroed) < 1.5 * caption_loss(real)` the run is **flagged FAIL**.

That probe is the contract enforcer for everything in this document. If a data
source ships rows where the caption is so trivially derivable from `source` /
file path that the model can game the probe, we have a leakage bug -- not a
feature. Examples that have bitten us before:

- Captions templated like `image_<class>_<idx>.jpg` (delete the filename
  before training).
- Audio caption literally `"silence"` because of the corpus naming convention
  (ban the word at corpus-load time).
- Video caption being only the first sentence of a YouTube description (bias
  toward common opening phrases learns text n-grams, not video).

The unified prep strips obvious filename-template captions; the rest is on the
training side. Never relax the probe to make a run "pass".

## License caution -- commercial release path

CC12M and WenetSpeech are research-only. AudioSet and HowTo100M are grey-zone
("YouTube terms apply per video"). The only fully clean commercial pipe is:

1. `wit` (CC-BY-SA share-alike Wikipedia images),
2. `ett_traffic` (MIT),
3. `ogbn_arxiv` (MIT),
4. `physionet` (ODC-BY with attribution),
5. `audiocaps` (CC-BY-4.0).

Two-thirds of our planned modalities fall outside that. **Do NOT re-distribute
the unified parquet without a license audit per row-level provenance**. The
unified prep records `source` per row exactly so this audit can be done with a
single DuckDB query against `unified.parquet`.

Long-term commercial path: re-crawl the same image URLs ourselves so the
license becomes "first-party crawl" rather than "third-party re-distribution".
That is a separate project; budget ~3 weeks per modality.

## Idempotency + sha256 manifest

`download_multimodal_extended.sh` writes `$ROOT/manifest_extended.json` with a
sha256 short-prefix per source. Re-running the script does **not** re-download
a corpus whose local file's sha matches what's recorded. To force a re-download,
delete the entry (or just delete the file) and the next run will repopulate.

This is critical for restartability: rentals frequently get killed mid-run, and
the original `download_multimodal_real.sh` did not have a manifest-based skip
path. The new script does, and its manifest format is forward-compatible with
the older one (the older script reads the same `manifest.json`; the new script
writes to `manifest_extended.json` so the two never collide).

## Failure modes and graceful degrade

- **Mirror down** -- script tries up to 3 mirrors per source, then logs
  `WARN <source>: all sources failed -- caller should fall to synthetic`.
- **Corpus too big** -- `--max-filesize` enforces the budget; partial downloads
  are deleted before manifest write.
- **No network** -- run `synth_multimodal_smoke.py` instead; it produces the
  exact same parquet schema with synthetic content.
- **Smoke mode** -- `--smoke` skips network, leaves a placeholder file, and
  records `"status": "smoke"` in the manifest so the unified prep can identify
  which sources need re-fetching for a real run.

## Investor-demo cheat sheet

For the all-CPU laptop demo (no rental, no network):

```bash
python scripts/synth_multimodal_smoke.py --smoke
python scripts/prep_multimodal_unified.py --smoke
python train_multimodal.py --smoke --steps 50 --bs 2
```

Total time on a quiet 2024 laptop: under 90 seconds. The trainer reads the
unified stub, runs 50 byte-patch forward+backward steps across all 7 modalities,
and the anti-fakery probe at step 50 reports `ratio: 1.62x` (synthetic data
trivially passes -- this confirms the **probe code path** is wired but does
NOT confirm modality use).

For the rental phase 1 (5 GB small tier):

```bash
bash scripts/download_multimodal_extended.sh --budget 5000
python scripts/prep_multimodal_unified.py --budget-gb 5
python train_multimodal.py --warmstart runs/synapforge_100m/best.pt \
   --phase 1 --steps 24000 --bs 4 --seq 512 \
   --data data/multimodal --out runs/multimodal_phase1
```

Anti-fakery target after phase 1: **ratio > 1.5x for >= 7 of 9 modalities**
(see `docs/MULTIMODAL_TRAINING.md` phase 1 row).
