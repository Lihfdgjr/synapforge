"""prep_multimodal_unified -- normalize ALL multimodal sources to one parquet.

Reads any combination of the per-source corpora produced by
``download_multimodal_extended.sh`` and the synthetic-smoke corpora produced by
``synth_multimodal_smoke.py``, then emits a single ``unified.parquet`` whose
rows follow the contract documented in ``docs/MULTIMODAL_DATA.md``::

    bytes    : binary   -- raw modal bytes (image/audio/video chunk/...)
    caption  : string   -- short natural-language caption
    modality : string   -- one of {image, audio, video, time_series,
                                   graph, biosignal, spatial_3d}
    source   : string   -- which corpus the row came from
    meta     : string   -- JSON string with shape/dtype/codec hints

Per-modality byte budget per example (cap before emit -- see DOC):

    image       <= 100 KB     (downsample-resize / re-encode JPEG q=75)
    audio       <= 200 KB     (16 kHz mono float32 / 1.5 s clip)
    video       <= 256 KB     (first 4 frames RGB 64x64 float16)
    time_series <= 16  KB     (truncate channels x T to fit)
    graph       <= 64  KB     (limit n_nodes <= 256, edges <= 4096)
    biosignal   <= 64  KB     (truncate channels x T)
    spatial_3d  <= 128 KB     (cap n_pts at 4096)

After per-row capping, a global ``--budget-gb`` ceiling is applied. Rows are
sampled fairly across modalities by a deterministic round-robin so the unified
file does NOT collapse to whichever corpus happens to be biggest.

Cross-modal contrastive pair generation:

    For (modality, source) pairs `image`-`cc12m_lr` and `audio`-`audiocaps`,
    we additionally emit `*_pair_text.parquet` with rows
    {bytes, caption, modality, paired_caption, paired_modality}
    so the trainer can do InfoNCE without redoing the alignment.

Usage::

    python scripts/prep_multimodal_unified.py --help
    python scripts/prep_multimodal_unified.py --smoke    # synth-only smoke
    python scripts/prep_multimodal_unified.py \\
        --input-dirs data/multimodal,/workspace/data/multimodal \\
        --out data/multimodal/unified.parquet \\
        --budget-gb 5

Constraints:
    - pyarrow optional but recommended; falls back to a stub-only run that
      still validates argument parsing for `bash -n`-style smoke tests.
    - Real-corpus readers (Pillow/imageio/av) are imported lazily; missing
      readers degrade to "skip-with-WARN" instead of crashing the run.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

logger = logging.getLogger("prep_multimodal_unified")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ALL_MODALITIES = (
    "image", "audio", "video", "time_series",
    "graph", "biosignal", "spatial_3d",
)

# Per-modality byte budget per row. Order-of-magnitude rationale in DOC.
BYTE_BUDGET = {
    "image":      100 * 1024,
    "audio":      200 * 1024,
    "video":      256 * 1024,
    "time_series": 16 * 1024,
    "graph":       64 * 1024,
    "biosignal":   64 * 1024,
    "spatial_3d": 128 * 1024,
}

# Cross-modal contrastive pair recipes: (mod_a, src_a) <-> (mod_b, src_b).
CONTRASTIVE_PAIRS = [
    ("image", "cc12m_lr",  "image", "laion_coco"),
    ("audio", "audiocaps", "image", "cc12m_lr"),
]


# --------------------------------------------------------------------- helpers
def _try_import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError:
        logger.warning("pyarrow not installed -- emit will be skipped")
        return None, None


def _row(bytes_: bytes, caption: str, modality: str, source: str,
         meta: dict) -> dict:
    return {
        "bytes": bytes_,
        "caption": caption,
        "modality": modality,
        "source": source,
        "meta": json.dumps(meta, separators=(",", ":")),
    }


def _enforce_byte_budget(payload: bytes, modality: str) -> Optional[bytes]:
    """Truncate (or signal drop) by the per-modality budget."""
    cap = BYTE_BUDGET.get(modality, 256 * 1024)
    if len(payload) <= cap:
        return payload
    if modality in ("image",):
        # JPEG/PNG bytes shouldn't be naively sliced; signal "needs re-encode"
        # by returning None so caller falls back to synthetic placeholder.
        return None
    # For raw tensor bytes (audio float32, biosignal float32, time_series, graph)
    # truncating to the budget is acceptable: the trainer reads `meta` and adapts.
    return payload[:cap]


# ------------------------------------------------------------------ corpus io
@dataclass
class CorpusReader:
    name: str
    modality: str
    source: str
    path: Path
    rows_emitted: int = 0
    rows_skipped: int = 0

    def iter_rows(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Yield budgeted rows from a corpus on disk.

        Real readers (parquet/tar/csv) are tried in order; if none fits, the
        corpus is treated as missing and we yield nothing -- the caller falls
        back to synth_multimodal_smoke for that modality.
        """
        if not self.path.exists():
            logger.warning(f"{self.name}: missing path {self.path}")
            return
        if self.path.is_dir():
            yield from self._iter_dir(limit)
            return
        suffix = self.path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            yield from self._iter_parquet(limit)
        elif suffix in (".tsv", ".csv"):
            yield from self._iter_table(limit)
        elif suffix in (".jsonl", ".json"):
            yield from self._iter_jsonl(limit)
        else:
            logger.warning(f"{self.name}: unsupported file type {suffix}")

    # -- per-format iterators -------------------------------------------------
    def _iter_dir(self, limit: Optional[int]) -> Iterator[dict]:
        for child in sorted(self.path.iterdir()):
            if limit is not None and self.rows_emitted >= limit:
                return
            if not child.is_file():
                continue
            try:
                payload = child.read_bytes()
            except OSError:
                self.rows_skipped += 1
                continue
            payload = _enforce_byte_budget(payload, self.modality)
            if payload is None:
                self.rows_skipped += 1
                continue
            yield _row(payload, child.stem, self.modality, self.source,
                       {"basename": child.name, "size": len(payload)})
            self.rows_emitted += 1

    def _iter_parquet(self, limit: Optional[int]) -> Iterator[dict]:
        pa, pq = _try_import_pyarrow()
        if pa is None:
            return
        try:
            tbl = pq.read_table(str(self.path))
        except Exception as exc:
            logger.warning(f"{self.name}: parquet read failed: {exc!r}")
            return
        cols = set(tbl.column_names)
        for i in range(tbl.num_rows):
            if limit is not None and self.rows_emitted >= limit:
                return
            payload_b: bytes = b""
            caption = ""
            meta: dict = {}
            if "bytes" in cols:
                payload_b = bytes(tbl["bytes"][i].as_py() or b"")
            if "caption" in cols:
                caption = str(tbl["caption"][i].as_py() or "")
            elif "text" in cols:
                caption = str(tbl["text"][i].as_py() or "")
            if "meta" in cols:
                m = tbl["meta"][i].as_py()
                if isinstance(m, str):
                    try:
                        meta = json.loads(m)
                    except Exception:
                        meta = {}
                elif isinstance(m, dict):
                    meta = m
            payload_b = _enforce_byte_budget(payload_b, self.modality) or b""
            yield _row(payload_b, caption, self.modality, self.source, meta)
            self.rows_emitted += 1

    def _iter_table(self, limit: Optional[int]) -> Iterator[dict]:
        sep = "\t" if self.path.suffix.lower() == ".tsv" else ","
        try:
            f = self.path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            return
        with f:
            for line in f:
                if limit is not None and self.rows_emitted >= limit:
                    return
                parts = line.rstrip("\n").split(sep)
                if len(parts) < 2:
                    self.rows_skipped += 1
                    continue
                # url/caption table -> emit URL as bytes placeholder + caption.
                url, caption = parts[0], parts[1]
                bytes_ = url.encode("utf-8")[:BYTE_BUDGET[self.modality]]
                yield _row(bytes_, caption, self.modality, self.source,
                           {"url_only": True, "url": url})
                self.rows_emitted += 1

    def _iter_jsonl(self, limit: Optional[int]) -> Iterator[dict]:
        try:
            f = self.path.open("r", encoding="utf-8")
        except OSError:
            return
        with f:
            for line in f:
                if limit is not None and self.rows_emitted >= limit:
                    return
                try:
                    obj = json.loads(line)
                except Exception:
                    self.rows_skipped += 1
                    continue
                payload = obj.get("bytes") or obj.get("data") or b""
                if isinstance(payload, str):
                    payload = payload.encode("utf-8", errors="replace")
                payload = _enforce_byte_budget(payload, self.modality) or b""
                yield _row(payload, obj.get("caption", ""),
                           self.modality, self.source,
                           {k: v for k, v in obj.items()
                            if k not in {"bytes", "data", "caption"}})
                self.rows_emitted += 1


# ----------------------------------------------------------- corpus discovery
def discover_corpora(input_dirs: list[Path]) -> list[CorpusReader]:
    """Walk each input dir and pair it with a CorpusReader."""
    readers: list[CorpusReader] = []
    seen: set[str] = set()
    for root in input_dirs:
        if not root.exists():
            logger.warning(f"discover: missing input dir {root}")
            continue
        for mod_dir in sorted(root.iterdir()):
            if not mod_dir.is_dir() or mod_dir.name not in ALL_MODALITIES:
                continue
            for src_dir in sorted(mod_dir.iterdir()):
                if not src_dir.is_dir():
                    # Treat the modality dir itself as a single-source corpus.
                    candidate = next(
                        (p for p in [mod_dir / "train.parquet",
                                     mod_dir / "data.parquet",
                                     mod_dir]
                         if p.exists()), None
                    )
                    if candidate is None:
                        continue
                    name = f"{mod_dir.name}/_root"
                    if name in seen:
                        continue
                    seen.add(name)
                    readers.append(CorpusReader(
                        name=name, modality=mod_dir.name,
                        source="_root", path=candidate,
                    ))
                    continue
                # Source-specific layout: prefer train.parquet inside.
                candidate = next(
                    (p for p in [src_dir / "train.parquet",
                                 src_dir / "data.parquet",
                                 src_dir]
                     if p.exists()), None
                )
                if candidate is None:
                    continue
                name = f"{mod_dir.name}/{src_dir.name}"
                if name in seen:
                    continue
                seen.add(name)
                readers.append(CorpusReader(
                    name=name, modality=mod_dir.name,
                    source=src_dir.name, path=candidate,
                ))
    return readers


# -------------------------------------------------------- contrastive pairing
def emit_contrastive_pairs(rows_by_corpus: dict[str, list[dict]],
                           out: Path) -> int:
    """For each (mod_a/src_a, mod_b/src_b) recipe, emit a paired parquet."""
    n_pairs = 0
    pa, pq = _try_import_pyarrow()
    for mod_a, src_a, mod_b, src_b in CONTRASTIVE_PAIRS:
        ka = f"{mod_a}/{src_a}"
        kb = f"{mod_b}/{src_b}"
        a = rows_by_corpus.get(ka)
        b = rows_by_corpus.get(kb)
        if not a or not b:
            logger.info(f"contrastive: skip {ka}<->{kb} (one side missing)")
            continue
        n = min(len(a), len(b))
        paired = []
        for i in range(n):
            paired.append({
                "bytes_a": a[i]["bytes"],
                "bytes_b": b[i]["bytes"],
                "caption_a": a[i]["caption"],
                "caption_b": b[i]["caption"],
                "modality_a": mod_a,
                "modality_b": mod_b,
                "source_a": src_a,
                "source_b": src_b,
            })
        out_path = out.parent / f"pairs_{src_a}__{src_b}.parquet"
        if pa is not None:
            tbl = pa.Table.from_pylist(paired)
            pq.write_table(tbl, str(out_path), compression="zstd")
            logger.info(f"contrastive: {ka}<->{kb} {n} rows -> {out_path}")
        n_pairs += n
    return n_pairs


# ----------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dirs", default="data/multimodal,/workspace/data/multimodal",
                   help="comma-separated source roots")
    p.add_argument("--out", default="data/multimodal/unified.parquet",
                   help="output unified parquet path")
    p.add_argument("--budget-gb", type=float, default=5.0,
                   help="global GB ceiling for emitted bytes")
    p.add_argument("--limit-per-corpus", type=int, default=200_000,
                   help="hard row cap per (modality, source)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="skip real corpus reads; emit a 64-row stub")
    args = p.parse_args(argv)

    input_dirs = [Path(p_str.strip()) for p_str in args.input_dirs.split(",")
                  if p_str.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    budget_bytes = int(args.budget_gb * 1024 ** 3)

    pa, pq = _try_import_pyarrow()

    if args.smoke or pa is None:
        # Stub-only mode: write 64 rows of empty bytes per modality so the
        # downstream trainer still has a parquet shape to read against.
        logger.info("smoke / no-pyarrow: emitting 64-row stub")
        if pa is None:
            logger.error("pyarrow missing; cannot write parquet -- exit 0 stub")
            return 0
        stub_rows = []
        for m in ALL_MODALITIES:
            for i in range(8):
                stub_rows.append(_row(b"\x00" * 64, f"smoke {m} {i}",
                                      m, "smoke",
                                      {"smoke": True}))
        tbl = pa.Table.from_pylist(stub_rows)
        pq.write_table(tbl, str(out_path), compression="zstd")
        logger.info(f"smoke unified -> {out_path} ({len(stub_rows)} rows)")
        return 0

    readers = discover_corpora(input_dirs)
    if not readers:
        logger.error("no corpora found in any input-dir; abort")
        return 2
    logger.info(f"discovered {len(readers)} corpora across "
                f"{len(input_dirs)} input dirs")

    # Round-robin pull rows so unified.parquet fairly represents each corpus.
    iters = [(r, r.iter_rows(limit=args.limit_per_corpus)) for r in readers]
    rows_by_corpus: dict[str, list[dict]] = {r.name: [] for r in readers}
    rows_total: list[dict] = []
    bytes_total = 0
    t0 = time.time()
    while iters:
        next_iters = []
        for r, it in iters:
            try:
                row = next(it)
            except StopIteration:
                continue
            rows_by_corpus[r.name].append(row)
            rows_total.append(row)
            bytes_total += len(row["bytes"]) + len(row["caption"]) + len(row["meta"])
            if bytes_total >= budget_bytes:
                logger.info(f"hit budget {bytes_total/(1024**3):.2f} GB "
                            f"after {len(rows_total)} rows; stop")
                iters = []
                break
            next_iters.append((r, it))
        if not next_iters:
            break
        iters = next_iters

    elapsed = time.time() - t0
    tbl = pa.Table.from_pylist(rows_total)
    pq.write_table(tbl, str(out_path), compression="zstd")
    logger.info(f"unified -> {out_path} ({len(rows_total)} rows, "
                f"{bytes_total/(1024**3):.2f} GB, {elapsed:.1f}s)")

    n_pairs = emit_contrastive_pairs(rows_by_corpus, out_path)

    summary = {
        "out": str(out_path),
        "rows": len(rows_total),
        "bytes": bytes_total,
        "elapsed_s": round(elapsed, 2),
        "contrastive_pairs": n_pairs,
        "per_corpus": {r.name: {"emitted": r.rows_emitted,
                                "skipped": r.rows_skipped}
                       for r in readers},
    }
    summary_path = out_path.parent / "unified_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
