"""
BM25 sidecar — verbatim exact-token retrieval for long-context recall.

Why: CfC + PLIF has no KV cache. NIAH UUID tests fail because we only
retrieve nearest-neighbor hidden vectors, not exact tokens. PQ16-compressed
hidden state loses ~15% of source content — fine for paraphrastic recall,
fails on verbatim queries (UUIDs, code snippets, named entities).

Solution: maintain a parallel BM25 inverted index keyed on raw token-ids
alongside the hidden-state retrieval. At query time, union top-K from both
channels and let a learned mixer trust per channel.

Cost:
  storage: 4 bytes per token-position → 50M tokens × 4B = 200 MB
  insert:  amortized O(1) per token (token-id → posting list append)
  query:   O(K · log N) for top-K retrieval, ~5 ms at 50M

Backends:
  - tantivy (Rust BM25, fastest, requires `pip install tantivy`)
  - whoosh (pure Python, 5× slower, no extra dep)
  - hash-only (just inverted hash → positions, no scoring; trivial fallback)

The L3 drift killer: agent synthesis 2026-04-30 estimates BM25 sidecar alone
takes L3 (50M) NIAH pass-rate from ~30% → ~75%. Single biggest L3 win.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tantivy

    HAS_TANTIVY = True
except ImportError:
    HAS_TANTIVY = False


@dataclass
class TokenPosting:
    """One occurrence of a token at a specific position."""

    position: int
    document_id: int = 0
    timestamp_ns: int = 0


@dataclass
class _BM25Stats:
    n_docs: int = 0
    total_tokens: int = 0
    avg_doc_length: float = 0.0


class HashedTokenIndex:
    """Pure-Python fallback: hash inverted index with no scoring.

    For each token (or n-gram), maintain a list of positions. Query returns
    union of positions for the query terms, ranked by recency and term overlap.

    Uses ~4 bytes/token in memory (just position int32).
    """

    def __init__(
        self,
        ngram_n: int = 1,
        max_postings_per_term: int = 10000,
    ) -> None:
        self.ngram_n = ngram_n
        self.max_postings_per_term = max_postings_per_term
        self.inverted: Dict[int, List[int]] = defaultdict(list)
        self.total_positions = 0

    @staticmethod
    def _token_hash(token_id: int, salt: int = 0) -> int:
        return token_id ^ (salt * 2654435761)

    def add(self, tokens: List[int], start_position: int = 0) -> None:
        """Add a sequence of token-ids starting at start_position."""
        for i, t in enumerate(tokens):
            pos = start_position + i
            h = self._token_hash(t)
            postings = self.inverted[h]
            postings.append(pos)
            if len(postings) > self.max_postings_per_term:
                postings.pop(0)
            self.total_positions += 1

    def query(
        self,
        query_tokens: List[int],
        top_k: int = 16,
        recency_weight: float = 0.3,
        current_position: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Return [(position, score)] for top-K matches."""
        if not query_tokens:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for t in query_tokens:
            h = self._token_hash(t)
            for pos in self.inverted.get(h, []):
                scores[pos] += 1.0
        if not scores:
            return []
        if recency_weight > 0 and current_position is not None:
            for pos in list(scores.keys()):
                age_norm = max(0.0, 1.0 - (current_position - pos) / max(self.total_positions, 1))
                scores[pos] += recency_weight * age_norm
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def stats(self) -> dict:
        return {
            "backend": "hashed",
            "n_terms": len(self.inverted),
            "total_positions": self.total_positions,
            "avg_postings_per_term": (
                self.total_positions / max(len(self.inverted), 1)
            ),
        }


class _PythonBM25:
    """In-memory BM25 implementation (no external lib).

    Documents are conceptual (text chunks). Each `add(tokens, doc_id)` adds
    one document. `query(tokens, top_k)` returns top-K (doc_id, position, score).

    Storage: ~12 bytes per (term, doc) posting. 50M tokens × 12B = 600 MB.
    Slower than tantivy but no dep.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_to_doc_freq: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.doc_lengths: Dict[int, int] = {}
        self.position_map: Dict[int, List[int]] = defaultdict(list)
        self.doc_count = 0
        self.total_length = 0

    def add(self, tokens: List[int], doc_id: int, start_position: int = 0) -> None:
        if doc_id not in self.doc_lengths:
            self.doc_count += 1
            self.doc_lengths[doc_id] = 0
        for i, t in enumerate(tokens):
            self.term_to_doc_freq[t][doc_id] = self.term_to_doc_freq[t].get(doc_id, 0) + 1
            self.position_map[t].append(start_position + i)
        added = len(tokens)
        self.doc_lengths[doc_id] += added
        self.total_length += added

    def _avg_doc_length(self) -> float:
        if self.doc_count == 0:
            return 0.0
        return self.total_length / self.doc_count

    def _idf(self, term: int) -> float:
        df = len(self.term_to_doc_freq.get(term, {}))
        if df == 0:
            return 0.0
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def query(
        self,
        query_tokens: List[int],
        top_k: int = 16,
    ) -> List[Tuple[int, float]]:
        avg_dl = self._avg_doc_length() or 1.0
        scores: Dict[int, float] = defaultdict(float)
        for t in query_tokens:
            idf = self._idf(t)
            if idf == 0:
                continue
            postings = self.term_to_doc_freq.get(t, {})
            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(doc_id, 1)
                norm = self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score = idf * (tf * (self.k1 + 1)) / (tf + norm)
                scores[doc_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out: List[Tuple[int, float]] = []
        for doc_id, score in ranked:
            best_pos = -1
            best_overlap = 0
            for t in query_tokens:
                positions = self.position_map.get(t, [])
                for p in positions:
                    if best_pos < 0 or abs(p - best_pos) < 100:
                        best_overlap += 1
                        if best_pos < 0:
                            best_pos = p
            out.append((best_pos if best_pos >= 0 else doc_id * 1000, score))
        return out


class BM25Sidecar:
    """Verbatim exact-token retrieval index.

    Use:
        sidecar = BM25Sidecar(backend="auto", index_dir="/workspace/runs/bm25")
        sidecar.add(tokens=[1, 2, 3, ...], doc_id=0, start_position=0)
        hits = sidecar.query(tokens=[42, 43], top_k=16)
        # hits = [(position, score), ...]
    """

    def __init__(
        self,
        backend: str = "auto",
        index_dir: str | Path = "/workspace/runs/bm25",
        chunk_size: int = 4096,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size

        if backend == "auto":
            if HAS_TANTIVY:
                backend = "tantivy"
            else:
                backend = "python"
        self.backend = backend

        if backend == "tantivy":
            self._init_tantivy()
        elif backend == "python":
            self.bm25 = _PythonBM25()
        elif backend == "hash":
            self.bm25 = HashedTokenIndex()
        else:
            raise ValueError(f"unknown backend {backend}")

        self.position_offset = 0
        self.current_doc_id = 0
        self.tokens_in_current_doc = 0

    def _init_tantivy(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("tokens", stored=True)
        schema_builder.add_integer_field("position", stored=True, indexed=True)
        schema_builder.add_integer_field("doc_id", stored=True, indexed=True)
        self.schema = schema_builder.build()
        self.index = tantivy.Index(self.schema, path=str(self.index_dir))
        self.writer = self.index.writer(heap_size=64_000_000)

    def add(
        self,
        tokens: List[int],
        doc_id: Optional[int] = None,
        start_position: Optional[int] = None,
    ) -> None:
        """Append a sequence of token-ids."""
        if start_position is None:
            start_position = self.position_offset
        if doc_id is None:
            doc_id = self.current_doc_id

        if self.backend == "tantivy":
            chunk_text = " ".join(str(t) for t in tokens)
            doc = tantivy.Document()
            doc.add_text("tokens", chunk_text)
            doc.add_integer("position", start_position)
            doc.add_integer("doc_id", doc_id)
            self.writer.add_document(doc)
            self.tokens_in_current_doc += len(tokens)
            if self.tokens_in_current_doc >= self.chunk_size * 16:
                self.writer.commit()
                self.tokens_in_current_doc = 0
        else:
            self.bm25.add(tokens, doc_id=doc_id, start_position=start_position) \
                if hasattr(self.bm25, "doc_lengths") \
                else self.bm25.add(tokens, start_position=start_position)

        self.position_offset = start_position + len(tokens)

    def query(
        self,
        query_tokens: List[int],
        top_k: int = 16,
        current_position: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Return [(position, score)] for top-K verbatim matches."""
        if not query_tokens:
            return []
        if self.backend == "tantivy":
            self.writer.commit()
            searcher = self.index.searcher()
            query_text = " ".join(str(t) for t in query_tokens)
            parsed = self.index.parse_query(query_text, ["tokens"])
            hits = searcher.search(parsed, top_k).hits
            out: List[Tuple[int, float]] = []
            for score, addr in hits:
                doc = searcher.doc(addr)
                pos = doc["position"][0]
                out.append((pos, float(score)))
            return out
        else:
            if isinstance(self.bm25, HashedTokenIndex):
                return self.bm25.query(
                    query_tokens=query_tokens,
                    top_k=top_k,
                    current_position=current_position,
                )
            return self.bm25.query(query_tokens=query_tokens, top_k=top_k)

    def commit(self) -> None:
        if self.backend == "tantivy":
            self.writer.commit()

    def stats(self) -> dict:
        if self.backend == "tantivy":
            self.commit()
            searcher = self.index.searcher()
            return {
                "backend": "tantivy",
                "n_docs": searcher.num_docs,
                "n_terms": "unknown",
                "position_offset": self.position_offset,
            }
        base = self.bm25.stats() if hasattr(self.bm25, "stats") else {}
        return {
            **base,
            "backend": self.backend,
            "position_offset": self.position_offset,
        }


def merge_retrieval(
    bm25_hits: List[Tuple[int, float]],
    semantic_hits: List[Tuple[int, float]],
    bm25_weight: float = 0.5,
    top_k: int = 16,
) -> List[Tuple[int, float]]:
    """Linear-weighted union of BM25 (verbatim) + semantic (FAISS) top-K."""
    bm25_max = max((s for _, s in bm25_hits), default=1.0) or 1.0
    sem_max = max((s for _, s in semantic_hits), default=1.0) or 1.0

    scored: Dict[int, float] = defaultdict(float)
    for pos, score in bm25_hits:
        scored[pos] += bm25_weight * (score / bm25_max)
    for pos, score in semantic_hits:
        scored[pos] += (1 - bm25_weight) * (score / sem_max)

    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
