"""
Long-context memory subsystem.

Modules (some scaffold, some wiring of existing pieces):
  bm25_sidecar.py          Verbatim exact-token BM25/hash index for NIAH-class
                            queries. Mitigates "no KV cache → no verbatim recall".
                            ~200 LOC, 0 GPU-h, ~1 day. Highest-ROI L3 fix.
  trained_pq_codebook.py   Train Product Quantization codebook on actual hidden
                            state distribution (not random). FAISS recall@10
                            73% → 88% typical.
  retrieval_mixer.py       Linear gate over (BM25 score, FAISS cosine, STDP
                            recency) → trust score per query. +256 params.
  multitau_plif_binder.py  Wire existing MultiBandTau into PLIF.tau_log so
                            slow-band channels keep 100K+ token horizon.
  dual_path_gate.py        At pos > 32K AND PLIF spike-rate < 8%: switch to
                            dense CfC for next 4K tokens (PLIF observe_only).
"""

from .bm25_sidecar import BM25Sidecar, HashedTokenIndex

__all__ = [
    "BM25Sidecar",
    "HashedTokenIndex",
]
