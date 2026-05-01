"""sf.data — minimal parquet -> token-pair stream for synapforge training.

Goal: feed sf.train without pulling in the torch.utils.data DataLoader
multi-process machinery (which interacts badly with our plasticity hooks
and CUDA contexts in v0.1). Just a generator yielding (in, out) tensors.

Pipeline:
  parquet rows (string text)
    -> AutoTokenizer encode (any HF-compatible tokenizer; vocab is
       whatever the tokenizer reports, e.g. 50257 for gpt2 / 151643 for
       qwen2.5)
    -> append EOT token between docs (auto-derived from the tokenizer's
       eos_token_id; no hard-coded 50256)
    -> slide a window of seq_len+1 tokens, batch up B of them
    -> emit (tokens_in[B,T], tokens_out[B,T]) where out = in shifted left by 1.

Files are streamed row-by-row (pyarrow batch_size=64) so memory stays bounded
even on the 900k-row WT-103 train shard. The vocab_size argument is purely
documentation; the actual vocab is whatever the tokenizer reports.

Public API:
    >>> ds = ParquetTokenStream("/workspace/data/wt103_raw/train-*.parquet",
    ...                         seq_len=256, batch_size=32)
    >>> it = iter(ds)
    >>> x, y = next(it)              # both (32, 256) int64 cpu tensors
    >>> assert (x[:, 1:] == y[:, :-1]).all()  # next-token target shape
"""

from __future__ import annotations

import glob
from collections.abc import Iterator

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None
import torch

_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(name: str = "gpt2"):
    """Lazy-load tokenizer (any HF AutoTokenizer-compatible); per-name cached.

    Cached PER tokenizer name -- the previous single-global cache returned
    a stale tokenizer when the trainer switched between GPT-2 (vocab=50257)
    and Qwen (vocab=151643) within the same process, silently mis-tokenising
    the second stream.
    """
    cached = _TOKENIZER_CACHE.get(name)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER_CACHE[name] = tok
    return tok


class ParquetTokenStream:
    """Infinite-loop iterator over (tokens_in, tokens_out) batches.

    Args:
        glob_pattern: shell-style glob of parquet files. Each file must have
            a single string column (we auto-pick the first string-typed col).
        seq_len: tokens per training example T. We slide a length-T+1 window
            so we can split into in/out cleanly.
        batch_size: examples per batch B.
        vocab_size: documentation only; tokenizer vocab is authoritative.
        text_column: optional explicit column name; auto-detect if None.
        eot_id: end-of-text token id used as document separator. If None
            (default) auto-derives from the tokenizer's eos_token_id so
            we don't accidentally inject GPT-2's 50256 magic into a
            different vocab (e.g. Qwen).
        loop: if True, loop over files forever (training loop never raises
            StopIteration).
    """

    def __init__(
        self,
        glob_pattern: str,
        seq_len: int = 256,
        batch_size: int = 32,
        vocab_size: int = 50257,
        text_column: str | None = None,
        eot_id: int | None = None,
        loop: bool = True,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.files = sorted(glob.glob(glob_pattern))
        if not self.files:
            raise FileNotFoundError(f"glob {glob_pattern!r} matched no files")
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.vocab_size = int(vocab_size)
        self.text_column = text_column
        self.loop = bool(loop)
        self.tokenizer_name = tokenizer_name
        self._tok = _get_tokenizer(tokenizer_name)
        # Default EOT to the tokenizer's eos_token_id, NOT the GPT-2 magic
        # number 50256 -- using a hard-coded id with a Qwen tokenizer would
        # inject a spurious mid-vocab token between docs and corrupt
        # training. eot_id can still be passed explicitly to override.
        if eot_id is None:
            eos = getattr(self._tok, "eos_token_id", None)
            if eos is None:
                # Fall back to GPT-2 magic only if the tokenizer truly has
                # no eos (vanishingly rare).
                eos = 50256
            self.eot_id = int(eos)
        else:
            self.eot_id = int(eot_id)

    # ---------------------------------------------------------------- internal

    def _iter_text_rows(self) -> Iterator[str]:
        """Yield text strings from all parquet files, looping if `self.loop`."""
        while True:
            for path in self.files:
                pf = pq.ParquetFile(path)
                col = self.text_column
                if col is None:
                    # auto-pick first string column
                    for f in pf.schema_arrow:
                        if str(f.type) == "string":
                            col = f.name
                            break
                    if col is None:
                        raise RuntimeError(
                            f"no string column in {path}; "
                            f"got {pf.schema_arrow}"
                        )
                # batch_size here is parquet-row batches, not training batches
                for batch in pf.iter_batches(batch_size=64, columns=[col]):
                    for s in batch.column(col).to_pylist():
                        if s:
                            yield s
            if not self.loop:
                return

    def _iter_token_chunks(self) -> Iterator[list[int]]:
        """Yield length-(seq_len+1) integer windows from the token stream."""
        win = self.seq_len + 1
        buf: list[int] = []
        for txt in self._iter_text_rows():
            ids = self._tok.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(self.eot_id)
            while len(buf) >= win:
                yield buf[:win]
                buf = buf[win:]

    # ---------------------------------------------------------------- public

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (tokens_in [B,T], tokens_out [B,T]) int64 cpu tensors."""
        chunks: list[list[int]] = []
        for c in self._iter_token_chunks():
            chunks.append(c)
            if len(chunks) >= self.batch_size:
                arr = torch.tensor(chunks, dtype=torch.long)  # (B, T+1)
                tokens_in = arr[:, :-1].contiguous()
                tokens_out = arr[:, 1:].contiguous()
                yield tokens_in, tokens_out
                chunks = []

    def __repr__(self) -> str:
        return (
            f"ParquetTokenStream(files={len(self.files)}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"tokenizer={self.tokenizer_name!r})"
        )


__all__ = ["ParquetTokenStream"]
