"""sf.data — minimal parquet -> token-pair stream for synapforge training.

Goal: feed sf.train without pulling in the torch.utils.data DataLoader
multi-process machinery (which interacts badly with our plasticity hooks
and CUDA contexts in v0.1). Just a generator yielding (in, out) tensors.

Pipeline:
  parquet rows (string text)
    -> GPT-2 BPE encode (transformers.GPT2TokenizerFast, vocab=50257)
    -> append EOT token (<|endoftext|> = 50256) between docs
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

_TOKENIZER = None


def _get_tokenizer(name: str = "gpt2"):
    """Lazy-load GPT-2 BPE; cached locally on first call."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import GPT2TokenizerFast
        _TOKENIZER = GPT2TokenizerFast.from_pretrained(name)
    return _TOKENIZER


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
        eot_id: end-of-text token id used as document separator. 50256 is
            GPT-2's <|endoftext|>.
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
        eot_id: int = 50256,
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
        self.eot_id = int(eot_id)
        self.loop = bool(loop)
        self.tokenizer_name = tokenizer_name
        self._tok = _get_tokenizer(tokenizer_name)

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
