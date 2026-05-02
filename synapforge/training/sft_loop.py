"""synapforge.training.sft_loop -- shared SFT helpers.

Phase 2 SFT (T9.4): Switch from LM-only training to instruction-tune
SFT in order to break the val ppl plateau (~1000-2000 on raw LM data
per scaling laws) toward <= 10. The user 铁律 (memory:
``feedback_50m_context_monotonic_quality.md`` + the Phase autopilot in
DEEP_MAINT_QUEUE H5) requires SFT to be a separate trainer entry point
that reuses the production model + warmstart contract.

This module hosts the shared bits between ``train_100m_kd.py`` (LM)
and ``train_100m_sft.py`` (SFT) so we don't duplicate the entire
trainer:

* :class:`InstructionParquetStream` -- streaming reader for Alpaca-zh
  / Qwen-tokenized parquet files. Handles two on-disk schemas:
  (a) ``prompt_input_ids`` + ``response_input_ids`` columns (preferred);
  (b) single ``input_ids`` + ``response_mask`` column (compat).
  Yields ``(tokens_in [B, T], tokens_out [B, T], loss_mask [B, T])``
  with the loss mask zeroed on prompt tokens when
  ``response_only_loss=True``.

* :func:`response_only_ce_loss` -- masked next-token cross-entropy
  over only the positions where ``loss_mask == 1``. Equivalent to
  full-CE when ``loss_mask`` is all-ones, so the off-path
  (``--no-response-only-loss``) is bit-exact CE.

Both are pure CPU + pure-torch + pure-pyarrow so the unit tests run
on a stock dev laptop without GPU / transformers / network.

References
----------
* memory ``v2.6 response-only loss masking`` -- the 2026-04 task
  #214 that originally landed response-masking on the v2.6 trainer.
  This refactor lifts that recipe up to a reusable module and a
  first-class trainer entry point.
* Alpaca-zh format: each row encodes one instruction example with
  the prompt (instruction + optional input) tokenised separately
  from the response, so the loss mask is trivial.
"""
from __future__ import annotations

import glob
import os
import random
from collections.abc import Iterator
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover -- caught at constructor time
    pq = None


# ---------------------------------------------------------------------------
# Schema detection: alpaca-zh + qwen-tokenized parquet may use one of two
# layouts. We auto-detect at __init__ so callers don't have to specify.
# ---------------------------------------------------------------------------

_PROMPT_RESPONSE_COLS = ("prompt_input_ids", "response_input_ids")
_INPUT_MASK_COLS = ("input_ids", "response_mask")


def _detect_schema(parquet_path: str) -> str:
    """Return ``"prompt_response"`` or ``"input_mask"`` or raise.

    Looks at the first parquet's schema and decides which column layout
    we have. Both layouts encode the same information; the schema name
    here is a stable string the rest of the loop dispatches on.
    """
    if pq is None:
        raise RuntimeError("pyarrow is required for SFT data loading")
    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    if all(c in names for c in _PROMPT_RESPONSE_COLS):
        return "prompt_response"
    if all(c in names for c in _INPUT_MASK_COLS):
        return "input_mask"
    raise RuntimeError(
        f"parquet {parquet_path!r} has neither {_PROMPT_RESPONSE_COLS} "
        f"nor {_INPUT_MASK_COLS}; got {sorted(names)}"
    )


# ---------------------------------------------------------------------------
# InstructionParquetStream
# ---------------------------------------------------------------------------


class InstructionParquetStream:
    """Streaming iterator over (tokens_in, tokens_out, loss_mask) batches.

    Parameters
    ----------
    glob_pattern:
        Shell-style glob of parquet files. Each file must satisfy one of
        the two supported schemas (auto-detected from the FIRST matched
        file; all files in the glob MUST share the same schema -- the
        constructor validates this so a typo in --data-glob can't
        silently mix layouts).
    seq_len:
        Max tokens per training example T. Examples longer than
        ``seq_len + 1`` are right-truncated; shorter ones are padded with
        the tokenizer's pad/eos id. The exact pad id is governed by
        ``pad_id`` (default = ``eos_id``, which itself defaults to 0 if
        the tokenizer doesn't ship one in the parquet metadata).
    batch_size:
        Examples per batch B.
    response_only_loss:
        When True (default), the emitted ``loss_mask`` is 1 only on
        positions whose label is a response token, 0 on prompt tokens
        and pad. This is the SFT recipe (Stanford Alpaca / DeepSeek v2
        / Qwen-Chat). When False, mask is 1 on all non-pad positions
        (full instruction LM) -- the ablation path.
    pad_id, eos_id:
        Pad / EOS token ids used when materialising fixed-length
        tensors. Default -1 is replaced at __init__ time by either
        the parquet metadata or fallback values from the tokenizer.
    loop:
        If True, loop over the parquet files forever (training stream).
        If False, raise StopIteration at end (val stream).
    shuffle_buffer:
        Reservoir size for streaming Fisher-Yates shuffle. <=1 yields
        deterministic order (val). Default 10000 is the same recipe as
        :class:`synapforge.data.ParquetTokenStream` (P24 in
        ``MASTER_PLAN.md``).
    shuffle_seed:
        Deterministic seed for the reservoir RNG.
    """

    def __init__(
        self,
        glob_pattern: str,
        *,
        seq_len: int = 512,
        batch_size: int = 16,
        response_only_loss: bool = True,
        pad_id: int = -1,
        eos_id: int = -1,
        loop: bool = True,
        shuffle_buffer: int = 10000,
        shuffle_seed: int = 42,
    ) -> None:
        if pq is None:
            raise RuntimeError("pyarrow is required for SFT data loading")
        self.files = sorted(glob.glob(glob_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"glob {glob_pattern!r} matched no parquets"
            )
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.response_only_loss = bool(response_only_loss)
        self.loop = bool(loop)
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.shuffle_seed = int(shuffle_seed)

        self._schema = _detect_schema(self.files[0])
        # Cross-check: every file must share the schema so we never
        # silently mix prompt_response with input_mask layouts mid-epoch.
        for f in self.files[1:]:
            other = _detect_schema(f)
            if other != self._schema:
                raise RuntimeError(
                    f"parquet schema mismatch: {self.files[0]!r} is "
                    f"{self._schema!r} but {f!r} is {other!r}; SFT loop "
                    f"requires uniform schema across the data glob"
                )

        # Default pad/eos ids: when the parquet has no metadata, fall
        # back to common values. Qwen tokenizer's eos = 151643;
        # GPT-2's = 50256. We can't know without the tokenizer here, so
        # we accept whatever the caller passes. The CLI in
        # ``train_100m_sft.py`` derives them from the loaded tokenizer.
        self.pad_id = int(pad_id) if pad_id >= 0 else 0
        self.eos_id = int(eos_id) if eos_id >= 0 else self.pad_id

    # ------------------------------------------------------------------
    # Internal: yield one (input_ids, response_mask) pair per row.
    # ------------------------------------------------------------------

    def _iter_rows_raw(self) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(input_ids, response_mask)`` lists from parquet files.

        The mask is 1 on response tokens, 0 on prompt tokens. We keep
        the rows AT ROW GRANULARITY (no merging across docs) because
        instruction examples are inherently per-row -- merging would
        bleed prompt mass into the next response and corrupt SFT.
        """
        epoch = 0
        while True:
            files = list(self.files)
            if self.shuffle_buffer > 1:
                rng = random.Random(self.shuffle_seed + epoch)
                rng.shuffle(files)
            for path in files:
                pf = pq.ParquetFile(path)
                cols = (
                    list(_PROMPT_RESPONSE_COLS)
                    if self._schema == "prompt_response"
                    else list(_INPUT_MASK_COLS)
                )
                for batch in pf.iter_batches(batch_size=64, columns=cols):
                    if self._schema == "prompt_response":
                        prompts = batch.column("prompt_input_ids").to_pylist()
                        responses = batch.column("response_input_ids").to_pylist()
                        for p_ids, r_ids in zip(prompts, responses):
                            if not p_ids or not r_ids:
                                continue
                            input_ids = list(p_ids) + list(r_ids)
                            mask = [0] * len(p_ids) + [1] * len(r_ids)
                            yield input_ids, mask
                    else:  # input_mask schema
                        input_ids_col = batch.column("input_ids").to_pylist()
                        mask_col = batch.column("response_mask").to_pylist()
                        for ids, m in zip(input_ids_col, mask_col):
                            if not ids:
                                continue
                            yield list(ids), list(m)
            epoch += 1
            if not self.loop:
                return

    def _iter_rows(self) -> Iterator[tuple[list[int], list[int]]]:
        """Same as ``_iter_rows_raw`` but with optional reservoir shuffle.

        Identical algorithm to :class:`synapforge.data.ParquetTokenStream`
        but applied at the (input_ids, mask) tuple level.
        """
        raw = self._iter_rows_raw()
        if self.shuffle_buffer <= 1:
            yield from raw
            return
        rng = random.Random(self.shuffle_seed)
        K = self.shuffle_buffer
        buffer: list[tuple[list[int], list[int]]] = []
        for row in raw:
            if len(buffer) < K:
                buffer.append(row)
                continue
            idx = rng.randrange(K)
            yield buffer[idx]
            buffer[idx] = row
        rng.shuffle(buffer)
        yield from buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _pad_row(
        self, ids: list[int], mask: list[int]
    ) -> tuple[list[int], list[int]]:
        """Right-truncate or right-pad to ``seq_len + 1`` tokens.

        We need ``seq_len + 1`` ids so that the (in, out) shift gives
        ``seq_len`` positions in each. Mask follows the same schedule;
        mask positions corresponding to padding land at 0 (no loss).
        """
        T = self.seq_len + 1
        if len(ids) >= T:
            return ids[:T], mask[:T]
        # Pad. The loss mask is 0 on pads regardless of
        # response_only_loss because pads are not real tokens.
        pad = T - len(ids)
        ids = list(ids) + [self.pad_id] * pad
        mask = list(mask) + [0] * pad
        return ids, mask

    def _build_batch(
        self,
        rows: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise a batch of fixed-length tensors.

        Returns ``(tokens_in [B, T], tokens_out [B, T], loss_mask [B, T])``
        where T == seq_len. The loss mask is built from each row's
        response mask SHIFTED LEFT BY 1 (so it aligns with
        ``tokens_out``: the position whose target is a response token
        is the position WHOSE LABEL is in the response).

        When ``self.response_only_loss == False`` we override the mask
        to be 1 on all non-pad output positions, recovering full-CE
        (the ablation path documented in the trainer's --no-response-
        only-loss flag).
        """
        ids_padded: list[list[int]] = []
        masks_padded: list[list[int]] = []
        for ids, mask in rows:
            ip, mp = self._pad_row(ids, mask)
            ids_padded.append(ip)
            masks_padded.append(mp)
        arr = torch.tensor(ids_padded, dtype=torch.long)             # (B, T+1)
        m_arr = torch.tensor(masks_padded, dtype=torch.long)         # (B, T+1)
        tokens_in = arr[:, :-1].contiguous()                         # (B, T)
        tokens_out = arr[:, 1:].contiguous()                         # (B, T)

        if self.response_only_loss:
            # Mask for tokens_out (next-token target): position j has
            # mask 1 iff the LABEL (arr[:, j+1]) is a response token,
            # which is exactly m_arr[:, 1:] -- the original mask shifted
            # left by 1 to align with tokens_out.
            loss_mask = m_arr[:, 1:].contiguous()
        else:
            # Ablation: full instruction LM. Mask is 1 everywhere
            # except on pad positions in tokens_out (so we never
            # train the model to predict pad given pad).
            loss_mask = (tokens_out != self.pad_id).long()
        # Mask must be float for the multiply in response_only_ce_loss.
        return tokens_in, tokens_out, loss_mask.float()

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rows: list[tuple[list[int], list[int]]] = []
        for r in self._iter_rows():
            rows.append(r)
            if len(rows) >= self.batch_size:
                yield self._build_batch(rows)
                rows = []

    def __repr__(self) -> str:
        return (
            f"InstructionParquetStream(files={len(self.files)}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"response_only_loss={self.response_only_loss}, "
            f"schema={self._schema!r}, "
            f"shuffle_buffer={self.shuffle_buffer})"
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def response_only_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Mean cross-entropy over positions whose ``loss_mask`` is non-zero.

    Parameters
    ----------
    logits:
        ``(B, T, V)`` student logits in fp32 (caller upcasts before
        passing in -- we don't do it here so the autocast boundary
        stays explicit at the trainer level).
    labels:
        ``(B, T)`` int64 target token ids (the next-token labels =
        ``tokens_out`` from :class:`InstructionParquetStream`).
    loss_mask:
        ``(B, T)`` float32 mask. 1 on positions to count, 0 elsewhere.
        Pad positions and (when response-only) prompt positions arrive
        here as 0.
    label_smoothing:
        Optional CE label smoothing alpha; passed through to
        :func:`torch.nn.functional.cross_entropy`. Default 0 = no
        smoothing.
    eps:
        Floor for the denominator (number of unmasked positions). Stops
        the corner case where a degenerate batch has zero unmasked
        positions from blowing up to NaN -- in that case the loss is 0
        and the trainer step is effectively a no-op.

    Returns
    -------
    Scalar tensor: ``sum(per_token_ce * mask) / max(sum(mask), eps)``.

    Notes
    -----
    When ``loss_mask`` is all-ones this is bit-exact equivalent to
    ``F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
    label_smoothing=label_smoothing)`` -- the test
    ``test_response_only_loss_masks_prompt`` asserts that property.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be (B, T, V); got shape {tuple(logits.shape)}"
        )
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(flat_logits.dtype)
    # Per-token CE without reduction so we can mask and renormalise.
    per_tok = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    masked = per_tok * flat_mask
    denom = flat_mask.sum().clamp(min=eps)
    return masked.sum() / denom


# ---------------------------------------------------------------------------
# Tiny synth parquet writer for tests + smoke runs
# ---------------------------------------------------------------------------


def write_synth_alpaca_parquet(
    out_path: str,
    *,
    n_rows: int = 32,
    prompt_len: int = 8,
    response_len: int = 12,
    vocab: int = 100,
    schema: str = "prompt_response",
    seed: int = 42,
) -> str:
    """Write a tiny alpaca-zh-shaped parquet for unit tests.

    Returns the path written. The default schema is
    ``"prompt_response"`` (two list-of-int columns); pass
    ``"input_mask"`` to write the alternate layout.
    """
    if pq is None:
        raise RuntimeError("pyarrow is required to write synth parquets")
    import pyarrow as pa

    rng = random.Random(seed)
    if schema == "prompt_response":
        prompts = [
            [rng.randrange(1, vocab) for _ in range(prompt_len)]
            for _ in range(n_rows)
        ]
        responses = [
            [rng.randrange(1, vocab) for _ in range(response_len)]
            for _ in range(n_rows)
        ]
        table = pa.table(
            {
                "prompt_input_ids": prompts,
                "response_input_ids": responses,
            }
        )
    elif schema == "input_mask":
        ids_col = []
        mask_col = []
        for _ in range(n_rows):
            p = [rng.randrange(1, vocab) for _ in range(prompt_len)]
            r = [rng.randrange(1, vocab) for _ in range(response_len)]
            ids_col.append(p + r)
            mask_col.append([0] * prompt_len + [1] * response_len)
        table = pa.table(
            {"input_ids": ids_col, "response_mask": mask_col}
        )
    else:
        raise ValueError(
            f"schema must be 'prompt_response' or 'input_mask', "
            f"got {schema!r}"
        )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pq.write_table(table, out_path)
    return out_path


__all__ = [
    "InstructionParquetStream",
    "response_only_ce_loss",
    "write_synth_alpaca_parquet",
]
