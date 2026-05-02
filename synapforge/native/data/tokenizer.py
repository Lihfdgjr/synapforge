"""Native Qwen-2.5-compatible BPE tokenizer wrapper. Zero torch.

This module provides ``NativeTokenizer`` -- a thin compatibility shim
around three possible backends, picked in priority order at construction:

1. **HF ``transformers``** -- if the ``transformers`` package is
   importable, defer to ``AutoTokenizer.from_pretrained(name)``. Behaves
   identically to ``synapforge.data._get_tokenizer(name)``. This is the
   first-choice path because every cell of the legacy training stack
   has been validated against it.
2. **HF ``tokenizers``** -- the lower-level Rust-backed tokenizer
   (which ``transformers`` itself uses under the hood). Loads a
   ``tokenizer.json`` produced by the HF spec via
   ``tokenizers.Tokenizer.from_file``. Smaller dependency footprint than
   ``transformers``.
3. **Pure-Python JSON fallback** -- when neither package is installed,
   load ``tokenizer.json`` ourselves, parse the byte-level BPE model
   ``{vocab, merges}`` block, and run a minimal Qwen-style byte-level
   BPE (regex pre-tokenize -> bytes-to-unicode -> rank-merge -> id
   lookup). Slower than the Rust implementations but produces the same
   token IDs for any ASCII / standard UTF-8 input the trainer sees.

The public surface is intentionally tiny -- the trainer only ever needs
``encode(text) -> list[int]`` and ``eos_token_id``. We do NOT implement
``decode``, special-token logic, padding, truncation, etc.; the legacy
``ParquetTokenStream`` doesn't use them either, so re-implementing them
just to throw away test coverage would be net-negative.

NO ``import torch``. NO ``import sentencepiece`` (Qwen-2.5 ships a BPE
``tokenizer.json``, not an SP model -- using SP would silently mis-tokenise
half the corpus).
"""

from __future__ import annotations

import functools
import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# byte-level BPE -- shared helpers (used by JSON fallback)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _bytes_to_unicode() -> "dict[int, str]":
    """GPT-2 / Qwen byte-to-unicode permutation.

    The HF byte-level BPE pre-tokenizer remaps the 256 byte values onto a
    set of 256 *printable* unicode codepoints. The mapping is the same one
    GPT-2 invented (also used by Qwen-2.5, RoBERTa, etc.).

    Returns: dict mapping byte (0..255) -> single-char string.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: tuple) -> "set[tuple[str, str]]":
    """Return the set of adjacent symbol pairs in a tuple of strings."""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


# Qwen-2.5 / GPT-2 pre-tokenization regex (matches the HF spec verbatim).
# Falls back to ``re`` if ``regex`` is unavailable; the tradeoff is that
# ``re`` doesn't support \p{L}/\p{N}, so we approximate with [^\W_]/\d.
try:
    import regex as _re  # type: ignore[import-not-found]

    _PRETOK = _re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
        r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
        r"""|\p{N}"""
        r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
        r"""|\s*[\r\n]+"""
        r"""|\s+(?!\S)|\s+"""
    )
    _HAVE_REGEX = True
except ImportError:
    # Approximation: \p{L} -> [^\W\d_], \p{N} -> \d. Good enough for ASCII;
    # multibyte unicode tokens may split slightly differently. If the
    # caller cares, install ``regex``.
    _PRETOK = re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
        r"""|[^\r\n\w]?[^\W\d_]+"""
        r"""|\d"""
        r"""| ?[^\s\w]+[\r\n]*"""
        r"""|\s*[\r\n]+"""
        r"""|\s+(?!\S)|\s+"""
    )
    _HAVE_REGEX = False


# ---------------------------------------------------------------------------
# JSON-fallback BPE (pure-python)
# ---------------------------------------------------------------------------


class _PyBPE:
    """Minimal byte-level BPE that loads a HF tokenizer.json directly.

    The ``model`` block of a tokenizer.json with type ``"BPE"`` carries
    everything we need:

      * ``vocab``: dict[str, int] mapping merged-token-string -> id
      * ``merges``: list[str] of "a b" merge rules in priority order
      * ``byte_fallback``: when True, an unmapped byte gets the literal
        ``<0xXX>`` token. We honour this for Qwen.
      * ``unk_token``: id used for unknown sequences (rarely hit on
        byte-level BPE, but spec says we must wire it).
    """

    def __init__(self, model_block: dict, added_tokens: "list[dict]") -> None:
        if model_block.get("type") not in ("BPE", None):
            # Spec says ``type`` may be omitted iff vocab+merges present.
            raise ValueError(
                f"unsupported tokenizer.json model type: {model_block.get('type')!r}"
            )
        self.vocab: dict[str, int] = dict(model_block.get("vocab", {}))
        merges_raw = model_block.get("merges", [])
        # Two formats per HF spec: list[str "a b"] OR list[[a, b]].
        merges: list[tuple[str, str]] = []
        for m in merges_raw:
            if isinstance(m, str):
                # ``"abc def"`` -- exactly one space; defend against
                # multi-space tokens (rare but spec-permitted).
                a, b = m.split(" ", 1)
                merges.append((a, b))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                merges.append((str(m[0]), str(m[1])))
            else:
                raise ValueError(f"unrecognised merge entry: {m!r}")
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        # Honour byte_fallback; Qwen-2.5 uses it.
        self.byte_fallback: bool = bool(model_block.get("byte_fallback", False))
        # Cache: token-string -> id (with byte-fallback path).
        self._byte_tokens = {
            i: f"<0x{i:02X}>" for i in range(256)
        }
        # Index of added/special tokens by surface form.
        self.added_tokens: dict[str, int] = {}
        for t in added_tokens:
            self.added_tokens[t["content"]] = int(t["id"])
        # Reverse: id -> str (only needed for diagnostics; we keep it tiny).
        self.id_to_token: dict[int, str] = {i: s for s, i in self.vocab.items()}
        for s, i in self.added_tokens.items():
            self.id_to_token[i] = s
        # Byte mapper.
        self._b2u = _bytes_to_unicode()
        self._unk_token = model_block.get("unk_token") or ""
        # Cache of bpe-merged token strings to amortise ``encode``.
        self._bpe_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ BPE

    def _bpe(self, token: str) -> str:
        """Run merge rules on a single pre-token. Return space-joined symbols."""
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        if len(word) <= 1:
            self._bpe_cache[token] = token
            return token
        pairs = _get_pairs(word)
        while True:
            # Pick the lowest-rank pair (ties broken by first occurrence).
            best = None
            best_rank = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best = p
            if best is None:
                break
            a, b = best
            new_word: list[str] = []
            i = 0
            while i < len(word):
                # Find next occurrence of ``a`` from i forward.
                try:
                    j = word.index(a, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self._bpe_cache[token] = out
        return out

    # ------------------------------------------------------------------ public

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` to a list of integer ids. No special tokens."""
        if not text:
            return []
        ids: list[int] = []
        # Pre-tokenize with the GPT-2/Qwen regex.
        for piece in _PRETOK.findall(text):
            if not piece:
                continue
            # Bytes-to-unicode permutation on raw UTF-8 bytes.
            mapped = "".join(self._b2u[b] for b in piece.encode("utf-8"))
            # BPE-merge.
            merged = self._bpe(mapped).split(" ")
            for tok in merged:
                tid = self.vocab.get(tok)
                if tid is not None:
                    ids.append(tid)
                    continue
                # Vocab miss. Fallback options:
                #  1. byte_fallback=True -> emit ``<0xXX>`` per byte.
                #  2. else look up unk_token if registered.
                if self.byte_fallback:
                    for b in tok.encode("utf-8"):
                        bt = self._byte_tokens[b]
                        bt_id = self.vocab.get(bt) or self.added_tokens.get(bt)
                        if bt_id is None:
                            # Truly unknown; skip rather than crash.
                            continue
                        ids.append(bt_id)
                else:
                    unk_id = self.vocab.get(self._unk_token)
                    if unk_id is None:
                        unk_id = self.added_tokens.get(self._unk_token)
                    if unk_id is not None:
                        ids.append(unk_id)
        return ids


# ---------------------------------------------------------------------------
# unified front-door tokenizer
# ---------------------------------------------------------------------------


# Shared cache so successive ``NativeTokenizer(name)`` calls in one process
# don't reload + re-parse the 100 MB tokenizer.json. Per-name keyed because
# the trainer can flip between GPT-2 (vocab=50257) and Qwen (vocab=151643)
# in the same run (legacy ``synapforge.data._TOKENIZER_CACHE`` is also
# per-name for the same reason).
_TOK_CACHE: dict[str, "NativeTokenizer"] = {}


class NativeTokenizer:
    """Qwen-2.5-compatible BPE encoder with three backend tiers.

    Constructor arguments
    ---------------------
    name_or_path
        Either an HF model name (``"Qwen/Qwen2.5-1.5B"``) or a filesystem
        directory containing ``tokenizer.json``. The HF path is only
        used when the ``transformers`` or ``tokenizers`` packages are
        installed; otherwise pass a directory.
    backend
        One of ``"auto"`` (default), ``"transformers"``, ``"tokenizers"``,
        or ``"json"``. ``"auto"`` picks the first available in the order
        documented at the top of this file.

    Public API
    ----------
    ``encode(text) -> list[int]`` -- emit token ids without specials.
    ``eos_token_id`` -- end-of-text id (used by the streamer to separate
        documents). Mirrors HF's ``tokenizer.eos_token_id``.
    ``vocab_size`` -- documentation-only int. The actual vocab is what
        the underlying backend reports.
    """

    def __init__(
        self,
        name_or_path: str = "gpt2",
        backend: str = "auto",
    ) -> None:
        self.name = str(name_or_path)
        self.backend_used = "?"
        self._impl: Any = None
        self._eos: int = 0
        self._vocab: int = 0

        if backend not in ("auto", "transformers", "tokenizers", "json"):
            raise ValueError(
                f"backend must be auto/transformers/tokenizers/json; got {backend!r}"
            )

        # 1. transformers
        if backend in ("auto", "transformers"):
            try:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]

                tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                self._impl = tok
                self.backend_used = "transformers"
                self._eos = int(tok.eos_token_id) if tok.eos_token_id is not None else 50256
                self._vocab = int(tok.vocab_size)
                return
            except Exception:
                if backend == "transformers":
                    raise
                # fall through to tokenizers/json

        # 2. tokenizers (HF rust backend)
        if backend in ("auto", "tokenizers"):
            try:
                from tokenizers import Tokenizer  # type: ignore[import-not-found]

                # ``Tokenizer.from_pretrained`` does HTTP; restrict to
                # local-file path to keep tests offline.
                tj = self._resolve_tokenizer_json(name_or_path)
                tok = Tokenizer.from_file(tj)
                self._impl = tok
                self.backend_used = "tokenizers"
                # Best-effort eos; tokenizers exposes added_tokens but no
                # ``eos_token_id`` accessor -- read it from the JSON.
                with open(tj, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._eos = self._eos_from_json(raw)
                self._vocab = tok.get_vocab_size()
                return
            except Exception:
                if backend == "tokenizers":
                    raise
                # fall through

        # 3. json fallback
        tj = self._resolve_tokenizer_json(name_or_path)
        with open(tj, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._impl = _PyBPE(raw["model"], raw.get("added_tokens", []))
        self.backend_used = "json"
        self._eos = self._eos_from_json(raw)
        self._vocab = len(self._impl.vocab) + len(self._impl.added_tokens)

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tokenizer_json(name_or_path: str) -> str:
        """Locate a tokenizer.json given a model name or a directory path."""
        if os.path.isfile(name_or_path):
            return name_or_path
        if os.path.isdir(name_or_path):
            cand = os.path.join(name_or_path, "tokenizer.json")
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"NativeTokenizer JSON fallback needs a local tokenizer.json. "
            f"Pass a directory containing tokenizer.json or install "
            f"transformers/tokenizers. Tried: {name_or_path!r}"
        )

    @staticmethod
    def _eos_from_json(raw: dict) -> int:
        """Pull the EOS id out of a parsed tokenizer.json."""
        # 1. Look for an added token marked ``special=True`` whose surface
        #    is ``<|endoftext|>`` (Qwen) or similar. Priority order:
        #    dedicated EOS surface first, then im_end (Qwen chat), then
        #    classic </s>.
        added = raw.get("added_tokens", []) or []
        eos_candidates = ("<|endoftext|>", "<|im_end|>", "</s>", "<eos>")
        for surf in eos_candidates:
            for t in added:
                if t.get("content") == surf:
                    return int(t["id"])
        # 2. Look up in vocab.
        vocab = raw.get("model", {}).get("vocab", {}) or {}
        for surf in eos_candidates:
            if surf in vocab:
                return int(vocab[surf])
        # 3. Last resort: GPT-2 magic 50256.
        return 50256

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Tokenize text -> list of int ids. No special tokens added."""
        if not text:
            return []
        backend = self.backend_used
        if backend == "transformers":
            return list(self._impl.encode(text, add_special_tokens=False))
        if backend == "tokenizers":
            enc = self._impl.encode(text, add_special_tokens=False)
            return list(enc.ids)
        # json fallback
        return self._impl.encode(text)

    @property
    def eos_token_id(self) -> int:
        return int(self._eos)

    @property
    def vocab_size(self) -> int:
        return int(self._vocab)

    def __repr__(self) -> str:
        return (
            f"NativeTokenizer(name={self.name!r}, backend={self.backend_used!r}, "
            f"vocab_size={self._vocab}, eos={self._eos})"
        )


def get_tokenizer(name: str, backend: str = "auto") -> NativeTokenizer:
    """Cached front door -- mirrors ``synapforge.data._get_tokenizer``."""
    key = f"{name}::{backend}"
    cached = _TOK_CACHE.get(key)
    if cached is not None:
        return cached
    tok = NativeTokenizer(name, backend=backend)
    _TOK_CACHE[key] = tok
    return tok
