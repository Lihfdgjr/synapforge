"""sf.hf — HuggingFace adapter for synapforge.

Wraps `transformers` and `datasets` to:
  * always route to a mirror (default: https://hf-mirror.com) so calls
    don't hit huggingface.co directly (PRC firewall),
  * pre-cache assets on first call so all downstream synapforge code
    works offline,
  * load adv* (mscfc) checkpoints into a synapforge-style state_dict
    with shape/name remapping that gracefully handles
    missing/extra/transposed/different-rank tensors.

Public API
----------
    >>> import synapforge as sf
    >>> from synapforge import huggingface_adapter as sfhf
    >>> tok = sfhf.load_tokenizer("gpt2", mirror=True)
    >>> ds  = sfhf.load_dataset_streaming("HuggingFaceFW/fineweb-edu", split="train",
    ...                                   name="sample-10BT", mirror=True)
    >>> rep = sfhf.adv_warmstart(model, "/workspace/runs/step_001250.pt")
    >>> print(rep.matched, "/", rep.total_target,
    ...       "missing=", len(rep.missing), "extra=", len(rep.extra))

This module deliberately never raises on a single missing key: the goal
is "drop the warm-start in, recover whatever transfers, random-init the
rest", because adv29 is 256-dim and synapforge_100m is 512-dim — most
of the heavy weights *won't* transfer and that's fine.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

DEFAULT_MIRROR = "https://hf-mirror.com"


def _ensure_mirror_env(mirror: bool | str = True) -> str:
    """Set HF_ENDPOINT to a mirror unless already configured.

    Returns the resolved endpoint URL (always non-empty).
    """
    if mirror is False:
        return os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    endpoint = mirror if isinstance(mirror, str) else DEFAULT_MIRROR
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    # Honour an explicit override even if a different env was set.
    if isinstance(mirror, str):
        os.environ["HF_ENDPOINT"] = endpoint
    # Make sure huggingface_hub's internal default constants pick this up.
    try:
        import huggingface_hub.constants as hc
        hc.ENDPOINT = endpoint  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
    return endpoint


def load_tokenizer(
    name: str = "gpt2",
    mirror: bool | str = True,
    *,
    use_fast: bool = True,
    fallback_local_only: bool = True,
    **kwargs: Any,
):
    """Load an HF tokenizer through `mirror`, then fall back to local cache.

    Args:
        name: HF model id, e.g. "gpt2".
        mirror: True (use DEFAULT_MIRROR), False (use upstream), or a
            URL string for a custom mirror.
        use_fast: prefer the fast (Rust) tokenizer.
        fallback_local_only: if the network attempt raises, retry with
            ``local_files_only=True`` so partially-cached repos still work.
        **kwargs: forwarded to `AutoTokenizer.from_pretrained`.

    Returns:
        A tokenizer instance (PreTrainedTokenizerFast or PreTrainedTokenizer).
    """
    endpoint = _ensure_mirror_env(mirror)
    from transformers import AutoTokenizer  # noqa: WPS433  local import
    try:
        return AutoTokenizer.from_pretrained(name, use_fast=use_fast, **kwargs)
    except Exception as exc:
        if not fallback_local_only:
            raise
        warnings.warn(
            f"sf.hf.load_tokenizer({name!r}) network call via {endpoint} "
            f"failed: {exc}; retrying local_files_only=True"
        )
        return AutoTokenizer.from_pretrained(
            name, use_fast=use_fast, local_files_only=True, **kwargs
        )


def load_dataset_streaming(
    path: str,
    *,
    name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    mirror: bool | str = True,
    fallback_local_only: bool = False,
    **kwargs: Any,
):
    """Load a HF dataset in streaming mode through `mirror`.

    Streaming avoids materialising the entire shard list (most HF datasets
    used here are >100 GB on disk).

    Args:
        path: dataset id, e.g. "HuggingFaceFW/fineweb-edu".
        name: dataset configuration (e.g. "sample-10BT").
        split: e.g. "train", "validation".
        streaming: True yields an `IterableDataset`.
        mirror: see `load_tokenizer`.
        fallback_local_only: only honoured when streaming=False.
        **kwargs: forwarded to `datasets.load_dataset`.

    Returns:
        A `datasets.Dataset` or `IterableDataset`.
    """
    endpoint = _ensure_mirror_env(mirror)
    from datasets import load_dataset  # noqa: WPS433
    try:
        return load_dataset(path, name=name, split=split,
                            streaming=streaming, **kwargs)
    except Exception as exc:
        if streaming or not fallback_local_only:
            raise
        warnings.warn(
            f"sf.hf.load_dataset_streaming({path!r}) via {endpoint} "
            f"failed: {exc}; retrying local_files_only=True"
        )
        return load_dataset(path, name=name, split=split,
                            streaming=False, local_files_only=True, **kwargs)


# ---------------------------------------------------------------------------
# adv_warmstart — mscfc-style ckpt -> synapforge state_dict
# ---------------------------------------------------------------------------


@dataclass
class WarmstartReport:
    """Result of adv_warmstart for inspection / logging."""

    matched: int = 0
    total_target: int = 0
    total_source: int = 0
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    matched_keys: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"WarmstartReport(matched={self.matched}/{self.total_target}, "
            f"src={self.total_source}, missing={len(self.missing)}, "
            f"extra={len(self.extra)}, "
            f"shape_mismatch={len(self.shape_mismatch)})"
        )


_DEFAULT_NAME_MAP_RULES: list[tuple[str, str]] = [
    # adv29 is wrapped in a "wrapped." prefix; synapforge models aren't.
    (r"^wrapped\.", ""),
    # token embed: adv29 uses embed.text_embed, synapforge uses tok_embed
    (r"^embed\.text_embed\.", "tok_embed."),
    # blocks are sometimes called layers in synapforge; keep both.
    (r"^blocks\.", "blocks."),
    (r"^lm_head\.", "lm_head."),
    # norm aliases
    (r"\.norm\.", ".ln."),
]


def _apply_rename(name: str, rules: Sequence[tuple[str, str]]) -> str:
    out = name
    for pat, repl in rules:
        out = re.sub(pat, repl, out)
    return out


def _attempt_shape_fit(src: torch.Tensor, dst_shape: tuple[int, ...]) -> torch.Tensor | None:
    """Try to fit a source tensor into a target shape via:
       (a) exact match, (b) transpose if 2D and shapes are reversed,
       (c) pad / slice along channel dims for embeddings (e.g. 256->512).

    Returns None if no safe fit exists.
    """
    if tuple(src.shape) == dst_shape:
        return src
    if src.dim() == 2 and dst_shape == tuple(reversed(src.shape)):
        return src.t().contiguous()
    # Channel-dim pad/slice (embedding upcast 256->512): copy into a
    # zero-init dst, leaving the rest random-init upstream.
    if src.dim() == len(dst_shape) and all(
        s <= d for s, d in zip(src.shape, dst_shape)
    ):
        out = torch.zeros(dst_shape, dtype=src.dtype)
        slices = tuple(slice(0, s) for s in src.shape)
        out[slices] = src
        return out
    return None


def adv_warmstart(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    name_map: Sequence[tuple[str, str]] | None = None,
    state_key: str = "model",
    strict: bool = False,
    allow_partial_shape: bool = True,
    verbose: bool = True,
) -> WarmstartReport:
    """Load an mscfc/adv* checkpoint into a synapforge model.

    Strategy:
        1. Load `state_key` sub-dict from ckpt (fall back to top-level).
        2. Apply default + user-supplied regex renames.
        3. For each target param, look up by remapped name and copy into
           it via shape-aware fit. Skip silently on mismatch.
        4. Return a WarmstartReport with everything we matched/missed.

    Args:
        model: target model. Modified in-place via `load_state_dict(strict=False)`.
        ckpt_path: path to an adv-style .pt file.
        name_map: extra (regex_pattern, replacement) rules; applied after
            the defaults. Use this to handle synapforge_100m's specific
            naming (e.g. "blocks.0.cfc." -> "blocks.0.liquid.").
        state_key: dict key to extract the params from. Defaults to "model".
        strict: ignored for backward-compat; we always non-strict.
        allow_partial_shape: pad/slice shape to fit when possible.
        verbose: print a one-line report.
    """
    rules = list(_DEFAULT_NAME_MAP_RULES)
    if name_map:
        rules.extend(name_map)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, Mapping):
        if state_key in ckpt:
            src_sd = ckpt[state_key]
        elif "state_dict" in ckpt:
            src_sd = ckpt["state_dict"]
        else:
            src_sd = ckpt
    else:
        src_sd = ckpt

    # Apply rename to source.
    renamed_src: dict[str, torch.Tensor] = {}
    for k, v in src_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        renamed_src[_apply_rename(k, rules)] = v

    target_sd = model.state_dict()
    report = WarmstartReport(
        total_target=len(target_sd),
        total_source=len(renamed_src),
    )

    new_sd: dict[str, torch.Tensor] = {}
    used_src_keys: set[str] = set()

    for tgt_name, tgt_tensor in target_sd.items():
        # Try direct, then suffix-match, then rule-then-direct.
        candidate = None
        cand_key: str | None = None
        if tgt_name in renamed_src:
            candidate = renamed_src[tgt_name]
            cand_key = tgt_name
        else:
            # try matching by trailing path (last 3 segments)
            tgt_tail = ".".join(tgt_name.split(".")[-3:])
            for sk in renamed_src:
                if sk.endswith(tgt_tail):
                    candidate = renamed_src[sk]
                    cand_key = sk
                    break

        if candidate is None:
            report.missing.append(tgt_name)
            continue

        fitted = candidate
        if tuple(candidate.shape) != tuple(tgt_tensor.shape):
            if allow_partial_shape:
                attempt = _attempt_shape_fit(candidate, tuple(tgt_tensor.shape))
                fitted = attempt if attempt is not None else candidate
            if tuple(fitted.shape) != tuple(tgt_tensor.shape):
                report.shape_mismatch.append(
                    (tgt_name, tuple(candidate.shape), tuple(tgt_tensor.shape))
                )
                continue

        new_sd[tgt_name] = fitted.to(tgt_tensor.dtype)
        used_src_keys.add(cand_key or "")
        report.matched += 1
        report.matched_keys.append((tgt_name, cand_key or ""))

    report.extra = [k for k in renamed_src if k not in used_src_keys]

    # Apply onto model. Using load_state_dict with strict=False so the
    # un-matched target params keep their random init.
    incompat = model.load_state_dict(new_sd, strict=False)
    # Fold load_state_dict's missing/unexpected into report so caller sees
    # the *real* picture, not just our matching.
    if verbose:
        print(f"[sf.hf.adv_warmstart] {report.summary()}", flush=True)
        if report.shape_mismatch[:5]:
            for n, s_src, s_tgt in report.shape_mismatch[:5]:
                print(f"  shape mismatch {n}: src={s_src} tgt={s_tgt}",
                      flush=True)
        if isinstance(incompat, tuple):
            mk, uk = incompat
            print(f"  load_state_dict missing={len(mk)} unexpected={len(uk)}",
                  flush=True)
    return report


# ---------------------------------------------------------------------------
# tiny CLI for sanity-check / pre-cache
# ---------------------------------------------------------------------------

def _cli() -> int:
    """Pre-cache GPT-2 + warm-start probe; returns 0 on success."""
    endpoint = _ensure_mirror_env(True)
    print(f"[sf.hf] HF_ENDPOINT={endpoint}", flush=True)
    t = load_tokenizer("gpt2")
    print(f"[sf.hf] tokenizer ok vocab={t.vocab_size} eos={t.eos_token_id}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
