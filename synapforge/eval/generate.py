"""sf.eval.generate — prompt-completion utility shared by synapforge / mscfc.

Usage (CLI):
    python -m synapforge.eval.generate --backend sf --ckpt /workspace/runs/synapforge_100m/step_000300.pt \
        --prompt "The capital of France is" --max-new 64

Two backends are supported behind the same `generate(...)` API:
    * "sf"     -> synapforge.model_100m.SynapForge100M  (forward(ids) -> logits)
    * "mscfc"  -> mscfc.model.MSpikingCfCLoop           (forward(tokens=...) -> (logits, aux))

The sampling loop is identical: top-k -> top-p (nucleus) -> multinomial.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make /workspace importable so both `synapforge` and `mscfc` resolve.
_WS = "/workspace"
if _WS not in sys.path:
    sys.path.insert(0, _WS)

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ sampling


@torch.no_grad()
def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then top-p (nucleus) filtering. Operates on a 1D tensor."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(
            logits >= kth, logits, torch.full_like(logits, float("-inf"))
        )
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum > top_p
        # Always keep at least the highest-probability token.
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def _forward_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """Backend-aware single forward. Returns logits[B, T, V]."""
    # Try the synapforge signature first (positional `ids`).
    try:
        out = model(ids)
    except TypeError:
        out = model(tokens=ids)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new: int = 64,
    top_k: int = 40,
    top_p: float = 0.9,
    temperature: float = 0.7,
    device: str | None = None,
    max_seq: int | None = None,
) -> str:
    """Generate up to `max_new` tokens after `prompt`. Returns decoded string.

    `max_seq` (if given) caps the context length we feed back to the model — useful
    because both backbones have positional-embed limits (256 for sf-100m).
    """
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    out = list(ids)

    # Detect positional-embed limit on the synapforge model.
    if max_seq is None:
        max_seq = getattr(model, "max_seq", None)

    for _ in range(max_new):
        ctx = out
        if max_seq is not None and len(ctx) > max_seq:
            ctx = ctx[-max_seq:]
        inp = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits = _forward_logits(model, inp)
        last = logits[0, -1].float() / max(temperature, 1e-6)
        last = _filter_logits(last, top_k=top_k, top_p=top_p)
        probs = F.softmax(last, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            break
        tok = int(torch.multinomial(probs, 1).item())
        out.append(tok)
        if tok == eos:
            break
    return tokenizer.decode(out, skip_special_tokens=False)


# ------------------------------------------------------------------ loaders


def load_tokenizer(name: str = "gpt2"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    return tok


def load_synapforge(ckpt_path: str | None = None,
                    fresh_warmstart_from: str | None = None,
                    device: str = "cuda"):
    """Load a synapforge 100M model.

    * If `ckpt_path` is given, loads that state_dict on top of a fresh build.
    * If `fresh_warmstart_from` (an mscfc ckpt) is given, copies overlapping
      tensors by name (best-effort) so the model has a vaguely english-ish
      embedding+head rather than pure noise.
    * Otherwise returns a fresh (Xavier-init) model.
    """
    from synapforge.model_100m import build_synapforge_100m

    model = build_synapforge_100m()
    src = "fresh-init"

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        src = f"ckpt={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}"
    elif fresh_warmstart_from and os.path.exists(fresh_warmstart_from):
        ckpt = torch.load(fresh_warmstart_from, map_location="cpu", weights_only=False)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        if isinstance(state, dict) and state and all(k.startswith("wrapped.") for k in state.keys()):
            state = {k[len("wrapped."):]: v for k, v in state.items()}
        copied = 0
        own = model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                copied += 1
        model.load_state_dict(own, strict=False)
        src = f"warmstart={fresh_warmstart_from} copied={copied}/{len(state)}"

    model.to(device).eval()
    return model, src


def load_mscfc(ckpt_path: str, device: str = "cuda"):
    """Reuse evaluate.build_model_from_ckpt to load an adv29-style ckpt."""
    sys.path.insert(0, "/workspace")
    from evaluate import build_model_from_ckpt, load_checkpoint  # type: ignore

    ckpt = load_checkpoint(ckpt_path)
    model, cfg = build_model_from_ckpt(ckpt)
    model.to(device).eval()
    return model, cfg


# ------------------------------------------------------------------ CLI


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sf", "mscfc"], default="sf")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--warmstart-from", default=None,
                    help="(sf only) copy overlapping tensors from an mscfc ckpt.")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--device", default="cuda:1")
    return ap.parse_args()


def main():
    args = _parse()
    if args.backend == "sf":
        model, src = load_synapforge(
            args.ckpt, fresh_warmstart_from=args.warmstart_from, device=args.device,
        )
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt required for mscfc backend")
        model, src = load_mscfc(args.ckpt, device=args.device)
    tok = load_tokenizer(args.tokenizer)
    print(f"[gen] backend={args.backend} src={src}")
    t0 = time.time()
    out = generate(
        model, tok, args.prompt,
        max_new=args.max_new, top_k=args.top_k, top_p=args.top_p,
        temperature=args.temperature, device=args.device,
    )
    dt = time.time() - t0
    print(f"=== prompt: {args.prompt!r}")
    print(f"    completion: {out}")
    print(f"    wall: {dt:.2f}s")


if __name__ == "__main__":
    main()
