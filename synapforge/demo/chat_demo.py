"""Chat demo: 5 EN + 5 ZH prompts against SynapForge100M (Qwen vocab).

If a v24h-style ckpt is available, load it and generate live; otherwise
replay a recorded transcript captured during v4.1 training (ppl ~44, 100M
LNN+SNN). The recorded outputs are coherent but limited — that's honest
for a 100M model. Runs on CPU.

Usage:
    synapforge-demo chat
    synapforge-demo chat --ckpt /path/to/v24h_chat.pt --tokenizer-path Qwen/Qwen2.5-0.5B
    python -m synapforge.demo.chat_demo --save chat_demo.json
    # T1.1 deep-maintenance probe (queue prompts, JSON with lang):
    python -m synapforge.demo.chat_demo \
        --ckpt /workspace/runs/v24h_qwen3/step_004000.pt \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --max-new 60 --temperature 0.7 \
        --save /tmp/chat_HHMM.json
    # T8.4 — load EMA-smoothed weights at inference (uses sibling
    # step_<N>_ema.pt next to step_<N>.pt, or `ema_state` key inside
    # step_<N>.pt — whichever exists):
    python -m synapforge.demo.chat_demo \
        --ckpt /workspace/runs/v24h_qwen3/step_004000.pt \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --use-ema --verbose --save /tmp/chat_HHMM_ema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------- Prompt sets ----------
#
# T1 prompts (docs/DEEP_MAINT_QUEUE.md T1.1) — used by the live deep-maintenance
# chat sample CLI. Short open-ended completions, designed for the
# `tokenizer.encode(prompt, ...)` -> raw continuation pattern (NOT the
# instruction template). Do not change these without updating the queue
# validation gates ([一-鿿]+ regex, len > 5 tokens).
EN_PROMPTS_T1 = [
    "The capital of France is",
    "In the morning, I like to",
    "Photosynthesis is the process",
    "def reverse_string(s):",
    "Once upon a time,",
]

ZH_PROMPTS_T1 = [
    "中国的首都是",
    "今天天气",
    "光合作用是",
    "我喜欢吃",
    "从前有一个",
]

# Legacy prompts (chat_recorded.json) — kept for the recorded-replay
# fallback (docs/INSURANCE_NATIVE.md Option B). The recorded transcript
# was captured against these and has zero overlap with EN_PROMPTS_T1, so
# replaying with T1 prompts would produce all "<no recorded response>".
EN_PROMPTS_LEGACY = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "Q: Who wrote Romeo and Juliet?\nA:",
    "Translate to Chinese: Hello",
]

ZH_PROMPTS_LEGACY = [
    "你好,",
    "中国的首都是",
    "请解释什么是机器学习。",
    "1+1等于几?",
    "今天天气真好,",
]

# Back-compat module-level aliases (used by older callers / tests).
EN_PROMPTS = EN_PROMPTS_LEGACY
ZH_PROMPTS = ZH_PROMPTS_LEGACY

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"


def _default_ckpt() -> str:
    return str(Path.home() / ".synapforge" / "v24h_chat.pt")


def _default_recorded() -> Path:
    return Path(__file__).resolve().parent / "chat_recorded.json"


def _resolve_device(device: str | None) -> str:
    """auto -> 'cuda' if available else 'cpu'. Anything else passes through."""
    if device is None or device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _strip_module_prefix(sd: dict) -> dict:
    """Strip leading 'module.' from DDP-saved state_dicts."""
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()}
    return sd


# Fallback architecture values used when the ckpt has no "config" dict
# (legacy ckpts written before P12 / 2026-05-01). Update these together
# with `train_100m_kd.py::MODEL_*` constants if the canonical architecture
# ever changes — but the right path is always to write a new ckpt that
# carries its own config dict.
_FALLBACK_CFG = {
    "vocab": 151936, "d": 512, "n_layers": 10, "loop_depth": 1,
    "max_seq": 2048, "ffn_ratio": 8.0, "sparsity": 0.95,
    "dropout": 0.0, "tie_lm_head": True,
}


def _resolve_ema_source(ckpt: str, raw_ckpt: dict) -> tuple[str, dict | None]:
    """Locate EMA weights for ``--use-ema`` (T8.4). Returns ``(source, sd)``.

    Order of resolution:
      1. ``raw_ckpt["ema_state"]`` — EMA state_dict embedded in the live ckpt
         (newer trainer behavior). Preferred because it's bit-exact paired
         with the live model checkpoint.
      2. Sibling file ``step_<N>_ema.pt`` next to ``ckpt``.
      3. Bare ``<ckpt>_ema.pt`` next to a non-step-prefixed ckpt name.

    Returns ``("none", None)`` when nothing was found so the caller can
    log a warning and fall through to the live weights.
    """
    if isinstance(raw_ckpt, dict):
        emb = raw_ckpt.get("ema_state")
        if isinstance(emb, dict) and emb:
            return ("embedded", emb)

    p = Path(ckpt)
    candidates: list[Path] = []
    # step_<N>.pt -> step_<N>_ema.pt
    if p.name.endswith(".pt"):
        candidates.append(p.with_name(p.stem + "_ema.pt"))
    candidates.append(p.with_name(p.stem + "_ema" + p.suffix))

    for cand in candidates:
        if cand.is_file():
            try:
                import torch
                ema_raw = torch.load(str(cand), map_location="cpu")
            except Exception:
                continue
            if isinstance(ema_raw, dict) and "model" in ema_raw:
                return (str(cand), ema_raw["model"])
    return ("none", None)


def _try_load_live(
    ckpt: str,
    tokenizer_path: str | None,
    device: str = "cpu",
    verbose: bool = False,
    use_ema: bool = False,
):
    """Best-effort load of model + tokenizer. Returns (model, tok, meta) or None.

    The ckpt is expected to carry a ``"config"`` dict written by the trainer
    (see ``train_100m_kd.py::_build_config_dict``). If absent (legacy ckpt)
    we fall back to ``_FALLBACK_CFG`` and emit a WARNING so the demo doesn't
    silently load garbage when architecture drifts. P12 — see
    docs/MASTER_PLAN.md §6.

    ``meta`` carries auxiliary info (step, ppl, ...) parsed from the ckpt
    dict so callers can stamp it into JSON output.

    ``use_ema`` (T8.4): after the live state_dict is loaded, optionally
    overlay EMA-smoothed weights from either ``raw_ckpt["ema_state"]``
    (embedded) or a sibling ``step_<N>_ema.pt`` file. Strict=False so
    legacy EMA dumps that miss a few keys still apply cleanly.
    """
    if not ckpt or not Path(ckpt).is_file():
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer
        from synapforge.model_100m import SynapForge100M
    except Exception:
        return None
    if not tokenizer_path:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception:
        return None
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    import torch
    # weights_only=False is the torch <2.6 default; setting it explicitly so
    # the same call path survives a future torch upgrade where the default
    # flips to True (which would refuse our optim_state pickled blobs).
    try:
        raw = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        # torch <2.0 doesn't accept weights_only.
        raw = torch.load(ckpt, map_location=device)
    if isinstance(raw, dict) and "config" in raw and isinstance(raw["config"], dict):
        cfg = dict(_FALLBACK_CFG)
        cfg.update(raw["config"])  # ckpt config wins
    else:
        cfg = dict(_FALLBACK_CFG)
        if verbose:
            print(
                f"[chat_demo] WARNING: ckpt has no config; using fallback values "
                f"d={cfg['d']} n_layers={cfg['n_layers']} loop_depth={cfg['loop_depth']} "
                f"ffn_ratio={cfg['ffn_ratio']} sparsity={cfg['sparsity']} "
                f"max_seq={cfg['max_seq']} vocab={cfg['vocab']} "
                f"tie_lm_head={cfg['tie_lm_head']} -- shape drift may cause garbage output"
            )

    model = SynapForge100M(
        vocab=int(cfg["vocab"]),
        d=int(cfg["d"]),
        n_layers=int(cfg["n_layers"]),
        loop_depth=int(cfg["loop_depth"]),
        max_seq=int(cfg["max_seq"]),
        ffn_ratio=float(cfg["ffn_ratio"]),
        sparsity=float(cfg["sparsity"]),
        dropout=float(cfg["dropout"]),
        tie_lm_head=bool(cfg["tie_lm_head"]),
    )
    sd = raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw
    sd = _strip_module_prefix(sd)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except Exception:
        return None
    if verbose and len(missing) + len(unexpected) > 5:
        print(
            f"[chat_demo] WARNING: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys -- ckpt may not match model "
            f"architecture (cfg from ckpt: {'yes' if 'config' in (raw if isinstance(raw, dict) else {}) else 'no'})"
        )
    try:
        model = model.to(device)
    except Exception:
        # Unknown device -- fall back to CPU silently rather than crash.
        model = model.to("cpu")

    # ---------------- T8.4 — optional EMA overlay ----------------
    # When --use-ema is set, replace the live weights with the EMA-smoothed
    # copy. The trainer writes EMA state to two places (either is fine):
    #   * ``raw["ema_state"]`` inside step_<N>.pt (embedded form)
    #   * sibling file step_<N>_ema.pt with format from EMATracker.save()
    # If neither is present we log and fall through to the live weights so
    # the demo still works on pre-T8.4 ckpts.
    ema_source = "none"
    ema_missing = 0
    ema_unexpected = 0
    if use_ema:
        ema_source, ema_sd = _resolve_ema_source(
            ckpt, raw if isinstance(raw, dict) else {}
        )
        if ema_sd is None:
            if verbose:
                print(
                    "[chat_demo] --use-ema: no EMA weights found "
                    "(no 'ema_state' key in ckpt + no sibling _ema.pt); "
                    "falling back to live weights."
                )
        else:
            ema_sd = _strip_module_prefix(ema_sd)
            try:
                ema_missing_keys, ema_unexpected_keys = model.load_state_dict(
                    ema_sd, strict=False
                )
                ema_missing = len(ema_missing_keys)
                ema_unexpected = len(ema_unexpected_keys)
                if verbose:
                    print(
                        f"[chat_demo] EMA loaded from {ema_source} "
                        f"({ema_missing} missing, {ema_unexpected} unexpected)"
                    )
            except Exception as exc:
                if verbose:
                    print(
                        f"[chat_demo] EMA load failed ({exc!r}); "
                        f"falling back to live weights."
                    )
                ema_source = "none"

    model.eval()
    meta = {
        "step": int(raw["step"]) if isinstance(raw, dict) and "step" in raw else None,
        "loss": float(raw["loss"]) if isinstance(raw, dict) and "loss" in raw else None,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "had_config": bool(isinstance(raw, dict) and "config" in raw),
        "ema_source": ema_source,
        "ema_missing_keys": ema_missing,
        "ema_unexpected_keys": ema_unexpected,
    }
    return model, tok, meta


def _generate_one(
    model,
    tok,
    prompt: str,
    max_new: int,
    temperature: float,
    use_template: bool = True,
    device: str = "cpu",
    rfold_inference: bool = False,
    coconut_k: int = 0,
) -> str:
    """Greedy / temperature-sampled completion. Returns just the suffix.

    When ``rfold_inference=True`` the per-token decode goes through
    :func:`synapforge.inference.generate_rfold` which carries CfC + PLIF
    state across calls (O(1) per token, no prefix re-processing).
    Default ``False`` preserves the legacy per-step model.forward path
    bit-identically.

    When ``coconut_k > 0`` AND ``rfold_inference=True``, the decode is
    routed through :func:`generate_with_coconut` so the latent thinking
    loop fires at any ``<bot>`` markers. ``coconut_k=0`` (default) is a
    strict no-op.
    """
    import torch
    text = INSTRUCTION_TEMPLATE.format(instruction=prompt) if use_template else prompt
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if im_end is not None and im_end >= 0:
        eos_ids.add(im_end)

    base_len = ids.size(1)

    if rfold_inference:
        # New path: stateful incremental decode. Token sequence is bit-
        # identical to the legacy path at fp32 + temperature=0; verified
        # in tests/inference/test_rfold_chat.py.
        if coconut_k > 0:
            from synapforge.inference import generate_with_coconut
            bot_id = (
                tok.convert_tokens_to_ids("<bot>")
                if hasattr(tok, "convert_tokens_to_ids") else None
            )
            eot_id = (
                tok.convert_tokens_to_ids("<eot>")
                if hasattr(tok, "convert_tokens_to_ids") else None
            )
            bot_id = bot_id if bot_id is not None and bot_id >= 0 else None
            eot_id = eot_id if eot_id is not None and eot_id >= 0 else None
            out_ids, _ = generate_with_coconut(
                model, ids, max_new=max_new,
                temperature=temperature, coconut_k=coconut_k,
                bot_id=bot_id, eot_id=eot_id,
                eos_ids=list(eos_ids),
            )
        else:
            from synapforge.inference import generate_rfold
            out_ids, _ = generate_rfold(
                model, ids, max_new=max_new,
                temperature=temperature, eos_ids=list(eos_ids),
            )
        out = tok.decode(out_ids[0, base_len:], skip_special_tokens=True)
    else:
        for _ in range(max_new):
            if ids.size(1) >= model.max_seq:
                break
            with torch.no_grad():
                logits = model(ids)
            last = logits[:, -1, :]
            if temperature <= 0:
                nxt = last.argmax(dim=-1, keepdim=True)
            else:
                probs = (last / temperature).softmax(dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            if int(nxt.item()) in eos_ids:
                break
            ids = torch.cat([ids, nxt], dim=1)
        out = tok.decode(ids[0, base_len:], skip_special_tokens=True)
    if "###" in out:
        out = out.split("###", 1)[0]
    return out.strip()


def _load_recorded() -> list[dict]:
    p = _default_recorded()
    if not p.is_file():
        raise FileNotFoundError(f"recorded transcript missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _print_pair(prompt: str, response: str) -> None:
    # one-line prompt header (truncated), full response indented
    head = prompt.replace("\n", " \\n ")
    if len(head) > 60:
        head = head[:57] + "..."
    print(f"  > {head}")
    for line in response.splitlines() or [""]:
        print(f"      {line}")
    print()


def _select_prompts(prompt_set: str) -> tuple[list[str], list[str]]:
    if prompt_set == "t1":
        return EN_PROMPTS_T1, ZH_PROMPTS_T1
    if prompt_set == "legacy":
        return EN_PROMPTS_LEGACY, ZH_PROMPTS_LEGACY
    raise ValueError(f"unknown prompt_set: {prompt_set!r}")


def run_demo(
    ckpt: str | None = None,
    tokenizer_path: str | None = None,
    max_new: int = 80,
    temperature: float = 0.7,
    save_path: str | None = None,
    quiet: bool = False,
    device: str = "auto",
    verbose: bool = False,
    prompt_set: str = "auto",
    use_template: bool | None = None,
    use_ema: bool = False,
    rfold_inference: bool = False,
    coconut_k: int = 0,
) -> dict:
    """Run the chat demo.

    ``prompt_set``:
      - ``"t1"``   : queue T1.1 prompts (live raw completion)
      - ``"legacy"``: chat_recorded.json prompts (back-compat replay)
      - ``"auto"``  : t1 if a live ckpt loads, legacy if we fall through
                      to the recorded replay path

    ``use_template``:
      - ``True``   : wrap each prompt with INSTRUCTION_TEMPLATE
      - ``False``  : pass prompt verbatim to tokenizer (raw completion)
      - ``None``   : auto — False for ``t1`` (queue prompts are open-ended
                     completions), True for ``legacy`` (instruction-tuned
                     prompts that need the response template).

    ``use_ema`` (T8.4): if the ckpt was trained with ``--ema-decay > 0``,
      load the EMA-smoothed weights at inference instead of the raw live
      weights. Looks for ``raw["ema_state"]`` first, then a sibling
      ``step_<N>_ema.pt`` file. No-op if neither exists.
    """
    ckpt = ckpt or os.environ.get("SYNAPFORGE_CKPT") or _default_ckpt()
    resolved_device = _resolve_device(device)

    live = _try_load_live(ckpt, tokenizer_path, device=resolved_device,
                          verbose=verbose, use_ema=use_ema)

    # Pick prompt set after we know whether live mode is available.
    if prompt_set == "auto":
        active_prompt_set = "t1" if live is not None else "legacy"
    else:
        active_prompt_set = prompt_set
    en_prompts, zh_prompts = _select_prompts(active_prompt_set)
    prompts = [(p, "EN") for p in en_prompts] + [(p, "ZH") for p in zh_prompts]

    if use_template is None:
        # T1 = raw completion; legacy recorded transcripts were generated
        # with the instruction template.
        effective_use_template = (active_prompt_set == "legacy")
    else:
        effective_use_template = bool(use_template)

    samples: list[dict] = []
    t0 = time.time()
    meta: dict = {}
    if live is not None:
        model, tok, meta = live
        if verbose and not quiet:
            print(f"  ckpt loaded: {ckpt}")
            print(f"  tokenizer:   {tokenizer_path}")
            print(f"  device:      {resolved_device}")
            print(f"  prompt_set:  {active_prompt_set}")
            if use_ema:
                ema_src = meta.get("ema_source", "none")
                print(f"  ema:         {ema_src}")
            print()
        for p, lang in prompts:
            try:
                resp = _generate_one(
                    model, tok, p, max_new=max_new,
                    temperature=temperature,
                    use_template=effective_use_template,
                    device=resolved_device,
                    rfold_inference=rfold_inference,
                    coconut_k=coconut_k,
                )
            except Exception as e:
                resp = f"<generation error: {e}>"
            samples.append({"lang": lang, "prompt": p, "response": resp})
            if verbose and not quiet:
                _print_pair(p, resp)
        mode = "live"
    else:
        if verbose and not quiet:
            # docs/INSURANCE_NATIVE.md Option B: when the live ckpt is
            # unloadable we play the recorded v4.x transcript -- but loud
            # disclosure first so the investor knows it's not live. Same
            # architecture (LNN+SNN), older ckpt; ANTI_LORA.md compliant.
            try:
                from .disclose import disclose_replay
                print(disclose_replay())
                print()
            except Exception:
                pass
            print("  ckpt unavailable, replaying recorded transcript:")
            print(f"  source: {_default_recorded()}")
            print()
        try:
            recorded = {r["prompt"]: r["response"] for r in _load_recorded()}
        except FileNotFoundError:
            recorded = {}
        for p, lang in prompts:
            resp = recorded.get(p, "<no recorded response>")
            samples.append({"lang": lang, "prompt": p, "response": resp})
            if verbose and not quiet:
                _print_pair(p, resp)
        mode = "recorded"

    dt = time.time() - t0
    out = {
        "mode": mode,
        "ckpt": ckpt,
        "tokenizer_path": tokenizer_path,
        "device": resolved_device,
        "prompt_set": active_prompt_set,
        "wall_time_s": dt,
        "n_prompts": len(prompts),
        "step": meta.get("step") if meta else None,
        "use_ema": bool(use_ema),
        "ema_source": (meta.get("ema_source") if meta else "none"),
        "samples": samples,
        # Back-compat: older callers (cli.py JSON dump, downstream analysis
        # scripts) read out["pairs"] without the "lang" key. Mirror samples
        # into "pairs" keeping just prompt + response so old code paths
        # don't break while new ones can read the richer "samples" list.
        "pairs": [{"prompt": s["prompt"], "response": s["response"]}
                  for s in samples],
    }
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if verbose and not quiet:
            print(f"  saved transcript -> {save_path}")
    if verbose and not quiet:
        print(f"  done ({mode}) in {dt:.1f}s, {len(samples)} prompts.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synapforge-demo chat")
    ap.add_argument("--ckpt", default=None,
                    help="path to v24h-style ckpt (default ~/.synapforge/v24h_chat.pt)")
    ap.add_argument("--tokenizer-path", default=None,
                    help="HF tokenizer path or repo id (default Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--save", default="chat_demo.json")
    ap.add_argument("--device", default="auto",
                    help="cuda / cpu / auto (default: auto)")
    ap.add_argument("--verbose", action="store_true", default=False,
                    help="print per-prompt output and load diagnostics")
    ap.add_argument("--prompt-set", default="auto",
                    choices=["auto", "t1", "legacy"],
                    help="auto picks t1 for live, legacy for recorded")
    ap.add_argument("--use-ema", action="store_true", default=False,
                    dest="use_ema",
                    help="T8.4: load EMA-smoothed weights (from "
                         "raw['ema_state'] or sibling step_<N>_ema.pt) "
                         "in place of the raw live weights at inference. "
                         "No-op if neither EMA payload exists.")
    ap.add_argument(
        "--rfold-inference", action="store_true", default=False,
        dest="rfold_inference",
        help="Use stateful R-fold incremental decode (constant-time per "
             "token). Bit-identical to the default sequential path at "
             "fp32 + temperature=0; default OFF preserves legacy behaviour.")
    ap.add_argument(
        "--coconut-k", type=int, default=0,
        dest="coconut_k",
        help="Coconut latent thinking depth (arxiv:2412.06769). When > 0 "
             "and combined with --rfold-inference, fires K hidden-state "
             "passes on every <bot> token. 0 = OFF (default).")
    args = ap.parse_args(argv)
    run_demo(
        ckpt=args.ckpt,
        tokenizer_path=args.tokenizer_path or "Qwen/Qwen2.5-0.5B",
        max_new=args.max_new,
        temperature=args.temperature,
        save_path=args.save,
        device=args.device,
        verbose=args.verbose,
        prompt_set=args.prompt_set,
        use_ema=args.use_ema,
        rfold_inference=args.rfold_inference,
        coconut_k=args.coconut_k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
