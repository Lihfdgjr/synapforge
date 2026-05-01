"""Chat demo: 5 EN + 5 ZH prompts against SynapForge100M (Qwen vocab).

If a v24h-style ckpt is available, load it and generate live; otherwise
replay a recorded transcript captured during v4.1 training (ppl ~44, 100M
LNN+SNN). The recorded outputs are coherent but limited — that's honest
for a 100M model. Runs on CPU.

Usage:
    synapforge-demo chat
    synapforge-demo chat --ckpt /path/to/v24h_chat.pt --tokenizer-path Qwen/Qwen2.5-0.5B
    python -m synapforge.demo.chat_demo --save chat_demo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EN_PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "Q: Who wrote Romeo and Juliet?\nA:",
    "Translate to Chinese: Hello",
]

ZH_PROMPTS = [
    "你好,",
    "中国的首都是",
    "请解释什么是机器学习。",
    "1+1等于几?",
    "今天天气真好,",
]

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"


def _default_ckpt() -> str:
    return str(Path.home() / ".synapforge" / "v24h_chat.pt")


def _default_recorded() -> Path:
    return Path(__file__).resolve().parent / "chat_recorded.json"


def _try_load_live(ckpt: str, tokenizer_path: str | None):
    """Best-effort load of model + tokenizer. Returns (model, tok) or None."""
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
    model = SynapForge100M(
        vocab=151936, d=512, n_layers=10, loop_depth=1, max_seq=2048,
        ffn_ratio=8.0, sparsity=0.95, dropout=0.0, tie_lm_head=True,
    )
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        return None
    model.eval()
    return model, tok


def _generate_one(model, tok, prompt: str, max_new: int, temperature: float) -> str:
    import torch
    text = INSTRUCTION_TEMPLATE.format(instruction=prompt)
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if im_end is not None and im_end >= 0:
        eos_ids.add(im_end)

    base_len = ids.size(1)
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


def run_demo(
    ckpt: str | None = None,
    tokenizer_path: str | None = None,
    max_new: int = 80,
    temperature: float = 0.7,
    save_path: str | None = None,
    quiet: bool = False,
) -> dict:
    ckpt = ckpt or os.environ.get("SYNAPFORGE_CKPT") or _default_ckpt()
    prompts = EN_PROMPTS + ZH_PROMPTS

    live = _try_load_live(ckpt, tokenizer_path)
    pairs: list[dict] = []
    t0 = time.time()
    if live is not None:
        model, tok = live
        if not quiet:
            print(f"  ckpt loaded: {ckpt}")
            print(f"  tokenizer:   {tokenizer_path}")
            print()
        for p in prompts:
            try:
                resp = _generate_one(model, tok, p, max_new=max_new,
                                     temperature=temperature)
            except Exception as e:
                resp = f"<generation error: {e}>"
            pairs.append({"prompt": p, "response": resp})
            if not quiet:
                _print_pair(p, resp)
        mode = "live"
    else:
        if not quiet:
            print("  ckpt unavailable, replaying recorded transcript:")
            print(f"  source: {_default_recorded()}")
            print()
        recorded = {r["prompt"]: r["response"] for r in _load_recorded()}
        for p in prompts:
            resp = recorded.get(p, "<no recorded response>")
            pairs.append({"prompt": p, "response": resp})
            if not quiet:
                _print_pair(p, resp)
        mode = "recorded"

    dt = time.time() - t0
    out = {
        "mode": mode,
        "ckpt": ckpt,
        "tokenizer_path": tokenizer_path,
        "wall_time_s": dt,
        "n_prompts": len(prompts),
        "pairs": pairs,
    }
    if save_path:
        Path(save_path).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not quiet:
            print(f"  saved transcript -> {save_path}")
    if not quiet:
        print(f"  done ({mode}) in {dt:.1f}s, {len(pairs)} prompts.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synapforge-demo chat")
    ap.add_argument("--ckpt", default=None,
                    help="path to v24h-style ckpt (default ~/.synapforge/v24h_chat.pt)")
    ap.add_argument("--tokenizer-path", default=None,
                    help="HF tokenizer path or repo id (default Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--save", default="chat_demo.json")
    args = ap.parse_args(argv)
    run_demo(
        ckpt=args.ckpt,
        tokenizer_path=args.tokenizer_path or "Qwen/Qwen2.5-0.5B",
        max_new=args.max_new,
        temperature=args.temperature,
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
