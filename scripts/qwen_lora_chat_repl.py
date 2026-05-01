"""qwen_lora_chat_repl — interactive chat REPL for the Qwen 0.5B + LoRA v0 frontend.

Loads Qwen2.5-0.5B-Instruct + the LoRA adapter trained by
`scripts/train_qwen_lora.py`, applies the Qwen chat template
(`<|im_start|>system / user / assistant<|im_end|>`), and runs a generation
loop. Save-flag dumps every (prompt, response) to JSON for demo replay.

Companion to `scripts/chat_repl.py`, which targets the native SynapForge
100M LNN+SNN. Same UX, different backbone.

Usage:
    python scripts/qwen_lora_chat_repl.py \\
        --adapter ~/.synapforge/release/qwen_lora_v0 \\
        --base-path /workspace/teachers/qwen2.5-0.5b \\
        --temperature 0.7 \\
        --max-new 200 \\
        --save chat_qwen_lora.json

    # smoke (no Qwen, no adapter): falls back to a mock model
    python scripts/qwen_lora_chat_repl.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


SYSTEM_PROMPT = (
    "You are a helpful, harmless, honest bilingual assistant. "
    "Answer in the language of the user's question (English or Chinese)."
)


def _qwen_chat_template(messages: list[dict]) -> str:
    """Manual Qwen chat template, used when tokenizer.apply_chat_template
    isn't available (smoke / older transformers)."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _load_smoke():
    """Return (model, tok, device) using the smoke fake-Qwen from the
    trainer module so this REPL is runnable without any deps."""
    from train_qwen_lora import SmokeQwen, SmokeTokenizer  # type: ignore
    model = SmokeQwen(vocab=256, d=32, n_layers=2)
    tok = SmokeTokenizer(vocab=256)
    return model, tok, "cpu"


def load_qwen_lora(adapter_dir: str, base_path: str, device: str) -> tuple:
    """Load Qwen base + LoRA adapter. Strategy:

    1. If `<adapter_dir>/merged.pt` exists and is non-empty, load base then
       overwrite the merged proj weights -> single merged model (fastest).
    2. Else if peft is installed and `<adapter_dir>/adapter` looks like
       a peft save_pretrained dir, use PeftModel.from_pretrained.
    3. Else if `<adapter_dir>/adapter/lora_state.pt` exists (inline LoRA),
       attach inline LoRA modules then load_state_dict.
    4. Else: just return base model + warning.
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers required for live load; pass --smoke")

    print(f"[load] base={base_path}")
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    adapter = Path(adapter_dir)
    merged_pt = adapter / "merged.pt"
    peft_dir = adapter / "adapter"
    inline_pt = peft_dir / "lora_state.pt"

    if merged_pt.is_file() and merged_pt.stat().st_size > 0:
        print(f"[load] applying merged ckpt {merged_pt}")
        sd = torch.load(merged_pt, map_location="cpu")
        own = model.state_dict()
        n_apply = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n_apply += 1
        model.load_state_dict(own)
        print(f"[load] applied {n_apply} merged tensors")
    elif HAS_PEFT and (peft_dir / "adapter_config.json").is_file():
        print(f"[load] peft adapter {peft_dir}")
        model = PeftModel.from_pretrained(model, str(peft_dir))
    elif inline_pt.is_file():
        print(f"[load] inline LoRA from {inline_pt}")
        from train_qwen_lora import _attach_inline_lora  # type: ignore
        _attach_inline_lora(model, rank=16, alpha=32,
                            targets=("q_proj", "k_proj", "v_proj", "o_proj"))
        sd = torch.load(inline_pt, map_location="cpu")
        own = model.state_dict()
        for k, v in sd.items():
            if k in own:
                own[k] = v
        model.load_state_dict(own, strict=False)
    else:
        print(f"[load] WARN: no adapter found under {adapter_dir}, using base only")

    model = model.to(device).eval()
    return model, tok, device


@torch.no_grad()
def generate_chat(model, tok, prompt: str, max_new: int, temperature: float,
                  device: str, history: list[dict] | None = None) -> str:
    history = history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": prompt}]

    # try the official template first; fall back to manual
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        text = _qwen_chat_template(messages)

    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    base_len = ids.size(1)

    eos_ids: set[int] = set()
    if getattr(tok, "eos_token_id", None) is not None:
        eos_ids.add(int(tok.eos_token_id))
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            eos_ids.add(im_end)
    except Exception:
        pass

    for _ in range(max_new):
        try:
            logits = model(input_ids=ids).logits
        except TypeError:
            logits = model(ids).logits
        last = logits[:, -1, :]
        if temperature <= 0:
            nxt = last.argmax(dim=-1, keepdim=True)
        else:
            probs = (last / temperature).softmax(dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        if int(nxt.item()) in eos_ids:
            break
        ids = torch.cat([ids, nxt], dim=1)

    return tok.decode(ids[0, base_len:], skip_special_tokens=True).strip()


def repl(model, tok, max_new: int, temperature: float, save_path: str | None, device: str):
    saved: list[dict] = []
    if save_path and Path(save_path).is_file():
        try:
            saved = json.loads(Path(save_path).read_text(encoding="utf-8"))
        except Exception:
            saved = []
    history: list[dict] = []
    print("Qwen 0.5B + LoRA chat. Ctrl-D / Ctrl-C to exit. EN + ZH OK.\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in ("/clear", "/reset"):
            history = []
            print("(history cleared)")
            continue
        t0 = time.time()
        try:
            out = generate_chat(model, tok, prompt, max_new=max_new,
                                temperature=temperature, device=device,
                                history=history)
        except Exception as e:
            out = f"<generation error: {e}>"
        dt = time.time() - t0
        print(f"{out}\n  ({dt:.1f}s)\n")
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": out})
        saved.append({"ts": time.time(), "prompt": prompt, "response": out, "duration_s": dt})
        if save_path:
            Path(save_path).write_text(
                json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        # cap history at last 6 turns (12 messages) to fit context
        if len(history) > 12:
            history = history[-12:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qwen_lora_chat_repl")
    ap.add_argument("--adapter", default=str(Path.home() / ".synapforge" / "release" / "qwen_lora_v0"),
                    help="dir produced by scripts/train_qwen_lora.py")
    ap.add_argument("--base-path", default=os.environ.get("QWEN_BASE_PATH", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="use mock Qwen (no real ckpt needed)")
    ap.add_argument("--once", default=None,
                    help="run a single prompt and exit (smoke-friendly)")
    args = ap.parse_args(argv)

    if args.smoke or not HAS_TRANSFORMERS:
        if not args.smoke:
            print("[repl] transformers unavailable, falling back to smoke", file=sys.stderr)
        model, tok, dev = _load_smoke()
        args.device = dev
    else:
        model, tok, dev = load_qwen_lora(args.adapter, args.base_path, args.device)
        args.device = dev

    if args.once is not None:
        out = generate_chat(model, tok, args.once,
                            max_new=min(args.max_new, 32 if args.smoke else args.max_new),
                            temperature=args.temperature, device=args.device)
        print(f"prompt: {args.once}")
        print(f"response: {out}")
        if args.save:
            Path(args.save).write_text(
                json.dumps([{"prompt": args.once, "response": out, "ts": time.time()}],
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return 0

    repl(model, tok, args.max_new, args.temperature, args.save, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
