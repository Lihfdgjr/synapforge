"""chat_repl — interactive REPL for SynapForge100M after pretrain + SFT.

Loads a SynapForge100M ckpt + Qwen-compatible tokenizer, runs a generation
loop with the same instruction template that prep_alpaca_qwen.py emits.

Usage:
    python scripts/chat_repl.py \
        --ckpt /workspace/runs/v24h_qwen_sft/best_step000.pt \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --temperature 0.7 \
        --max-new 80 \
        --save chat_demo.json

Type a prompt at the `>` prompt; Ctrl-D / Ctrl-C exits. All (prompt, output)
pairs are appended to --save (JSON list) so a demo run produces a
shareable transcript even if the SSH session drops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path so this works invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n"
    "### Response:\n"
)


def load_model(ckpt_path: str, vocab: int = 151936, device: str = "cuda"):
    from synapforge.model_100m import SynapForge100M
    model = SynapForge100M(
        vocab=vocab, d=512, n_layers=10, loop_depth=1, max_seq=2048,
        ffn_ratio=8.0, sparsity=0.95, dropout=0.0, tie_lm_head=True,
    )
    if ckpt_path and Path(ckpt_path).is_file():
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[chat] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[chat] WARN: ckpt {ckpt_path!r} not found, using RANDOM INIT (output will be garbage)")
    model.to(device).eval()
    return model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 80, temperature: float = 0.7,
             device: str = "cuda") -> str:
    """Greedy or temperature-sampling generation. Stops at EOS / im_end."""
    text = INSTRUCTION_TEMPLATE.format(instruction=prompt)
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    # eos_token_id is sometimes None on un-configured tokenizers; build a
    # set of ints only.
    eos_ids: set[int] = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)
    # `\n###` is the next-instruction marker — also a soft stop
    stop_str = "###"
    # Cache prompt length once -- the previous code re-encoded the prompt
    # every iteration to compute the slice offset (O(N^2) decode cost).
    prompt_len = ids.size(1)

    generated_text = ""
    for _ in range(max_new):
        # SynapForge100M.forward(ids) returns hidden; we tie lm head to embedding
        try:
            logits = model(ids)
        except Exception:
            h = model.tok_embed(ids)
            pos = model.pos_embed[: h.size(1)].unsqueeze(0)
            h = h + pos
            h = model._run_blocks(h)
            h = model.ln_f(h)
            logits = (model.lm_head(h) if getattr(model, "lm_head", None) is not None
                      else h @ model.tok_embed.weight.T)
        last = logits[:, -1, :]
        if temperature <= 0:
            next_id = last.argmax(dim=-1, keepdim=True)
        else:
            probs = (last / temperature).softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        if int(next_id.item()) in eos_ids:
            break
        ids = torch.cat([ids, next_id], dim=1)
        # detokenize incrementally so we can stop on `###`
        generated_text = tok.decode(ids[0, prompt_len:], skip_special_tokens=True)
        if stop_str in generated_text:
            generated_text = generated_text.split(stop_str)[0]
            break
    return generated_text.strip()


def repl(model, tok, max_new: int, temperature: float, save_path: str | None, device: str):
    saved = []
    if save_path and Path(save_path).is_file():
        try:
            saved = json.loads(Path(save_path).read_text())
        except Exception:
            saved = []
    print("Enter a prompt (Ctrl-D / Ctrl-C to exit). Both English and Chinese OK.\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        t0 = time.time()
        try:
            out = generate(model, tok, prompt, max_new=max_new,
                           temperature=temperature, device=device)
        except Exception as e:
            out = f"<generation error: {e}>"
        dt = time.time() - t0
        print(f"{out}\n  ({dt:.1f}s)\n")
        saved.append({"ts": time.time(), "prompt": prompt, "response": out, "duration_s": dt})
        if save_path:
            Path(save_path).write_text(json.dumps(saved, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("SYNAPFORGE_CKPT", ""))
    ap.add_argument("--tokenizer-path", default="/workspace/teachers/qwen2.5-0.5b")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--save", default=None, help="dump (prompt,response) pairs to JSON")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_path)
    model = load_model(args.ckpt, vocab=args.vocab, device=args.device)
    repl(model, tok, args.max_new, args.temperature, args.save, args.device)


if __name__ == "__main__":
    main()
