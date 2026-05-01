"""qwen_lora_demo — investor-demo wrapper around the Qwen 0.5B + LoRA v0 frontend.

Honesty contract:
- Native architecture claim     -> SynapForge 100M LNN+SNN (`chat_demo.py`)
- v0 chat frontend (today)      -> Qwen 0.5B + LoRA (this file)

`run_demo(adapter_path, n_samples=5)` runs 5 EN + 5 ZH canned prompts and
prints a transcript. If the adapter directory is missing, prints the tail
of the live training log (`<adapter_path>/train.log`) plus a clear
"training in progress" notice — that's a deliberate fall-back so the
investor demo runs even before the LoRA finishes training.

Wired into `synapforge-demo qwenchat`. Independent of `synapforge-demo
chat`, which targets the native 100M LNN+SNN.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# canned prompts -- bilingual, light coding, simple QA
EN_PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "Translate to Chinese: Time flies like an arrow.",
    "Q: Who wrote the play 'Hamlet'?\nA:",
]

ZH_PROMPTS = [
    "请用一句话介绍量子计算。",
    "写一首关于秋天的四行诗。",
    "1+1 等于几?为什么?",
    "把这句话翻译成英文:今天天气真好。",
    "中国的首都是哪里?",
]


def _default_adapter() -> Path:
    return Path.home() / ".synapforge" / "release" / "qwen_lora_v0"


def _print_pair(prompt: str, response: str) -> None:
    head = prompt.replace("\n", " \\n ")
    if len(head) > 60:
        head = head[:57] + "..."
    print(f"  > {head}")
    for line in (response or "").splitlines() or [""]:
        print(f"      {line}")
    print()


def _print_training_progress(adapter_dir: Path) -> dict:
    """Adapter not ready yet. Show user where training is + tail of log."""
    print("  Qwen-LoRA adapter not found at:")
    print(f"    {adapter_dir}")
    print("  Live training is in progress. Tail of train.log below.")
    print()
    log = adapter_dir / "train.log"
    tail_lines: list[str] = []
    if log.is_file():
        try:
            text = log.read_text(encoding="utf-8")
            tail_lines = text.splitlines()[-25:]
        except Exception as e:
            tail_lines = [f"<could not read log: {e}>"]
    else:
        # fall back to top-level synapforge log if present
        alt = Path.home() / ".synapforge" / "train.log"
        if alt.is_file():
            try:
                tail_lines = alt.read_text(encoding="utf-8").splitlines()[-25:]
            except Exception:
                pass
    if tail_lines:
        for ln in tail_lines:
            print(f"  | {ln}")
    else:
        print("  | (no train.log yet — training has not started, or is on a remote rental)")
    print()
    print("  Run scripts/train_qwen_lora.py to start the LoRA fine-tune.")
    print("  The native SynapForge 100M LNN+SNN is shown in `synapforge-demo all`.")
    return {
        "mode": "training_in_progress",
        "adapter": str(adapter_dir),
        "tail_lines": tail_lines,
    }


def _load_live(adapter_dir: Path, base_path: str | None, smoke: bool) -> tuple | None:
    """Load Qwen + adapter via the REPL helper. Returns (model, tok, device)
    or None on any failure (so demo gracefully falls back)."""
    try:
        # add scripts to path
        repo_root = Path(__file__).resolve().parent.parent.parent
        scripts = repo_root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from qwen_lora_chat_repl import _load_smoke, load_qwen_lora  # type: ignore
    except Exception as e:
        print(f"  [demo] could not import REPL helper: {e}")
        return None
    try:
        if smoke:
            return _load_smoke()
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        return load_qwen_lora(str(adapter_dir),
                              base_path or os.environ.get("QWEN_BASE_PATH",
                                                          "Qwen/Qwen2.5-0.5B-Instruct"),
                              device)
    except Exception as e:
        print(f"  [demo] live load failed: {e}")
        return None


def _generate(model, tok, prompt: str, max_new: int, temperature: float, device: str) -> str:
    try:
        from qwen_lora_chat_repl import generate_chat  # type: ignore
        return generate_chat(model, tok, prompt, max_new=max_new,
                             temperature=temperature, device=device)
    except Exception as e:
        return f"<generation error: {e}>"


def run_demo(
    adapter_path: str | None = None,
    base_path: str | None = None,
    n_samples: int = 5,
    max_new: int = 80,
    temperature: float = 0.7,
    save_path: str | None = "chat_qwen_lora_demo.json",
    smoke: bool = False,
    quiet: bool = False,
) -> dict:
    """Run 5 EN + 5 ZH prompts against Qwen + LoRA. Returns result dict."""
    adapter_dir = Path(adapter_path).expanduser() if adapter_path else _default_adapter()
    en = EN_PROMPTS[:n_samples]
    zh = ZH_PROMPTS[:n_samples]
    prompts = en + zh

    if not quiet:
        print("=== Qwen 0.5B + LoRA chat demo (v0 frontend) ===")
        print()

    # smoke path skips the not-found check
    if smoke:
        live = _load_live(adapter_dir, base_path, smoke=True)
    elif not adapter_dir.exists():
        prog = _print_training_progress(adapter_dir)
        prog["pairs"] = []
        prog["n_prompts"] = len(prompts)
        return prog
    else:
        live = _load_live(adapter_dir, base_path, smoke=False)

    if live is None:
        prog = _print_training_progress(adapter_dir)
        prog["pairs"] = []
        prog["n_prompts"] = len(prompts)
        return prog

    model, tok, device = live
    if not quiet:
        print(f"  base + adapter: {adapter_dir}")
        print(f"  device:         {device}")
        print()

    pairs: list[dict] = []
    t0 = time.time()
    for p in prompts:
        resp = _generate(model, tok, p, max_new=max_new, temperature=temperature, device=device)
        pairs.append({"prompt": p, "response": resp})
        if not quiet:
            _print_pair(p, resp)

    dt = time.time() - t0
    out = {
        "mode": "live_qwen_lora",
        "adapter": str(adapter_dir),
        "device": device,
        "n_prompts": len(prompts),
        "wall_time_s": dt,
        "pairs": pairs,
    }
    if save_path:
        try:
            Path(save_path).write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if not quiet:
                print(f"  saved transcript -> {save_path}")
        except Exception as e:
            if not quiet:
                print(f"  save failed: {e}")
    if not quiet:
        print(f"  done in {dt:.1f}s, {len(pairs)} prompts.")
    return out


# allow `python -m synapforge.demo.qwen_lora_demo`
def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="synapforge-demo qwenchat")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-path", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--save", default="chat_qwen_lora_demo.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    run_demo(adapter_path=args.adapter, base_path=args.base_path,
             n_samples=args.n, max_new=args.max_new,
             temperature=args.temperature, save_path=args.save,
             smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
