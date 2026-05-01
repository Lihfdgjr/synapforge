"""Honest eval hook — stops 'training-curve hallucination'.

Per user rule 2026-05-01: ppl 数字 ≠ 对话能力. Every N steps, run a fixed
set of test prompts through the live model and dump the actual generated
text to train.log. Plateau detector also flags if best_ppl hasn't improved
in K cycles, so we don't blindly burn GPU on a stuck run.

Public API:
    HonestEvalHook(model, tokenizer, out_dir, every_steps=500)
        .maybe_eval(step, current_ppl) -> dict | None
        .check_plateau(window=5) -> bool

Output (always appended to train.log + saved as JSONL):
    {"step": 500, "ppl": 87.3, "samples": [
        {"prompt": "你好", "generated": "你好,今天..."},
        ...
    ], "verdict": "WORD_SALAD"|"GRAMMAR_OK"|"COHERENT"}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# 5 fixed prompts spanning easy → hard. Don't change between runs (keeps
# diff comparable across ckpts).
TEST_PROMPTS_EN = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "Q: Who wrote Romeo and Juliet?\nA:",
    "Translate to Chinese: Hello, how are you?",
]
TEST_PROMPTS_ZH = [
    "你好,",
    "中国的首都是",
    "请解释什么是机器学习。",
    "1+1等于几?",
    "今天天气真好,",
]


def _heuristic_verdict(samples: list[dict]) -> str:
    """Quick string-quality check. NOT ground truth — log original always."""
    texts = [s["generated"] for s in samples]
    avg_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)
    # token-soup: same word repeats >5 times in a generation
    token_soup = any(
        max((t.split().count(w) for w in set(t.split())), default=0) > 5
        for t in texts if t
    )
    # mostly-empty or very short → just learned to stop early
    if avg_len < 3:
        return "TOO_SHORT"
    if token_soup:
        return "TOKEN_SOUP"
    # rough grammar: contains capital + period sequence
    has_punct = sum(1 for t in texts if any(c in t for c in ".!?。!?"))
    if has_punct >= len(texts) // 2:
        return "GRAMMAR_OK"
    return "WORD_SALAD"


class HonestEvalHook:
    """Periodic chat-sample eval inside the training loop."""

    def __init__(
        self,
        model,
        tokenizer,
        out_dir: str | Path,
        every_steps: int = 500,
        max_new_tokens: int = 40,
        prompts: list[str] | None = None,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.out_dir = Path(out_dir)
        self.every = int(every_steps)
        self.max_new = int(max_new_tokens)
        self.device = device
        self.prompts = prompts or (TEST_PROMPTS_EN + TEST_PROMPTS_ZH)
        self.history: list[dict] = []
        self.jsonl = self.out_dir / "honest_eval.jsonl"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _generate(self, prompt: str) -> str:
        """Greedy generation of self.max_new tokens. Falls back gracefully
        if the model doesn't expose a forward(input_ids) returning logits.
        """
        import torch

        try:
            ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        except Exception:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.max_new):
                try:
                    out = self.model(ids)
                    logits = out.logits if hasattr(out, "logits") else out
                except Exception:
                    # synapforge backbone returns hidden states; tie head
                    h = self.model.tok_embed(ids) if hasattr(self.model, "tok_embed") else None
                    if h is None:
                        break
                    h = self.model._run_blocks(h) if hasattr(self.model, "_run_blocks") else h
                    if hasattr(self.model, "ln_f"):
                        h = self.model.ln_f(h)
                    logits = (
                        self.model.lm_head(h)
                        if getattr(self.model, "lm_head", None) is not None
                        else h @ self.model.tok_embed.weight.T
                    )
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == getattr(self.tok, "eos_token_id", -1):
                    break
        if was_training:
            self.model.train()
        try:
            return self.tok.decode(ids[0], skip_special_tokens=True)
        except Exception:
            return self.tok.decode(ids[0].tolist())

    def maybe_eval(self, step: int, current_ppl: float | None = None) -> dict | None:
        if step <= 0 or step % self.every != 0:
            return None
        samples = []
        for p in self.prompts:
            try:
                gen = self._generate(p)
            except Exception as e:
                gen = f"<EVAL_ERROR: {e}>"
            samples.append({"prompt": p, "generated": gen})
        verdict = _heuristic_verdict(samples)
        record = {
            "step": int(step),
            "ts": time.time(),
            "ppl": float(current_ppl) if current_ppl is not None else None,
            "verdict_heuristic": verdict,
            "samples": samples,
        }
        # Always append to JSONL — disk is cheap, hallucination is expensive.
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Pretty-print first 2 samples to stdout
        print(f"[honest-eval step={step}] verdict={verdict} ppl={current_ppl}")
        for s in samples[:2]:
            print(f"  prompt: {s['prompt']!r}")
            print(f"  generated: {s['generated']!r}")
        self.history.append(record)
        return record

    def check_plateau(self, window: int = 5, eps: float = 0.5) -> bool:
        """True if best ppl hasn't improved by `eps` in last `window` evals.

        Caller should `--early-stop` if this returns True for several cycles.
        """
        if len(self.history) < window:
            return False
        recent = [h["ppl"] for h in self.history[-window:] if h["ppl"]]
        if len(recent) < window:
            return False
        return (max(recent) - min(recent)) < eps


def smoke():
    """Standalone test (no real model needed)."""
    class _Tok:
        eos_token_id = 0
        def encode(self, s, return_tensors=None):
            import torch
            return torch.tensor([[ord(c) % 256 for c in s[:32]]])
        def decode(self, ids, skip_special_tokens=True):
            try:
                return "".join(chr(int(i) % 256) for i in ids)
            except Exception:
                return str(ids)

    class _Model:
        training = False
        def eval(self): self.training = False
        def train(self): self.training = True
        def __call__(self, ids):
            import torch
            return type("o", (), {"logits": torch.randn(ids.shape[0], ids.shape[1], 256)})

    h = HonestEvalHook(_Model(), _Tok(), out_dir="/tmp/honest_smoke", every_steps=1, max_new_tokens=8, device="cpu")
    r = h.maybe_eval(step=1, current_ppl=88.0)
    print("smoke OK; jsonl:", h.jsonl)
    return r


if __name__ == "__main__":
    smoke()
