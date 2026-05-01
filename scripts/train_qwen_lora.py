"""train_qwen_lora — LoRA SFT on Qwen2.5-0.5B-Instruct for the v0 chat frontend.

This is the "Plan C" demo trainer. Frozen Qwen 0.5B Instruct (already chat-
tuned by the Qwen team) gets a small LoRA adapter on the attention
projections; we SFT it on alpaca-en + alpaca-zh for one epoch. 30 min on
A800 / 2-3h on CPU.

The native SynapForge 100M LNN+SNN backbone is the architecture claim and
is trained from scratch in `train_100m.py`. THIS script is the demo
frontend: it ships a chat-able artifact in 30 min so the investor demo
has live bilingual chat alongside the 100M telemetry.

Honest framing matters:
- v0 demo frontend = Qwen 0.5B + LoRA  (this file)
- v1 native = SynapForge 100M LNN+SNN  (train_100m.py + train_100m_sft.py)

Usage:
    # real run (peft path, GPU)
    python scripts/train_qwen_lora.py \\
        --base-path /workspace/teachers/qwen2.5-0.5b \\
        --data data/sft/alpaca_combined.parquet \\
        --out ~/.synapforge/release/qwen_lora_v0 \\
        --steps 5000 --bs 8 --lr 2e-4

    # smoke (no GPU, no real Qwen): runs ~5 steps on a fake mini-Qwen
    python scripts/train_qwen_lora.py --smoke

CLI:
    --base-path PATH    HF AutoModelForCausalLM-loadable path or repo id
    --data PATH         parquet from prep_alpaca_qwen.py (input_ids, loss_mask)
    --out PATH          adapter + merged ckpt directory
    --steps N           total training steps (5000 ≈ 1 epoch on 100K examples)
    --bs N              batch size (8 fits A800 80GB at seq=1024)
    --lr FLOAT          peak learning rate (2e-4 standard for r=16 LoRA)
    --rank N            LoRA rank (default 16)
    --alpha N           LoRA alpha (default 32, i.e. 2× rank)
    --max-seq N         max sequence length (default 1024)
    --warmup N          linear warmup steps (default 100)
    --log-every N       loss log frequency (default 50)
    --sample-every N    dump 5 EN + 5 ZH samples (default 500)
    --save-every N      checkpoint adapter (default 1000)
    --smoke             use fake mini-Qwen + tiny mock dataset, 5 steps

Output artifacts under --out:
    adapter/                LoRA adapter (peft .save_pretrained or numpy)
    merged.pt               merged base+LoRA ckpt for fast inference
    tokenizer/              tokenizer files (so REPL only needs --out)
    train.log               step / loss / lr / sample dumps
    config.json             reproducibility metadata
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Repo root on sys.path so this works invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- robust imports: peft optional, torch required, transformers preferred ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    print(f"FATAL: torch is required. {e}", file=sys.stderr)
    sys.exit(2)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel  # noqa: F401
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


# ---------------------------------------------------------------------------
# Inline LoRA fallback (so smoke runs without peft installed)
# ---------------------------------------------------------------------------

class InlineLoRALinear(nn.Module):
    """Wraps an nn.Linear with frozen base + LoRA A·B low-rank update.

    forward(x) = base(x) + (alpha/rank) * (x @ A.T) @ B.T

    Matches peft's lora math. ~20 LOC, exact same numerics. Used when peft
    isn't installed (smoke + minimal env).
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # peft default
        nn.init.zeros_(self.lora_B)  # zero init -> identity at step 0
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        x = self.drop(x)
        out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out

    def merged_weight(self) -> torch.Tensor:
        # base.weight + scaling * (B @ A) -> base.weight shape
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def _attach_inline_lora(model: nn.Module, rank: int, alpha: int,
                       targets: tuple[str, ...]) -> list[str]:
    """Walk model, replace any Linear whose attribute name is in `targets`
    with an InlineLoRALinear. Returns list of touched FQNs."""
    touched: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                wrapped = InlineLoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, child_name, wrapped)
                fqn = f"{parent_name}.{child_name}" if parent_name else child_name
                touched.append(fqn)
    # freeze everything else
    for name, p in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            p.requires_grad = False
    return touched


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet_dataset(path: str, max_seq: int) -> list[tuple[list[int], list[int]]]:
    """Load (input_ids, loss_mask) pairs from prep_alpaca_qwen output."""
    if not HAS_PYARROW:
        raise RuntimeError("pyarrow required to read alpaca parquet; pip install pyarrow")
    table = pq.read_table(path)
    df = table.to_pylist()
    out: list[tuple[list[int], list[int]]] = []
    for row in df:
        ids = list(row["input_ids"])[:max_seq]
        mask = list(row["loss_mask"])[:max_seq]
        if not ids or sum(mask) == 0:
            continue
        out.append((ids, mask))
    return out


def make_smoke_dataset(vocab: int, n: int = 64, seq: int = 64) -> list[tuple[list[int], list[int]]]:
    """Tiny synthetic data for smoke tests."""
    rng = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        ids = torch.randint(1, vocab, (seq,), generator=rng).tolist()
        # mask first 16 tokens as prompt, rest as response
        mask = [0] * 16 + [1] * (seq - 16)
        out.append((ids, mask))
    return out


def collate_batch(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    max_len = max(len(b[0]) for b in batch)
    bs = len(batch)
    ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    msk = torch.zeros((bs, max_len), dtype=torch.long)
    attn = torch.zeros((bs, max_len), dtype=torch.long)
    for i, (a, m) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        msk[i, :len(m)] = torch.tensor(m, dtype=torch.long)
        attn[i, :len(a)] = 1
    return {"input_ids": ids, "loss_mask": msk, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Smoke fake-Qwen model (so this file runs end-to-end with no Qwen ckpt)
# ---------------------------------------------------------------------------

class _SmokeAttn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # toy attention: row-norm dot-product (good enough for shape correctness)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = att.softmax(dim=-1)
        return self.o_proj(att @ v)


class SmokeQwen(nn.Module):
    """8-layer toy transformer with q_proj/k_proj/v_proj/o_proj names so
    `target_modules` matches both inline LoRA and peft."""

    def __init__(self, vocab: int = 256, d: int = 32, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_SmokeAttn(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits, "loss": None})()


class SmokeTokenizer:
    def __init__(self, vocab: int = 256):
        self.vocab_size = vocab
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"

    def encode(self, s, add_special_tokens=False, return_tensors=None):
        ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in s][:64]
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr((i - 2) % 95 + 32) for i in ids if i > 1)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "smoke_tokenizer.json").write_text(
            json.dumps({"vocab_size": self.vocab_size}), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainCfg:
    base_path: str
    data: str
    out: str
    steps: int = 5000
    bs: int = 8
    lr: float = 2e-4
    rank: int = 16
    alpha: int = 32
    max_seq: int = 1024
    warmup: int = 100
    log_every: int = 50
    sample_every: int = 500
    save_every: int = 1000
    smoke: bool = False


def cosine_lr(step: int, peak: float, total: int, warmup: int) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def cross_entropy_with_mask(logits: torch.Tensor, ids: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """Causal LM loss on positions where loss_mask==1 in the *next* token.

    logits: [B, T, V]
    ids:    [B, T]
    mask:   [B, T]    1 = supervise the prediction OF this token
    """
    # shift: predict token t+1 at position t
    sh_logits = logits[:, :-1, :].contiguous()
    sh_ids = ids[:, 1:].contiguous()
    sh_mask = mask[:, 1:].contiguous().to(sh_logits.dtype)
    flat_logits = sh_logits.view(-1, sh_logits.size(-1))
    flat_ids = sh_ids.view(-1)
    flat_mask = sh_mask.view(-1)
    losses = F.cross_entropy(flat_logits, flat_ids, reduction="none")
    denom = flat_mask.sum().clamp_min(1.0)
    return (losses * flat_mask).sum() / denom


def sample_canned(model, tok, device: str, max_new: int = 32) -> str:
    """5 EN + 5 ZH canned prompts, return joined transcript string."""
    en = [
        "Once upon a time",
        "The capital of France is",
        "def fibonacci(n):",
        "Q: Who wrote Romeo and Juliet?\nA:",
        "Translate to Chinese: Hello",
    ]
    zh = ["你好,", "中国的首都是", "请解释什么是机器学习。", "1+1等于几?", "今天天气真好,"]
    out_lines = []
    model.eval()
    with torch.no_grad():
        for p in en + zh:
            ids = tok.encode(p, add_special_tokens=False, return_tensors="pt").to(device)
            base_len = ids.size(1)
            for _ in range(max_new):
                logits = model(input_ids=ids).logits
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if int(nxt.item()) == getattr(tok, "eos_token_id", -1):
                    break
            resp = tok.decode(ids[0, base_len:], skip_special_tokens=True)
            head = p.replace("\n", " \\n ")
            out_lines.append(f"  > {head[:60]}")
            out_lines.append(f"      {resp[:200]}")
    model.train()
    return "\n".join(out_lines)


def save_artifacts(model, tok, cfg: TrainCfg, step: int, log_path: Path) -> None:
    """Save adapter (peft format if available, else numpy), merged ckpt,
    tokenizer, and config."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # adapter
    if HAS_PEFT and isinstance(model, PeftModel):
        try:
            model.save_pretrained(str(adapter_dir))
        except Exception as e:
            (adapter_dir / "save_error.txt").write_text(str(e), encoding="utf-8")
    else:
        # inline LoRA: save just the A/B matrices
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
              if "lora_A" in k or "lora_B" in k}
        torch.save(sd, adapter_dir / "lora_state.pt")

    # merged base+LoRA (for fast inference w/o peft)
    try:
        merged_state = {}
        for name, mod in model.named_modules():
            if isinstance(mod, InlineLoRALinear):
                merged_state[name + ".weight"] = mod.merged_weight().detach().cpu()
        if merged_state:
            torch.save(merged_state, out / "merged.pt")
    except Exception as e:
        (out / "merge_error.txt").write_text(str(e), encoding="utf-8")

    # tokenizer
    try:
        tok.save_pretrained(str(out / "tokenizer"))
    except Exception:
        pass

    # config
    cfg_dict = asdict(cfg)
    cfg_dict["step"] = step
    cfg_dict["framework"] = "peft" if (HAS_PEFT and isinstance(model, PeftModel)) else "inline"
    (out / "config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[save] step={step} -> {out}\n")


def train(cfg: TrainCfg) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    log_path.write_text(f"[start] {time.ctime()}\ncfg={asdict(cfg)}\n", encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() and not cfg.smoke else "cpu"
    print(f"[train] device={device} smoke={cfg.smoke} peft={HAS_PEFT}")

    # --- model + tokenizer ---
    if cfg.smoke or not HAS_TRANSFORMERS:
        if not cfg.smoke:
            print("[train] WARN: transformers unavailable, falling back to smoke model")
        model = SmokeQwen(vocab=256, d=32, n_layers=2).to(device)
        tok = SmokeTokenizer(vocab=256)
    else:
        if not Path(cfg.base_path).exists() and "/" not in cfg.base_path and "\\" not in cfg.base_path:
            # treat as HF hub id
            pass
        elif not Path(cfg.base_path).exists():
            print(f"[train] WARN: base path {cfg.base_path} missing, falling back to smoke")
            cfg.smoke = True
            model = SmokeQwen(vocab=256, d=32, n_layers=2).to(device)
            tok = SmokeTokenizer(vocab=256)
        if not cfg.smoke:
            tok = AutoTokenizer.from_pretrained(cfg.base_path, trust_remote_code=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device)

    # --- attach LoRA ---
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    if HAS_PEFT and not cfg.smoke and HAS_TRANSFORMERS:
        peft_cfg = LoraConfig(
            r=cfg.rank, lora_alpha=cfg.alpha,
            target_modules=list(targets),
            lora_dropout=0.0, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    else:
        touched = _attach_inline_lora(model, cfg.rank, cfg.alpha, targets)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[train] inline LoRA on {len(touched)} layers; trainable={n_tr:,} / total={n_all:,}")

    model.train()

    # --- data ---
    if cfg.smoke or not Path(cfg.data).exists():
        if not cfg.smoke:
            print(f"[train] WARN: data {cfg.data} missing, falling back to smoke data")
        vocab = getattr(tok, "vocab_size", 256)
        dataset = make_smoke_dataset(vocab=vocab, n=64, seq=64)
        cfg.smoke = True
    else:
        dataset = load_parquet_dataset(cfg.data, cfg.max_seq)
    n_examples = len(dataset)
    print(f"[train] dataset: {n_examples:,} examples")
    pad_id = getattr(tok, "pad_token_id", 0) or 0

    # --- optim ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # --- loop ---
    step = 0
    epoch = 0
    t0 = time.time()
    losses_window: list[float] = []
    rng = torch.Generator().manual_seed(42)
    while step < cfg.steps:
        # shuffled index list
        perm = torch.randperm(n_examples, generator=rng).tolist()
        for i in range(0, n_examples, cfg.bs):
            if step >= cfg.steps:
                break
            batch_rows = [dataset[perm[j]] for j in range(i, min(i + cfg.bs, n_examples))]
            batch = collate_batch(batch_rows, pad_id=pad_id)
            ids = batch["input_ids"].to(device)
            msk = batch["loss_mask"].to(device)
            attn = batch["attention_mask"].to(device)

            try:
                out_obj = model(input_ids=ids, attention_mask=attn)
                logits = out_obj.logits
            except TypeError:
                # smoke path doesn't accept attention_mask kw
                out_obj = model(input_ids=ids)
                logits = out_obj.logits

            loss = cross_entropy_with_mask(logits, ids, msk)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            lr_now = cosine_lr(step, cfg.lr, cfg.steps, cfg.warmup)
            for g in optim.param_groups:
                g["lr"] = lr_now
            optim.step()

            losses_window.append(float(loss.item()))
            step += 1

            if step % cfg.log_every == 0 or step == 1:
                avg = sum(losses_window[-cfg.log_every:]) / max(len(losses_window[-cfg.log_every:]), 1)
                tps = step * cfg.bs / max(time.time() - t0, 1e-3)
                line = (f"[step {step:5d}/{cfg.steps}] loss={loss.item():.4f} "
                        f"avg{cfg.log_every}={avg:.4f} lr={lr_now:.2e} "
                        f"tok/s≈{tps:.0f} epoch={epoch}")
                print(line)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

            if step % cfg.sample_every == 0:
                try:
                    text = sample_canned(model, tok, device, max_new=24 if cfg.smoke else 64)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"\n--- samples at step {step} ---\n{text}\n---\n\n")
                    print(f"[sample] step {step} dumped 10 samples to {log_path}")
                except Exception as e:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[sample-error] {e}\n")

            if step % cfg.save_every == 0:
                save_artifacts(model, tok, cfg, step, log_path)

        epoch += 1

    save_artifacts(model, tok, cfg, step, log_path)
    summary = {
        "step": step,
        "wall_time_s": time.time() - t0,
        "final_loss": losses_window[-1] if losses_window else None,
        "out": str(out),
        "smoke": cfg.smoke,
    }
    print(f"[done] {summary}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[end] {time.ctime()} {json.dumps(summary, ensure_ascii=False)}\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_qwen_lora")
    ap.add_argument("--base-path", default=os.environ.get("QWEN_BASE_PATH", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--data", default="data/sft/alpaca_combined.parquet")
    ap.add_argument("--out", default=str(Path.home() / ".synapforge" / "release" / "qwen_lora_v0"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true",
                    help="run with mock Qwen + mock data (5 steps)")
    args = ap.parse_args(argv)

    if args.smoke:
        # crank everything to tiny in smoke
        args.steps = min(args.steps, 5)
        args.bs = min(args.bs, 4)
        args.log_every = 1
        args.sample_every = 5
        args.save_every = 5

    cfg = TrainCfg(
        base_path=args.base_path, data=args.data, out=args.out,
        steps=args.steps, bs=args.bs, lr=args.lr,
        rank=args.rank, alpha=args.alpha, max_seq=args.max_seq,
        warmup=args.warmup, log_every=args.log_every,
        sample_every=args.sample_every, save_every=args.save_every,
        smoke=args.smoke,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
