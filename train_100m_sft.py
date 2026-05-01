"""train_100m_sft -- SFT (instruction-tune) of synapforge_100m on alpaca parquet.

Trimmed copy of train_100m_kd.py:
  * No KD teacher (alpaca SFT doesn't need one; KD already happened in pretrain).
  * Loads SFT parquet with input_ids + loss_mask columns (see scripts/prep_alpaca_qwen.py).
  * Response-only loss: cross_entropy with ignore_index=-100 over masked labels.
  * Same SynapForge100M architecture (vocab=151643 d=512 n_layers=10 loop_depth=1)
    -- must match pretrain to load warmstart cleanly.
  * Smaller LR (1e-5), shorter run (2-4k steps).

Pad with eos_token_id to seq_len; loss_mask 0 over pad too.

CLI:
    python3 train_100m_sft.py \
        --warmstart /workspace/runs/v24h_qwen/best_step_XXXX.pt \
        --sft-parquet /workspace/data/alpaca_sft/alpaca.parquet \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --out /workspace/runs/v24h_qwen_sft \
        --backend triton_block --batch-size 16 --steps 3000 --lr 1e-5
"""

from __future__ import annotations

import os as _os_early
_os_early.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import math
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

from synapforge.surrogate import PLIFCell  # noqa: E402
from synapforge.huggingface_adapter import adv_warmstart  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.optim import build_optimizer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SEQ_LEN = 1024


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/runs/v24h_qwen_sft")
    p.add_argument("--backend", default="triton_block",
                   choices=["gpu_dense", "triton_block"])
    p.add_argument("--warmstart", required=True,
                   help="pretrain ckpt (e.g. /workspace/runs/v24h_qwen/best_*.pt)")
    p.add_argument("--sft-parquet", required=True,
                   help="output of scripts/prep_alpaca_qwen.py")
    p.add_argument("--tokenizer-path", required=True,
                   help="for eos_token_id and chat sample")
    p.add_argument("--vocab", type=int, default=151643)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--loop-depth", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SFTBatcher:
    """Stream input_ids+loss_mask rows from parquet, pad to a common length per batch.

    Pads with `pad_id`; loss_mask 0 over pad. Right-pad. Truncates to max_seq.
    """

    def __init__(self, parquet_path: str, batch_size: int, max_seq: int,
                 pad_id: int, seed: int = 0):
        import pyarrow.parquet as pq
        self.pq = pq
        self.path = parquet_path
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.pad_id = int(pad_id)
        self.seed = seed
        # Load whole table -- alpaca is ~50K rows, fits in RAM.
        table = pq.read_table(parquet_path)
        self.ids = [list(x) for x in table["input_ids"].to_pylist()]
        self.msk = [list(x) for x in table["loss_mask"].to_pylist()]
        self.n = len(self.ids)
        _log(f"[sft-data] loaded {self.n:,} examples from {parquet_path}")
        self._order = list(range(self.n))
        random.Random(seed).shuffle(self._order)
        self._cursor = 0

    def _next_idx(self) -> int:
        if self._cursor >= self.n:
            random.Random(self.seed + self._cursor).shuffle(self._order)
            self._cursor = 0
        i = self._order[self._cursor]
        self._cursor += 1
        return i

    def __iter__(self):
        return self

    def __next__(self):
        ids_b, msk_b = [], []
        Lmax = 0
        for _ in range(self.batch_size):
            i = self._next_idx()
            ids = self.ids[i][: self.max_seq]
            msk = self.msk[i][: self.max_seq]
            ids_b.append(ids)
            msk_b.append(msk)
            Lmax = max(Lmax, len(ids))
        # right-pad
        for k in range(len(ids_b)):
            pad = Lmax - len(ids_b[k])
            ids_b[k] = ids_b[k] + [self.pad_id] * pad
            msk_b[k] = msk_b[k] + [0] * pad
        x = torch.tensor(ids_b, dtype=torch.long)
        m = torch.tensor(msk_b, dtype=torch.long)
        return x, m


def lr_at(step, peak, warmup, total):
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    _log(f"device={DEVICE} dtype={DTYPE} out={args.out} backend={args.backend}")
    _log(f"steps={args.steps} bs={args.batch_size} seq={args.max_seq} lr={args.lr}")

    # --- model (must mirror pretrain config) ---
    model = build_synapforge_100m(
        vocab=args.vocab, d=args.d, n_layers=args.n_layers,
        loop_depth=args.loop_depth, max_seq=args.max_seq,
        ffn_ratio=8.0, sparsity=0.95, dropout=0.0,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    n_params = model.num_parameters()
    _log(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- warmstart from pretrain (REQUIRED for SFT to make sense) ---
    if not os.path.exists(args.warmstart):
        _log(f"FATAL: warmstart {args.warmstart!r} not found -- SFT without "
             f"pretrain will not produce chat-quality output.")
        return 2
    # Mirror the kd trainer's remap so we accept either modern (.liquid./
    # .tok_embed.) or legacy (.cfc./.embed.text_embed.) ckpt key naming.
    rep = adv_warmstart(
        model, args.warmstart,
        name_map=[
            (r"\.cfc\.", ".liquid."),
            (r"\.embed\.text_embed\.", ".tok_embed."),
        ],
    )
    _log(f"warmstart matched={rep.matched}/{rep.total_target} "
         f"missing={len(rep.missing)} extra={len(rep.extra)}")
    model = model.to(DEVICE)
    model.train()

    plif_cells = [m for m in model.modules() if isinstance(m, PLIFCell)]

    # --- backend ---
    if args.backend == "triton_block":
        try:
            from synapforge.backends.triton_block import TritonBlockBackend
            from synapforge.backends.triton_block_kernel import _HAS_TRITON
            be = TritonBlockBackend()
            stats = be.compile(model)
            _log(f"[backend] triton_block: avail={_HAS_TRITON} pairs={stats.get('n_pairs_fused', 0)}")
        except Exception as e:
            _log(f"[backend] triton_block FAILED: {e}; falling back to gpu_dense")
            args.backend = "gpu_dense"

    # --- optimizer + warmstart momentum if present ---
    optim = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    try:
        ck = torch.load(args.warmstart, map_location="cpu")
        if isinstance(ck, dict) and "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
            _log("warmstart: loaded optim_state from pretrain ckpt")
    except Exception as e:
        _log(f"warmstart optim load skipped: {e}")

    # --- tokenizer (just for pad_id) ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    # --- data ---
    train_iter = SFTBatcher(args.sft_parquet, args.batch_size, args.max_seq,
                            pad_id=pad_id, seed=args.seed)

    t0 = time.time()
    cum_tok = 0
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        x, mask = next(train_iter)
        # Compute response-token count BEFORE the device move so we don't
        # need a GPU->CPU sync barrier each step just for cum_tok.
        # mask[:, 1:] is the slice that corresponds to label_mask after the
        # next-token shift inside the forward path.
        n_resp_step = int(mask[:, 1:].sum().item())
        x = x.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        # next-token LM: predict x[:, t+1] from x[:, :t+1]
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE,
                                enabled=DEVICE == "cuda"):
            logits = model(x[:, :-1])  # (B, T-1, V)
            labels = x[:, 1:].clone()
            label_mask = mask[:, 1:]
            # response-only: -100 where mask==0
            labels = labels.masked_fill(label_mask == 0, -100)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
        optim.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0
        cum_tok += n_resp_step

        if step % args.log_every == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            mem_str = (
                f" mem_GB={torch.cuda.memory_allocated()/1e9:.2f}"
                if DEVICE == "cuda" else ""
            )
            _log(f"step {step:5d} loss={loss.item():.4f} lr={cur_lr:.2e} "
                 f"step_ms={step_ms:.1f} resp_tok/s={tok_s:.0f}{mem_str}")

        # ckpt + best tracking
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "optim_state": optim.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "config": {
                    "vocab": args.vocab, "d": args.d, "n_layers": args.n_layers,
                    "loop_depth": args.loop_depth, "max_seq": args.max_seq,
                },
            }
            cur = float(loss.item())
            if cur < best_loss:
                best_loss = cur
                best_path = os.path.join(args.out, f"best_step_{step:06d}.pt")
                torch.save(ckpt, best_path)
                _log(f"saved BEST ckpt {best_path} (loss={cur:.4f})")
            torch.save(ckpt, os.path.join(args.out, f"step_{step:06d}.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
