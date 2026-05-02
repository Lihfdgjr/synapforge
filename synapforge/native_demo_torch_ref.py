"""native_demo_torch_ref.py -- torch reference for ``native_demo.py``.

This is the SAME architecture and the SAME synthetic-data generator as
``native_demo.py``, but written using ``torch.nn.Module`` +
``torch.optim.AdamW``. We use it as the ground-truth oracle for the
final-loss accuracy gate (native within 5% of torch on the same seed).

The numerical match isn't guaranteed bit-for-bit (numpy uses fp32
everywhere; torch defaults to fp32 too but cudnn/blas paths can
differ), but on the same seed + same shapes the final loss should
land within ~5%.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Match native_demo.py
VOCAB = 256
D_MODEL = 64
N_LAYERS = 2
SEQ_LEN = 16
BATCH = 4
FFN_RATIO = 4
N_STEPS = 100
LR = 3e-3
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.95
ADAMW_EPS = 1e-8
ADAMW_WD = 0.01
SEED = 1234

PLIF_ALPHA = 2.0
PLIF_THRESHOLD = 0.3
PLIF_TAU_INIT = 1.5


# ---------------------------------------------------------------------------
# Synthetic data -- byte-identical generator to native_demo.py
# ---------------------------------------------------------------------------

def synth_batch(rng: np.random.Generator, batch: int, seq_len: int,
                vocab: int) -> Tuple[np.ndarray, np.ndarray]:
    """See ``native_demo.synth_batch`` for the exact rule."""
    x = rng.integers(0, vocab, size=(batch, seq_len), dtype=np.int64)
    y = np.empty_like(x)
    for t in range(seq_len):
        if t == 0:
            y[:, t] = (x[:, t] + 1) % vocab
        elif t % 2 == 0:
            y[:, t] = (x[:, t] + 1) % vocab
        else:
            y[:, t] = x[:, t - 1]
    return x, y


# ---------------------------------------------------------------------------
# torch building blocks
# ---------------------------------------------------------------------------

class _ATanSurrogate(torch.autograd.Function):
    """ATan surrogate for the spike indicator (Fang et al. 2021).

    Forward: hard indicator (mem >= thr).
    Backward: alpha / (2 * (1 + (pi/2 * alpha * x)^2)).
    """

    @staticmethod
    def forward(ctx, mem: torch.Tensor, thr: torch.Tensor,
                alpha: float) -> torch.Tensor:
        ctx.save_for_backward(mem, thr)
        ctx.alpha = float(alpha)
        return (mem >= thr).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mem, thr = ctx.saved_tensors
        alpha = ctx.alpha
        x = alpha * (mem - thr)
        surrogate = alpha / (2.0 * (1.0 + (math.pi / 2.0 * x).pow(2)))
        grad_mem = grad_output * surrogate
        # grad_thr broadcast over leading dims:
        grad_thr = -(grad_output * surrogate).reshape(-1, mem.shape[-1]).sum(0)
        return grad_mem, grad_thr, None


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sq = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(sq + self.eps)
        return self.weight * x * rstd


class LiquidCellRef(nn.Module):
    """Sequential CfC matching ``cfc_step_fwd`` in native_demo."""

    def __init__(self, in_d: int, hidden_d: int) -> None:
        super().__init__()
        self.in_d = in_d
        self.hidden_d = hidden_d
        self.W_delta = nn.Parameter(torch.empty(hidden_d, in_d))
        self.W_b = nn.Parameter(torch.empty(hidden_d, in_d))
        self.A_log = nn.Parameter(torch.empty(hidden_d))
        nn.init.normal_(self.W_delta, std=0.02)
        nn.init.normal_(self.W_b, std=0.02)
        nn.init.uniform_(self.A_log, math.log(0.5), math.log(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_d) -> out: (B, T, hidden_d)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_d, dtype=x.dtype, device=x.device)
        outs = []
        expA = torch.exp(self.A_log)
        for t in range(T):
            xt = x[:, t]
            delta_in = xt @ self.W_delta.t()
            delta_t = F.softplus(delta_in)
            A_t = torch.exp(-delta_t * expA)
            B_t = delta_t * (xt @ self.W_b.t())
            h = A_t * h + B_t
            outs.append(torch.tanh(h))
        return torch.stack(outs, dim=1)


class PLIFRef(nn.Module):
    """PLIF matching ``plif_step_fwd`` in native_demo."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.tau_log = nn.Parameter(torch.full((d,), math.log(PLIF_TAU_INIT)))
        self.threshold = nn.Parameter(torch.full((d,), PLIF_THRESHOLD))

    def forward(self, current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # current: (B, T, d) -> spike: (B, T, d)
        B, T, d = current.shape
        mem = torch.zeros(B, d, dtype=current.dtype, device=current.device)
        outs = []
        tau = torch.exp(self.tau_log)
        decay = torch.exp(-dt / tau)
        for t in range(T):
            mem_pre = mem * decay + current[:, t]
            spike = _ATanSurrogate.apply(mem_pre, self.threshold, PLIF_ALPHA)
            mem = mem_pre - spike * self.threshold
            outs.append(spike)
        return torch.stack(outs, dim=1)


class SwiGLU(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class HybridBlockRef(nn.Module):
    def __init__(self, d: int, ffn_h: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.cfc = LiquidCellRef(d, d)
        self.plif = PLIFRef(d)
        self.syn = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.syn.weight, std=0.02)
        self.ln2 = RMSNorm(d)
        self.ffn = SwiGLU(d, ffn_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.ln1(x)
        h = self.cfc(a)
        s = self.plif(h)
        x = x + self.syn(s)
        x = x + self.ffn(self.ln2(x))
        return x


class SynapForgeTinyRef(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL,
                 n_layers: int = N_LAYERS, ffn_ratio: int = FFN_RATIO) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([
            HybridBlockRef(d, ffn_ratio * d) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(n_steps: int = N_STEPS, seed: int = SEED) -> Dict[str, Any]:
    print(f"[torch_ref] init model seed={seed} d={D_MODEL} "
          f"n_layers={N_LAYERS} V={VOCAB} T={SEQ_LEN} B={BATCH}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SynapForgeTinyRef()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[torch_ref] n_params={n_params:,}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=LR, betas=(ADAMW_BETA1, ADAMW_BETA2),
        eps=ADAMW_EPS, weight_decay=ADAMW_WD,
    )
    rng = np.random.default_rng(seed + 1)

    losses: List[float] = []
    step_times: List[float] = []
    t_total_start = time.time()
    for step in range(n_steps):
        t0 = time.time()
        x_np, y_np = synth_batch(rng, BATCH, SEQ_LEN, VOCAB)
        x = torch.from_numpy(x_np).long()
        y = torch.from_numpy(y_np).long()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        dt = time.time() - t0
        step_times.append(dt)
        losses.append(float(loss.item()))
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[torch_ref] step={step + 1:3d}/{n_steps} "
                  f"loss={loss.item():.4f} dt={dt * 1000:.1f}ms")

    total_dt = time.time() - t_total_start
    win = 10
    rolling = []
    for i in range(len(losses) - win + 1):
        rolling.append(float(np.mean(losses[i:i + win])))
    first_half = float(np.mean(rolling[:len(rolling) // 2])) if rolling else float("nan")
    second_half = float(np.mean(rolling[len(rolling) // 2:])) if rolling else float("nan")

    results = dict(
        impl="torch_reference",
        seed=seed,
        n_steps=n_steps,
        n_params=int(n_params),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        vocab=VOCAB,
        seq_len=SEQ_LEN,
        batch=BATCH,
        losses=losses,
        rolling_10=rolling,
        first_half_mean=first_half,
        second_half_mean=second_half,
        loss_decreased_monotonically=bool(second_half < first_half),
        final_loss=float(losses[-1]),
        wall_time_sec=float(total_dt),
        ms_per_step_mean=float(np.mean(step_times) * 1000),
        ms_per_step_p50=float(np.median(step_times) * 1000),
    )
    out_path = Path(__file__).resolve().parent / "_native_demo_torch_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[torch_ref] wrote {out_path}")
    print(f"[torch_ref] final_loss={results['final_loss']:.4f} "
          f"first_half={first_half:.4f} second_half={second_half:.4f} "
          f"wall={total_dt:.1f}s ms/step={results['ms_per_step_mean']:.1f}")
    return results


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    train()
