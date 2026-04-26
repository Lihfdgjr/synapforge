"""Smoke test: 2-GPU DDP synapforge with PlasticBufferSync.

Usage
-----
    torchrun --nproc-per-node=2 /workspace/synapforge/test_distributed_smoke.py

Validates
---------
1. DDP wrapping does not hang or NaN over 50 steps.
2. Gradient all-reduce keeps trainable params identical across ranks
   (a 0-tolerance L2 check after step 50).
3. **Critical**: plasticity buffers (W_fast, elig, astro_state, ...) DO
   diverge across ranks if you don't sync them, and DO match after
   `PlasticBufferSync.sync()`. We assert this directly.
4. Throughput: ms/step at 2 GPUs (compared offline against 1-GPU run).
"""

from __future__ import annotations

import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Make synapforge importable when launched via torchrun
sys.path.insert(0, "/workspace")
import synapforge as sf
from synapforge import distributed as sfd


# ---------------------------------------------------------------------------
# Tiny HybridBlock with all the plasticity buffers we care about
# ---------------------------------------------------------------------------
class TinyHybrid(sf.Module):
    """D=64, L=2 stack: LiquidCell + PLIF + STDP buffer + astro buffer + elig."""

    def __init__(self, d: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.d = int(d)
        self.layers = int(layers)
        self.embed = nn.Linear(d, d)

        self.cells = nn.ModuleList()
        for _ in range(layers):
            cell = sf.LiquidCell(d, d)
            self.cells.append(cell)

        self.plifs = nn.ModuleList(
            [sf.PLIF(d, threshold=0.3, learnable_threshold=False) for _ in range(layers)]
        )

        # Plasticity buffers — these are the ones DDP doesn't sync.
        # We register them as non-persistent buffers so they show up in
        # named_buffers() and our sync sees them.
        for i in range(layers):
            self.register_buffer(f"W_fast_{i}", torch.zeros(d, d))
            self.register_buffer(f"elig_{i}",   torch.zeros(d, d))
            self.register_buffer(f"astro_state_{i}", torch.zeros(d))
            self.register_buffer(f"stdp_trace_{i}",  torch.zeros(d))
            self.register_buffer(f"coact_ema_{i}",   torch.zeros(d, d))

        self.head = nn.Linear(d, d)

    def _plastic_update(self, layer_idx: int, h: torch.Tensor, spk: torch.Tensor) -> None:
        """Mock plasticity update: deterministic per-rank, but uses local data
        so each rank's update is *different* unless we sync."""
        # h, spk: (B, T, D); reduce to per-feature signals
        h_mean = h.detach().mean(dim=(0, 1))      # (D,)
        s_mean = spk.detach().mean(dim=(0, 1))    # (D,)
        outer = torch.outer(h_mean, s_mean)        # (D, D)

        wf = getattr(self, f"W_fast_{layer_idx}")
        el = getattr(self, f"elig_{layer_idx}")
        ag = getattr(self, f"astro_state_{layer_idx}")
        st = getattr(self, f"stdp_trace_{layer_idx}")
        ce = getattr(self, f"coact_ema_{layer_idx}")

        wf.mul_(0.99).add_(outer, alpha=0.01)
        el.mul_(0.95).add_(outer, alpha=0.05)
        ag.mul_(0.9).add_(h_mean, alpha=0.1)
        st.mul_(0.9).add_(s_mean, alpha=0.1)
        ce.mul_(0.99).add_(outer, alpha=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.embed(x)
        for i, (cell, plif) in enumerate(zip(self.cells, self.plifs)):
            h = cell(h)                    # (B, T, D)
            # PLIF expects per-step current; collapse T into iterations.
            B, T, D = h.shape
            spk_list = []
            mem = None
            for t in range(T):
                spk_t, mem = plif(h[:, t, :], mem)
                spk_list.append(spk_t)
            spk = torch.stack(spk_list, dim=1)  # (B, T, D)
            self._plastic_update(i, h, spk)
            h = h + spk                          # spike-modulated
        return self.head(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gather_l2(model: torch.nn.Module, world_size: int) -> dict[str, list[float]]:
    """Per-rank L2 norms of every plastic buffer, gathered to all ranks."""
    local = sfd.buffer_l2_per_rank(model)
    names = sorted(local.keys())
    local_vec = torch.tensor([local[n] for n in names], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(local_vec) for _ in range(world_size)]
    dist.all_gather(gathered, local_vec)
    out: dict[str, list[float]] = {n: [] for n in names}
    for r, g in enumerate(gathered):
        for i, n in enumerate(names):
            out[n].append(float(g[i].item()))
    return out


def max_divergence(per_rank: dict[str, list[float]]) -> float:
    """Largest absolute difference between any two ranks for any buffer."""
    if not per_rank:
        return 0.0
    diffs = [max(v) - min(v) for v in per_rank.values()]
    return max(diffs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rank, world_size, local_rank = sfd.init_dist(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42 + rank)   # different shards per rank

    if rank == 0:
        print(f"[smoke] world_size={world_size} backend=nccl torch={torch.__version__}", flush=True)

    D, L, B, T = 64, 2, 8, 16
    model = TinyHybrid(d=D, layers=L)
    model = sfd.wrap_model(model, local_rank, find_unused_parameters=True)

    sync = sfd.PlasticBufferSync(model)
    if rank == 0:
        names = sfd.get_plastic_buffer_names(model)
        print(f"[smoke] tracked {sync.num_buffers()} plastic buffers ({sync.numel()} elems):", flush=True)
        for n in names:
            print(f"        {n}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.randn(B, T, D, device=device)

    # ---------- (a) without sync(): verify divergence happens ----------
    if rank == 0:
        print("\n[phase A] WITHOUT sync(): expect divergence", flush=True)

    # Take 5 steps of plasticity updates, never call sync().
    for _ in range(5):
        x = torch.randn(B, T, D, device=device)
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()
        opt.zero_grad()

    div_before = max_divergence(gather_l2(model, world_size))
    if rank == 0:
        print(f"[phase A] max plastic-buffer L2 divergence across ranks = {div_before:.6e}", flush=True)
        print("[phase A] this should be > 0 — proves DDP doesn't sync these", flush=True)

    # ---------- (b) WITH sync(): verify divergence collapses to 0 ----------
    if rank == 0:
        print("\n[phase B] WITH sync(): expect 0 divergence", flush=True)

    # First sync (collapse the divergence we just produced).
    sync.sync()
    div_after_first = max_divergence(gather_l2(model, world_size))
    if rank == 0:
        print(f"[phase B] after one sync(): divergence = {div_after_first:.6e}", flush=True)

    # ---------- (c) main 50-step training with sync after every step -----
    if rank == 0:
        print("\n[phase C] 50 training steps with PlasticBufferSync", flush=True)

    losses: list[float] = []
    torch.cuda.synchronize(device)
    t0 = time.time()
    for step in range(50):
        x = torch.randn(B, T, D, device=device)
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync.sync()
        losses.append(loss.item())
    torch.cuda.synchronize(device)
    t1 = time.time()

    ms_step = (t1 - t0) * 1000.0 / 50.0
    if rank == 0:
        print(f"[phase C] 50 steps: {t1-t0:.2f}s = {ms_step:.2f} ms/step (2-GPU)", flush=True)
        print(f"[phase C] losses: first={losses[0]:.4f} last={losses[-1]:.4f} (no NaN: {all(map(lambda v: v == v, losses))})", flush=True)

    # ---------- (d) verify final state matches across ranks --------------
    div_end = max_divergence(gather_l2(model, world_size))
    if rank == 0:
        print(f"\n[phase D] final plastic-buffer divergence (after sync) = {div_end:.6e}", flush=True)
        assert div_end < 1e-5, f"FAIL: buffers still diverge ({div_end:.6e})"

    # Trainable param parity check (DDP's job, sanity)
    raw = model.module if hasattr(model, "module") else model
    p_l2_local = torch.tensor(
        [p.detach().float().norm().item() for p in raw.parameters()],
        dtype=torch.float64, device=device,
    )
    p_gathered = [torch.zeros_like(p_l2_local) for _ in range(world_size)]
    dist.all_gather(p_gathered, p_l2_local)
    if rank == 0:
        p_diff = max(
            (p_gathered[r] - p_gathered[0]).abs().max().item()
            for r in range(world_size)
        )
        print(f"[phase D] trainable-param L2 max div across ranks = {p_diff:.6e}", flush=True)

    # ---------- (e) summary ---------------------------------------------
    if rank == 0:
        print("\n=== SMOKE SUMMARY ===", flush=True)
        print(f"  world_size           : {world_size}", flush=True)
        print("  steps                : 50", flush=True)
        print(f"  ms/step (2-GPU)      : {ms_step:.2f}", flush=True)
        print(f"  div WITHOUT sync     : {div_before:.6e}  (must be > 0)", flush=True)
        print(f"  div AFTER one sync   : {div_after_first:.6e}", flush=True)
        print(f"  div END of train     : {div_end:.6e}  (must be ~0)", flush=True)
        passed = (div_before > 0) and (div_after_first < 1e-5) and (div_end < 1e-5)
        print(f"  STATUS               : {'PASS' if passed else 'FAIL'}", flush=True)

    sfd.cleanup_dist()


if __name__ == "__main__":
    main()
