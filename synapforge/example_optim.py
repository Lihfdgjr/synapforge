"""End-to-end example: sf.optim + sf.plasticity wired into a hybrid model.

Two parameter groups in one model:
  * `proj`: a vanilla nn.Linear (BP-only)
  * `syn`:  a sf.SparseSynapse whose weight is BP+STDP (sf.Param tagged)

We run a short training loop where:
  - loss.backward() populates BP grads on every requires_grad parameter
  - an STDP rule emits ΔW for `syn.weight` every step (attached via the
    ms_param_table on the optimizer)
  - opt.step() merges both streams atomically — no version conflicts, no NaN

Both update streams should be active every step; we assert each non-zero.
"""
from __future__ import annotations

import torch

import synapforge as sf
from synapforge.optim import build_optimizer


class HybridBlock(sf.Module):
    """tiny hybrid: BP-only Linear → SparseSynapse (BP+STDP)."""

    def __init__(self, d: int = 32, sparsity: float = 0.5) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d, d)  # BP-only, default
        self.syn = sf.SparseSynapse(d, d, sparsity=sparsity, bias=False)
        # Re-tag the SparseSynapse weight with multi-source metadata.
        # We can't replace the Parameter object on a registered module without
        # losing the registration, but we CAN attach the marker attributes:
        self.syn.weight._sf_grad_source = ["bp", "stdp"]
        self.syn.weight._sf_weight_per_source = {"bp": 1.0, "stdp": 0.05}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(x))
        return self.syn(h)


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    d, batch, n_steps = 32, 64, 100

    model = HybridBlock(d=d, sparsity=0.5).to(dev)
    opt = build_optimizer(model, lr=3e-3, weight_decay=0.01)

    # Pull the MultiSourceParam wrapper for the synapse weight so the STDP
    # rule can attach its delta directly:
    syn_msp = opt.get_ms_param(model.syn.weight)
    assert syn_msp is not None and "stdp" in syn_msp.sources

    # Build an STDP rule on the right dimension (acts on syn input/output):
    stdp = sf.HebbianPlasticity(dim=d, eta=0.01).to(dev)  # legacy buffer trace (diagnostic)

    # Synthetic regression target
    X = torch.randn(batch, d, device=dev)
    A = torch.randn(d, d, device=dev) * 0.3
    y_target = X @ A

    # Track that BOTH update streams are non-zero
    bp_norms, stdp_norms, losses = [], [], []

    for step in range(n_steps):
        opt.zero_grad()
        out = model(X)
        loss = ((out - y_target) ** 2).mean()
        loss.backward()

        # --- STDP-style plasticity stream ---
        # Pre = output of proj (post-tanh), Post = output of syn.
        # In a real bio model, pre/post are spike trains; here we use the
        # continuous activations as analog "rates" — the STDP rule still
        # produces a valid (out_dim, in_dim) ΔW.
        with torch.no_grad():
            pre_act = torch.tanh(model.proj(X))           # (batch, d)
            post_act = model.syn(pre_act)                  # (batch, d)
            # Standard outer-product Hebbian-style ΔW (analog STDP proxy)
            delta_W = 0.001 * (post_act.t() @ pre_act) / batch
            syn_msp.attach_plast_delta("stdp", delta_W)
            # Also tick the buffer-only STDP rule so its W_fast trace evolves
            # (purely diagnostic — not consumed by optim.step here):
            stdp(pre_act.float(), post_act.float())  # buffer-only Hebbian, not consumed by optim

        # Snapshot the contributions BEFORE step() resets the caches
        bp_grad = model.syn.weight.grad
        bp_norm = bp_grad.norm().item() if bp_grad is not None else 0.0
        stdp_norm = syn_msp.plast_delta.get(
            "stdp", torch.zeros(1, device=dev)
        ).norm().item()
        bp_norms.append(bp_norm)
        stdp_norms.append(stdp_norm)

        opt.step()
        losses.append(loss.item())

    # ----- assertions: both streams active, loss decreased -----
    avg_bp = sum(bp_norms) / len(bp_norms)
    avg_stdp = sum(stdp_norms) / len(stdp_norms)
    assert avg_bp > 1e-4, f"BP grad stream empty: avg_norm={avg_bp:.2e}"
    assert avg_stdp > 1e-6, f"STDP delta stream empty: avg_norm={avg_stdp:.2e}"
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not converge: {losses[0]:.3f} → {losses[-1]:.3f}"
    )

    print(f"[example] device              = {dev}")
    print(f"[example] sparsity (active)   = {model.syn.density():.3f}")
    print(f"[example] avg BP grad norm    = {avg_bp:.4e}")
    print(f"[example] avg STDP delta norm = {avg_stdp:.4e}")
    print(f"[example] loss {losses[0]:.4f}  ->  {losses[-1]:.4f}  "
          f"(drop {100*(1-losses[-1]/losses[0]):.1f}%)")
    print(f"[example] sf.Hebbian buffer W_fast ‖ = {stdp.W_fast.norm().item():.4f}")
    print("[example] both update streams active, mixed-source training OK")


if __name__ == "__main__":
    main()
