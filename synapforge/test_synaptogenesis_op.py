"""test_synaptogenesis_op — verify M4 (Synaptogenesis as compile-time op).

Two-part check:

  PART 1  IR structure:
          compile_module(model) emits grow_op + prune_op IRNodes for
          every SparseSynapse(growth=...). Without growth, no extra nodes.

  PART 2  Mask evolution under training:
          Train a tiny model with SparseSynapse(growth=RigL(...)) for
          100 steps. After every step, call maybe_update_masks(graph, step).
          Verify density tracks the target and that connectivity actually
          churns (Hamming distance from initial mask is non-trivial).

Pass criteria
-------------
  * grow_op + prune_op nodes present in trace
  * density at step 100 within [target - 0.05, target + 0.05]
  * Hamming distance(mask_0, mask_100) >= 5% of mask.numel()
  * legacy: SparseSynapse(W) with no growth kwarg still works
"""

from __future__ import annotations

import os
import sys

import torch

# Use GPU 1 only per task constraints.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/workspace")

import synapforge as sf
from synapforge.ir import RigL, compile_module, maybe_update_masks


# ---------------------------------------------------------------------------
# Tiny model: 2 SparseSynapse layers, one with RigL, one without.
# ---------------------------------------------------------------------------
class TinyModel(sf.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = sf.SparseSynapse(
            in_dim=64, out_dim=128, sparsity=0.10,
            growth=RigL(target=0.15, period=10, drop_frac=0.30, decay_steps=200),
        )
        # Note: l2 has NO growth — backward compat smoke test.
        self.l2 = sf.SparseSynapse(in_dim=128, out_dim=32, sparsity=0.20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    torch.manual_seed(7)
    model = TinyModel().to(device)
    x = torch.randn(16, 64, device=device)
    target = torch.randn(16, 32, device=device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    # ---- Part 1: compile and inspect IR ----
    graph = compile_module(model)
    print("\n[graph trace]")
    print(graph.summary())

    grow_nodes = [n for n in graph.nodes if n.op == "grow_op"]
    prune_nodes = [n for n in graph.nodes if n.op == "prune_op"]
    assert len(grow_nodes) == 1, f"expected 1 grow_op (l1 only), got {len(grow_nodes)}"
    assert len(prune_nodes) == 1, f"expected 1 prune_op (l1 only), got {len(prune_nodes)}"
    g0 = grow_nodes[0]
    p0 = prune_nodes[0]
    assert g0.attrs.get("rule") == "rigl"
    assert g0.attrs.get("synapse") == "l1"
    assert g0.attrs.get("criterion") == "gradient_magnitude"
    assert p0.attrs.get("criterion") == "weight_magnitude"
    assert g0.inputs == ["l1"]
    print("\n[part 1] OK -- grow_op + prune_op visible as IR nodes")
    print(f"  grow_op:  {g0}")
    print(f"  prune_op: {p0}")

    # ---- Idempotency: rerun pass, no extra nodes ----
    from synapforge.ir.synaptogenesis import SynaptogenesisPass
    graph2 = SynaptogenesisPass().run(graph)
    assert len([n for n in graph2.nodes if n.op == "grow_op"]) == 1, "pass not idempotent"
    print("[idempotent] OK -- pass safe to re-run")

    # ---- Part 2: train 100 steps, track density ----
    initial_mask = model.l1.mask.clone()
    initial_density = float(initial_mask.float().mean().item())
    print("\n[part 2] training 100 steps, target density=0.15")
    print(f"  step   0  density={initial_density:.4f}")

    densities = [(0, initial_density)]
    fired_records = []
    n_steps = 100
    for step in range(1, n_steps + 1):
        out = model(x)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        # M4: structural plasticity is now an IR-level op fired between steps.
        fired = maybe_update_masks(graph, step)
        if fired:
            for f in fired:
                fired_records.append(f)
                densities.append((step, f["density_after"]))
        if step % 20 == 0 or step == 1:
            d_now = model.l1.density()
            print(f"  step {step:3d}  density={d_now:.4f}  loss={loss.item():.4f}")

    final_mask = model.l1.mask.clone()
    final_density = float(final_mask.float().mean().item())
    hamming = int((initial_mask != final_mask).sum().item())
    hamming_frac = hamming / final_mask.numel()

    print("\n[results]")
    print(f"  initial density:  {initial_density:.4f}")
    print(f"  final   density:  {final_density:.4f}  (target=0.15)")
    print(f"  rigl fires count: {len(fired_records)}")
    print(f"  hamming(mask_0,mask_100) = {hamming} / {final_mask.numel()}"
          f" = {hamming_frac:.3%}")

    # Density convergence
    assert 0.10 <= final_density <= 0.20, (
        f"density {final_density} drifted outside [0.10, 0.20]"
    )
    # Connectivity churn
    assert hamming_frac >= 0.04, (
        f"mask only churned {hamming_frac:.1%}; growth rule is silent"
    )
    # RigL fired the right number of times (period=10 ⇒ ~10 fires over 100 steps)
    assert len(fired_records) >= 8, f"expected ~10 fires, got {len(fired_records)}"
    print("\n[part 2] OK -- mask evolves, density tracks target")

    # ---- Density evolution table ----
    print("\n[density evolution]")
    print(f"  {'step':>5} | {'density':>8} | {'pruned':>6} | {'grown':>5}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*6} | {'-'*5}")
    for rec in fired_records[:15]:
        print(f"  {rec['step']:>5d} | {rec['density_after']:>8.4f} | "
              f"{rec['n_pruned']:>6d} | {rec['n_grown']:>5d}")

    # ---- Backward compat: synapse without growth ----
    legacy = sf.SparseSynapse(in_dim=8, out_dim=16, sparsity=0.10)
    out_legacy = legacy(torch.randn(2, 8))
    assert out_legacy.shape == (2, 16)
    g_legacy = compile_module(legacy)
    assert all(n.op != "grow_op" for n in g_legacy.nodes), (
        "synapse without growth should not emit grow_op"
    )
    print("[legacy] OK -- SparseSynapse without growth kwarg still works")

    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
