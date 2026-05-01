# MoE / Chain-of-Experts (CoE) Audit — T2.10

**Date**: 2026-05-02 01:38 (cron fire)
**Trainer in scope**: `train_100m_kd.py`
**Production model factory**: `synapforge.model_100m.build_synapforge_100m`

---

## Verdict: **ORPHAN**

`ChainOfExperts` / `attach_coe_to_block` are fully implemented as library
primitives in `synapforge/routers/coe.py` and re-exported from
`synapforge/__init__.py`, but they are **NOT** instantiated by
`build_synapforge_100m` and **NOT** imported by `train_100m_kd.py`. The 100M
model the production trainer actually constructs has **zero**
`ChainOfExperts` / `DeepSeekMoE` / `FineGrainedExpert` / `SharedExpertGroup`
/ `TopKRouter` modules in its `.modules()` graph.

Task #113 (`adv28-#1: CoE 每步路由实装`) was finished in the legacy `adv28`
trainer but **never carried into the live `train_100m_kd.py` pipeline**.

---

## Smoke check (per T2.10 brief)

Command run with project venv (`.venv/Scripts/python.exe`, torch 2.5.1+cu121),
from `D:/ai_tool/investor_demo/synapforge`:

```python
from synapforge.model_100m import build_synapforge_100m
m = build_synapforge_100m(d=128, n_layers=2, vocab=512)
types = [
    type(x).__name__ for x in m.modules()
    if 'expert' in type(x).__name__.lower()
    or 'moe'    in type(x).__name__.lower()
    or 'coe'    in type(x).__name__.lower()
    or 'router' in type(x).__name__.lower()
]
print('expert/moe/coe/router modules in build_synapforge_100m:', types)
from synapforge.routers.coe import ChainOfExperts, attach_coe_to_block
print('CoE module imports OK; ChainOfExperts:', ChainOfExperts)
print('top-level children:', [n for n, _ in m.named_children()])
```

Output:

```
expert/moe/coe/router modules in build_synapforge_100m: []
CoE module imports OK; ChainOfExperts: <class 'synapforge.routers.coe.ChainOfExperts'>
top-level children: ['tok_embed', 'blocks', 'ln_f']
```

The empty `[]` confirms the orphan. The library import succeeds (so the code
exists and is loadable) but the live model factory ignores it.

---

## File:line evidence

### A. Library exists (orphan code)
- `synapforge/routers/coe.py:51-152`  — `ChainOfExperts` class definition
  (per-step routers + shared expert pool, arXiv 2506.18945).
- `synapforge/routers/coe.py:213-242` — `attach_coe_to_block(block, hidden,
  n_routed, n_shared, top_k, n_steps, aux_alpha)` factory.
- `synapforge/routers/coe.py:165-210` — `_BlockWithCoE` composition wrapper
  with cross-step aux-loss accumulation, drives `_coe_step_t` set by
  `RDTLoop` before each step.
- `synapforge/routers/__init__.py:33,59-60` — re-export.
- `synapforge/__init__.py:202,205,219,226-230,236-241` — package-level
  re-export at `sf.routers.ChainOfExperts` and back-compat alias
  `ChainOfExpertsMoE`.
- `synapforge/test_routers.py:35-41,165,205,303` — standalone unit tests
  exist for the primitive (CoE attach test on a synthetic body).

### B. Live trainer does NOT use it
- `train_100m_kd.py` (1696 LOC) — full grep on
  `chain_of_experts | CoE | top_k_routing | n_experts | expert_router |
  MoE | router | Router | expert | Expert | coe | moe | chain | Chain`
  returns **0 hits** (case-insensitive).
- `synapforge/model_100m.py:146-274` — `class SynapForge100M.__init__`
  builds:
    - `self.tok_embed` (nn.Embedding)
    - `self.pos_embed` (nn.Parameter)
    - `self.blocks = nn.ModuleList(HybridBlock(...) for _ in range(n_layers))`
    - `self.ln_f` (`_RMSNorm`)
    - optionally `self.lm_head` (untied) and `self.latent_thinker`
      (T2.9 Coconut, default off).
  No router, no expert pool, no `attach_coe_to_block` call.
- `synapforge/model_100m.py:283-293` — `_run_blocks` is a plain double `for`:
  `for blk in self.blocks: for _ in range(self.loop_depth): x = blk(x)`.
  This is the legacy "shared router" depth-loop pattern that CoE was
  intended to replace.
- Same-file grep on `from synapforge.routers | attach_coe | attach_moe |
  ChainOfExperts` returns **0 hits**.

### C. No call site anywhere on the live training path
- `synapforge/**/*.py` (live tree only, excluding `.claude/worktrees/` and
  `legacy/`) grep on `attach_coe_to_block` reveals **definitions and tests
  only** — no production caller.

---

## Why "ORPHAN" is the right label (not ABSENT)

- ABSENT would mean code does not exist → false: `routers/coe.py` is 248 LOC
  fully implemented, tested, exported.
- LIVE would mean `build_synapforge_100m` produces ≥1 CoE / MoE / Router
  module → false: smoke output is `[]`.
- ORPHAN: code exists and is loadable, but the production trainer's model
  factory never instantiates it. This matches T2.10's worded outcome
  exactly.

---

## Defer to phase 1

Task #113 stays `[completed]` because the **adv28 trainer** ran the CoE
path; T2.10's audit verdict is that the **production `train_100m_kd.py`
trainer does not yet wire it.** Per T2.10 brief, mark deferred to phase 1.

### Phase-1 implementation plan (1 page)

**Objective**: wire `attach_coe_to_block` into `build_synapforge_100m` so
each `HybridBlock` becomes a `_BlockWithCoE` driven by per-loop-step
routing. Cost ~16K params per block at default `n_steps=4 / n_routed=16`
(<2% block overhead).

**ArXiv ref**: Wang et al. 2506.18945 (2025-06) "Chain-of-Experts" —
recurrent shared backbone where each loop step uses an INDEPENDENT router.

**Wiring steps** (≤ 60 LOC delta):

1. `synapforge/model_100m.py::SynapForge100M.__init__`:
   - new kwargs `coe_enabled: bool = False`, `coe_n_routed: int = 16`,
     `coe_top_k: int = 2`, `coe_n_shared: int = 2`, `coe_n_steps: int = 4`
     (default off → bit-identical to current).
   - when enabled, after `self.blocks` is built, replace each block with
     `attach_coe_to_block(blk, hidden=d, n_routed=coe_n_routed, ...)`.
2. `synapforge/model_100m.py::SynapForge100M._run_blocks`:
   - before each inner loop iteration `r in range(self.loop_depth)`, set
     `blk._coe_step_t = r` on every wrapped block. Required because the
     wrapper picks `routers[step_t]` from that attribute.
3. `synapforge/model_100m.py::SynapForge100M.forward` (or wherever total
   loss is composed in `train_100m_kd.py`):
   - sum `wrapped.last_moe_aux` across blocks, scale by `coe_aux_weight`
     (default `1e-2`), expose as a new component in the loss dict.
4. `train_100m_kd.py` argparse:
   - add `--coe-enabled`, `--coe-n-routed`, `--coe-top-k`, `--coe-n-shared`,
     `--coe-n-steps`, `--coe-aux-weight` flags wired through to
     `build_synapforge_100m(...)`.
5. `tests/integration/test_coe_wired_in_100m.py` (new):
   - assert that with `--coe-enabled` on, `build_synapforge_100m(d=128,
     n_layers=2)` exposes ≥ 2 `_BlockWithCoE` modules and forward stays
     non-NaN at `seq_len ∈ {32, 256}`.
   - assert `last_moe_aux` is a finite scalar per-block after one forward.
   - smoke 100-step train identity test: with `--coe-aux-weight 0.0` and
     all CoE expert weights frozen, loss curve must match baseline within
     numerical noise.

**Validation gate**: at A800 rental, after 1000 steps
ce_with_coe / ce_baseline within ±1%, with `pct_step_routing_diversity
> 0.3` (i.e. histogram per step is non-degenerate). If routing collapses
(one expert wins all steps), tune `aux_alpha` from 1e-2 → 1e-1.

---

## Conclusion

**Verdict**: ORPHAN.
**Evidence**: `synapforge/routers/coe.py:51,213` defines + factory; smoke
`expert/moe/coe/router modules in build_synapforge_100m: []`; zero
`router|expert|coe|moe` matches in `train_100m_kd.py` and
`synapforge/model_100m.py`.
**Action**: T2.10 closed as ORPHAN, defer wiring into production trainer
to phase 1 per the implementation plan above.
