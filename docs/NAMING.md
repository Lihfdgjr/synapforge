# Naming

**Project / framework**: `synapforge` (突触锻造 — the codebase, the open-source library)
**Model**: **Synap-1** (突触一号 — the trained artifact)

The repo and pip package stay `synapforge` because that's the framework. The
trained model that comes out of the framework is **Synap-1**.

## Why "Synap-1"

Three reasons.

1. **Continuity with the framework name.** `SynapForge` is the forge; `Synap` is
   what comes off the anvil. Same root, different role.
2. **Direct connection to the two headline claims.** Both NeuroMCP (synaptic
   growth replaces tool-calling) and inference-time STDP (Hebbian forward-only
   weight updates) are *about synapses*. The name front-loads the thesis.
3. **Short, bilingual, room to grow.** "Synap-1" reads cleanly in English; 突触
   一号 in Chinese. One syllable + a number — fits in any pitch deck or chat
   handle.

## Roadmap of model variants

| Codename            | Params | Spec                                              | Status   |
|---------------------|--------|---------------------------------------------------|----------|
| **Synap-1 Base**    | 100M   | d=512, n_layers=10, loop_depth=1, vocab 151936, Qwen-0.5B KD | training; Run 3l/3m/3n plateaued at val ppl ~3700 (2026-05-02) |
| Synap-1-SFT (Base)  | 100M   | Synap-1 Base + alpaca-zh response-only SFT        | phase 3 gate |
| Synap-1-RL (Base)   | 100M   | Synap-1-SFT + GRPO sympy verifier RL              | phase 4 gate |
| **Synap-1 Pro**     | 300M   | d=1024, n_layers=14, loop_depth=2, vocab 151936, Qwen-0.5B KD; ~175M useful backbone vs Base's ~25M | launching post-Run 3o (ETA 14:00 May 2) |
| Synap-1 Ultra       | 500M   | d=1280, n_layers=16, loop_depth=2, vocab 151936   | next variant after Pro validates |
| Synap-Edge          | 100M   | BitNet b1.58 ternary quantization                 | post-pitch |
| Synap-Air           | 30M    | Distilled from Synap-1, mobile target             | future   |

### Synap-1 Pro spec (300M)

The Pro tier is the architecture-identical scaling response to the Base
plateau. Same vocab, same KD recipe, same NeuroMCP / R-fold / STDP /
backup stack — only the backbone width and depth change.

- **Hidden dim** `d = 1024` (Base: 512)
- **Layers** `n_layers = 14` (Base: 10)
- **Loop depth** `loop_depth = 2` (Base: 1)
- **Vocab** 151936 (Qwen tokenizer, unchanged)
- **Total params** ~300M, of which Qwen embedding eats ~75M and the
  **useful backbone is ~175M — 7× the ~25M useful backbone in Base**.
  This 7× gap is the structural reason we expect Pro to push past Base's
  val ppl ~3700 plateau.
- **Teacher KD** Qwen2.5-0.5B logits (unchanged)
- **Training cost target** ~80 GPU-h on A800 80GB (~¥600) for first
  usable ckpt; 7 days (~¥1,200) for chat-fluent target.

### Synap-1 Ultra spec (500M, next variant)

If Pro validates, Ultra is the next size step on the same recipe:

- **Hidden dim** `d = 1280`
- **Layers** `n_layers = 16`
- **Loop depth** `loop_depth = 2`
- **Vocab** 151936 (unchanged)
- **Total params** ~500M; useful backbone ~325M after Qwen embedding.
- Status: design-only; not committed until Pro Run 3o results land.

Tag at GitHub Release: `synap-1-v0.1` (early), `synap-1-v1.0` (post-RL).

## What does NOT change

- Pip package: `pip install synapforge`.
- Python imports: `from synapforge.cells import ...`, `synapforge-demo` CLI.
- GitHub repo: `Lihfdgjr/synapforge`.
- Memory entries pointing to "synapforge".

The framework keeps the verb-y forge name; the model gets the noun.

## When to use which

- Talking about **the architecture, the trainer, the demos** → `SynapForge`.
- Talking about **the trained ckpt (any tier), the chat behavior, what an
  investor loads and runs** → **Synap-1**, with the tier suffix when the
  size matters: **Synap-1 Base** (100M), **Synap-1 Pro** (300M), or
  **Synap-1 Ultra** (500M, design-only).

The `synapforge-demo chat` CLI loads Synap-1 — which is consistent: a SynapForge
*demo* runs on a Synap *model*. The same CLI accepts ckpts at any tier.
