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

| Codename       | Params | Spec                                      | Status   |
|----------------|--------|-------------------------------------------|----------|
| **Synap-1**    | 100M   | LNN+SNN, vocab 151936, Qwen-0.5B KD       | training (2026-05-01) |
| Synap-1-SFT    | 100M   | Synap-1 + alpaca-zh response-only SFT     | phase 3 gate |
| Synap-1-RL     | 100M   | Synap-1-SFT + GRPO sympy verifier RL      | phase 4 gate |
| Synap-Pro      | 300M   | Same recipe, scaled                       | post-pitch |
| Synap-Edge     | 100M   | BitNet b1.58 ternary quantization         | post-pitch |
| Synap-Air      | 30M    | Distilled from Synap-1, mobile target     | future   |

Tag at GitHub Release: `synap-1-v0.1` (early), `synap-1-v1.0` (post-RL).

## What does NOT change

- Pip package: `pip install synapforge`.
- Python imports: `from synapforge.cells import ...`, `synapforge-demo` CLI.
- GitHub repo: `Lihfdgjr/synapforge`.
- Memory entries pointing to "synapforge".

The framework keeps the verb-y forge name; the model gets the noun.

## When to use which

- Talking about **the architecture, the trainer, the demos** → `SynapForge`.
- Talking about **the trained 100M ckpt, the chat behavior, what an investor
  loads and runs** → **Synap-1**.

The `synapforge-demo chat` CLI loads Synap-1 — which is consistent: a SynapForge
*demo* runs on a Synap *model*.
