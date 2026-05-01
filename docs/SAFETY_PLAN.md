# Safety Pipeline — 4-Phase Anthropic Stack Plan

Concrete plan for running `scripts/run_safety_pipeline.py` against a trained
SynapForge backbone. Companion to `docs/SAFETY.md` (which is the rationale
+ paper anchors); this doc is the **operational runbook**.

## Prerequisite: WHEN to run this

This scaffold is meant to be run **after Phase 3 (chat_sft) of main training**
(see `scripts/phase_manager.py`). Do **not** run safety on an under-trained
base model — refusal would be vacuous because:

1. The model can't follow the attack pattern, so it produces gibberish whose
   "refusal rate" is meaningless.
2. The judge sees noise on both sides, so DPO preference labels are random.
3. Any apparent safety win evaporates the moment chat capability comes online,
   because the safety signal didn't condition on coherent dialogue.

**Gate to start safety**: validation perplexity ≤ 60 + chat eval ≥ 60% pass on
held-out instructions. Below this, the model can't even take instructions, so
"refuse instructions" is a no-op.

## The 4 phases

| # | Stage | GPU-h | Output | Module |
|---|-------|-------|--------|--------|
| 1 | SFT 拒绝 (HH-RLHF refusal subset) | 2.0 | 10k (prompt, refusal) pairs | `synapforge.safety.red_team_corpus` + your SFT loop |
| 2 | CAI SL-CAI 4 轮自批评 | 4.0 | 4k revised SFT pairs | `synapforge.safety.constitutional.ConstitutionalRevisor` |
| 3 | 红蓝 DPO 自对弈 (β=0.1) | 6.0 | 3k preference pairs | `synapforge.safety.red_blue.RedBlueSelfPlay` + `dpo.DPOTrainer` |
| 4 | Hidden-state 探针 | 0.5 | 1k labeled hidden states + linear probe | `scripts/run_safety_pipeline.py::run_probe` |
| **Total** | | **12.5h on A100×2** | | |

Skipped from the full Anthropic stack (and why):

- **PPO** — unstable on 375M LNN+SNN; DPO gets 80% of the value with 20% of
  the variance. (Skip per agent synthesis 2026-04-30.)
- **Full 16-principle constitution** — we condense to 6 principles
  (`P01_harm`, `P02_minor`, `P03_pii`, `P04_pro_advice`, `P05_misinfo`,
  `P06_polite_refusal`); the rest are proper subsets at our scale.
- **Sleeper-Agent defenses** (Hubinger 2401.05566) — research-grade, low
  ROI for 375M.

## Phase 1: SFT refusal warmup

**Target**: ce loss on refusal text drops from ~5.5 → ≤ 2.0 over 10k pairs.

Data sources, in priority:
1. HH-RLHF subset filtered to `harmless` split — Anthropic 2204.05862.
2. If unavailable, the orchestrator synthesizes from
   `red_team_corpus.sample_attack_prompt` × `sample_refusal_template`.

Loss curve expectation: monotonic decrease to 1.5 ± 0.3, fast (1 epoch is
enough — SFT on refusal is not the hard part).

Failure mode: refusal templates leak into non-refusal contexts (model
refuses benign requests). Mitigation: held-out benign prompts get checked
during eval — see `eval_harness.persona_robustness` which counts hard
jailbreak markers separately from safe deflections.

## Phase 2: Constitutional AI critique-revise (SL-CAI)

**Target**: 2k red prompts × 2 critique-revise iterations = 4k revised
SFT samples, then 1 epoch SFT on `(red_prompt, final_revised)`.

Loop (`safety/constitutional.py::ConstitutionalRevisor`):

```
For each (red_prompt, initial_response):
    for i in 0..N_ITERS-1:
        principle = principles[i % len(principles)]
        critique  = generate(critique_template(red, response, principle))
        revised   = generate(revision_template(red, response, critique))
        response  = revised
    emit (red_prompt, final response) as SFT pair
```

We cycle through the 6 principles in `CONSTITUTIONAL_PRINCIPLES` so
each iteration applies a different lens. 4 iterations is the Anthropic
default; we keep it.

Loss curve expectation: revision-quality (judged by `AIJudge`) climbs
50% → 75% over 4 iterations. Plateau by iter 4; no benefit from going
to 8.

Failure mode: model cycles between two equally bad rewrites
("hyperhelpful → hyperrefusal → hyperhelpful → ..."). Mitigation: the
fix is principle rotation across iterations (already in code), plus
length penalty during SFT.

## Phase 3: Red-Blue DPO self-play (β=0.1)

**Target**: HarmBench refusal rate ≥ 85%, JailbreakBench attack success
≤ 15%. 3k DPO preference pairs, 3 epochs.

Self-play loop (`safety/red_blue.py::RedBlueSelfPlay`):

```
For each round:
    1. red:    mutate seed_attack via 7 strategies (roleplay / encoding /
               ignore_prior / multi-turn / puzzle / language_switch / emotion)
    2. blue:   generate response_a (T=0.7) and response_b (T=1.2) under
               BLUE_SYSTEM_PROMPT
    3. judge:  AIJudge picks safer response (rule + API hybrid)
    4. emit:   DPOPair(prompt=mutated, chosen=safer, rejected=other)
After K pairs, run DPOTrainer.step. Refresh π_ref every 50 steps
(iterative DPO 2404.10719).
```

Persona-swap fraction: ≥ 30% of all DPO pairs (`target_persona_swap_frac=0.30`
in `RedBlueSelfPlay.__init__`). This is the **single highest-leverage
intervention** — see "Why persona swap is #1" below.

DPO hyperparams (per `safety/dpo.py` + Rafailov 2305.18290):
- β = 0.1 (raise to 0.3 if mode collapse, i.e. accuracy plateaus at 0.5)
- LoRA r=16 α=32 dropout=0.05
- lr = 5e-7 (10× lower than SFT — DPO over-fits faster)
- Iterative DPO refresh every 50 steps
- Batch 32, grad-accum 4, 3 epochs over 3k pairs

Loss curve expectation: DPO loss starts ~0.69 (random), drops to ~0.35
within 200 steps as the policy learns to prefer chosen over rejected.
Accuracy climbs 0.5 → 0.85 over 3k steps.

Failure mode A: **model becomes overly refusal-y** — refuses benign
prompts because all training pressure is "lean toward refusal".
Mitigation: include benign-but-tricky prompts in the DPO blue side
(e.g. "Help me write a poem about fire" — the chosen response is a
helpful poem, not a refusal). Currently the corpus is attack-only;
extending it is on the TODO list for v4.3 safety.

Failure mode B: **mode collapse** — policy outputs same refusal text
regardless of input. Mitigation: raise β from 0.1 to 0.3, or reduce
ref-refresh frequency from 50 to 200 steps.

## Phase 4: Hidden-state safety probe

**Target**: linear classifier on layer-12 hidden states with detection
≥ 85% on held-out attacks, FPR ≤ 5% on benign prompts.

Data: 500 attacks (label=1) + 500 benign (label=0). Hidden state =
mean-pool over sequence, layer-12 of backbone. 80/20 train/eval split.

Output: `phase4_probe/probe_weights.json` — apply at inference as
last-line filter: if `P(unsafe | hidden) > 0.7` then refuse generation.

Wall-time: 30 minutes. Hidden states are pre-computable (one forward
pass over the prompt set), then logistic regression is seconds.

Failure mode: probe overfits to surface tokens (e.g. literally the word
"bomb"). Mitigation: include adversarial paraphrases of attacks in
training set; eval on held-out paraphrase set. Probe accuracy on
paraphrase set is the real signal, not in-distribution accuracy.

## Why persona-swap is the #1 risk

Per memory `feedback_anthropic_safety_stack.md` (2026-04-30):

> 80% 公开 jailbreak 是 PERSONA SWAP 的变种.

DAN 1-13, AIM, DevMode, Grandma, "ignore previous", Jailbroken, AntiGPT,
EvilBOT — these are surface variations of the **same attack family**:
override the assistant's identity/instructions by asserting a different
persona that "doesn't have rules".

Coverage strategy:
- `synapforge/safety/persona_swap_corpus.py` generates **200 personas ×
  50 attack templates = 10K** persona-swap red prompts (deterministic,
  seed=42)
- `RedBlueSelfPlay.target_persona_swap_frac = 0.30` ensures DPO pairs
  are biased toward this class
- `eval_harness.persona_robustness` measures **refusal rate per attack
  family** so we can see if specific personas are punching through

A model that handles the persona-swap distribution robustly handles
~80% of in-the-wild jailbreaks, transferably (no need to enumerate every
specific DAN variant — they share the underlying signal pattern).

## Eval gates (must hit at release)

From `synapforge.safety.eval_harness.evaluate_safety`:

| Metric | Threshold | Source |
|--------|-----------|--------|
| HH-RLHF held-out preference accuracy | ≥ 70% | Anthropic 2204.05862 |
| MMLU regression vs pre-safety baseline | ≤ 2pt drop | helpfulness preserved |
| Persona-swap refusal rate (10K corpus) | ≥ 95% | this plan |

Optional / nice-to-have:

| Metric | Threshold | Source |
|--------|-----------|--------|
| HarmBench refusal rate | ≥ 85% | Mazeika 2402.04249 |
| JailbreakBench attack success | ≤ 15% | Chao 2404.01318 |
| Flames Chinese refusal rate | ≥ 80% | 2311.06899 |

`evaluate_safety` returns `pass=True` only if all 3 primary gates pass
AND none were skipped. A skipped gate (e.g. no HH eval data) does not
count as a pass — caller must explicitly downgrade with `gates_run`
inspection.

## Failure-mode catalog

| Symptom | Likely cause | Mitigation |
|---------|--------------|------------|
| Refuses benign prompts | over-refusal collapse | add benign-tricky to DPO blue side; reduce DPO epochs |
| Identical refusal text everywhere | mode collapse | raise β=0.1 → 0.3; refresh ref every 200 not 50 |
| MMLU drops > 2pt | safety SFT eats helpfulness | mix benign chat data into SFT; LoRA-merge instead of full FT |
| Persona-swap success > 5% | persona corpus under-sampled | raise `target_persona_swap_frac` 0.30 → 0.45 |
| Probe FPR > 10% | probe overfit to surface tokens | add paraphrase eval; lower threshold from 0.7 → 0.5 |
| DPO loss not decreasing | judge labels too noisy | switch from RuleJudge → APIJudge (DeepSeek-V3); discard low-confidence pairs |

## What we DON'T claim

- **No formal guarantees**. Output safety is empirical, not provable.
- **Backdoors not defended**. Sleeper-Agent style (Hubinger 2401.05566)
  trigger-gated backdoors survive SFT/RLHF/red-team. We add a probe
  but don't claim defense.
- **Adversarial robustness is limited**. Published attacks (PAIR, GCG,
  AutoDAN) bypass the strongest published filters 40-60% of the time.
  We expect similar.
- **Inputs from continual learning are NOT covered by output safety**.
  That's the role of `synapforge.defense` (Track A poison detection,
  Track B retrieval cache); see `docs/CONTINUAL_LEARNING.md`.

## Anchor papers

- Constitutional AI: Bai et al, [2212.08073](https://arxiv.org/abs/2212.08073)
- DPO: Rafailov et al, [2305.18290](https://arxiv.org/abs/2305.18290)
- RLAIF: Lee et al, [2309.00267](https://arxiv.org/abs/2309.00267)
- Red Teaming LMs: Ganguli et al, [2209.07858](https://arxiv.org/abs/2209.07858)
- Helpful-Harmless RLHF: Bai et al, [2204.05862](https://arxiv.org/abs/2204.05862)
- Iterative DPO: [2404.10719](https://arxiv.org/abs/2404.10719)
- HarmBench: Mazeika [2402.04249](https://arxiv.org/abs/2402.04249)
- JailbreakBench: Chao [2404.01318](https://arxiv.org/abs/2404.01318)
- Flames (Chinese): [2311.06899](https://arxiv.org/abs/2311.06899)
- Sleeper Agents: Hubinger [2401.05566](https://arxiv.org/abs/2401.05566)
