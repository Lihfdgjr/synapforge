# SFT Data — corpora, mix strategy, and honest limitations

This document covers the *supervised fine-tuning* (SFT) corpora that
``train_100m_sft.py`` consumes after Phase 0/1 pretraining. It complements:

* `docs/PHASE_TRAINING.md` — when each phase is opened, including the SFT
  phase that consumes the parquet produced here.
* `scripts/chat_eval_gate.py` — the 50-prompt heuristic tripwire that runs
  on every SFT checkpoint.
* `synapforge/action/neuromcp.py` + `synapforge/data/tool_use_traces.py` —
  the tool-use-traces story that backs the NeuroMCP claim.

The data flow:

```
download_alpaca.sh           ┐
download_sft_extended.sh     ├──► /workspace/data/sft/{name}/train.json
synapforge.data.tool_use_traces │
                             ┘
                              │
                              ▼
prep_sft_combined.py  ──►  /workspace/data/sft/combined.parquet
                                  (input_ids + loss_mask + source + language)
                              │
                              ▼
                      train_100m_sft.py  ──►  ckpt/sft/best.pt
                              │
                              ▼
                      chat_eval_gate.py  (heuristic tripwire)
```

---

## Per-corpus cards

### 1. `alpaca_en` — Stanford Alpaca (English)

* **Source:** `tatsu-lab/alpaca` (52K) and `yahma/alpaca-cleaned` (51K).
* **Anchor:** Taori et al. 2023 (no arxiv; Stanford CRFM blog); training
  data described in Wang et al. *Self-Instruct* 2305.13735.
* **License:** CC BY-NC 4.0 (research only).
* **Format:** ``{instruction, input, output}`` JSON list.
* **Teaches:** baseline english instruction-following, short-form QA.
* **Quality:** medium. ~10% of the original 52K is buggy
  (truncated outputs, wrong facts). The "cleaned" mirror fixes the worst.
* **Mix ratio:** ``1.0×`` (default, neither up- nor down-sampled).

### 2. `alpaca_zh` — Chinese GPT-4 distilled Alpaca

* **Source:** `silk-road/alpaca-data-gpt4-chinese` and
  `shibing624/alpaca-zh` mirrors (51K).
* **License:** CC BY-NC 4.0 (research only).
* **Format:** Same as alpaca-en.
* **Teaches:** baseline Chinese instruction-following.
* **Quality:** **mediocre.** Translation artifacts ("作为一个AI模型..."),
  occasional code-switching, robotic responses. We upsample 1.5× to
  compensate for English dominance, but production demos need
  human-curated zh data eventually.
* **Mix ratio:** ``1.5×``.

### 3. `lima` — Less Is More for Alignment (1K curated)

* **Source:** `GAIR/lima` train.jsonl (1030 examples).
* **Anchor:** Zhou et al. 2305.11206 *LIMA: Less Is More for Alignment*.
* **License:** non-commercial, research only.
* **Format:** Conversation array; we extract the first turn.
* **Teaches:** LIMA's claim is that **alignment is mostly format learning**;
  these 1K hand-curated examples teach the chat skin without polluting
  knowledge from the pretraining stage.
* **Mix ratio:** ``2.0×`` — tiny but very high quality.

### 4. `sharegpt_zh` — Real Chinese ChatGPT conversations

* **Source:** `xitao/sharegpt_zh` (90K conversations, scraped from
  ShareGPT.com Chinese subset).
* **License:** unclear. Treat as research-only; do not redistribute.
* **Format:** ``{conversations: [{from: human|gpt, value: ...}, ...]}``.
  We currently use only the first ``(human, gpt)`` pair; multi-turn is
  left for v2.
* **Teaches:** real conversational style, longer responses, emoji
  artifacts, code-mixed phrases.
* **Quality:** **messy.** The data was scraped, so contains: HTML
  artefacts, JavaScript fragments, conversation truncation, and the
  occasional jailbreak attempt. We downsample to 0.8× to partially
  compensate.
* **Mix ratio:** ``0.8×``.

### 5. `coig` — Chinese Open Instruction Generalist

* **Source:** `BAAI/COIG` (191K).
* **Anchor:** Zhang et al. 2304.07987 *Chinese Open Instruction Generalist*.
* **License:** Apache-2.0.
* **Format:** alpaca-like, with a ``task_type`` field we currently drop.
* **Teaches:** broader Chinese instruction taxonomy than alpaca-zh —
  includes counterfactual reasoning, exam questions, value alignment.
* **Quality:** mid-high. Clean license, decent diversity.
* **Mix ratio:** ``1.0×``.

### 6. `oa_zh` — OpenAssistant translated Chinese

* **Source:** `OpenAssistant/oasst1` filtered to ``lang=zh`` (~40K).
  Mirror: `sunzeyeah/chinese_chatgpt_corpus`.
* **Anchor:** Köpf et al. 2304.07327 *OpenAssistant Conversations*.
* **License:** Apache-2.0.
* **Format:** OASST tree — we use prompter→assistant root pairs only.
* **Teaches:** community-curated assistant style (volunteer-built).
* **Quality:** high in EN; the ZH translations vary. We do not
  upsample.
* **Mix ratio:** ``1.0×``.

### 7. `wizard_zh` — WizardLM evol-instruct (Chinese)

* **Source:** `silk-road/wizard_vicuna_70k_chinese`.
* **Anchor:** Xu et al. 2304.12244 *WizardLM: Empowering Large Language
  Models to Follow Complex Instructions*.
* **License:** non-commercial.
* **Format:** alpaca-like.
* **Teaches:** **complex** multi-step instructions via evol-instruct
  bootstrapping.
* **Quality:** repetitive — evol-instruct generates surface-form
  variations of the same instruction. We downsample heavily.
* **Mix ratio:** ``0.5×``.

### 8. `gsm8k_cot` — GSM8K with chain-of-thought

* **Source:** `openai/gsm8k` (7.5K train).
* **Anchor:** Cobbe et al. 2110.14168 *Training Verifiers to Solve Math
  Word Problems*.
* **License:** MIT.
* **Format:** ``{question, answer}`` where ``answer`` includes a
  natural-language CoT and a ``#### N`` final-answer marker.
* **Teaches:** elementary-school multi-step arithmetic.
* **Quality:** the gold standard for small-model math eval.
* **Mix ratio:** ``1.0×``.

### 9. `codealpaca` — CodeAlpaca

* **Source:** `sahil2801/CodeAlpaca-20k` (20K).
* **License:** non-commercial (alpaca-derived).
* **Format:** alpaca-like ``{instruction, input, output}``.
* **Teaches:** small-program writing, debugging, refactoring.
* **Quality:** noisy — many examples have buggy outputs. We
  downsample to 0.7× to limit code-quality regression.
* **Mix ratio:** ``0.7×``. Memory `feedback_pure_code_kills_english.md`
  documents that pure code training crashes English perplexity, so
  even a 30/70 code/text mix is the upper bound here.

### 10. `self_instruct` — 200 high-quality seeds

* **Source:** `yizhongw/self_instruct` ``seed_tasks.jsonl`` (175 seeds).
* **Anchor:** Wang et al. 2305.13735 *Self-Instruct*.
* **License:** Apache-2.0.
* **Format:** JSONL, one seed per line.
* **Teaches:** the *meta-instructions* the original paper used to
  bootstrap a 52K-example dataset from scratch. Useful for
  self-bootstrap experiments (currently parked).
* **Mix ratio:** ``3.0×`` — only ~200 rows total.

### 11. `tool_use_traces` — synthetic agent traces (NeuroMCP)

* **Source:** ``synapforge/data/tool_use_traces.py`` — generated locally,
  not downloaded.
* **License:** Apache-2.0 (our code).
* **Format:** alpaca-like; ``output`` contains explicit
  ``Thought: / Action: / Observation:`` lines so the tokenised stream
  still encodes the loop structure.
* **Teaches:** the agent loop *as text*. The model later relearns the
  same loop via the ActionHead in
  ``synapforge/action/neuromcp.py``, where actions are emitted from
  hidden state directly, NOT as ``<tool_call>`` tokens.
* **Volume:** 200 traces, 5–15 actions each ⇒ ~1500 (action, obs)
  pairs total.

---

## Default mix and budget

After per-source ratio scaling (see ``DEFAULT_RATIOS`` in
``scripts/prep_sft_combined.py``), the unfiltered pool is roughly:

| source         | raw rows  | × ratio | post-resample |
|----------------|-----------|---------|---------------|
| alpaca_en      |    52,000 |   1.0   |       52,000  |
| alpaca_zh      |    51,000 |   1.5   |       76,500  |
| lima           |     1,030 |   2.0   |        2,060  |
| sharegpt_zh    |    90,000 |   0.8   |       72,000  |
| coig           |   191,000 |   1.0   |      191,000  |
| oa_zh          |    40,000 |   1.0   |       40,000  |
| wizard_zh      |    70,000 |   0.5   |       35,000  |
| gsm8k_cot      |     7,500 |   1.0   |        7,500  |
| codealpaca     |    20,000 |   0.7   |       14,000  |
| self_instruct  |       175 |   3.0   |          525  |
| tool_use_traces|       200 |   1.0   |          200  |
| **total**      | **522,905**|        |  **~490,785** |

Length filter: 64 ≤ tokens ≤ 1024 drops ~5–10% as too-short or too-long.
Final example count after filtering: **~440K–470K**.

Disk budget after parquet+zstd: **~5 GB**.

### Recommended SFT mix strategies

The default ratios above produce a **balanced chat** model. For other
profiles, override `DEFAULT_RATIOS` via a config patch:

| profile     | upweight                                             | downweight     |
|-------------|------------------------------------------------------|----------------|
| chat        | sharegpt_zh, oa_zh, lima, alpaca_zh                  | wizard, code  |
| instruct    | wizard_zh ×1, alpaca_en, coig                        | sharegpt_zh   |
| coder       | codealpaca ×3.0, gsm8k_cot                           | sharegpt_zh, oa_zh |
| tool-use    | tool_use_traces ×10, multi_step subset of multi_step | most chat data |

---

## Honest limitations

1. **alpaca-zh is mediocre.** Translation artefacts dominate. A
   production demo would need ≥10K hand-curated Chinese SFT examples
   layered on top — see Anthropic's published practice in
   ``feedback_anthropic_safety_stack.md``.

2. **sharegpt-zh is messy.** Scraped HTML/JS, partial truncations,
   the occasional jailbreak attempt. We downsample but do not filter.
   A future revision should add an LLM-judge filter before
   tokenisation.

3. **tool_use_traces are synthetic.** They teach the *form* of the
   agent loop but the actions/observations are deterministic
   templates. Real agent logs collected via
   ``synapforge.tools.WebSearchTool`` and the executors in
   ``synapforge.action`` should be merged in for v2.

4. **Multi-turn handling is single-turn.** sharegpt and oasst trees
   contain rich multi-turn conversations; our normaliser collapses to
   the first ``(human, gpt)`` pair. v2 should preserve full turns and
   add explicit ``USER:``/``ASSISTANT:`` separators with response-only
   loss masking on every assistant turn.

5. **License hygiene.** Several corpora are non-commercial (alpaca,
   wizard, lima). Our trained checkpoints inherit those constraints;
   a permissively-licensed reproduction would need to swap to
   `coig`-only + `gsm8k` + `tool_use_traces`.

6. **No human preferences.** This is supervised fine-tuning only.
   The DPO/RLHF stage (`docs/SAFETY.md`,
   `feedback_anthropic_safety_stack.md`) is a separate phase that
   consumes ``(prompt, chosen, rejected)`` triples not produced here.

7. **The 50-prompt eval is heuristic.** ``scripts/chat_eval_gate.py``
   detects WORD_SALAD vs COHERENT, not chat quality. The investor demo
   plus manual eyeballing remains the final gate. Memory
   `feedback_chat_eval_gate.md` documents typical failure modes.

---

## Reproducing the dataset

```bash
# 1. Download (5–30 min on a fast link; falls back to synthetic stubs).
bash scripts/download_alpaca.sh
bash scripts/download_sft_extended.sh --size full

# 2. Generate tool-use traces (instant; no network).
python -m synapforge.data.tool_use_traces \
       --n 0 --save /workspace/data/sft/tool_use_traces/train.json

# 3. Tokenize + mix into a single parquet.
python scripts/prep_sft_combined.py \
       --in-root /workspace/data/sft \
       --out /workspace/data/sft/combined.parquet \
       --tokenizer /workspace/teachers/qwen2.5-0.5b \
       --max-seq 1024
```

For smoke testing without internet:

```bash
bash scripts/download_sft_extended.sh --size smoke   # synthetic stubs only
python -m synapforge.data.tool_use_traces --n 5      # prints 5 sample traces
```

---

## References

* **Alpaca**: Taori et al. 2023 (Stanford CRFM, no arxiv) and the
  *Self-Instruct* paper Wang et al. 2305.13735 on which it builds.
* **LIMA**: Zhou et al. 2305.11206.
* **COIG**: Zhang et al. 2304.07987.
* **OpenAssistant**: Köpf et al. 2304.07327.
* **WizardLM**: Xu et al. 2304.12244.
* **GSM8K**: Cobbe et al. 2110.14168.
* **CodeAlpaca**: derived from Stanford Alpaca; no separate paper.
* **NeuroMCP** (our own): see ``synapforge/action/neuromcp.py`` and
  the v4.1 design notes in ``docs/NEUROMCP_UNIVERSAL.md``.
