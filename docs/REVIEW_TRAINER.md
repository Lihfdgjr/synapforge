# Trainer + 4 Silent-Killer Review (2026-05-01)

Review scope: `train_100m_kd.py`, `train_100m_sft.py`, `synapforge/data.py`,
`synapforge/phase_signal.py`, `scripts/honest_eval_hook.py`,
`scripts/triple_backup_daemon.py`, `scripts/phase_manager.py`,
`synapforge/trainer_mixins.py`, `scripts/prep_alpaca_qwen.py`,
`scripts/chat_repl.py`, `scripts/cpu_pilot_inference_stdp.py`.

**Public CLI / function signatures preserved** — every fix is internal so other
review agents (modal/bench/demo) keep building cleanly on top.


## HIGH severity (was actively corrupting training)

### H1. KD-skip step silently scaled LM gradient by `(1-α)` on 75% of steps
**File:** `train_100m_kd.py` ~line 575
**Symptom:** When `--kd-every=4`, only 1 in 4 steps actually computes the
teacher KD term. The previous code unconditionally computed
`loss = (1-α) * base_loss + α * kd` — and on KD-skip steps `kd=0`,
so the LM signal was scaled to `(1-α) * base_loss = 0.3 * base_loss` for the
default `α=0.7`. Effect: an effective 0.3× LR drop on 75% of steps. This is
almost certainly the wrong behaviour given the docstring intent.
**Fix:** Branch on `step % kd_every == 0`. KD-active steps still mix
`(1-α)*base + α*kd`; KD-skip steps use bare `base_loss`. Help string of
`--kd-every` updated to document the new semantics so a future reader doesn't
"undo" the fix as an inconsistency.

### H2. NameError landmine in `evaluate()` (module-level helper used `_log`)
**File:** `train_100m_kd.py` line 193 (eval `_log(...)` calls)
**Symptom:** `evaluate()` is defined at module scope; it called `_log` which
was previously only defined inside `main()`'s closure. Python looks up bare
names by global / module scope, not caller-closure. The first time the
spike-rate logging branch fires (every val cycle when there are PLIF cells),
training would crash with `NameError: name '_log' is not defined`.
**Fix:** Promoted `_log` to a module-level fallback that just `print()`s with a
timestamp. `main()` still defines its own `_log` closure that additionally
appends to `log_lines` for the live.log dump — the closure shadows the module
fallback for everything inside `main()`, so on-disk logging is unchanged.

### H3. Wrong default EOT id silently poisoned the training stream
**File:** `synapforge/data.py` `ParquetTokenStream.__init__`
**Symptom:** `eot_id=50256` is GPT-2's `<|endoftext|>`. The trainer now uses
the Qwen tokenizer (`/workspace/teachers/qwen2.5-0.5b`, vocab=151643) and
token id 50256 in Qwen vocab is *not* an EOS — it's a regular Chinese
sub-word. So between every doc the stream injected a junk token. Possibly a
cause of the long-running ppl-floor / word-salad complaints in memory.
**Fix:** Default is now `eot_id=None` → auto-derive from
`tokenizer.eos_token_id`. Explicit `eot_id` still overrides if a caller
genuinely wants a different separator. Docstring updated.

### H4. `_get_tokenizer` cache returned a stale tokenizer when name changed
**File:** `synapforge/data.py`
**Symptom:** Single `_TOKENIZER` global cache, keyed on nothing — once
loaded with `gpt2`, a later call with `qwen2.5-0.5b` re-used the GPT-2
tokenizer silently. Mostly latent because the trainer only uses one
tokenizer per process, but it would explode the moment a script wanted both
(e.g. teacher+student probe). Replaced with `dict[name, tokenizer]` cache.

## MED severity (latent bugs, fragile defaults)

### M1. SFT trainer's warmstart ignored ckpt-key remapping
**File:** `train_100m_sft.py` line 171
**Symptom:** SFT called `adv_warmstart(model, args.warmstart, name_map=[])`.
The pretrain trainer (kd) DOES pass a name_map for legacy `.cfc.` →
`.liquid.` and `.embed.text_embed.` → `.tok_embed.`. SFT loading a pretrain
ckpt that came from an even older codepath would silently miss layers and
produce garbage chat output.
**Fix:** Pass the same `name_map` the kd trainer uses.

### M2. `phase_manager.parse_log` treated `last_ce==0.0` as missing
**File:** `scripts/phase_manager.py`
**Symptom:** `math.exp(last_ce) if last_ce else None` is False for 0.0
(impossible in practice, but a code smell). Same `or`-chain bug in
`decide_phase`. Both replaced with explicit `is None` checks.

### M3. `triple_backup_daemon` `e.stderr.decode()` could crash on None
**File:** `scripts/triple_backup_daemon.py`
**Symptom:** `subprocess.CalledProcessError.stderr` may be `None`; the
`.decode()` call would raise `AttributeError: 'NoneType' object has no
attribute 'decode'` and break the daemon's "never die" contract. Same in
`(create.stderr or upload.stderr).decode()` for the gh-release path.
**Fix:** Coerce to `b""` before decode, with `errors="replace"` for safety.

### M4. `chat_repl.generate` re-encoded the prompt every iteration (O(N²))
**File:** `scripts/chat_repl.py`
**Symptom:** Inside the generation loop, `tok.decode(ids[0,
len(tok.encode(text, add_special_tokens=False)):], ...)` re-encoded the
prompt to compute the slice offset. For an 80-token reply this is 80 prompt
encodes. Cached `prompt_len = ids.size(1)` once before the loop. Also fixed
the `tok.eos_token_id` set having a `None` member (would silently skip EOS
detection — `int x not in {None}`).

### M5. `sample()` fallback eos was the GPT-2 magic number
**File:** `train_100m_kd.py` `sample()`
**Symptom:** Empty-prompt fallback used `tokenizer.eos_token_id or 50256`.
If the tokenizer has no eos_token_id (very rare), 50256 is wrong for any
non-GPT-2 vocab and would emit garbage. Use 0 as a defensive last-resort.

## LOW severity (style / docstring drift)

### L1. `_kd_loss` averaged chunk-means, biased on uneven last chunk
**File:** `train_100m_kd.py` `_kd_loss`
**Symptom:** Inner-loop divided each chunk's KL `sum` by `B_chunk*T` then
the outer divided by `n_chunks` — when `bs % chunk != 0` the last partial
chunk's per-token mean carried equal weight to a full chunk, slightly
biasing the result. Negligible at typical bs=32 chunk=8 (no leftover) but
fragile. Replaced with `total_sum / total_n_tokens * T**2`. Numerics are
identical when chunks divide evenly; correct when they don't. Also removed
the redundant in-function `import torch` / `import torch.nn.functional as F`
(both already imported at module top).

### L2. `synapforge/data.py` docstring still mentioned GPT-2 BPE specifically
Updated header + `eot_id` docstring to reflect AutoTokenizer + auto-derived
EOT.

### L3. `train_100m_sft.py` GPU→CPU sync per step on `label_mask.sum().item()`
Computed `n_resp_step` on the CPU-side `mask` BEFORE the device transfer so
`cum_tok` accumulation no longer forces a sync barrier each step.

## Things I'd do with more time (not done)

* **`HonestEvalHook` plateau detector** uses spread (`max - min < eps`) rather
  than "best ppl hasn't decreased" — a slowly-rising curve looks the same as
  a plateau. Worth distinguishing rising-from-plateau (still bad, different
  fix) vs flat-plateau (early-stop candidate). Did not change because the
  current behaviour is what the trainer already wires in, and changing the
  semantics needs a cross-call review with whatever consumes
  `check_plateau`.
* **`triple_backup_daemon` retry/backoff** — currently a transient network
  blip on cycle N causes the cycle to fail and we wait `interval` seconds
  for the next attempt. A short exponential retry-inside-cycle (3 tries with
  10/30/60s) would close that gap, but adding it cleanly without breaking
  the "1 of 3 success = OK" semantics is non-trivial.
* **`prep_alpaca_qwen` END_MARK robustness** — when the tokenizer's
  byte-fallback splits `<|im_end|>` into multiple sub-tokens, the loss_mask
  still covers them as 1s but the SFT model never learns to predict the
  stop-marker as a single token. A clean fix would resolve `<|im_end|>` to
  a single special-token id at the tokenizer level (or fall back to
  `eos_token_id` for non-Qwen tokenizers). Out of scope.
* **`train_100m_kd.py` `loop_depth=1` literal** — memory note flagged that
  `model_100m.SynapForge100M` defaults to `loop_depth=4` but the trainer
  passes `1` explicitly. This is intentional per the LoopLM design (loop is
  applied via RDT routers, not core stack), but it makes the explicit `1`
  feel like a bug. Worth a comment-only PR.
* **`MultimodalMixin._next_sample` blocking torch.load on hot path** — every
  step pulls one `.pt` from disk synchronously. A small pre-fetch worker
  would eliminate the I/O stall on the autocast critical path. Out of scope.

## "Looks scary, actually fine"

* **`evaluate` nested f-strings calling `.item()` on each PLIF rate** — looks
  like a sync barrier per cell, but only fires inside `@torch.no_grad()` at
  validation time, not in the hot training loop. Acceptable.
* **`MultimodalMixin.smoke()` swaps `mix._next_sample` then restores it** —
  monkey-patch on a method, but the test runs in isolation and explicitly
  restores. No leak across smoke runs.
* **`CuriosityMixin.curiosity_loss` returns `(h_any.float().sum() * 0.0)`
  on disabled paths** — the `* 0.0` looks like a bug ("you forgot to
  multiply by something") but is the standard "give me a zero scalar with
  the same autograd graph" pattern. Confirmed against PyTorch's own
  testing patterns.
* **`ParquetTokenStream._iter_text_rows` `while True: ...; if not loop:
  return` looks infinite** — it's structurally correct: outer while loops
  until `not loop`, inner for-loop iterates files; `return` only fires
  after a complete pass.
* **`SFTBatcher.__next__` re-shuffles using `seed + cursor`** — the seed
  drift across resets is intentional so each epoch isn't an exact replay.
  Verified by tracing.
* **`phase_signal.consume_phase` keeps the malformed file with `.malformed.`
  suffix instead of deleting it** — looks like a leak, but the prefix lets
  forensics see what poisoned the trainer; new `.phase` writes don't
  collide with `.malformed.<ts>` names. Intentional.
* **`triple_backup_daemon._sha256_short` and `_sha256_full` look duplicated**
  — they're not: `_short` truncates to 16 hex chars (cheap dedup key),
  `_full` returns the whole digest (manifest field). Both wrap the same
  IO loop; combining them would just add an `if return_short:` branch.
  Acceptable.

## Smoke results (this env, no torch / no cuda)

* `python synapforge/phase_signal.py` → **PASS** ("phase_signal smoke OK")
* AST syntax check across all 11 in-scope files → **PASS** (all OK).
* `python -c "import synapforge.phase_signal"` would need a torch import
  chain (synapforge/__init__.py → action → torch); skipped per env.
* `train_100m_kd.py --help` and `train_100m_sft.py --help` would also need
  torch; skipped — argparse layout is unchanged so any prior `--help`
  capture still applies. Argparse sanity-check on the new `%%` escape in
  `--kd-every` help passed in a 1-line repro.

## Files touched

* `train_100m_kd.py` — H1, H2, M5, L1, `--kd-every` help text
* `train_100m_sft.py` — M1, L3
* `synapforge/data.py` — H3, H4, L2
* `scripts/phase_manager.py` — M2
* `scripts/triple_backup_daemon.py` — M3
* `scripts/chat_repl.py` — M4
* `docs/REVIEW_TRAINER.md` (this file, new)
