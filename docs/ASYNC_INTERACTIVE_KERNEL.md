# Async Interactive Kernel

User feedback (2026-05-02): "能像人类一样说话主动发消息打断人类说话你又是怎么解决的"
— the model should listen and interrupt like a human, not request-response.

This kernel is the answer.  It lives in `synapforge/interactive/` and is
the live wire between user input and the LNN+SNN backbone.  Task #237
("async interactive kernel") is now actually live in the chat path:
`synapforge-chat --async --proactive` instead of the synchronous
`synapforge-demo chat`.

## Why LNN+SNN naturally streams

The chat path exploits a structural advantage of the LNN+SNN backbone:

| Property                | Transformer | SynapForge LNN+SNN          |
|-------------------------|-------------|-----------------------------|
| Per-token forward       | O(T) (KV refresh) or O(T^2) (full prefill) | O(1)        |
| State after token t     | KV cache (`B*T*D*L*2`)                     | `(h_cfc, v_plif)` per block |
| Cancel mid-token        | Throw KV away              | Token-boundary clean        |
| Pause / resume          | Hard (KV memory live)      | Trivial — state is small    |
| Hidden-state checkpoint | Heavy                      | Two `B*D` tensors per layer |

CfC + PLIF give us:
* `h_t = CfC(h_{t-1}, x_t)` — per-token incremental update.
* `v_t = PLIF(v_{t-1}, h_t)` — event-driven spike accumulator.

Both are recurrent.  The full conversation history is summarised in those
two tensors; we never re-prefill prior context to "restore" the model.
That's why streaming-with-cancel-and-pause works without surgery on a KV
cache.

## Architecture

```
[InputStream]                            [model backbone]
  keyboard / VAD / file fd                continuous CfC step
  partial tokens, pauses, EOF             h_t = CfC(h_{t-1}, x_t)
        │                                          │
        ↓                                          ↓
  [TurnTakingPolicy] ←── confidence/urgency ──  [ProactiveTrigger]
        │                                       decides: speak / silent
        ↓
  [Decision: speak | listen | interrupt | silent]
        │
        ↓
  [StreamingGen]: tokens.send() pace-controlled, cancellable
```

Source files in `synapforge/interactive/`:

| Module                 | Responsibility                                  |
|------------------------|-------------------------------------------------|
| `streaming.py`         | Cancellable, pausable, paceable token-by-token  |
| `listener.py`          | stdin / VAD / file-fd → event queue             |
| `proactive_trigger.py` | Curiosity / GoalMemory / silence-relevance rules|
| `turn_taking.py`       | Speak / queue / abort / silent decision         |
| `async_chat.py`        | Session glue — listener + policy + generator   |

CLI: `synapforge/demo/async_chat_cli.py` — exposes the kernel as
`synapforge-chat`.

## Politeness / interruption policy

| Situation                                          | Action            |
|----------------------------------------------------|-------------------|
| User submitted EOT, model idle                     | Speak after 300ms |
| User submitted EOT, model still speaking           | Abort, then speak |
| User typing while model speaking                   | Abort immediately |
| Proactive trigger, user mid-typing (<2 s pause)    | Queue for later   |
| Proactive trigger, model speaking, urgency <0.95   | Queue for later   |
| Proactive trigger, model speaking, urgency ≥0.95   | Interrupt model   |
| Proactive trigger, idle channel                    | Speak now         |
| User explicitly muted                              | Stay silent       |

Defaults are conservative.  The model NEVER interrupts unless urgency
clears 0.95.  CLI flag `--interrupt-aggressive` drops the threshold to
0.7 for users who want a more forward agent.

## Settings

| Knob                         | Default    | Where                         |
|------------------------------|------------|-------------------------------|
| `politeness_pause_ms`        | 300        | TurnTakingPolicy ctor / CLI   |
| `interrupt_threshold`        | 0.95       | TurnTakingPolicy ctor / CLI   |
| `user_typing_grace_s`        | 2.0        | TurnTakingPolicy ctor         |
| `proactive_silence_window`   | 5.0 s      | ProactiveTrigger ctor / CLI   |
| `silence_relevance_threshold`| 0.4        | ProactiveTrigger ctor         |
| `goal_success_prob_threshold`| 0.7        | ProactiveTrigger ctor         |
| `min_confidence` (proactive) | 0.7        | ProactiveTrigger ctor (gate)  |
| `min_relevance` (proactive)  | 0.5        | ProactiveTrigger ctor (gate)  |
| `target_tokens_per_sec`      | 80         | StreamingGen ctor / CLI       |
| `check_cancel_every`         | 4 tokens   | StreamingGen ctor             |

## Quality guards

Every proactive utterance must clear two gates **before** the policy even
sees it:

* `confidence > 0.7` — the rule must be firing on real signal, not noise.
* `relevance > 0.5` — the utterance must connect to the live conversation.

Triggers that fail either gate are dropped silently.  See
`ProactiveTrigger.tick()` for the rule details.

## Demo recipe

The kernel ships with a deterministic transcript-driven demo.  Run from
the repo root (or any directory with `synapforge` installed):

```bash
synapforge-chat --async --proactive \
    --transcript synapforge/demo/async_chat_demo.transcript.jsonl \
    --save /tmp/sf_async_session.json \
    --target-tps 0
```

Each line of the transcript is one event:
* JSON-encoded string (e.g. `"hello"`) → user keystroke / paste.
* `<EOT>` or `"<EOT>"` → end of turn (Enter pressed).
* `<EOF>` or `"<EOF>"` → close listener (model stops listening).

Output: model response chunks stream to stdout; session JSON
(transcript + trigger log + policy stats) lands in `--save`.

For a real model, point `--ckpt` at any SynapForge checkpoint:

```bash
synapforge-chat --async --proactive \
    --ckpt /path/to/v24h_chat.pt \
    --tokenizer-path Qwen/Qwen2.5-0.5B
```

Live mode: stdin is the listener, stdout is the streaming output.
ENTER ends a user turn; Ctrl+C cancels the in-flight model response.

## Tuning the thresholds

The defaults are *placeholders informed by industry priors*, not
empirically tuned for SynapForge yet:

* `politeness_pause_ms = 300` — taken from typical voice agent pause
  studies.  Should be revisited once we have real voice input.
* `interrupt_threshold = 0.95` — deliberately very high; we'd rather
  miss a proactive interrupt than annoy the user.  Consider lowering to
  0.85 once curiosity scoring is calibrated against held-out chats.
* `user_typing_grace_s = 2.0` — feels right in synthetic transcripts;
  needs A/B against real users.
* `min_confidence / min_relevance` — placeholders matching the spec
  request; the actual P(user values this proactive utterance) is
  unmeasured.

These will move.  See `tests/interactive/` for the regression baseline.

## What works today vs. future work

Works:
* Streaming, cancel, pause, resume, flush_partial.
* Listener for stdin lines, deterministic iterables, manual `push()`.
* Proactive trigger evaluating curiosity / goal_memory / silence-rel.
* Turn-taking policy with overlap prevention + politeness pause.
* End-to-end async chat session with on-chunk callback.
* CLI `synapforge-chat` with `--async --proactive --transcript` etc.
* Backwards-compat: `synapforge-demo chat` still works (synchronous).

Future:
* **Streaming-train path** — current training-time CfC step batches
  T=256, the inference path runs per-token.  Streaming-during-training
  is not yet implemented; that would let the model learn from human
  pauses / interruptions in real time.
* **Audio mode** — `InputListener.with_vad()` returns the right shape,
  but the actual audio capture (PyAudio / sounddevice) is left to the
  caller.
* **Curiosity feed** — the proactive trigger expects pushed curiosity
  scores; the live wiring from `synapforge.curiosity.CuriosityScorer`
  to the trigger is a follow-up, not yet wired in `async_chat_cli.py`.
* **Persona swap defenses** — the trigger has no defense against an
  adversarial user injecting `[Proactive]` markers.  Add a
  trigger-source assertion in the on-chunk callback layer.

## Hard constraints (this kernel obeys)

* `synapforge/interactive/*.py` — zero module-level torch imports.
  The only file that touches torch is `streaming.py::_run`, and it's
  lazy-imported inside that function.
* The synchronous `synapforge.demo.chat_demo` is untouched; this is a
  parallel path, not a replacement.
* Conservative-by-default; user opts in to aggressive interrupts.

## Industry precedent

* Anthropic Claude Realtime — interruptible response design.
* OpenAI Realtime API (gpt-4o-realtime) — voice with mid-speech interrupt.
* Inflection Pi (Pi.ai) — ambient chat UX.
* Character.ai — proactive memory recall + backchannel.

We're not voice-mode but the protocol is identical for text:
WebSocket / SSE streaming + cancellation + proactive queue.

## Testing

```bash
pytest tests/interactive/ -v
```

| Test                                | Verifies                                    |
|-------------------------------------|---------------------------------------------|
| `test_streaming_cancel.py`          | Cancel / pause / resume / flush_partial     |
| `test_proactive_trigger_fires.py`   | All three rules + quality guards            |
| `test_turn_taking_no_overlap.py`    | No-overlap, politeness-pause, abort rules   |
| `test_async_chat_smoke.py`          | 5-turn synthetic conversation completes     |
