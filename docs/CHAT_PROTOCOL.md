# Chat Protocol — Async Like a Real Person

The user's complaint: "我希望能像真人一样倾听打断, 不是一问一答" — model should listen
and interrupt like a real person, not strict request-response.

This document describes the new `synapforge.chat.*` async kernel that replaces
the synchronous protocol. It works like real-time conversation:

- User can type while model is generating (no turn-taking lock)
- Model can be cancelled mid-sentence by user typing
- Model can speak unprompted (proactive triggers, tagged `[Proactive]`)
- User can mute model entirely
- Cron / web events can wake the model up
- Long pauses → backchannel ("嗯", "OK"...) optional

## Architecture

```
                        ┌─────────────────────────────┐
   user keystrokes ───▶│ inbox queue                  │
   submit signals  ───▶│  (asyncio.Queue)             │
   cron ticks      ───▶│                              │
   web events      ───▶│                              │
                        └─────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                        ▼
            ┌────────────────────┐    ┌─────────────────────────┐
            │ TurnTakingDetector │    │ ProactiveMessenger      │
            │  is user done?     │    │  IdleLoop / Cron / Web  │
            └────────────────────┘    └─────────────────────────┘
                          │                        │
                          └───────────┬────────────┘
                                      ▼
                        ┌─────────────────────────────┐
                        │ InterruptPolicy             │
                        │  speak / interrupt-self /   │
                        │  silence / backchannel ?    │
                        └─────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │ StreamingGenerator          │
                        │  cancellable token-by-token │
                        └─────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │ outbox queue                │
                        │   (async iterator to UI)    │
                        └─────────────────────────────┘
```

## Components

### `TurnTakingDetector` (text-VAD equivalent)
Decides if the user's current input is a complete turn or just mid-thought.
Tracks: keystroke timestamps, paren/quote depth, code block state.

States:
- `EMPTY` — no input yet
- `TYPING` — user actively typing
- `PAUSE` — > 1.2s since last keystroke
- `COMPLETE` — terminal punctuation (`.`, `!`, `?`, `。`, `！`, `？`, newline)
  AND no open structures
- `EXPLICIT_SUBMIT` — Enter pressed / send button

"Definitely incomplete" overrides:
- Trailing comma / dash / "..."
- Open paren / bracket / quote
- Inside un-closed ```code block```

### `InterruptPolicy` (the speak-or-not brain)
4 critical decisions per event:

| Situation | Decision rule |
|-----------|---------------|
| User submitted message | Speak immediately |
| User typed mid-stream | Interrupt self if urgency > 0.7 (markers: stop / wait / 停 / ! / ?) |
| User has been silent N min + I have insight | Proactive speak if frequency cap not hit (default 1/min, 30/hour) |
| User muted | Stay silent always |

Frequency caps prevent annoying the user. Industry standard from
Anthropic / Inflection / character.ai.

### `StreamingGenerator` (cancellable per-token)
Wraps `model.encode + model.lm_logits` with:
- Token-by-token loop with sampling (top-k 50, top-p 0.95, temperature 0.7)
- `CancellationToken` checked every `check_cancel_every` tokens (default 4)
- Async queue for chunks → UI consumes via `async for`
- EOS token detection
- Top-k / top-p / temperature configurable per call

### `ProactiveMessenger` (outbound triggers)
3 sub-triggers:
- `IdleLoopTrigger`: user silent 10 min → emit interesting thought
- `CronTrigger`: user said "remind me at 3pm" → fire at scheduled time
- `WebEventTrigger`: autonomous_daemon found relevant new content
  (matches user's recent topics in retrieval cache)

Each trigger has urgency [0, 1]. Policy decides actual emission.

### `ConversationKernel` (the loop)
Owns inbox, outbox, state. Two background tasks:
- Main loop: drains inbox, dispatches events
- Tick loop: every 5s, polls ProactiveMessenger for due triggers

## Usage example

```python
import asyncio
from transformers import AutoTokenizer
from synapforge.model_chat_600m import SynapForgeChat600M
from synapforge.chat import ConversationKernel, StreamingGenerator
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = SynapForgeChat600M().cuda()
model.load_state_dict(torch.load("/path/to/best.pt")["model"])
model.eval()

kernel = ConversationKernel(
    generator=StreamingGenerator(model, tok),
)

async def main():
    await kernel.start()

    # Consumer: render output to UI
    async def render():
        async for chunk in kernel.outbox_stream():
            kind = chunk.get("kind")
            text = chunk.get("text", "")
            if chunk.get("proactive"):
                print(f"\n[模型主动] {text}", end="", flush=True)
            elif kind == "interrupt":
                print(f"\n[模型被打断] {text}", flush=True)
            else:
                print(text, end="", flush=True)

    asyncio.create_task(render())

    # Publisher: simulate user
    await kernel.send_user_text("你好，请讲一下")
    await asyncio.sleep(0.5)
    # User keeps typing while waiting for model
    await kernel.send_user_text("线性代数的特征值")
    await kernel.send_user_submit()

    # Wait, then user interrupts mid-answer
    await asyncio.sleep(2.0)
    await kernel.send_user_text("等等")  # → urgency markers trigger interrupt

    # Schedule a cron reminder
    kernel.proactive.cron.schedule(
        fire_at_ts=time.time() + 60,
        message="提醒: 你说过一分钟后让我提醒你回去训练",
    )

    # Run forever (or until user disconnects)
    await asyncio.sleep(3600)
    await kernel.stop()

asyncio.run(main())
```

## What "feels like a real person"

After running v4.2 ckpt through this kernel, the user sees:
- Model starts responding the moment user finishes typing (no Enter required)
- If user types "等等" or "wait" mid-answer, model **stops in the middle of its sentence**
- Model occasionally says things on its own ("[Proactive] 顺便, 你之前问的那个特征值...")
- User can mute model with one button, model goes quiet but keeps listening
- Multiple back-to-back user messages don't pile up — model handles the latest

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Model interrupts too often → annoying | Frequency cap: 1 proactive per minute, 30 per hour |
| Model loses context after self-interrupt | Always retain last 10 user messages in prompt; cancellation reason logged |
| Concurrent messages cause race condition | Single inbox queue, single dispatcher |
| Streaming generation breaks on slow GPU | Drop top-p / top-k, use greedy sampling, batch 1 |
| User types while model is mid-token sampling | Cancel checks every 4 tokens, < 50ms p95 |

## Wiring to UI

The kernel exposes 6 methods that any UI can drive:

```python
# Publishers (UI → kernel)
await kernel.send_user_char(c)        # per-keystroke
await kernel.send_user_text(text)     # paste / batch
await kernel.send_user_submit()       # Enter pressed
await kernel.send_mute(seconds)       # mute button
await kernel.send_unmute()
await kernel.emit_proactive(trigger)  # external system fires

# Consumer (kernel → UI)
async for chunk in kernel.outbox_stream():
    # chunk = {"kind": "model"|"interrupt"|"error",
    #         "text": "...",
    #         "ts": float,
    #         "proactive": bool}
    render(chunk)
```

WebSocket / SSE / native widget — all work with this contract.

## Inspiration

- Anthropic Claude interruptible response design (2024-2025 client UX)
- OpenAI Realtime API (gpt-4o-realtime, voice with mid-speech interrupt)
- Inflection Pi (Pi.ai) chat UX
- Character.ai backchannel + proactive memory recall

We're not voice-mode but the protocol is identical for text. WebSocket
streaming + cancellation token + proactive queue is the standard pattern.
