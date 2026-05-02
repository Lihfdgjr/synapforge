# NeuroMCP Closed-Loop Computer Use

Status: **Layer 0..5 implemented + closed-loop trainer wire-in** (2026-05-02).
Branch: `feature/neuromcp-closed-loop`.

This is the SynapForge realisation of the user铁律 in memory
`feedback_neural_action_no_token_no_mcp` and
`feedback_synaptogenesis_replaces_mcp`:

> No `<tool_call>` tokens. No MCP JSON schema registration. Pure neural
> emergence — neurons grow connections that *are* the tools.

## 1. The 5-layer stack

```
Layer 0  Primitives (24 fixed)            click_at(x,y), type_text(s),
                                           scroll, press_key, screenshot,
                                           wait, drag, focus_window, ...
                                           [synapforge/neuromcp/primitives.py]

Layer 1  ActionHead -> primitive_logits   Linear(d, 24) over backbone last hidden
                                           argmax = primitive_id
                                           [synapforge/neuromcp/action_head.py]

Layer 2  ParamHead  -> primitive_params   Linear(d, 8) for the 8 universal slots
                                           (x, y, x2, y2, token_id, keysym, dx, dy)
                                           sigmoid -> [0,1] for coords
                                           [synapforge/neuromcp/action_head.py]

Layer 3  DynamicCodebook                  K growing prototypes; novel slot spawned
                                           when sim < 0.7 to all alive prototypes
                                           [delegated to synapforge.action.neuromcp]

Layer 4  CompoundGrowth                   Sliding window of last 50 firings.
                                           Same N-gram fires >= 5 times within
                                           100-step window -> new compound prototype
                                           added to codebook + hyperedge in
                                           SparseSynapticLayer linking those prims
                                           [synapforge/neuromcp/compound_growth.py]

Layer 5  OSActuator                       backend dispatch:
                                           - sandbox     (default for training)
                                           - win32       (pyautogui, opt-in)
                                           - mcp_control (mcp__mcp-control_*)
                                           [synapforge/neuromcp/os_actuator.py]
```

## 2. The 24 primitives

| id  | name              | category | param slots          | sandbox-guarded |
| --- | ----------------- | -------- | -------------------- | --------------- |
| 0   | click_at          | pointer  | x, y                 |                 |
| 1   | double_click      | pointer  | x, y                 |                 |
| 2   | right_click       | pointer  | x, y                 |                 |
| 3   | middle_click      | pointer  | x, y                 |                 |
| 4   | drag              | pointer  | x, y, x2, y2         |                 |
| 5   | move_mouse        | pointer  | x, y                 |                 |
| 6   | hover             | pointer  | x, y                 |                 |
| 7   | scroll            | pointer  | dx, dy               |                 |
| 8   | type_text         | keyboard | token_id             |                 |
| 9   | press_key         | keyboard | keysym               |                 |
| 10  | key_chord         | keyboard | keysym               |                 |
| 11  | key_down          | keyboard | keysym               |                 |
| 12  | key_up            | keyboard | keysym               |                 |
| 13  | select_all        | keyboard | -                    |                 |
| 14  | copy              | keyboard | -                    |                 |
| 15  | paste             | keyboard | -                    |                 |
| 16  | screenshot        | screen   | -                    |                 |
| 17  | wait              | screen   | dx (ms = abs(dx)*1k) |                 |
| 18  | focus_window      | screen   | keysym (title hash)  |                 |
| 19  | minimize_window   | screen   | keysym               |                 |
| 20  | get_active_window | screen   | -                    |                 |
| 21  | get_screen_size   | screen   | -                    |                 |
| 22  | file_delete       | system   | token_id             | YES             |
| 23  | exec_shell        | system   | token_id             | YES             |

Sandbox-guarded primitives **never** execute on the real OS without the
explicit `--neuromcp-real-os` runtime flag, regardless of which backend
the actuator is configured with.

## 3. How tools emerge without JSON

The brief: *"神经元生成新工具他又该如何去生成"*.

Answer: **Hebbian wire-together-fire-together**.  Every primitive firing
is recorded into a 50-step sliding window.  When the same N-gram (any
length 2..5) of primitive ids fires >= 5 times within a 100-step
window, a new compound prototype is minted:

1. `CompoundGrowth.observe(primitive_id)` returns a new
   `CompoundPrototype` with the canonical `primitive_seq`.
2. The trainer **commits** the prototype: it allocates a new slot in
   `DynamicActionCodebook.alive_mask` (Layer 3) and writes a mean-pooled
   hidden embedding from the firings into the prototype.
3. `SparseSynapticLayer` adds a hyperedge linking the source primitive
   slots to the new prototype slot via `update_coactivation`.

There is no JSON registration anywhere in this pipeline.  The compound
"name" is its primitive_seq tuple — that's the canonical id and the
human-readable inspection token.

If the new compound is reused, STDP strengthens its codebook prototype
further.  If it goes 1000 steps without re-firing, `CompoundGrowth.tick`
returns it in the dead-list and the codebook unbinds the slot (garbage
collection).

## 4. Sandbox vs real-OS gating policy

| Mode                | Default | Real-OS pointer ops | Sandbox-guarded prims |
| ------------------- | ------- | ------------------- | --------------------- |
| `backend="sandbox"` | YES     | impossible          | always sandbox        |
| `backend="win32"`, `allow_real_os=False` | NO  | sandbox-fallback    | always sandbox        |
| `backend="win32"`, `allow_real_os=True`  | NO  | real pyautogui      | always sandbox        |
| `backend="mcp_control"` | NO  | mcp__mcp-control_*  | always sandbox        |

**Even with `allow_real_os=True`, primitives 22 and 23 (`file_delete`,
`exec_shell`) NEVER touch the real OS.**  They are routed back through
the `SandboxBackend` for their full lifetime.  The host application
that launches the agent is responsible for any policy that lets these
primitives escape.

The `--neuromcp-real-os` flag on `train_100m_kd.py` is the single
opt-in gate.  Without it, the closed-loop trainer always uses
`backend="sandbox"`.

## 5. Quality guards

* **Confidence < 0.5 -> halt.**  The `NeuroActionHead` exposes a
  per-token confidence sigmoid.  When the policy emits an action with
  confidence below `halt_threshold`, `ClosedLoopEnv.step` returns a
  `StepResult(halted=True)` with `success=False` and zero reward — no
  actuator dispatch.
* **Sandbox by default.**  All training runs happen in
  `synapforge.neuromcp.sandbox.VirtualDesktop`.
* **No torch import at module top-level.**  The whole
  `synapforge/neuromcp/` package can be imported on a torch-free
  toolbox; only `NeuroActionHead`'s constructor pulls torch in.

## 6. Closed-loop training

```
.\.venv\Scripts\python -m synapforge.training.train_100m_kd \
    --neuromcp-weight 0.05 \
    --neuromcp-closed-loop \
    --neuromcp-loop-every 10 \
    --neuromcp-loop-steps 20
```

Every 10 outer training steps, the trainer runs a 20-step rollout in
the sandbox.  The rollout:

1. Draws a primitive_id from the policy (NeuroActionHead during
   production; uniform-random during the imitation seed).
2. Calls `OSActuator.execute(primitive_id, params)`.
3. Records `(primitive_id, params, reward)` into the
   `synapforge.plasticity` STDP engine.
4. Tracks compound emergence and logs new compound ids:
   `[compound] step=2500 new_compound_id=42 = primitives(0,8,9)`

Reward back-propagates through STDP only.  Adam never sees the
closed-loop signal, by design (memory铁律
`feedback_neural_action_no_token_no_mcp`).

## 7. Demo recording (imitation seed)

```python
from synapforge.neuromcp import OSActuator, DemoRecorder

actuator = OSActuator(backend="sandbox")
rec = DemoRecorder(actuator=actuator)
rec.record_event(0, [0.1, 0.1] + [0]*6)  # click_at(0.1, 0.1)
rec.record_event(8, [0, 0, 0, 0, 1, 0, 0, 0])  # type_text(' ')
rec.record_event(9, [0, 0, 0, 0, 0, 26, 0, 0])  # press_key('enter')
rec.save("demos/seed_001.parquet")
```

Replay seeds the policy:

```python
from synapforge.neuromcp import DemoReplayer, ClosedLoopEnv

env = ClosedLoopEnv()
rep = DemoReplayer.from_file("demos/seed_001.parquet")
observations = rep.replay(env)
```

The recorder writes parquet via pyarrow when available, JSONL fallback
otherwise.  Each row has timestamp, primitive_id, params, before/after
PNG bytes, and success flag.

## 8. Honest performance estimate

* **At Run 7 (PLIF dead, 2026-Q2):** the codebook + ActionHead train
  fine — they are dense Linear projections that don't depend on PLIF
  spikes.  Compound emergence works because it's pure Python sliding-
  window logic.  But the *policy* sampling primitive_ids from the LM
  hidden state is meaningless when PLIF spike rate is 0.0001%, so
  the closed-loop loop runs through a uniform-random fallback policy.
  Sandbox success rate at random policy: ~5-15% on the 4-button
  VirtualDesktop (one of 4 buttons hit by a random click within
  ~200x200 zone).
* **When PLIF revives:** the LM hidden state finally carries non-trivial
  signal, the NeuroActionHead's `primitive_proj` begins to specialise,
  and STDP edges link "I want to type X" -> primitive 8 type_text.
  Compound emergence accelerates: instead of random N-grams, the model
  starts firing real (focus_input, type_text, press_enter) sequences
  from the imitation demos.
* **Steps to a 5-step desktop task ("open chrome, type URL, press
  enter"):** estimate 5-10k closed-loop steps after PLIF revival, on
  top of `train_ppl<=250` LM weight quality, plus 100-500 human demo
  rows for the imitation seed.  Without a quality LM backbone the
  ceiling is ~2-3 step compounds (like (focus_input, type_text)).

## 9. File layout

```
synapforge/neuromcp/
  __init__.py                # public API surface
  primitives.py              # Layer 0: 24 primitives + slot map (torch-free)
  action_head.py             # Layers 1-3: ActionHead/ParamHead/Codebook (lazy torch)
  os_actuator.py             # Layer 5: sandbox/win32/mcp dispatch (torch-free)
  sandbox.py                 # VirtualDesktop for safe training (torch-free)
  compound_growth.py         # Layer 4: Hebbian compound emergence (torch-free)
  closed_loop.py             # ClosedLoopEnv glue (torch-free)
  demo_record.py             # human-demo parquet seed (torch-free)

tests/neuromcp/
  test_primitives.py         # 24-primitive sandbox sanity
  test_compound_growth.py    # Hebbian emergence + GC
  test_closed_loop.py        # 10-step rollout + halt + reward
  test_os_actuator_dispatch.py # sandbox/win32/mcp_control routing
```

## 10. Next steps

* **Phase manager hook:** auto-enable `--neuromcp-closed-loop` when
  `train_ppl_holdout < 250` (Phase 1 gate, per memory
  `feedback_phased_training_2026q2`).
* **Real demo collection script:** `scripts/record_neuromcp_demo.py`
  with pynput keyboard+mouse listener writing parquet on disk.
* **HierarchicalCodebook integration:** wire
  `synapforge.action.compositional_codebook.HierarchicalCodebook` into
  `NeuroActionHead` so L2 compounds share the L1 cosine routing path.
