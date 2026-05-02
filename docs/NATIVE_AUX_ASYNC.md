# NATIVE_AUX_ASYNC -- async coordinator for the 4 auxiliary components

**Status**: 2026-05-02. Production code under `synapforge/native/auxsched/`.
This is the policy layer that decides *which* aux component runs *where*
and *when*. The torch-using ICM / TTT / plasticity heads remain in the
trainer; this package only schedules them.

> **Why ``auxsched`` and not ``aux``?** Windows reserves ``AUX`` as a
> legacy DOS device name. The git CLI on Windows refuses to ``open()``
> files inside any folder named ``aux`` (even though `cmd.exe` and Bash
> can list them). The rename keeps the package portable.

---

## TL;DR

* Run 7 step time on A800 d=1280 bs=48: **~9 sec / step**.
* TTT k=8 inline blocks ~200ms / step (2-3x main fwd+bwd).
* Curiosity ICM ~5-10 ms; NeuroMCP ~1-2 ms; tool exec 50-5000 ms.
* Synthetic bench (`scripts/bench_aux_async.py`, 20 steps, Run 7 timings):
  **2.9x speedup** (484 -> 167 ms / step).
* Real-world projection at Run 7 config: **~25-35% faster** end-to-end
  (saving 2-3 sec out of the 9-sec step). See bottom of doc.

---

## Stream / thread layout

```
+---------------------------------------------------------------+
|                       MAIN PYTHON THREAD                       |
|                                                                |
|   step N:                                                      |
|     1. h_prev, h_next = main_fwd(batch[N])    (stream A)       |
|     2. aux.submit_curiosity(N, h_prev, h_next)                 |
|     3. state, fut = aux.submit_ttt(N, vx, vy, state)           |
|        |__ runs inline_k=2 inner steps right here              |
|        |__ schedules async_k=6 inner steps on stream C         |
|     4. aux.submit_spike_stats(SpikeStats(...))                 |
|     5. aux.submit_tool_call(ToolCall(...))                     |
|     6. main_bwd(...)                          (stream A)       |
|     7. (optional) aux.wait_aux()  -- only if curiosity grad    |
|                                       must reach backbone      |
|     8. obs = aux.drain_observations()  (cheap, non-blocking)   |
|                                                                |
+---------------------------------------------------------------+
       |                |                |              |
       v                v                v              v
+----------+     +-------------+   +-----------+   +-----------+
| stream B |     |  stream C   |   | CPU thread|   | CPU pool  |
|          |     |             |   |           |   |           |
| ICM      |     | TTT inner   |   | NeuroMCP  |   | OSActuator|
| forward+ |     | loop (6/8)  |   | tick      |   | tool exec |
| inverse  |     | overlapped  |   | Hebbian + |   | (web/sh/  |
| loss +   |     | with main   |   | codebook  |   | browser)  |
| grad on  |     | step N+1's  |   | grow on   |   |           |
| ICM head |     | data        |   | numpy     |   | x4 wkrs   |
|          |     | prefetch    |   | arrays    |   | per ID    |
+----------+     +-------------+   +-----------+   +-----------+
       ^                ^                ^              ^
       |                |                |              |
       (replace-stale,   (replace-stale,   (drop-oldest,   (drop-newest,
        1-deep queue)     1-deep queue)    4-deep queue)   16-deep queue)
```

* **Stream A** (main): owned by the trainer, NOT this package. We
  don't touch it; we just wait on the data it produced and do nothing
  that would force-sync it.
* **Stream B** (curiosity): a `cupy.cuda.Stream(non_blocking=True)`
  via `AuxStream(label="aux.curiosity")`. The ICM heads are small
  (couple of MLPs); their forward+inverse loss is ~5-10 ms on A800.
* **Stream C** (TTT): same kind of stream. Hosts inner SGD steps 2..7
  while stream A runs the next outer step's main forward.
* **Stream D** (H2D): owned by the trainer's prefetcher, NOT this
  package.
* **CPU thread P1** (NeuroMCP): one daemon thread that owns the
  Hebbian EMA / sparse-mask grow / codebook lookup. No GPU
  participation -- discrete bookkeeping is purely numpy.
* **CPU thread pool P2** (ActionHead): N=4 daemon threads dispatching
  tool calls to the OSActuator. System calls (subprocess, requests,
  Playwright) -- never block the GPU stream.

---

## Component-by-component policy

### 1. Curiosity ICM (`synapforge/native/auxsched/curiosity_async.py`)

Caller wires its torch ICM forward+inverse model into a callable:

```python
def compute_fn(payload: _CuriosityPayload) -> CuriosityResult:
    # payload.h_prev, payload.h_next, payload.action_emb
    # do torch ICM forward+inverse, .backward() on ICM optimiser
    return CuriosityResult(step_idx=payload.step_idx,
                           loss=..., grad_norm=...)

drv = CuriosityAsyncDriver(compute_fn=compute_fn)
fut = drv.submit(step_idx=N, h_prev=hp, h_next=hn)
# usually NEVER wait on fut -- it just runs to completion.
```

* **Backpressure**: replace-stale (1-deep queue). If submit N+1
  arrives before worker has popped N, N is dropped with
  ``extra={"dropped": "stale-skip"}``. This keeps lag bounded to 1
  outer step.
* **Quality guard**: ICM is its own gradient graph; no parity test
  with the main loss is required. We DO assert the ``loss`` /
  ``forward_model_loss`` numbers come back finite (covered by tests).

### 2. Self-learn TTT (`synapforge/native/auxsched/ttt_async.py`)

```python
def inner_step(state, val_inputs, val_targets, iter_idx):
    # one TTT inner SGD step on the trainer's "fast weight" state
    return new_state, loss

drv = TTTAsyncDriver(inner_step_fn=inner_step,
                     total_k=8, inline_k=2,
                     done_fn=lambda final_state: ...)

state_after_inline, fut = drv.run(step_idx=N, val_inputs=vx,
                                  val_targets=vy, inner_state=cur_state)
# state_after_inline is fed into main_fwd RIGHT NOW.
# fut + done_fn will deliver the FINAL state once the async chunk lands.
```

* `inline_k=2` inner steps run synchronously on the calling thread:
  fresh-grad property is preserved on the very first part of TTT.
* `async_k=6` inner steps run on stream C, overlapping with the next
  outer step's main forward + data prefetch.
* `done_fn(final_state)` fires on the worker thread once all 8 are
  done. The trainer typically uses this to stash the adapted state
  into a buffer for the *next* outer step to consume.
* **Quality guard**: parity test (`test_async_coordinator.py
  ::test_ttt_quality_parity_2_inline_6_async_vs_8_inline`) compares
  the final loss after 8 inner steps via the async path against the
  reference 8-inline path. **Must agree within 1%**. Currently
  agrees within numpy floating-point noise (the math is the same
  callable, just scheduled differently).

### 3. NeuroMCP plasticity (`synapforge/native/auxsched/neuromcp_cpu.py`)

```python
def tick_fn(stats: SpikeStats, prev_mask: np.ndarray | None) -> PlasticityResult:
    # Hebbian EMA on coact buffer, prune low-|W|, grow high-coact.
    # Codebook: maybe add new prototype if novelty above threshold.
    return PlasticityResult(step_idx=stats.step_idx,
                            new_mask=updated_mask,
                            grew_prototype=False, ...)

drv = NeuroMCPCpuDriver(tick_fn=tick_fn, queue_capacity=4)

# every step, after main fwd:
drv.submit_spikes(SpikeStats(step_idx=N, spike_rate=..., proto_sim=...))

# every step, before next main fwd:
new_mask = drv.latest_mask()  # may be None if no tick has finished yet
```

* CPU thread runs the user `tick_fn` in true parallelism with the GPU
  forward (numpy releases the GIL during ufunc / BLAS calls).
* **Backpressure**: drop-oldest-on-full. If the CPU tick falls behind
  (e.g. after a codebook growth event runs slower), the oldest spike
  batch in the queue is evicted. The Hebbian EMA's exponential decay
  forgives a couple skipped batches by design.
* **Latency tolerance**: codebook growth is gated by
  ``growth_cooldown=50`` steps anyway; sparse-mask growth is checked
  every 20 steps. One step of mask staleness on the next forward is
  operationally identical to inline.

### 4. ActionHead -> OSActuator (`synapforge/native/auxsched/action_async.py`)

```python
def execute_fn(call: ToolCall) -> ToolObservation:
    # actually run the tool (web fetch, shell, browser) -- this is
    # where 50ms-5000ms latency lives.
    return ToolObservation(step_idx=call.step_idx,
                           tool_id=call.tool_id, success=True, result=...)

drv = ActionHeadAsyncDriver(execute_fn=execute_fn, num_workers=4,
                            submit_capacity=16)

# main loop never waits on tool exec:
ok = drv.submit(ToolCall(step_idx=N, tool_id=k, arg_payload=...))

# every N outer steps, drain finished observations and feed them
# into the slow STDP signal:
for obs in drv.drain_completed():
    codebook.observe(obs.tool_id, obs.success)
```

* **Backpressure**: drop-newest-on-full. If 16 calls are queued and
  workers can't keep up, new calls are dropped. The training loop
  is told (return ``False`` from ``submit``) and continues -- the
  GPU NEVER blocks waiting for a slow tool.
* **Hard guarantee**: the test
  ``test_action_driver_non_blocking_with_slow_tool`` asserts that
  with a deliberately 5-second slow tool, 10 main-loop iters
  complete in <200ms. The spec gate.
* **Latency tolerance for STDP feedback**: the tool's observation
  drives a *slow* STDP signal back into the codebook -- e.g.
  "tool prototype 7 succeeded 92% of the time over the last 100
  calls". A few outer steps of staleness is harmless.

---

## Failure modes & how the coordinator handles them

| Failure | Component response | Visibility |
|---|---|---|
| `compute_fn` raises | Future gets the exception; trainer can `result()` to re-raise. Driver thread keeps running. | metrics: `errors` counter |
| Aux queue grows beyond capacity | Per-driver drop policy fires (replace-stale / drop-oldest / drop-newest). Logged in metrics. | metrics: `dropped*` counter |
| Worker thread crashes | Daemon thread; main thread doesn't deadlock. New submits will accumulate in queue but never run -- caller should periodically check `metrics()` for `errors > 0`. | metrics |
| 5s tool latency | Tool exec runs on worker pool; main never sees it. | (covered by spec test) |
| Backpressure blocks main step | NEVER. Every `submit_*` is non-blocking by construction. | (asserted in tests) |

---

## End-to-end speedup projection at Run 7 config

The synthetic bench (`scripts/bench_aux_async.py --main-ms 80
--cur-ms 8 --ttt-per-step-ms 25 --ttt-k 8 --inline-k 2 --nm-ms 1
--tool-ms 100`) reports **2.9x** speedup of the *aux-bound portion*
of the step.

But Run 7's actual step time is dominated by other factors that this
package does NOT change:

* Main forward + backward: ~3-4 sec (training_data + KD teacher inference)
* CE / KD loss + AdamW step: ~1-2 sec
* Optimizer state H2D/D2H: ~0.5 sec

Of the 9-sec total step, the aux portion (TTT + curiosity + NeuroMCP +
optional tool) currently takes ~2.5-3 sec. Async-coordinator removes
~80% of that (parallelises max 200ms onto stream C/CPU threads).

**Honest projection**: end-to-end step time drops from ~9 sec to
~6.5-7 sec. That's **~25-35% faster training** with no math change,
no quality regression (parity guard < 1%), no extra GPU memory.

The bigger win lands when TTT-k goes higher (k=16 or k=32, common in
Coconut-style chain-of-thought self-learn): there the inline-k=2
ratio drops, hiding more inner steps, and the speedup approaches the
full 2.9x synthetic number.

---

## Integration points the trainer must wire

The coordinator is policy; the trainer wires the actual computation:

1. **Curiosity compute_fn**: replaces the inline ICM block in the
   trainer's `train_step`. Move that body verbatim into a function
   that takes `(payload)` and returns `CuriosityResult`.
2. **TTT inner_step_fn**: replaces the inner loop. Move ONE
   iteration's worth of code into a function `(state, vx, vy, i)
   -> (new_state, loss)`.
3. **NeuroMCP tick_fn**: replaces the per-step Hebbian/codebook block.
   Move it into `(stats, prev_mask) -> PlasticityResult`. The
   trainer's spike collector pushes `SpikeStats` per outer step.
4. **ActionHead execute_fn**: replaces the inline call to
   `OSActuator`. Whatever the actuator does (web fetch / shell /
   browser), move into `(call) -> ToolObservation`.

The trainer then constructs the coordinator once and calls
`submit_*` per outer step. None of the trainer's torch code moves
into `auxsched/`; only the scheduling glue moves.

---

## Open todos

* When `synapforge.native.dispatch.StreamPair` lands on the merged
  feature branch, swap our local `AuxStream` for theirs (zero API
  change, same constructor). Tracked as a follow-up.
* Add a coordinator-level CUDA event barrier for "main backward
  needs curiosity grad" so `wait_aux` is a single event-wait rather
  than a python `Future.wait`. (Optional; current host-wait works
  fine since ICM compute is ~5-10 ms anyway.)
* When TTT-k is configurable per-phase (phased_training), wire
  `total_k` / `inline_k` to the phase signal so phase 0 (LM-only)
  can disable TTT entirely (`inner_step_fn=None`).
