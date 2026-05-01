# Multi-User Memory Isolation

Status: **shipped (2026-05-01)** — `RetrievalMemory` is namespace-aware,
`ConversationKernel` plumbs `user_id` through every chat turn, the CLI
exposes `--user-id`, and 19 integration tests guard the contract.

This doc closes the question:

> **是否有 类人说话 + 真记忆 + 多用户隔离 + 不占显存?**
> Does SynapForge chat like a human, remember things, keep users
> separate, and stay off the GPU?

Yes — see the demo at `scripts/demo_multi_user_chat.py` and the verbatim
transcript at [`MULTI_USER_DEMO.md`](MULTI_USER_DEMO.md).

---

## Architecture (one paragraph)

Each user gets a private namespace under
`~/.synapforge/memory/<user_id>/` with three files: `log.jsonl`
(conversation history, append-only), `prefs.json` (learned preferences),
`index.bin` (reserved for an HNSW/FAISS drop-in; today the lookup uses
recency-weighted lexical overlap so the demo runs CPU-only with zero
optional deps). One `RetrievalMemory` instance can serve many users; it
maintains a `dict[user_id, PerUserMemory]` and resolves the namespace on
every read/write. **No model weights, no CUDA tensors, no GPU memory
ever touch this stack** — the design and the static check
(`test_no_torch_imports_in_module`) make that contractually true.

```
                 ┌──────────────────────────────┐
   chat turn ──▶ │  ConversationKernel(user_id) │
                 └──────────────┬───────────────┘
                                │
                                │  retrieval_memory.add / .query
                                ▼
                 ┌──────────────────────────────┐
                 │  RetrievalMemory             │
                 │     dict[user_id ➜ PerUser]  │
                 └──────────────┬───────────────┘
                                │
                                ▼
            ~/.synapforge/memory/<user_id>/
                ├── log.jsonl       (CPU/disk)
                ├── prefs.json      (CPU/disk)
                └── index.bin       (reserved)
```

---

## Storage layout

```
$SYNAPFORGE_MEMORY_HOME            # default: ~/.synapforge/memory/
└── <user_id>/                     # one dir per user, OS-perm isolation
    ├── log.jsonl                  # append-only chat / fact log
    ├── prefs.json                 # {key: value} learned preferences
    └── index.bin                  # 0-byte marker today; reserved for
                                   # HNSW/FAISS dense index drop-in.
```

Override the root with `SYNAPFORGE_MEMORY_HOME=/var/lib/synapforge` or
the `root_dir=` constructor arg. The legacy flat layout
(`<cache_dir>/<user_hash>.jsonl`) used by `continual_daemon` and pre-2026
runs is still recognized for backward compat — see
`PerUserMemory.legacy_log_path` in
`synapforge/learn/retrieval_memory.py`.

---

## API

```python
from synapforge.learn import RetrievalMemory

# Single-user mode (CLI default, backward compat)
mem = RetrievalMemory()                         # user_id="default"
mem.add("我喜欢猫")
mem.query("你记得我喜欢什么吗")                  # → [{"text": "我喜欢猫", ...}]

# Multi-tenant mode (one process serves many users)
mem = RetrievalMemory(root_dir="/var/lib/synapforge/memory")
mem.add("alice fact", user_id="alice")
mem.add("bob fact",   user_id="bob")
mem.query("fact", user_id="alice")              # only alice's hits
mem.purge(user_id="alice")                      # bob unchanged
mem.set_pref("favorite_animal", "cats", user_id="alice")
mem.get_pref("favorite_animal", user_id="alice")
mem.list_users()                                # ['alice', 'bob']
mem.stats()                                     # {users, total_memories, root_dir}
```

Direct namespace handle (skip the dispatcher when you have one user):

```python
ns = mem.for_user("alice")        # PerUserMemory
ns.add("hello")
ns.query("hello", top_k=4)
ns.set_pref("k", "v")
ns.purge()
```

Legacy 3-arg form (`continual_daemon` etc.) keeps working:

```python
mem.add("alice_hash", "text", 17)
mem.query("alice_hash", "text", 4)
mem.delete_user("alice_hash")
```

---

## CLI

```
synapforge-demo chat --user-id alice
synapforge-demo all  --user-id bob
```

Each invocation scopes its retrieval cache, prefs, and log to
`~/.synapforge/memory/<user_id>/`. Two simultaneous CLI sessions with
different `--user-id` values cannot see each other's memory.

---

## ConversationKernel wiring

`ConversationKernel` accepts `user_id` and an optional `retrieval_memory`
in its constructor. On every user submit:

1. The user's text is appended to `log.jsonl` under the bound `user_id`.
2. The next prompt prepends a `[memory:<user_id>]` recall block built
   from `mem.query(text, user_id=...)` — only this user's hits.
3. The model has no awareness of any other user.

```python
from synapforge.chat import ConversationKernel
from synapforge.chat.streaming import StreamingGenerator
from synapforge.learn import RetrievalMemory

mem = RetrievalMemory()
gen = StreamingGenerator(model, tokenizer)
alice_kernel = ConversationKernel(generator=gen, user_id="alice", retrieval_memory=mem)
bob_kernel   = ConversationKernel(generator=gen, user_id="bob",   retrieval_memory=mem)
```

---

## Threat model

| Threat                                           | Mitigation                                                                  |
|--------------------------------------------------|-----------------------------------------------------------------------------|
| User A reads user B's recall via API             | Every read takes `user_id`; namespace dispatch is the only entry point.     |
| User A reads user B's data via filesystem        | Each user has its own directory; set OS perms (`chmod 700`) per directory. |
| Path-traversal `user_id="../escape"`             | `_resolve_uid` rejects `/`, `\`, `.`, `..`, empty.                          |
| User A's content leaks into the *model weights*  | We don't train on user chat. Memory is retrieval-only (Track B).            |
| Cross-user contamination via curiosity / TTT     | TTT and curiosity operate on the *generic* training stream, not user chat.  |
| Compliance / GDPR delete request                 | `mem.purge(user_id=...)` removes log + prefs + index in one call.           |

The `purge` operation is bounded — it deletes only the user's own files
and `rmdir`s the directory only when empty, so it can never recursive-delete
data the system didn't write.

---

## Zero-VRAM guarantee

Two mechanisms enforce this:

1. **Static check** — `tests/integration/test_multi_user_memory.py::
   test_no_torch_imports_in_module` greps `retrieval_memory.py` and
   asserts there is no `import torch`, no `.cuda(...)`, no
   `torch.cuda`, no `.to('cuda')`. Add a torch import to the file and
   CI fails immediately.
2. **Dynamic check** — when CUDA is available,
   `test_vram_does_not_grow_during_add` asserts
   `torch.cuda.memory_allocated()` is unchanged after 100 add+query
   operations.

The investor-facing demo (`scripts/demo_multi_user_chat.py`) prints the
delta on stdout and writes it to `docs/MULTI_USER_DEMO.md`.

---

## Tests

`tests/integration/test_multi_user_memory.py` (20 cases):

- Two-user isolation (5 facts × 2 users, no crosstalk on query)
- `lookup` alias matches `query`
- Purge only affects target user
- Purge idempotent (True then False)
- Persistence across instances (dump → reopen → 5 facts intact)
- Storage layout (`log.jsonl` / `prefs.json` / `index.bin`)
- `list_users` discovery
- Per-user prefs isolation
- Backward-compat 3-arg `add(user_hash, text, sample_id)`
- Default `user_id="default"` works without explicit kwarg
- Path-traversal `user_id`s rejected (`../`, `/`, `\`, `.`, `..`, empty)
- `user_id=None` falls back to bound default
- VRAM dynamic check (skipped without CUDA; structural check covers)
- VRAM static check (no torch in module)
- `PerUserMemory` standalone surface

Run:

```bash
pytest tests/integration/test_multi_user_memory.py -v
```

---

## Cross-references

- [`MASTER_PLAN.md`](MASTER_PLAN.md) §5 — feature checklist row
  "Multi-user memory isolation".
- [`MULTI_USER_DEMO.md`](MULTI_USER_DEMO.md) — verbatim alice + bob
  transcript dumped by the demo script.
- `synapforge/learn/retrieval_memory.py` — implementation.
- `synapforge/chat/event_loop.py` — `ConversationKernel(user_id=...)`.
- `synapforge/demo/cli.py` — `--user-id` flag.
- `scripts/demo_multi_user_chat.py` — runnable end-to-end demo.

---

## What this is **not**

- Not a knowledge-base shared across users (by design — privacy-first).
- Not a vector index today (lexical fallback ships; HNSW drop-in slot
  reserved at `index.bin`).
- Not a substitute for end-to-end encryption — OS file permissions are
  the trust boundary. Set `chmod 700 ~/.synapforge/memory/<user_id>/`
  on a multi-tenant box.
