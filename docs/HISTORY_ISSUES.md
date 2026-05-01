<!-- DOC_STAMP: LIVE 2026-05-02 -->
# Historical Training Bug Issues — Index

This file maps the GitHub Issues opened retrospectively (T6.4 in `DEEP_MAINT_QUEUE.md`)
to the underlying training failures, the commits that fixed them, and current status.

Issues are kept **open** — even RESOLVED ones — for stakeholder visibility. Each
issue carries a status comment so investors / reviewers can read the trajectory
without scrolling through commit logs.

Repository: <https://github.com/Lihfdgjr/synapforge>
Labels used: `bug`, `history`, `training`

---

## Issue index

| # | Title | Status | Fix commit | Source row |
|---|-------|--------|------------|------------|
| [#1](https://github.com/Lihfdgjr/synapforge/issues/1) | `[history] Adam stale momentum on warmstart (Run 3a)` | RESOLVED | [`2d8bb24`](https://github.com/Lihfdgjr/synapforge/commit/2d8bb24) | `TRAINING_ISSUES_RETROSPECTIVE.md` row 3 |
| [#2](https://github.com/Lihfdgjr/synapforge/issues/2) | `[history] Data ordering deterministic divergence at step 2500 (Run 3c)` | RESOLVED | [`bd1ac73`](https://github.com/Lihfdgjr/synapforge/commit/bd1ac73) | `TRAINING_ISSUES_RETROSPECTIVE.md` row 5 |
| [#3](https://github.com/Lihfdgjr/synapforge/issues/3) | `[history] KD softmax full-vocab OOM at bs=80` | RESOLVED | [`f32bc4a`](https://github.com/Lihfdgjr/synapforge/commit/f32bc4a) | `TRAINING_ISSUES_RETROSPECTIVE.md` row 8 |
| [#4](https://github.com/Lihfdgjr/synapforge/issues/4) | `[history] bs=80 backward OOM (z-loss full-vocab activations)` | RESOLVED | [`4d0d2a9`](https://github.com/Lihfdgjr/synapforge/commit/4d0d2a9) | `TRAINING_ISSUES_RETROSPECTIVE.md` row 7 |
| [#5](https://github.com/Lihfdgjr/synapforge/issues/5) | `[history] P30 indent regression in T2.7 grad-accum merge` | RESOLVED | [`5e5debe`](https://github.com/Lihfdgjr/synapforge/commit/5e5debe) | T2.7 / P30 fix arc |
| [#6](https://github.com/Lihfdgjr/synapforge/issues/6) | `[history] Run 3l divergence step 5500 val=2522 (killed)` | RESOLVED | [`2581da5`](https://github.com/Lihfdgjr/synapforge/commit/2581da5) | `auto-fire 01:10` summary commit `e183803` |
| [#7](https://github.com/Lihfdgjr/synapforge/issues/7) | `[history] Run 3m divergence step 15000 val=25965 (killed)` | **ONGOING** | _(Run 3n staging)_ | `auto-fire 02:35` summary commit `8209289` |

---

## Notes

* Each issue body contains: symptom, root cause, source location, fix commit (if any), and status.
* Each issue carries a status comment — `RESOLVED in <hash>` for the six fixed bugs and `ONGOING` for Run 3m.
* Cross-references to `feedback_training_root_causes_2026q2.md` and `feedback_data_ordering_divergence_2026q2.md` user-memory entries are embedded in the issue bodies.
* Do **not** close these issues — keep open for investor / stakeholder visibility.
