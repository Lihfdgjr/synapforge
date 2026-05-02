"""synapforge.neuromcp.compound_growth -- Hebbian compound emergence.

Layer 4 of the NeuroMCP closed-loop stack.  Watches primitive sequences
fired in a sliding window; when the same N-gram fires >= reuse_threshold
times within a window of total_window steps, a new compound prototype
is created and added to the dynamic codebook.  This is HOW new tools
emerge -- not via JSON registration, via Hebbian "wire-together-fire-
together".

Reuse and decay
---------------

    proposal_window  = 100   sliding window for proposal counting
    reuse_threshold  = 5     min repetitions inside window to mint
    n_gram_size      = 2..5  candidate compound length (configurable)
    gc_idle_steps    = 1000  no-reuse decay -- compound dies at K_alive

Public API
----------

    grower = CompoundGrowth(num_primitives=24, ...)
    grower.observe(primitive_id)
    if grower.has_pending_compound():
        compound = grower.next_compound()    # CompoundPrototype
        grower.commit(compound, slot_idx=K)  # tells grower this slot is live
    grower.tick()                            # increments step + GC

torch is **lazily imported** for the optional embedding pooler.  The
sliding-window logic is pure Python so the file is import-safe without
torch.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# CompoundPrototype -- a sequence of primitive ids + metadata.
# ---------------------------------------------------------------------------


@dataclass
class CompoundPrototype:
    compound_id: int
    primitive_seq: Tuple[int, ...]
    fire_count: int = 0
    last_fired_step: int = 0
    embedding: object = None  # torch.Tensor when pooled, else None

    def signature(self) -> Tuple[int, ...]:
        """Hashable signature -- what dedup keys on."""
        return tuple(int(p) for p in self.primitive_seq)


# ---------------------------------------------------------------------------
# CompoundGrowth -- watches primitive firings, mints new compounds.
# ---------------------------------------------------------------------------


@dataclass
class CompoundGrowth:
    """Hebbian compound emergence engine.

    Args
    ----
    num_primitives : int
        Size of the base primitive vocabulary (Layer-0 fixed set).
    proposal_window : int
        Sliding window of recent firings to count N-gram repetitions in.
    reuse_threshold : int
        Repetitions required inside the window to mint a new compound.
    n_gram_min, n_gram_max : int
        Inclusive range of compound lengths to scan.
    max_compounds : int
        Hard cap on live compounds.  When the cap is hit, GC runs first;
        if no slot is freed, mint requests are dropped silently.
    gc_idle_steps : int
        A compound that goes ``gc_idle_steps`` without re-firing is
        garbage-collected (returned by ``tick()`` so the caller can
        unbind its codebook slot).
    """

    num_primitives: int = 24
    proposal_window: int = 100
    reuse_threshold: int = 5
    n_gram_min: int = 2
    n_gram_max: int = 5
    max_compounds: int = 256
    gc_idle_steps: int = 1000

    history: Deque[Tuple[int, int]] = field(default_factory=collections.deque)
    # signature -> CompoundPrototype for already-minted compounds.
    compounds: Dict[Tuple[int, ...], CompoundPrototype] = field(default_factory=dict)
    # signature -> count window-pending compounds (already proposed,
    # not yet committed by the codebook).
    pending: List[CompoundPrototype] = field(default_factory=list)
    # next compound_id to hand out; starts at 0.
    _next_id: int = 0
    step: int = 0

    # -- observation -----------------------------------------------------
    def observe(self, primitive_id: int) -> Optional[CompoundPrototype]:
        """Record a primitive firing.  Return a new CompoundPrototype iff
        this firing closes an N-gram that crosses ``reuse_threshold``
        within ``proposal_window`` steps.

        The returned prototype is automatically added to ``self.compounds``
        and queued in ``self.pending`` so the caller can drain proposals.
        """
        primitive_id = int(primitive_id)
        if not (0 <= primitive_id < int(self.num_primitives)):
            return None
        self.history.append((self.step, primitive_id))
        # Trim history.
        cutoff = self.step - int(self.proposal_window)
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        # If we already have a live compound that ends with the most
        # recent firing, bump its fire_count and refresh last_fired_step.
        for c in self.compounds.values():
            if self._tail_matches(c.primitive_seq):
                c.fire_count += 1
                c.last_fired_step = self.step
        # Scan candidate N-grams ending at the most-recent firing.
        new_compound: Optional[CompoundPrototype] = None
        recent_seq = [p for _, p in self.history]
        for n in range(int(self.n_gram_min), int(self.n_gram_max) + 1):
            if len(recent_seq) < n:
                continue
            tail = tuple(recent_seq[-n:])
            if tail in self.compounds:
                continue
            count = self._count_in_history(recent_seq, tail)
            if count >= int(self.reuse_threshold):
                # We have a new compound candidate.
                if len(self.compounds) >= int(self.max_compounds):
                    # GC pass -- drop the oldest idle compound.
                    self._gc_one_idle()
                if len(self.compounds) >= int(self.max_compounds):
                    continue
                proto = CompoundPrototype(
                    compound_id=self._next_id,
                    primitive_seq=tail,
                    fire_count=count,
                    last_fired_step=self.step,
                )
                self._next_id += 1
                self.compounds[tail] = proto
                self.pending.append(proto)
                new_compound = proto
                # Stop at the first new compound for this step so the
                # caller can register it before the next observation
                # arrives.
                break
        return new_compound

    def tick(self) -> List[CompoundPrototype]:
        """Advance the step counter and run GC.

        Returns the list of compounds that were garbage-collected so the
        caller can unbind their codebook slots.
        """
        self.step += 1
        return self._gc_idle()

    # -- introspection ---------------------------------------------------
    def has_pending_compound(self) -> bool:
        return len(self.pending) > 0

    def next_compound(self) -> Optional[CompoundPrototype]:
        if not self.pending:
            return None
        return self.pending.pop(0)

    def commit(self, compound: CompoundPrototype, slot_idx: int,
               embedding: object = None) -> None:
        """Mark a proposed compound as live.

        The codebook caller should have allocated a real slot at
        ``slot_idx`` and (optionally) computed a mean-pooled hidden
        embedding for it -- pass it in via ``embedding`` so future
        cosine-routing inside the codebook can match it.
        """
        # We don't actually use slot_idx internally; we store it on the
        # prototype so persistence can save/restore the binding.
        compound.compound_id = int(slot_idx) if slot_idx >= 0 else compound.compound_id
        if embedding is not None:
            compound.embedding = embedding

    def get(self, primitive_seq: Sequence[int]) -> Optional[CompoundPrototype]:
        return self.compounds.get(tuple(int(p) for p in primitive_seq))

    def list_signatures(self) -> List[Tuple[int, ...]]:
        return list(self.compounds.keys())

    def stats(self) -> Dict[str, float]:
        return {
            "step": float(self.step),
            "n_compounds": float(len(self.compounds)),
            "n_pending": float(len(self.pending)),
            "history_len": float(len(self.history)),
            "next_compound_id": float(self._next_id),
        }

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _count_in_history(seq: Sequence[int],
                          subseq: Sequence[int]) -> int:
        n = len(subseq)
        if n == 0 or len(seq) < n:
            return 0
        count = 0
        for i in range(0, len(seq) - n + 1):
            if all(seq[i + j] == subseq[j] for j in range(n)):
                count += 1
        return count

    def _tail_matches(self, seq: Sequence[int]) -> bool:
        """True iff history ends with ``seq``."""
        n = len(seq)
        if n == 0 or len(self.history) < n:
            return False
        return all(
            self.history[-n + i][1] == seq[i]
            for i in range(n)
        )

    def _gc_one_idle(self) -> None:
        """Remove a single oldest-idle compound to make room."""
        if not self.compounds:
            return
        worst_sig = None
        worst_age = -1
        for sig, c in self.compounds.items():
            age = self.step - c.last_fired_step
            if age > worst_age:
                worst_age = age
                worst_sig = sig
        if worst_sig is not None:
            del self.compounds[worst_sig]

    def _gc_idle(self) -> List[CompoundPrototype]:
        dead: List[CompoundPrototype] = []
        for sig, c in list(self.compounds.items()):
            if (self.step - c.last_fired_step) > int(self.gc_idle_steps):
                dead.append(c)
                del self.compounds[sig]
        return dead


__all__ = ["CompoundPrototype", "CompoundGrowth"]
