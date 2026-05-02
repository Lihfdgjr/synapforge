"""Quality guard: roll back self-drive STDP weight changes that hurt val ppl.

The self-drive coordinator may run an inner loop of "imagined" pseudo-
batches where the only weight pathway is STDP (NOT AdamW; the STDP
trace is the bypass-optimizer learning rule per
``feedback_neural_action_no_token_no_mcp.md``). Because the pseudo
batches are synthetic, the trainer must verify they do not corrupt
val_ppl on the real holdout set. If post-step val regresses past the
configured tolerance (default 5%), restore the snapshotted weights.

Pure-Python coordination (no torch import). The actual snapshot/restore
is done via the user-supplied callbacks so we stay test-clean and
back-end-agnostic.

Usage:

    g = QualityGuard(max_regression=0.05)
    g.snapshot(snapshot_fn=lambda: trainer.snapshot_stdp())
    trainer.run_self_drive_step(...)        # mutates STDP weights
    pre = baseline_ppl
    post = trainer.eval_on_holdout()
    decision = g.verify(pre_ppl=pre, post_ppl=post,
                        restore_fn=lambda snap: trainer.restore_stdp(snap))
    log_msg = g.format_decision(decision)   # human-readable line
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class GuardDecision:
    """Result of one ``QualityGuard.verify`` call."""

    pre_ppl: float
    post_ppl: float
    threshold_ppl: float
    rolled_back: bool
    reason: str = ""


@dataclass
class QualityGuard:
    """Roll back STDP weight changes if val ppl regresses past tolerance.

    Parameters
    ----------
    max_regression : float
        Fractional tolerance. ``0.05`` rolls back when post_ppl > 1.05*pre_ppl.
    rollback_on_nan : bool
        If True (default), rollback when post is NaN/inf even if pre is finite.
    """

    max_regression: float = 0.05
    rollback_on_nan: bool = True

    # Snapshot is whatever the snapshot_fn returns (typically a dict of
    # cloned tensors held by the caller). We just pass it through to
    # restore_fn unchanged.
    _snapshot: Any = field(default=None, init=False)
    _has_snapshot: bool = field(default=False, init=False)

    n_calls: int = field(default=0, init=False)
    n_rollbacks: int = field(default=0, init=False)
    n_kept: int = field(default=0, init=False)

    # ------------------------------------------------------------------
    # snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, snapshot_fn: Callable[[], Any]) -> None:
        """Capture pre-self-drive weights via the supplied callback."""
        try:
            self._snapshot = snapshot_fn()
            self._has_snapshot = True
        except Exception:
            self._snapshot = None
            self._has_snapshot = False

    def discard_snapshot(self) -> None:
        """Release whatever the snapshot_fn returned (e.g. tensors)."""
        self._snapshot = None
        self._has_snapshot = False

    # ------------------------------------------------------------------
    # verify
    # ------------------------------------------------------------------
    def verify(
        self,
        pre_ppl: float,
        post_ppl: float,
        restore_fn: Optional[Callable[[Any], None]] = None,
    ) -> GuardDecision:
        """Decide rollback / keep based on (pre, post) val ppl.

        ``restore_fn(snapshot)`` is called when rollback fires. The
        snapshot kept inside the guard is then discarded.
        """
        self.n_calls += 1
        pre = float(pre_ppl)
        post = float(post_ppl)
        threshold = pre * (1.0 + float(self.max_regression))

        rolled_back = False
        reason = "within-tolerance"
        # Fail-open if pre is non-finite (no signal -> assume KEEP).
        if not _is_finite(pre):
            reason = "pre-non-finite"
        elif self.rollback_on_nan and not _is_finite(post):
            rolled_back = True
            reason = "post-non-finite"
        elif post > threshold:
            rolled_back = True
            reason = (
                f"post>{self.max_regression * 100:.1f}%-of-pre "
                f"(post={post:.2f} > {threshold:.2f})"
            )

        if rolled_back:
            if restore_fn is not None and self._has_snapshot:
                try:
                    restore_fn(self._snapshot)
                except Exception as exc:
                    reason += f" + restore_fn-failed:{type(exc).__name__}"
            self.n_rollbacks += 1
        else:
            self.n_kept += 1

        # Whether we rolled back or kept, the snapshot is consumed: the
        # caller's next self-drive cycle should snapshot afresh.
        self.discard_snapshot()

        return GuardDecision(
            pre_ppl=pre, post_ppl=post,
            threshold_ppl=threshold,
            rolled_back=rolled_back,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def format_decision(d: GuardDecision) -> str:
        verdict = "ROLLBACK" if d.rolled_back else "KEEP"
        return (
            f"[self-drive guard] pre_ppl={d.pre_ppl:.2f} "
            f"post_ppl={d.post_ppl:.2f} thr={d.threshold_ppl:.2f} "
            f"{verdict} ({d.reason})"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "n_rollbacks": self.n_rollbacks,
            "n_kept": self.n_kept,
            "rollback_rate": (self.n_rollbacks / self.n_calls)
            if self.n_calls else 0.0,
            "max_regression": self.max_regression,
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _is_finite(x: float) -> bool:
    """True iff ``x`` is finite (not NaN, not +/-inf)."""
    return x == x and x not in (float("inf"), float("-inf"))
