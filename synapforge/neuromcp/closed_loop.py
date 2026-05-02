"""synapforge.neuromcp.closed_loop -- step-the-world env loop.

Layer-5 dispatch + Layer-4 compound emergence + Layer-1/2 ActionHead.

The env loop is the production glue:

    obs0 = env.reset()
    for t in range(T):
        action = model.step(obs)            # primitive_id, params, conf
        obs    = env.step(action)           # OSActuator.execute -> obs
        env.update_compound_growth(action)  # record firing
        env.update_episodic_memory(obs)     # for ICM curiosity
        if obs["success"]: env.reward(+1)
        else:               env.reward(-0.5)

torch is **lazily imported** inside ``ClosedLoopEnv``.  The class does
not require torch when used in ``mode="record"`` (recording demos).

Reward / STDP wiring
--------------------

The brief specifies STDP-only credit assignment for the closed-loop
reward:

    on success: STDP strengthens (hidden, primitive_id) edge
    on failure: STDP weakens that edge + propose new prototype

We do *not* invoke an Adam-style backprop step from this loop -- that
is the point of the user铁律 ``feedback_neural_action_no_token_no_mcp``.
The trainer wire-in (see ``train_100m_kd.py --neuromcp-closed-loop``)
hooks ``ClosedLoopEnv.fire`` into the synapforge plasticity engine
directly; here we just expose the (hidden, primitive_id, reward) tuples.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .compound_growth import CompoundGrowth, CompoundPrototype
from .os_actuator import ObservationDict, OSActuator
from .primitives import NUM_PRIMITIVES


# ---------------------------------------------------------------------------
# StepResult -- one closed-loop transition record.
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    primitive_id: int
    params: List[float]
    confidence: float
    reward: float
    success: bool
    obs: ObservationDict
    new_compound: Optional[CompoundPrototype] = None
    halted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "primitive_id": int(self.primitive_id),
            "params": list(self.params),
            "confidence": float(self.confidence),
            "reward": float(self.reward),
            "success": bool(self.success),
            "obs": self.obs.as_dict() if self.obs else None,
            "new_compound": (
                {
                    "compound_id": self.new_compound.compound_id,
                    "primitive_seq": list(self.new_compound.primitive_seq),
                }
                if self.new_compound is not None else None
            ),
            "halted": bool(self.halted),
        }


# ---------------------------------------------------------------------------
# Action policy -- what the env asks for per step.
# ---------------------------------------------------------------------------


# A "policy" callable returns (primitive_id, params, confidence) given
# the previous ObservationDict.  We accept either a callable directly
# or a NeuroActionHead-shaped object.
PolicyFn = Callable[[ObservationDict], Tuple[int, Sequence[float], float]]


# ---------------------------------------------------------------------------
# ClosedLoopEnv
# ---------------------------------------------------------------------------


@dataclass
class ClosedLoopEnv:
    """Step-by-step closed-loop env.

    Args
    ----
    actuator : OSActuator
        Backend that executes primitives.  Default = sandbox OSActuator
        (1024x768).
    growth : CompoundGrowth | None
        Hebbian compound emergence engine.  Default None -> auto-build.
    episodic_capacity : int
        Ring buffer size for ICM-style curiosity reward (last K observations).
    success_reward : float
        Reward for ``obs.success=True``.  Default +1.0.
    failure_reward : float
        Reward for ``obs.success=False``.  Default -0.5.
    halt_reward : float
        Reward when policy returns confidence < halt_threshold.  Default 0.
    halt_threshold : float
        Confidence below which the loop halts and asks for human
        confirmation (per the brief).  Default 0.5.
    seed : int
        Optional rng seed for reproducible sandbox runs.
    """

    actuator: Optional[OSActuator] = None
    growth: Optional[CompoundGrowth] = None
    episodic_capacity: int = 32
    success_reward: float = 1.0
    failure_reward: float = -0.5
    halt_reward: float = 0.0
    halt_threshold: float = 0.5
    seed: int = 0

    episodic_memory: Deque[ObservationDict] = field(default_factory=collections.deque)
    history: List[StepResult] = field(default_factory=list)
    cumulative_reward: float = 0.0
    _rng: random.Random = field(default_factory=lambda: random.Random(0))
    _last_obs: Optional[ObservationDict] = None

    def __post_init__(self) -> None:
        if self.actuator is None:
            self.actuator = OSActuator(backend="sandbox")
        if self.growth is None:
            self.growth = CompoundGrowth(num_primitives=NUM_PRIMITIVES)
        if self.seed:
            self._rng = random.Random(int(self.seed))
        self.episodic_memory = collections.deque(maxlen=int(self.episodic_capacity))

    # -- core ------------------------------------------------------------
    def reset(self) -> ObservationDict:
        """Reset the env -- empties episodic memory and re-snapshots."""
        self.episodic_memory.clear()
        self.history.clear()
        self.cumulative_reward = 0.0
        # Take an initial screenshot so the policy has something to
        # condition on.
        obs = self.actuator.execute(16, [0.0] * 8)  # primitive 16 = screenshot
        self._last_obs = obs
        self.episodic_memory.append(obs)
        return obs

    def step(self, primitive_id: int, params: Sequence[float],
             confidence: float = 1.0) -> StepResult:
        """Run one closed-loop step.

        Side effects:
          - actuator.execute -> ObservationDict
          - growth.observe(primitive_id) -> maybe a new CompoundPrototype
          - episodic_memory.append(obs)
          - history.append(StepResult)

        Returns
        -------
        StepResult with reward + new_compound + halted flags filled.
        """
        if confidence < float(self.halt_threshold):
            # Halt path -- no actuator call.
            res = StepResult(
                primitive_id=int(primitive_id),
                params=list(params),
                confidence=float(confidence),
                reward=float(self.halt_reward),
                success=False,
                obs=ObservationDict(
                    success=False, primitive_id=int(primitive_id),
                    text=f"halted: confidence={confidence:.3f}",
                    error_msg="confidence_below_threshold",
                ),
                halted=True,
            )
            self.history.append(res)
            return res
        obs = self.actuator.execute(int(primitive_id), list(params))
        new_compound = self.growth.observe(int(primitive_id))
        self.growth.tick()
        self.episodic_memory.append(obs)
        self._last_obs = obs
        reward = self.success_reward if obs.success else self.failure_reward
        # Curiosity bonus: novelty against the last K observations.  We
        # use a cheap hash-based novelty (no torch) so this stays
        # zero-dep.  It complements the Adam-free STDP path the trainer
        # wires up separately.
        reward = float(reward) + self._novelty_bonus(obs)
        res = StepResult(
            primitive_id=int(primitive_id),
            params=list(params),
            confidence=float(confidence),
            reward=reward,
            success=bool(obs.success),
            obs=obs,
            new_compound=new_compound,
            halted=False,
        )
        self.history.append(res)
        self.cumulative_reward += reward
        return res

    # -- policy roll-outs ------------------------------------------------
    def rollout(self, policy: PolicyFn, n_steps: int = 20,
                reset: bool = True) -> List[StepResult]:
        """Run n_steps with a policy callable.

        ``policy(obs) -> (primitive_id, params, confidence)``.  This is
        synchronous and torch-free as long as the caller's policy is.
        """
        if reset:
            self.reset()
        results: List[StepResult] = []
        last_obs = self._last_obs
        for _ in range(int(n_steps)):
            primitive_id, params, conf = policy(last_obs)
            res = self.step(primitive_id, params, conf)
            last_obs = res.obs
            results.append(res)
        return results

    # -- helpers ---------------------------------------------------------
    def _novelty_bonus(self, obs: ObservationDict) -> float:
        """Cheap ICM proxy: +0.05 reward when obs differs from last K."""
        if not self.episodic_memory:
            return 0.0
        sig = (obs.primitive_id, obs.success, obs.text[:32], obs.error_msg[:32])
        seen = sum(
            1 for o in self.episodic_memory
            if (o.primitive_id, o.success, o.text[:32], o.error_msg[:32]) == sig
        )
        if seen <= 1:
            return 0.05
        return 0.0

    def fire(self) -> Tuple[int, List[float], bool]:
        """Return (primitive_id, params, success) of the most recent step.

        Trainer can subscribe to this to drive STDP updates: when the
        most recent step succeeded, strengthen the (hidden, primitive_id)
        edge; when it failed, weaken it.
        """
        if not self.history:
            return -1, [0.0] * 8, False
        last = self.history[-1]
        return int(last.primitive_id), list(last.params), bool(last.success)

    def stats(self) -> Dict[str, float]:
        n = len(self.history)
        wins = sum(1 for r in self.history if r.success)
        halts = sum(1 for r in self.history if r.halted)
        return {
            "n_steps": float(n),
            "n_success": float(wins),
            "n_halted": float(halts),
            "success_rate": float(wins) / max(1, n),
            "halt_rate": float(halts) / max(1, n),
            "cumulative_reward": float(self.cumulative_reward),
            "n_compounds": float(len(self.growth.compounds)) if self.growth else 0.0,
        }


__all__ = ["ClosedLoopEnv", "StepResult", "PolicyFn"]
