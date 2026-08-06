"""Synthetic HRI event stream generator with scripted conflict scenarios.

The simulator's role is to falsify the framework's σ-cycle balance
representation under known dynamics — not to replicate real HRI data.
The joint pilot study after the Nagoya visit is where real-data
validation lives.

Default behaviour over a triadic alice–bob–r1 coalition:

* Baseline (t < t_conflict_start): low-rate stream of positive
  observations across all three dyads (gaze_at, distance_close,
  shared_attention). Each frame ~3 observations.
* Conflict window (t_conflict_start ≤ t < t_conflict_end): a burst
  of negative observations between alice and bob
  (tone_negative, withdrawal).
* Repair window (after robot intervention): if the robot's policy
  has fired a ``signal_alignment`` action recently, increase the
  positive observation rate proportionally on the dyad that was
  imbalanced.

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coalition import Coalition
from .estimator import Observation


@dataclass
class ConflictScenario:
    """Scripted conflict between two named agents over a time window."""
    start: int
    end: int
    agent_a: str
    agent_b: str
    kinds: tuple[str, ...] = ("tone_negative", "withdrawal")


@dataclass
class SimulatorConfig:
    """Knobs for the synthetic generator."""
    n_frames: int = 600                  # total simulation length (frames)
    baseline_rate: float = 3.0           # avg observations per frame, baseline
    conflict_rate: float = 4.0           # avg observations per frame, in-conflict
    baseline_kinds: tuple[str, ...] = (
        "gaze_at", "distance_close", "shared_attention",
    )
    conflict_scenarios: tuple[ConflictScenario, ...] = ()
    repair_window: int = 30              # frames of elevated positives after repair_action
    repair_kinds: tuple[str, ...] = ("gaze_at", "shared_attention")
    seed: int = 0


def _pairs(coalition: Coalition) -> list[tuple[str, str]]:
    """All unordered agent pairs that have a defined relation."""
    out: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for r in coalition.relations.values():
        key = frozenset({r.src, r.dst})
        if key in seen:
            continue
        seen.add(key)
        out.append((r.src, r.dst))
    return out


class Simulator:
    """Generate per-frame observation lists for the rapport demo."""

    def __init__(self, coalition: Coalition, config: SimulatorConfig) -> None:
        self.coalition = coalition
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._pairs = _pairs(coalition)
        self._repair_until: int = -1   # frame index up to which repair-window holds

    def trigger_repair(self, t: int) -> None:
        """Called by the policy module when a repair action fires.

        Extends the simulator's repair window so the next ``repair_window``
        frames get an elevated rate of positive observations.
        """
        self._repair_until = max(self._repair_until, t + self.config.repair_window)

    def _conflict_active(self, t: int) -> ConflictScenario | None:
        for cs in self.config.conflict_scenarios:
            if cs.start <= t < cs.end:
                return cs
        return None

    def step(self, t: int) -> list[Observation]:
        """Emit the observation list for frame ``t``."""
        cs = self._conflict_active(t)
        observations: list[Observation] = []

        if cs is not None:
            # Conflict burst on the (a, b) dyad. If a repair action
            # has fired recently (within self._repair_until), the
            # robot's intervention dampens the conflict: the negative-
            # observation rate is multiplied by 0.3, and a small
            # positive stream on the conflict dyad is added (the
            # robot's signal_alignment encourages reattunement).
            damping = 0.3 if t < self._repair_until else 1.0
            n_neg = int(self._rng.poisson(self.config.conflict_rate * damping))
            for _ in range(n_neg):
                kind = self.config.conflict_scenarios[0].kinds[
                    int(self._rng.integers(len(cs.kinds)))
                ]
                observations.append(Observation(
                    t=t, kind=kind, src=cs.agent_a, dst=cs.agent_b,
                ))
            # Background positives elsewhere (low rate)
            n_bg = int(self._rng.poisson(self.config.baseline_rate / 2))
            for _ in range(n_bg):
                a, b = self._pairs[int(self._rng.integers(len(self._pairs)))]
                if frozenset({a, b}) == frozenset({cs.agent_a, cs.agent_b}):
                    continue
                kind = self.config.baseline_kinds[
                    int(self._rng.integers(len(self.config.baseline_kinds)))
                ]
                observations.append(Observation(t=t, kind=kind, src=a, dst=b))
            # During repair window: add positive nudges on the
            # conflict dyad itself (robot's intervention reattunes
            # the human-human relation, not just adjacent ones).
            if t < self._repair_until:
                n_repair_pos = int(self._rng.poisson(self.config.baseline_rate))
                for _ in range(n_repair_pos):
                    kind = self.config.repair_kinds[
                        int(self._rng.integers(len(self.config.repair_kinds)))
                    ]
                    observations.append(Observation(
                        t=t, kind=kind, src=cs.agent_a, dst=cs.agent_b,
                    ))
        else:
            # Baseline: random pairs, random positive observation kinds.
            rate = self.config.baseline_rate
            if t < self._repair_until:
                rate *= 1.6   # elevated rate during repair window
            n = int(self._rng.poisson(rate))
            for _ in range(n):
                a, b = self._pairs[int(self._rng.integers(len(self._pairs)))]
                kinds = self.config.repair_kinds if t < self._repair_until \
                    else self.config.baseline_kinds
                kind = kinds[int(self._rng.integers(len(kinds)))]
                observations.append(Observation(t=t, kind=kind, src=a, dst=b))

        return observations
