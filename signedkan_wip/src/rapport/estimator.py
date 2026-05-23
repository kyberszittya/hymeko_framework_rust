"""Observation → signed-edge update for the rapport coalition.

The estimator maintains a running signed magnitude w_ij(t) per
relation, updated by observation events of the form
``(observation_kind, src_agent, dst_agent)``. Each observation kind
contributes a directional signed nudge; the estimator applies
exponential-moving-average smoothing so transient noise doesn't
flip the cycle σ-balance.

Symmetric edges: an observation ``(kind, a, b)`` updates the
single signed edge between agents a and b regardless of direction.
The coalition's relation lookup is bidirectional.

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

from dataclasses import dataclass

from .coalition import Coalition


# Default observation schema: kind → (sign_direction, magnitude_nudge).
# Each observation event adds `sign * magnitude_nudge` to the EMA target.
DEFAULT_SCHEMA: dict[str, tuple[int, float]] = {
    "gaze_at":          (+1, 0.30),
    "distance_close":   (+1, 0.40),
    "shared_attention": (+1, 0.50),
    "latency_long":     (-1, 0.50),
    "tone_negative":    (-1, 0.70),
    "withdrawal":       (-1, 0.60),
}


@dataclass
class Observation:
    t: int                  # time step
    kind: str               # observation kind (key in schema)
    src: str                # source agent name
    dst: str                # destination agent name


class CoalitionEstimator:
    """EMA over the signed magnitude per relation.

    State per relation r:
        w[r] ← (1 - α) · w[r] + α · target[r]
    where target[r] is the signed magnitude implied by the latest
    observation, or the prior steady-state target if no observation
    fires this step. By default the relation's initial sign × magnitude
    from the .hymeko file is the prior target.
    """

    def __init__(
        self,
        coalition: Coalition,
        schema: dict[str, tuple[int, float]] | None = None,
        alpha: float = 0.15,
        decay_to_prior: float = 0.02,
    ) -> None:
        self.coalition = coalition
        self.schema = dict(schema or DEFAULT_SCHEMA)
        self.alpha = float(alpha)
        self.decay_to_prior = float(decay_to_prior)
        # Initial weights from the .hymeko spec.
        self.weights: dict[str, float] = {
            r.name: float(r.sign) * float(r.magnitude)
            for r in coalition.relations.values()
        }
        # Steady-state prior — what each relation decays back toward
        # in the absence of observations.
        self.prior: dict[str, float] = dict(self.weights)
        # Build adjacency: (frozenset({src, dst})) → relation name.
        self._edge_index: dict[frozenset[str], str] = {}
        for r in coalition.relations.values():
            self._edge_index[frozenset({r.src, r.dst})] = r.name

    def find_relation(self, a: str, b: str) -> str | None:
        """Return the relation name connecting agents a, b (or None)."""
        return self._edge_index.get(frozenset({a, b}))

    def step(self, observations: list[Observation]) -> dict[str, float]:
        """Apply one time step's worth of observations and decay.

        Returns the post-step weights dict (copy).
        """
        # Aggregate per-relation target nudges from this step's observations.
        target_delta: dict[str, float] = {}
        for obs in observations:
            kind = obs.kind
            if kind not in self.schema:
                continue
            rel = self.find_relation(obs.src, obs.dst)
            if rel is None:
                continue
            sign_dir, mag = self.schema[kind]
            target_delta[rel] = target_delta.get(rel, 0.0) + sign_dir * mag

        # Apply EMA toward the observed target (or back toward prior).
        for r_name, w in list(self.weights.items()):
            if r_name in target_delta:
                target = max(-1.0, min(1.0, target_delta[r_name]))
                self.weights[r_name] = (1.0 - self.alpha) * w + self.alpha * target
            else:
                # Decay toward prior.
                self.weights[r_name] = (
                    (1.0 - self.decay_to_prior) * w
                    + self.decay_to_prior * self.prior[r_name]
                )
            self.weights[r_name] = max(-1.0, min(1.0, self.weights[r_name]))

        return dict(self.weights)
