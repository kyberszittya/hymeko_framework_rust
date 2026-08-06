"""σ-cycle coherence over a signed coalition graph.

Implements the Cartwright–Harary (1956) signed-cycle balance score
restricted to a single cycle: the σ-product over the cycle's edges,
scaled by the minimum confidence magnitude (the weakest edge bounds
the cycle's coherence).

For a triadic 3-cycle on edges (e_1, e_2, e_3) with current
signed-magnitude weights w_i = sign_i × magnitude_i:

    σ(cycle) = sign(w_1 · w_2 · w_3) × min(|w_1|, |w_2|, |w_3|)

σ ∈ [-1, +1]. σ > 0 = balanced cycle (an even number of negative
edges). σ < 0 = imbalanced.

Plan: docs/plans/2026-05-18-rapport-coherence-demo-nagoya/.
"""
from __future__ import annotations

from dataclasses import dataclass

from .coalition import Coalition, SigmaCycle


def signed_magnitude(sign: int, magnitude: float) -> float:
    """Convert a (sign, magnitude) pair to a signed scalar w in [-1, +1]."""
    return float(sign) * float(magnitude)


def sigma_cycle(
    weights: dict[str, float],
    cycle: SigmaCycle,
) -> float:
    """Compute σ over a single cycle from a dict of edge weights.

    Args:
        weights: relation_name → signed weight in [-1, +1].
        cycle: the SigmaCycle whose ``members`` name relations.

    Returns:
        σ ∈ [-1, +1]. A cycle whose minimum |w_i| is 0 returns 0
        (no information).
    """
    if not cycle.members:
        return 0.0
    try:
        ws = [weights[m] for m in cycle.members]
    except KeyError as e:
        raise KeyError(
            f"cycle {cycle.name!r} references relation {e.args[0]!r} not in weights"
        ) from e
    abs_min = min(abs(w) for w in ws)
    sign_product = 1
    for w in ws:
        sign_product *= (1 if w >= 0 else -1)
    return float(sign_product) * abs_min


@dataclass
class CoherenceSnapshot:
    """A single time step's coherence over all cycles in a coalition."""
    t: int
    weights: dict[str, float]
    sigma: dict[str, float]   # cycle_name → σ value

    def sigma_of(self, cycle_name: str) -> float:
        return self.sigma.get(cycle_name, 0.0)


def coherence_snapshot(
    coalition: Coalition,
    weights: dict[str, float],
    t: int,
) -> CoherenceSnapshot:
    """Compute σ for every cycle in a coalition at the given t."""
    sigmas = {c.name: sigma_cycle(weights, c) for c in coalition.cycles.values()}
    return CoherenceSnapshot(t=t, weights=dict(weights), sigma=sigmas)
