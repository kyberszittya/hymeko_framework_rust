"""Attribution adapters over the frozen 5-component progress attribution + a C1 free-progress decomposition.

The frozen :class:`~hymeko_rl.train.coin_delivery_actor.Attribution` (α = [L, R, body, support, free]) is
authoritative and re-exported here as a vector for the runners. For intermittent (C1) delivery the coin can
coast between contacts; :func:`decompose_alpha_free` splits that free (``α_free``) progress by the last contact
that preceded it — {free-after-fingertip-impulse, free-after-body-impulse, unexplained} — using the per-step
contact trace. This is an APPROXIMATION (labelled): it cannot separate genuine coasting from an unmodeled
contribution, exactly as the frozen ``α_free`` note states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hymeko_rl.coin_delivery.scenarios.contact_policy import Contact, ContactSample
from hymeko_rl.train.coin_delivery_actor import Attribution, DeliveryResult

ALPHA_FIELDS = ("alpha_L", "alpha_R", "alpha_body", "alpha_support", "alpha_free")


def attribution_of(result: DeliveryResult) -> Attribution:
    """Reconstruct the frozen :class:`Attribution` from a :class:`DeliveryResult` (no recomputation)."""
    a = result.attribution
    return Attribution(a["alpha_L"], a["alpha_R"], a["alpha_body"], a["alpha_support"], a["alpha_free"],
                       a["total_progress"])


def alpha_vector(result: DeliveryResult) -> np.ndarray:
    """The α = [L, R, body, support, free] vector of a result."""
    a = result.attribution
    return np.array([a[k] for k in ALPHA_FIELDS], dtype=np.float32)


@dataclass(frozen=True)
class FreeDecomposition:
    """C1 approximate split of α_free progress by the last preceding contact (fractions of total free progress)."""

    after_fingertip: float
    after_body: float
    unexplained: float
    total_free_progress: float
    approximate: bool = True

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def decompose_alpha_free(trace: Sequence[ContactSample]) -> FreeDecomposition:
    """Split free (no-contact) toward-zone progress by the kind of the last contact before it.

    # Preconditions ``trace`` is time-ordered. # Postconditions the three fractions sum to ~1 when there is any
      free progress, else all zero; ``approximate`` is always True (coasting cannot be perfectly attributed)."""
    buckets = {"fingertip": 0.0, "body": 0.0, "unexplained": 0.0}
    last = "unexplained"
    for s in trace:
        if s.state is not Contact.NONE:
            last = "fingertip"
        elif s.body:
            last = "body"
        elif s.progress > 0.0:
            buckets[last] += s.progress
    total = sum(buckets.values())
    if total <= 0.0:
        return FreeDecomposition(0.0, 0.0, 0.0, 0.0)
    return FreeDecomposition(round(buckets["fingertip"] / total, 3), round(buckets["body"] / total, 3),
                             round(buckets["unexplained"] / total, 3), round(total, 4))
