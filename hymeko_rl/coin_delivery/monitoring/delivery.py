"""The single delivery monitor shared by every scenario — a thin adapter over the FROZEN monitor.

There is exactly ONE delivery verdict in this framework, and it is the frozen strict dwell-verified monitor
:func:`~hymeko_rl.train.coin_delivery_actor.rollout_delivery`. :class:`DeliveryMonitor` does not re-implement,
weaken, or reward-certify it; it wraps it so the runners can hold one monitor instance across all four scenarios
(the same object, asserted by the tests). Delivery success is NEVER conflated with contact continuity.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hymeko_rl.train.coin_delivery_actor import (
    ActorParams,
    DeliveryActor,
    DeliveryResult,
    rollout_delivery,
)


class DeliveryMonitor:
    """Wrap the frozen strict valid-delivery monitor. One instance is reused across all scenarios."""

    #: The frozen rollout the monitor delegates to (exposed so tests can assert the single source).
    rollout_fn: "Callable[..., DeliveryResult]" = staticmethod(rollout_delivery)

    def evaluate(self, env: Any, seed: int, actor: DeliveryActor, params: "ActorParams | None" = None, *,
                 max_steps: int = 60,
                 scramble: "Callable[[np.ndarray, int], np.ndarray] | None" = None) -> DeliveryResult:
        """Strict delivery verdict for one (env, seed, actor). # Postconditions delegates verbatim to
        :func:`rollout_delivery`; never mutates the monitor thresholds or the reward."""
        return type(self).rollout_fn(env, int(seed), actor, params or ActorParams(),
                                     max_steps=max_steps, scramble=scramble)
