"""Contact-trace collection — the env-coupled half of contact certification.

:meth:`ContactPolicy.certify` is a pure judge of a per-step trace; this module PRODUCES that trace by rolling
the frozen delivery actor through the env and recording the per-step fingertip / arm-body contact and toward-zone
progress. It computes NO delivery verdict, attribution vector, or mechanism — those stay with the frozen monitor
(:func:`~hymeko_rl.train.coin_delivery_actor.rollout_delivery`). Because the frozen rollout is history-independent
and reset exactly by seed, this trace corresponds step-for-step to the trajectory the monitor evaluates.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.scenarios.contact_policy import (
    ContactCertification,
    ContactPolicy,
    ContactSample,
)
from hymeko_rl.train.coin_delivery_actor import ActorParams, DeliveryActor, actor_action
from hymeko_rl.train.coin_grip_control import normal_contact_forces


def contact_trace(env: Any, seed: int, actor: DeliveryActor, params: "ActorParams | None" = None, *,
                  max_steps: int = 60, scramble: "Callable[[np.ndarray, int], np.ndarray] | None" = None,
                  ) -> list[ContactSample]:
    """Roll ``actor`` from a fresh reset and record the per-step contact trace (fingertips + arm body + progress).

    # Preconditions ``env`` is the delivery ``ContactFormationEnv`` (``env._env`` = ``PlanarGraspEnv``); ``seed``
      selects the reset exactly. # Postconditions returns a time-ordered list of :class:`ContactSample` of length
      <= ``max_steps``; delegates every VERDICT (delivery / mechanism / attribution) to the frozen monitor."""
    params = params or ActorParams()
    env.reset(seed=int(seed))
    inner = env._env
    prev_dtz = float(inner._planar_metrics.disk_to_zone)
    prev_body_steps = int(inner._arm_body_steps)
    samples: list[ContactSample] = []
    for t in range(max_steps):
        a = np.clip(actor_action(inner, t, actor, params), -1.0, 1.0).astype(np.float32)
        if scramble is not None:
            a = scramble(a, t)
        _o, _r, term, trunc, _info = env.step(a)
        mm = inner._planar_metrics
        dtz = float(mm.disk_to_zone)
        fl, fr = normal_contact_forces(inner)
        body = int(inner._arm_body_steps) > prev_body_steps
        prev_body_steps = int(inner._arm_body_steps)
        samples.append(ContactSample(t=t, left=bool(mm.left_contact or fl > 1e-4),
                                     right=bool(mm.right_contact or fr > 1e-4), body=body,
                                     progress=prev_dtz - dtz))
        prev_dtz = dtz
        if term or trunc:
            break
    return samples


def certify(policy: ContactPolicy, env: Any, seed: int, actor: DeliveryActor,
            params: "ActorParams | None" = None, *, max_steps: int = 60,
            scramble: "Callable[[np.ndarray, int], np.ndarray] | None" = None) -> ContactCertification:
    """Collect the trace and certify it against ``policy`` (contact-continuity only; NOT delivery success)."""
    trace = contact_trace(env, seed, actor, params, max_steps=max_steps, scramble=scramble)
    return policy.certify(trace)
