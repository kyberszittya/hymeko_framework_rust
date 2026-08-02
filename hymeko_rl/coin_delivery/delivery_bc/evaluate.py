"""R11.4B closed-loop evaluation — physical strict-K6 rollout of a predicted theta from the certified handoff state.

This is the honest test: NO per-scenario CEM, NO deliverability oracle, NO teacher lookup — just the unchanged transport
primitive driven by the policy's predicted theta, scored by the same strict-K6 + safety predicate the certification used.
The frozen-R2 incumbent is rolled out via ``characterize_delivery`` for the mandatory baseline comparison.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, clip_theta
from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import characterize_delivery
from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec
from hymeko_rl.coin_delivery.forward_displacement import ForwardConfig, delivery_success, rollout_primitive

_SPEC = full_transport_spec()
CLOSED_LOOP_CFG = dataclasses.replace(ForwardConfig(), deliver=True, horizon=_SPEC.horizon)

_SAFE_QDOT = 3.0
_SAFE_COIN_SPEED = 1.5


def rollout_theta(snap: Any, theta: np.ndarray, cfg: ForwardConfig = CLOSED_LOOP_CFG) -> dict[str, Any]:
    """Roll the transport primitive with a (clipped) predicted theta; return strict-K6, safety, and delivered dtz.

    Preconditions: ``snap`` is a certified post-capture handoff targeting the scenario zone.
    Postconditions: ``k6`` iff the physical rollout holds strict-K6 dwell under the joint-velocity limit; ``safe`` iff the
    same peak-velocity contract the certification enforced holds. No search, no oracle.
    """
    m = rollout_primitive(snap, clip_theta(theta), cfg)
    return {"k6": bool(delivery_success(m, cfg)),
            "safe": bool(m["peak_qdot"] <= _SAFE_QDOT and m["peak_coin_speed"] <= _SAFE_COIN_SPEED),
            "dtz_mm": round(float(m["dtz_end"]) * 1000, 2)}


def r2_delivery(snap: Any, down: Any) -> dict[str, Any]:
    """The frozen-R2 incumbent's delivery from this handoff (the mandatory baseline; theta-free)."""
    base = characterize_delivery(snap, down)
    return {"k6": bool(base.k6), "dtz_mm": round(float(base.min_dtz_mm), 2), "safe": True}


def theta_basin_survival(snap: Any, theta: np.ndarray, scales: "tuple[float, ...]", k: int = 6,
                         seed: int = 0) -> dict[float, float]:
    """Fraction of K6 survivors when a certified ``theta`` is perturbed by Gaussian noise at each RELATIVE box scale —
    the delivery-basin width around the teacher solution. ``scale`` is a fraction of the box range ``(hi-lo)``.

    This is the decisive diagnostic for whether the teacher schedules are smoothly assimilable: if survival collapses at
    a small scale, the certified theta are narrow-basin / chaotic and no smooth descriptor->theta policy can reproduce
    them (a learned policy's finite prediction error lands off the basin). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    span = THETA_HI - THETA_LO
    theta = np.asarray(theta, np.float64)
    out: dict[float, float] = {}
    for sc in scales:
        hits = sum(rollout_theta(snap, theta + sc * span * rng.standard_normal(theta.shape[0]))["k6"] for _ in range(k))
        out[sc] = round(hits / k, 3)
    return out
