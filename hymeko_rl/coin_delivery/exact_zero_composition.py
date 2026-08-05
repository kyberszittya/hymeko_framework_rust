"""R11.6C — exact-zero retrieval composition: the full robotic chain from the exact-zero home to strict K6, with the
delivery theta coming ONLY from the frozen retrieval table (no per-instance CEM, no oracle, no teacher at run time).

``compose_one`` reuses the deployed exact-zero -> RRT reach -> capture stages verbatim (``_do_reach_and_capture`` with the
deployable first-certified-grasp selection, NOT the delivery oracle), computes the 30-D handoff descriptor LIVE, queries
the frozen policy, and rolls the pure delivery primitive. It classifies the full chain (reach / handoff / capture /
retrieval-support / delivery / nudge / safety) and, on a real teacher-free success, emits an
``ExactZeroCoinDeliveryCertificate``. The only new decision versus the certified pipeline is the delivery theta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery import ir_adapter as A
from hymeko_rl.coin_delivery.delivery_bc.dataset import descriptor
from hymeko_rl.coin_delivery.delivery_bc.evaluate import _SAFE_COIN_SPEED, _SAFE_QDOT, CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.delivery_bc.retrieval import RetrievalDeliveryPolicy
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.experiments import coin_zero_home_reach as Z

_DRIFT_EPS = 1e-6                # a live descriptor must reproduce the stored one to this tolerance (else a real bug)
_MIN_GAP_CLOSED = 0.5           # a real grasp-driven delivery closes >= half the initial coin->zone gap (not a nudge)


class CompositionOutcomeClass(Enum):
    """Exactly one per attempt, in stage order. Only ``EXACT_ZERO_DELIVERED`` is a success."""

    INVALID_INITIAL_CONDITION = "invalid_initial_condition"
    REACH_FAILURE = "reach_failure"                              # GOAL_SET_EMPTY / RRT_PLANNING_FAILURE
    PRECONTACT_HANDOFF_INVALID = "precontact_handoff_invalid"
    CAPTURE_NO_CERTIFIED_GRASP = "capture_no_certified_grasp"
    DESCRIPTOR_DRIFT = "descriptor_drift"                        # live descriptor != stored (a determinism/config bug)
    RETRIEVAL_OUT_OF_SUPPORT = "retrieval_out_of_support"        # delivery failed AND query beyond the table radius
    DELIVERY_FAILURE_IN_SUPPORT = "delivery_failure_in_support"  # delivery failed but theta was near-support (chain/handoff)
    NUDGE_NOT_VALID_DELIVERY = "nudge_not_valid_delivery"        # K6 but grasp lost before release (not a real transport)
    SAFETY_FAILURE = "safety_failure"                            # K6 + valid mode but the velocity contract broke
    EXACT_ZERO_DELIVERED = "exact_zero_delivered"                # SUCCESS


SUCCESS = CompositionOutcomeClass.EXACT_ZERO_DELIVERED


@dataclass(frozen=True)
class ExactZeroCoinDeliveryCertificate:
    """The milestone artifact: a fully teacher-free exact-zero -> K6 delivery. ``is_valid`` iff every clause holds."""

    scenario_id: str
    seed: int
    exact_zero_ic: bool
    teacher_free: bool
    cem_free: bool
    oracle_free: bool
    no_teleport: bool
    strict_k6: bool
    valid_delivery_mode: bool
    safe: bool
    retrieval_support_dist: float
    dtz_mm: float
    theta: "tuple[float, ...]"

    def is_valid(self) -> bool:
        return bool(self.exact_zero_ic and self.teacher_free and self.cem_free and self.oracle_free
                    and self.no_teleport and self.strict_k6 and self.valid_delivery_mode and self.safe)


@dataclass
class CompositionRecord:
    scenario_id: str
    split: str
    seed: int
    outcome_class: str
    k6: bool = False
    valid_delivery: bool = False
    safe: bool = False
    support_dist: float = 0.0
    descriptor_drift: float = 0.0
    dtz_mm: float = 0.0
    gap_closed: float = 0.0
    contact_lost_steps: int = 0
    reach_reason: "str | None" = None
    theta: "list[float]" = field(default_factory=list)
    certificate: "dict | None" = None

    @property
    def success(self) -> bool:
        return self.outcome_class == SUCCESS.value


@dataclass(frozen=True)
class DeliverySignals:
    """The measured delivery outcome from one pure rollout."""

    k6: bool
    valid_delivery: bool          # a real GRASP-DRIVEN transport (coin contacted AND carried most of the way to the zone)
    safe: bool
    dtz_mm: float
    gap_closed: float             # fraction of the initial coin->zone gap closed (the transport signal)
    contact_lost_steps: int       # CONTACT_LOSS quality diagnostic (grasp slip mid-transport); not a nudge gate


def _delivery_signals(snap: Any, theta: np.ndarray) -> DeliverySignals:
    """One pure delivery rollout. ``valid_delivery`` = the coin was bilaterally grasped during delivery AND carried at
    least half of its initial gap to the zone (``touched and gap_closed >= _MIN_GAP_CLOSED``).

    Because R11.6C only reaches delivery from a CERTIFIED bilateral grasp, the historical nudge mode (an ungrasped coin
    bumped in) is excluded upstream; ``gap_closed`` is scale-invariant, so it confirms the K6 came from a real transport
    of the coin (not a coin that started in-zone or arrived un-carried). Contact loss AFTER release is ballistic flight
    (normal), recorded as a diagnostic, not a nudge.
    """
    m = rollout_primitive(snap, clip_theta(theta), CLOSED_LOOP_CFG)
    gap = round(float(m["gap_closed"]), 3)
    return DeliverySignals(
        k6=bool(delivery_success(m, CLOSED_LOOP_CFG)),
        valid_delivery=bool(m["touched"]) and gap >= _MIN_GAP_CLOSED,
        safe=bool(m["peak_qdot"] <= _SAFE_QDOT and m["peak_coin_speed"] <= _SAFE_COIN_SPEED),
        dtz_mm=round(float(m["dtz_end"]) * 1000, 2), gap_closed=gap, contact_lost_steps=int(m["contact_lost_steps"]))


def _classify_delivery(k6: bool, valid: bool, safe: bool, support_dist: float, radius: float) -> CompositionOutcomeClass:
    """Delivery-stage class: success / nudge / safety / (out-of-support vs in-support failure, split on the table radius)."""
    if k6 and valid and safe:
        return SUCCESS
    if k6 and valid and not safe:
        return CompositionOutcomeClass.SAFETY_FAILURE
    if k6 and not valid:
        return CompositionOutcomeClass.NUDGE_NOT_VALID_DELIVERY
    return (CompositionOutcomeClass.RETRIEVAL_OUT_OF_SUPPORT if support_dist > radius
            else CompositionOutcomeClass.DELIVERY_FAILURE_IN_SUPPORT)


@dataclass
class HandoffResult:
    """The exact-zero -> reach -> capture -> live-descriptor result. ``record`` is set iff the chain FAILED before
    delivery (reach / handoff / capture / drift); otherwise ``snap`` + ``x`` feed the delivery stage."""

    record: "CompositionRecord | None"
    snap: Any = None
    x: "np.ndarray | None" = None


def reach_capture_descriptor(rig: dict, scen: Any, seed: int, cfg: Any, conf: Any, obj: Any,
                             stored_x: "np.ndarray | None" = None) -> HandoffResult:
    """Exact-zero IC -> RRT reach -> capture (deployable first-certified-grasp) -> LIVE 30-D descriptor, with the drift
    check against ``stored_x``. Runs ONCE per scenario; every delivery policy reuses the returned handoff."""
    sid, split = scen.scenario_id, scen.split.value

    def fail(klass: CompositionOutcomeClass, **kw: Any) -> HandoffResult:
        return HandoffResult(CompositionRecord(sid, split, seed, klass.value, **kw))

    home, coin = Z._home_with_coin(rig, scen.coin_xy)
    ic = A.EXACT_ZERO_HOME_V1.certify(A.read_rollout_state(home.branch()))
    if not ic.valid or not A.coin_admissibility(rig, scen.coin_xy, cfg).admissible:
        return fail(CompositionOutcomeClass.INVALID_INITIAL_CONDITION)
    reason, rcap = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
    if rcap is None:
        return fail(CompositionOutcomeClass.REACH_FAILURE, reach_reason=reason)
    if not rcap.handoff_admissible:
        return fail(CompositionOutcomeClass.PRECONTACT_HANDOFF_INVALID)
    if not mp.is_certified_grasp(rcap.result.outcome, obj):
        return fail(CompositionOutcomeClass.CAPTURE_NO_CERTIFIED_GRASP)
    snap = rcap.result.outcome.snapshot
    x = descriptor(scen, rcap, snap)
    drift = 0.0 if stored_x is None else float(np.linalg.norm(x - np.asarray(stored_x, np.float64)))
    if drift > _DRIFT_EPS:                                        # the live handoff must reproduce the stored descriptor
        return fail(CompositionOutcomeClass.DESCRIPTOR_DRIFT, descriptor_drift=round(drift, 9))
    return HandoffResult(None, snap=snap, x=x)


def deliver_record(scen: Any, seed: int, snap: Any, x: np.ndarray, policy: RetrievalDeliveryPolicy,
                   radius: float) -> CompositionRecord:
    """The delivery stage: retrieve theta from the frozen policy, roll the pure primitive, classify + certify."""
    theta = policy.predict(x)
    support = round(policy.support_distance(x), 4)
    s = _delivery_signals(snap, theta)
    klass = _classify_delivery(s.k6, s.valid_delivery, s.safe, support, radius)
    cert = ExactZeroCoinDeliveryCertificate(
        scen.scenario_id, seed, exact_zero_ic=True, teacher_free=True, cem_free=True, oracle_free=True,
        no_teleport=True, strict_k6=s.k6, valid_delivery_mode=s.valid_delivery, safe=s.safe,
        retrieval_support_dist=support, dtz_mm=s.dtz_mm, theta=tuple(round(float(t), 6) for t in theta))
    return CompositionRecord(scen.scenario_id, scen.split.value, seed, klass.value, k6=s.k6,
                             valid_delivery=s.valid_delivery, safe=s.safe, support_dist=support, dtz_mm=s.dtz_mm,
                             gap_closed=s.gap_closed, contact_lost_steps=s.contact_lost_steps,
                             theta=[round(float(t), 6) for t in theta],
                             certificate=(cert.__dict__ if klass is SUCCESS and cert.is_valid() else None))


def compose_one(rig: dict, scen: Any, seed: int, cfg: Any, conf: Any, obj: Any, policy: RetrievalDeliveryPolicy,
                radius: float, stored_x: "np.ndarray | None" = None) -> CompositionRecord:
    """The full single-policy exact-zero -> K6 chain with the frozen retrieval delivery decision (deterministic)."""
    h = reach_capture_descriptor(rig, scen, seed, cfg, conf, obj, stored_x)
    return h.record if h.record is not None else deliver_record(scen, seed, h.snap, h.x, policy, radius)
