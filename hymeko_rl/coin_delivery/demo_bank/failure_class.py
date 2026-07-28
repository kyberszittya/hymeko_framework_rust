"""The R11.3 failure taxonomy and a stage-ordered classifier — never one generic ``FAILED`` label.

A rollout ends in exactly one of {SUCCESS, one FailureClass}. The classifier walks the pipeline stages in order and returns
the first that failed, so a capture failure is never mislabelled a delivery failure. Two subtleties this taxonomy gets right:

* **strict K6 is checked BEFORE the capture/delivery classes.** ``contacts`` is a *terminal* count — a successful delivery
  releases the coin, so terminal ``contacts == 0`` is not "no capture". K6 (with the safety gate) decides success.
* **the teacher-handoff split** (per the R11.3 teacher-generation contract): a geometrically valid, well-tracked RRT reach
  can still land *outside* the frozen CEM teacher's admissible handoff manifold. When the capture never moves the coin
  toward the zone (``coin_progress`` below threshold) and the handoff is geometrically outside the teacher envelope, that is
  ``PRECONTACT_HANDOFF_INVALID`` (the teacher cannot be expected to grasp) — distinct from ``CAPTURE_TEACHER_FAILURE`` (an
  admissible handoff the teacher still failed to grasp) and ``DELIVERY_TEACHER_FAILURE`` (grasped, coin approached the zone,
  but did not arrive). ``coin_progress = entry_dtz - min_dtz`` (a documented proxy for capture engagement, pending
  per-step first-contact instrumentation).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureClass(Enum):
    """The mutually-exclusive terminal states of a coin/target demonstration rollout (SUCCESS is the absence of these)."""

    INVALID_INITIAL_CONDITION = "invalid_initial_condition"
    GOAL_SET_EMPTY = "goal_set_empty"
    RRT_PLANNING_FAILURE = "rrt_planning_failure"
    RRT_EXECUTION_FAILURE = "rrt_execution_failure"
    PREMATURE_COIN_MOTION = "premature_coin_motion"
    PREMATURE_CONTACT = "premature_contact"
    PRECONTACT_HANDOFF_INVALID = "precontact_handoff_invalid"
    CAPTURE_TEACHER_FAILURE = "capture_teacher_failure"
    DELIVERY_TEACHER_FAILURE = "delivery_teacher_failure"
    TARGET_ENTRY_OVERSHOOT = "target_entry_overshoot"
    SETTLE_K6_FAILURE = "settle_k6_failure"
    SAFETY_FAILURE = "safety_failure"
    PROVENANCE_FAILURE = "provenance_failure"
    ENERGY_LEDGER_INCOMPLETE = "energy_ledger_incomplete"


@dataclass(frozen=True)
class ClassifyThresholds:
    """Bands for classification. Documented proxies for the handoff/capture/delivery/settle splits."""

    reach_qerr_tol_rad: float = 0.15      # joint tracking error to the SELECTED RRT goal (not canonical precontact)
    precontact_tol_mm: float = 1.0        # accepted-reach coin motion ceiling (Phase 3 contract)
    capture_engaged_mm: float = 5.0       # coin must approach the zone at least this much to count as an engaged capture
    delivery_far_mm: float = 40.0
    zone_tol_mm: float = 15.0


@dataclass(frozen=True)
class RolloutSignals:
    """The measured signals the classifier consumes. Reach *execution* is judged by ``reach_qerr_rad`` (tracking to the
    selected RRT goal), NOT tip distance to the canonical precontact — the RRT legitimately reaches a different valid
    straddle. ``handoff_admissible`` is a geometric READY-state predicate; ``coin_progress_mm = entry_dtz - min_dtz``."""

    goal_set_empty: bool
    reach_found: bool
    reach_qerr_rad: float
    precontact_coin_motion_mm: float
    premature_contacts: int
    safe: bool
    handoff_admissible: bool
    k6: bool
    coin_progress_mm: float
    min_dtz_mm: float


def _capture_or_delivery(sig: RolloutSignals, thr: ClassifyThresholds) -> FailureClass:
    """Split a non-K6, safe, cleanly-reached rollout between handoff/capture/delivery/settle."""
    if sig.coin_progress_mm < thr.capture_engaged_mm:
        return (FailureClass.PRECONTACT_HANDOFF_INVALID if not sig.handoff_admissible
                else FailureClass.CAPTURE_TEACHER_FAILURE)
    if sig.min_dtz_mm > thr.delivery_far_mm:
        return FailureClass.DELIVERY_TEACHER_FAILURE
    if sig.min_dtz_mm <= thr.zone_tol_mm:
        return FailureClass.TARGET_ENTRY_OVERSHOOT   # reached the zone but did not settle to strict K6
    return FailureClass.SETTLE_K6_FAILURE


def classify(sig: RolloutSignals, thr: ClassifyThresholds = ClassifyThresholds()) -> "FailureClass | None":
    """Return the failed stage, or ``None`` for SUCCESS (strict K6).

    Preconditions: ``sig`` reflects one rollout past a valid admissibility + integrity check (INVALID_INITIAL_CONDITION,
    PROVENANCE_FAILURE, ENERGY_LEDGER_INCOMPLETE are assigned by the pipeline, not here). Postconditions: returns ``None``
    iff ``sig.k6`` and every earlier stage passed.
    """
    if sig.goal_set_empty:
        return FailureClass.GOAL_SET_EMPTY
    if not sig.reach_found:
        return FailureClass.RRT_PLANNING_FAILURE
    if sig.reach_qerr_rad > thr.reach_qerr_tol_rad:
        return FailureClass.RRT_EXECUTION_FAILURE
    if sig.precontact_coin_motion_mm > thr.precontact_tol_mm:
        return FailureClass.PREMATURE_COIN_MOTION
    if sig.premature_contacts > 0:
        return FailureClass.PREMATURE_CONTACT
    if not sig.safe:
        return FailureClass.SAFETY_FAILURE
    if sig.k6:                                        # K6 wins before the handoff/capture split (terminal release is fine)
        return None
    return _capture_or_delivery(sig, thr)
