"""R11.5++ deliverability-ranked capture-grasp selection (teacher-only oracle).

The R11.5+ reproduction showed a certified grasp is not necessarily a *delivery-favourable* grasp: the same scenario
flips to K6 or failure across grasp realizations under the SAME frozen single-stage transport. This module ranks
CERTIFIED grasps by their *delivered* outcome (a full rollout of the frozen delivery teacher) with a class-based
lexicographic key — never a weighted sum, never letting a non-certified grasp outrank a certified one.

**Teacher-only.** Using ``delivered_dtz`` to select a grasp is a full-rollout ORACLE: legitimate for teacher-demonstration
GENERATION (whole rollouts are searched here), but NOT a deployment mechanism — at deployment one cannot simulate every
candidate's future delivery. A later policy/surrogate must predict deliverability from the pre-delivery handoff
descriptor. Every demonstration this ranking produces is labelled ``selection_kind=DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER,
teacher_only=True``. This module touches NO transport / capture-controller code — only selection among already-solved grasps.
"""
from __future__ import annotations

import dataclasses

SELECTION_KIND = "DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER"
DWELL_TARGET = 4                 # bilateral held-dwell threshold for the tier-3 STABILITY GATE (matches GraspObjective)


@dataclasses.dataclass(frozen=True)
class GraspDelivery:
    """One certified-grasp candidate together with its frozen-teacher delivery outcome (the ranking inputs)."""

    seed: int
    certified: bool                 # a certified bilateral held grasp (tier 2 — a non-certified grasp never outranks one)
    safe: bool                      # delivery within the motion / speed contracts (tier 1)
    bilateral_dwell: int            # grasp held-dwell (tier 3, stability)
    kinetic: bool                   # delivery entered KINETIC (coin contacted + moving) (tier 4)
    delivered_dtz_mm: float         # frozen-teacher delivered min/final dtz (tier 5 — the deliverability oracle)
    gap_closed: float               # target-direction progress (tier 6)
    contact_delay: int              # left-right seating delay (tier 7, existing regularization)
    k6: bool                        # strict K6 achieved (recorded, not a ranking tier — dtz/kinetic already encode it)


def deliverability_key(g: GraspDelivery) -> "tuple[int, int, int, int, float, float, int]":
    """Lexicographic ranking key (lower = better). Tiers, in order: (1) safety, (2) certified bilateral grasp,
    (3) grasp-stability GATE (dwell >= ``DWELL_TARGET`` — a *qualification*, NOT a maximizer), (4) delivery enters
    KINETIC, (5) delivered dtz, (6) target-direction progress, (7) seating regularization.

    Tier 3 is a binary gate on purpose: a raw ``-dwell`` there would let a slightly-higher-dwell grasp outrank a
    strictly-more-DELIVERABLE one before dtz is ever consulted (measured — it discarded an 18.6 mm K6 grasp for a 24.2 mm
    miss). As a gate, all adequately-stable grasps tie on tier 3 and **delivered dtz (tier 5) decides** — which is the
    whole point. A non-certified grasp can never beat a certified one on dtz."""
    return (0 if g.safe else 1,
            0 if g.certified else 1,
            0 if g.bilateral_dwell >= DWELL_TARGET else 1,
            0 if g.kinetic else 1,
            round(g.delivered_dtz_mm, 2),
            -round(g.gap_closed, 3),
            g.contact_delay)


def select_deliverable_grasp(candidates: "list[GraspDelivery]") -> GraspDelivery:
    """The best certified grasp by delivered outcome (the deliverability oracle).

    # Preconditions: ``candidates`` non-empty. # Postconditions: returns the lexicographic minimum; deterministic
    (ties broken by the full key, then by input order via ``min``'s stability)."""
    assert candidates, "no grasp candidates to rank"
    return min(candidates, key=deliverability_key)


def is_material_dtz_improvement(current_dtz_mm: float, ranked_dtz_mm: float, *, floor_mm: float = 5.0) -> bool:
    """Arm D vs Arm C: a materially better delivered distance (>= ``floor_mm`` closer)."""
    return current_dtz_mm - ranked_dtz_mm >= floor_mm
