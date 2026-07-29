"""R11.5+ residual failure taxonomy — the canonical 9-category set + a trajectory-derived classifier.

Every uncovered scenario receives exactly one primary class. The coverage ledger records only *final-state* fields
(``teacher_dtz_mm`` is ``dtz_end``, not the trajectory minimum), which cannot separate a coin that transiently entered
the 20 mm zone and drifted out (``ZONE_ENTRY_WITHOUT_DWELL`` / ``TARGET_ENTRY_SPEED_FAILURE``) from one that stalled
short (``INSUFFICIENT_TRANSPORT_PROGRESS``). The classifier therefore consumes a :class:`DeliveryTrace` built from a
deterministic replay (min-dtz over the trajectory, max K6 dwell, entry speed, pre-release contact loss).

``HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE`` is *not* emitted from the trace alone — it is a Phase-3 transport-audit
determination (an ``INSUFFICIENT_TRANSPORT_PROGRESS`` case proven coordinate-bound rather than under-driven), so the
rule classifier leaves it empty at Phase 1.
"""
from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL

ZONE_TOL_MM = CENTER_TOL * 1000.0                    # 20 mm — K6 dtz gate
_STILL_MM = 5.0                                      # |progress| below this with ~0 gap change = coin never moved
_STILL_GAP = 0.05
_LOST_FRAC = 0.5                                     # >half the pre-release window without bilateral contact = grasp lost


class FailureCategory(enum.Enum):
    """Canonical mutually-exclusive R11.5+ residual categories. Enum order is the classifier's precedence."""
    CAPTURE_SUPPORT_FAILURE = "CAPTURE_SUPPORT_FAILURE"                          # no certified grasp reached the teacher
    HANDOFF_TO_KINETIC_FAILURE = "HANDOFF_TO_KINETIC_FAILURE"                    # certified, but the coin never got moving
    CONTACT_LOSS_DURING_DELIVERY = "CONTACT_LOSS_DURING_DELIVERY"               # grasp lost pre-release (premature)
    DELIVERY_DIRECTIONAL_BIAS = "DELIVERY_DIRECTIONAL_BIAS"                     # driven net AWAY from target (gap < 0)
    TARGET_ENTRY_SPEED_FAILURE = "TARGET_ENTRY_SPEED_FAILURE"                   # reached the zone but too fast to settle
    ZONE_ENTRY_WITHOUT_DWELL = "ZONE_ENTRY_WITHOUT_DWELL"                       # entered the zone, slow, but no held dwell
    INSUFFICIENT_TRANSPORT_PROGRESS = "INSUFFICIENT_TRANSPORT_PROGRESS"        # moved toward target, closed part, stalled
    HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE = "HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE"  # Phase-3 audit only
    OTHER_MEASURED = "OTHER_MEASURED"                                           # only with a concrete trace reason


@dataclasses.dataclass(frozen=True)
class DeliveryTrace:
    """Trajectory-derived delivery features (from deterministic replay). ``None`` metrics only for uncertified grasps."""
    certified: bool
    coin_progress_mm: float = 0.0
    gap_closed_final: float = 0.0
    min_dtz_mm: float = float("inf")                 # trajectory MINIMUM dtz (not dtz_end)
    max_k6_dwell: int = 0
    entry_speed: float = 0.0                         # coin speed at the closest approach / zone entry
    lost_before_release: int = 0
    release_step: int = 1


def classify(trace: DeliveryTrace) -> FailureCategory:
    """Assign one canonical category from a replay-derived :class:`DeliveryTrace`.

    Preconditions: ``min_dtz_mm >= 0``; ``release_step >= 1``. Postcondition: exactly one category;
    ``HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE`` and ``OTHER_MEASURED`` are not inferred from the trace (Phase-3 / manual).
    """
    assert trace.min_dtz_mm >= 0.0 and trace.release_step >= 1
    if not trace.certified:
        return FailureCategory.CAPTURE_SUPPORT_FAILURE
    if abs(trace.coin_progress_mm) < _STILL_MM and abs(trace.gap_closed_final) < _STILL_GAP:
        return FailureCategory.HANDOFF_TO_KINETIC_FAILURE
    if trace.lost_before_release / trace.release_step > _LOST_FRAC:
        return FailureCategory.CONTACT_LOSS_DURING_DELIVERY
    if trace.gap_closed_final < 0.0:
        return FailureCategory.DELIVERY_DIRECTIONAL_BIAS
    if trace.min_dtz_mm <= ZONE_TOL_MM:                          # entered the zone at some trajectory step
        if trace.entry_speed > SETTLE_VEL:
            return FailureCategory.TARGET_ENTRY_SPEED_FAILURE
        if trace.max_k6_dwell < HELD_DWELL:
            return FailureCategory.ZONE_ENTRY_WITHOUT_DWELL
        return FailureCategory.OTHER_MEASURED                    # entered, slow, dwelled >= HELD but no K6 — needs a look
    return FailureCategory.INSUFFICIENT_TRANSPORT_PROGRESS


def cclass(sid: str) -> str:
    """Curriculum stage (c0..c3) from a bank scenario id."""
    tok = sid.split("_")[1]
    return tok[:2] if tok[:1] == "c" else tok


@dataclasses.dataclass(frozen=True)
class FailureRecord:
    scenario_id: str
    split: str
    cclass: str
    category: FailureCategory
    subtype: str                    # capture: "systematic_pp" | "stochastic_regen"; else ""
    dtz_mm: float | None            # min-dtz (trajectory) for certified; None for capture-support


def _capture_pick(records: list[FailureRecord]) -> list[FailureRecord]:
    """2 systematic-+/+ (hardest, never certified) + 2 stochastic-regen (likeliest to recover) capture-support cases."""
    cap = [r for r in records if r.category is FailureCategory.CAPTURE_SUPPORT_FAILURE]
    syst = sorted((r for r in cap if r.subtype == "systematic_pp"), key=lambda r: r.scenario_id)
    stoch = sorted((r for r in cap if r.subtype == "stochastic_regen"), key=lambda r: r.scenario_id)
    return syst[:2] + stoch[:2]


def _dedup(records: list[FailureRecord]) -> list[FailureRecord]:
    seen: set[str] = set()
    out: list[FailureRecord] = []
    for r in records:
        if r.scenario_id not in seen:
            seen.add(r.scenario_id)
            out.append(r)
    return out


# The negative-x / far-tail group: the transport fix's ALIGN target. The trajectory replay showed these fail by
# CONTACT_LOSS (the grasp squirts out when pushed off-axis toward a negative-x target); DIRECTIONAL_BIAS is the
# final-state-only label for the same cases, kept so the selector is robust to either classification.
_NEG_X = (FailureCategory.CONTACT_LOSS_DURING_DELIVERY, FailureCategory.DELIVERY_DIRECTIONAL_BIAS)
_TRANSPORT = (FailureCategory.INSUFFICIENT_TRANSPORT_PROGRESS, FailureCategory.ZONE_ENTRY_WITHOUT_DWELL,
              FailureCategory.TARGET_ENTRY_SPEED_FAILURE, FailureCategory.HARD_GEOMETRY_WITHIN_CURRENT_COORDINATE)


def _progress_pick(records: list[FailureRecord]) -> list[FailureRecord]:
    """4 representative transport-progress residuals (not capture, not the negative-x tail). Held-out first (one dev +
    one test if present — the transport fix must be validated off-train, since the negative-x tail is all-train), then
    the dtz range."""
    by_dtz = sorted((r for r in records if r.category in _TRANSPORT),
                    key=lambda r: (r.dtz_mm if r.dtz_mm is not None else 0.0))
    if not by_dtz:
        return []
    heldout = [c for want in ("dev", "test")
               if (c := next((r for r in by_dtz if r.split == want), None)) is not None]
    return _dedup(heldout + [by_dtz[0], by_dtz[-1]] + by_dtz)[:4]


def select_pilot(records: list[FailureRecord]) -> list[FailureRecord]:
    """Bounded 12-scenario recovery pilot: 4 capture-support, 4 negative-x/far-tail (CONTACT_LOSS/DIRECTIONAL — the ALIGN
    target), 4 representative transport-progress residuals. If a group is short, fills from the remaining residuals.
    Postcondition: <= 12 records, collectively including >= 1 dev and >= 1 test when the residual set contains them."""
    capture = _capture_pick(records)
    negx = sorted((r for r in records if r.category in _NEG_X), key=lambda r: r.scenario_id)[:4]
    progress = _progress_pick(records)
    picked = _dedup(capture + negx + progress)
    if len(picked) < 12:                                        # top up from any remaining residuals, dtz-sorted
        rest = sorted((r for r in records if r not in picked),
                      key=lambda r: (r.dtz_mm if r.dtz_mm is not None else 0.0))
        picked = _dedup(picked + rest)[:12]
    return picked[:12]


def summarize(records: list[FailureRecord]) -> "dict[str, list[str]]":
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_cat[r.category.value].append(r.scenario_id)
    return {k: sorted(v) for k, v in sorted(by_cat.items())}
