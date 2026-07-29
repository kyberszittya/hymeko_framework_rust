"""The capture->delivery contract: a DELIVERY_READY_GRASP_CERTIFICATE and the corrected delivery taxonomy (R11.4A0).

The R11.3 audit showed that "capture ran" was silently accepted as "delivery-ready", even when the coin was never
bilaterally grasped. The downstream then either entered KINETIC/R2 (a genuine grasp->delivery) or the APPROACH servo merely
*nudged* an ungrasped coin into the 20 mm zone (a K6 flag without any delivery-mode transition). This module makes the
contract explicit: a rollout is a *valid capture* only if it produces a delivery-ready grasp (bilateral contact held for N
steps, bounded relative slip, bounded coin speed, safe, continuous episode). Otherwise it is CAPTURE_INCOMPLETE and the
downstream must not start or be scored from it.

The corrected taxonomy separates the two mechanisms so a nudge is never counted as a delivery success:
K6_WITH_VALID_DELIVERY_MODE / K6_WITHOUT_DELIVERY_MODE_TRANSITION / CAPTURE_TO_DELIVERY_REGRASP_FAILURE /
DELIVERY_FAILURE_AFTER_VALID_GRASP / SETTLE_FAILURE_AFTER_VALID_GRASP.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

DELIVERY_BAND_MM = 40.0    # min_dtz > this = fell short (delivery); <= this = reached the vicinity (settle)


class DeliveryOutcomeClass(Enum):
    """Exactly one per attempt. A nudge-only K6 is NOT a delivery success."""

    K6_WITH_VALID_DELIVERY_MODE = "k6_with_valid_delivery_mode"
    K6_WITHOUT_DELIVERY_MODE_TRANSITION = "k6_without_delivery_mode_transition"
    CAPTURE_TO_DELIVERY_REGRASP_FAILURE = "capture_to_delivery_regrasp_failure"
    DELIVERY_FAILURE_AFTER_VALID_GRASP = "delivery_failure_after_valid_grasp"
    SETTLE_FAILURE_AFTER_VALID_GRASP = "settle_failure_after_valid_grasp"


@dataclass(frozen=True)
class GraspThresholds:
    """Delivery-ready grasp requirements. Slip/speed in SI; dwell in control steps."""

    dwell_n: int = 3               # consecutive bilateral-contact steps
    relative_slip_max: float = 0.02
    coin_speed_max: float = 0.20


@dataclass(frozen=True)
class DeliveryReadyGraspCertificate:
    """The certified precondition for starting the delivery controller. ``valid`` iff every clause holds."""

    bilateral_contacts: bool
    bilateral_dwell: int
    relative_slip: float
    coin_speed: float
    safe: bool
    continuous_episode: bool
    thresholds: GraspThresholds = GraspThresholds()

    @property
    def valid(self) -> bool:
        t = self.thresholds
        return (self.bilateral_contacts and self.bilateral_dwell >= t.dwell_n
                and self.relative_slip <= t.relative_slip_max and self.coin_speed <= t.coin_speed_max
                and self.safe and self.continuous_episode)

    def violations(self) -> tuple[str, ...]:
        t = self.thresholds
        checks = {
            "bilateral_contacts": self.bilateral_contacts,
            "bilateral_dwell": self.bilateral_dwell >= t.dwell_n,
            "relative_slip": self.relative_slip <= t.relative_slip_max,
            "coin_speed": self.coin_speed <= t.coin_speed_max,
            "safe": self.safe,
            "continuous_episode": self.continuous_episode,
        }
        return tuple(name for name, ok in checks.items() if not ok)


def classify_delivery(*, grasp_valid: bool, reaches_kinetic: bool, k6: bool,
                      min_dtz_mm: float) -> DeliveryOutcomeClass:
    """Corrected classification from the measured delivery mechanism.

    Preconditions: signals come from an instrumented delivery. Postconditions: a K6 reached without a delivery-mode
    transition is labelled a nudge, never a delivery success.
    """
    delivery_mode = grasp_valid and reaches_kinetic
    if k6 and delivery_mode:
        return DeliveryOutcomeClass.K6_WITH_VALID_DELIVERY_MODE
    if k6:                                                       # K6 flag but never entered delivery mode -> nudge
        return DeliveryOutcomeClass.K6_WITHOUT_DELIVERY_MODE_TRANSITION
    if not delivery_mode:
        return DeliveryOutcomeClass.CAPTURE_TO_DELIVERY_REGRASP_FAILURE
    return (DeliveryOutcomeClass.SETTLE_FAILURE_AFTER_VALID_GRASP if min_dtz_mm <= DELIVERY_BAND_MM
            else DeliveryOutcomeClass.DELIVERY_FAILURE_AFTER_VALID_GRASP)


def reclassify_from_bank_fields(*, contacts: int, k6: bool, min_dtz_mm: "float | None") -> DeliveryOutcomeClass:
    """A *derived* reclassification of an existing R11.3 bank record from its stored fields, using the measured proxy
    ``contacts == 2 <=> grasped handoff <=> reaches KINETIC`` (verified on the exact-replay set: bank_c0_3 s1 grasped ->
    KINETIC -> R2 -> 8.76 mm; released captures -> stuck APPROACH). This does NOT rewrite the bank; it is a versioned
    derived label. ``min_dtz`` None (no capture) is treated as fell-short."""
    grasped = contacts >= 2
    return classify_delivery(grasp_valid=grasped, reaches_kinetic=grasped, k6=bool(k6),
                             min_dtz_mm=1e9 if min_dtz_mm is None else float(min_dtz_mm))
