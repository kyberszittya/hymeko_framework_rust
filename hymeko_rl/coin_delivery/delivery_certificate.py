"""Two explicitly-separated coin-delivery certificates (task-semantics fix, 2026-07-21).

The user's original task is: *move the coin from a visibly outside starting position into the target zone and leave it
there in a stable certified state.* A push-and-coast mechanism is VALID for this task when the robot causes the
displacement and the mechanism / body-shove / settle conditions pass. **Force closure is NOT required for Coin
Delivery.** A separate, harder task — Grasp Delivery — additionally requires a bilateral force-closure hold.

This module makes the two certificates first-class and distinct, so a delivery success is never mislabeled a grasp
success (or vice versa), and historical results can be re-labeled by which certificate they satisfy:

- :data:`COIN_DELIVERY_STRICT`   — the existing delivery predicate (no force closure).
- :data:`COIN_GRASP_DELIVERY_STRICT` — delivery PLUS bilateral contact + minimum force + a force-closure interval.

A :class:`DeliveryCertifier` streams per-step :class:`CertStep` components and reports which certificate(s) passed.
It is reusable by the scripted oracle, the learned-policy evaluator, PickPlaceEnv, and future humanoid/AIBO work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

COIN_DELIVERY_STRICT = "COIN_DELIVERY_STRICT"
COIN_GRASP_DELIVERY_STRICT = "COIN_GRASP_DELIVERY_STRICT"


@dataclass(frozen=True)
class CertStep:
    """Per-step certificate inputs, read from the live env metrics (never invented).

    # Preconditions: ``disk_to_zone``/``disk_speed`` >= 0; contact/force fields from real MuJoCo contacts.
    """
    disk_to_zone: float          # in-plane coin-centre → zone-centre distance (m)
    disk_speed: float            # in-plane coin speed (m/s)
    left_fingertip: bool         # left fingertip touches the coin this step
    right_fingertip: bool        # right fingertip touches the coin this step
    arm_body_contact: bool       # a forbidden (non-fingertip) arm link touches the coin this step (a shove)
    arm_body_impulse: float      # magnitude of that illegal contact impulse this step (N·s-ish; 0 if none)
    force_left: float = 0.0      # left fingertip normal force (N) — only used by the GRASP certificate
    force_right: float = 0.0     # right fingertip normal force (N) — only used by the GRASP certificate


@dataclass
class DeliveryThresholds:
    """Certificate thresholds. Delivery terms mirror the existing oracle; grasp terms are the added force-closure gate."""
    center_tol: float = 0.02          # zone-entry / centered tolerance (m) — DeliveryRLConfig.center_tol
    settle_vel: float = 0.06          # settled coin-speed ceiling (m/s)
    dwell_req: int = 6                # consecutive centered+settled+clean steps required
    impulse_tol: float = 1e-3         # per-step illegal-impulse ceiling for a CLEAN mechanism
    grasp_force_min: float = 0.3      # per-side minimum normal force (N) for a force-closure interval


@dataclass
class DeliveryCertifier:
    """Streams :class:`CertStep`s and accumulates the two certificates.

    COIN_DELIVERY_STRICT passes when, for ``dwell_req`` consecutive steps, the coin is centered (``<= center_tol``),
    settled (``speed < settle_vel``) and the mechanism is CLEAN (no body-shove, illegal impulse ``<= impulse_tol``) —
    AND the displacement is robot-attributed (a fingertip touched the coin at some point) AND the initial footprints
    were disjoint (``initial_clearance > 0``). No force closure is required.

    COIN_GRASP_DELIVERY_STRICT additionally requires, across that same dwell, sustained BILATERAL fingertip contact
    with per-side force ``>= grasp_force_min`` (a force-closure interval).

    # Invariants: ``dwell`` counters reset on any violated step; ``*_certified`` latch True and never revert.
    """
    initial_clearance: float
    th: DeliveryThresholds = field(default_factory=DeliveryThresholds)
    robot_touched: bool = False
    delivery_dwell: int = 0
    grasp_dwell: int = 0
    best_delivery_dwell: int = 0
    best_grasp_dwell: int = 0
    delivery_certified: bool = False
    grasp_certified: bool = False
    _footprints_disjoint: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._footprints_disjoint = self.initial_clearance > 0.0

    def update(self, s: CertStep) -> None:
        self.robot_touched = self.robot_touched or s.left_fingertip or s.right_fingertip
        centered = s.disk_to_zone <= self.th.center_tol
        settled = s.disk_speed < self.th.settle_vel
        clean = (not s.arm_body_contact) and (s.arm_body_impulse <= self.th.impulse_tol)
        if centered and settled and clean:
            self.delivery_dwell += 1
            self.best_delivery_dwell = max(self.best_delivery_dwell, self.delivery_dwell)
            bilateral_force = (s.left_fingertip and s.right_fingertip
                               and min(s.force_left, s.force_right) >= self.th.grasp_force_min)
            self.grasp_dwell = self.grasp_dwell + 1 if bilateral_force else 0
            self.best_grasp_dwell = max(self.best_grasp_dwell, self.grasp_dwell)
        else:
            self.delivery_dwell = 0
            self.grasp_dwell = 0
        if (self.delivery_dwell >= self.th.dwell_req and self.robot_touched and self._footprints_disjoint):
            self.delivery_certified = True
        if self.grasp_dwell >= self.th.dwell_req and self.delivery_certified:
            self.grasp_certified = True

    def satisfies(self, certificate: str) -> bool:
        if certificate == COIN_DELIVERY_STRICT:
            return self.delivery_certified
        if certificate == COIN_GRASP_DELIVERY_STRICT:
            return self.grasp_certified
        raise ValueError(f"unknown certificate {certificate!r}")

    def summary(self) -> dict:
        return {"initial_clearance": round(self.initial_clearance, 5),
                "footprints_disjoint": self._footprints_disjoint,
                "robot_attributed": self.robot_touched,
                "delivery_certified": self.delivery_certified,
                "grasp_certified": self.grasp_certified,
                "best_delivery_dwell": self.best_delivery_dwell,
                "best_grasp_dwell": self.best_grasp_dwell}
