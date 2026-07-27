"""R10-C2 — the APPROACH_MOMENTUM_BUILD hybrid mode prepended to the frozen tip-transport scaffold.

Three converging audits (reach 46mm · M0 50mm · C0/C1) localised the s1 delivery failure to a MISSING trajectory phase: the
monolithic distance-proportional servo `v_ref = k_d·d_remain` slows near the target and cannot build the teacher's forward
momentum (scaffold peak v_par ≤ 0.154 even at max residual, vs teacher 0.322), so the coin arrives at the (working) brake
phase with too little energy. This adds ONE genuinely new mode whose control goal is DIFFERENT — build a bounded forward
impulse and reach a causal launch state — then hand off monotonically to the UNCHANGED HOLD/TRANSPORT → BRAKE/SETTLE →
SQUEEZE-DECAY → R6-certificate-guarded RELEASE.

    APPROACH_MOMENTUM_BUILD  →  (frozen tip-transport)  REGULATE/HOLD → BRAKE → SETTLE → RELEASE(R6 cert)

The APPROACH commands a distance-INDEPENDENT forward joint-velocity reference (bounded by the motion contract, NOT the S1-safe
settle cap) + an acquisition squeeze; it is NOT "raise k_d/v_max" (that stays distance-proportional and was ruled out by M0).
Exit is by monotone causal guards (safety ≻ launch ≻ reachability ≻ budget ≻ horizon); phase never regresses. No teacher θ,
no state edit, no release bit — the R6 certificate remains the sole release authority; all torque/slew/joint/motion limits
are the frozen scaffold's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import REGULATE, TipReferencedController, TipTransportParams

APPROACH = -1                                     # the momentum-build phase, BEFORE the scaffold's REGULATE(0)/RELEASE(1)
CARRY = -2                                        # the candidate handoff phase (held-carry or passive-release), before REGULATE


@dataclass
class ApproachParams:
    """Physically-interpretable APPROACH basis (frozen for a run). `qdot_approach` is distance-INDEPENDENT and may exceed the
    S1-safe settle cap (1.0) but stays under the motion-contract hard limit (governed to ≤ joint hard cap). The `carry_*`
    fields parameterise the C2.5 handoff phase-existence audit (held-momentum-carry vs teacher-like released-coast)."""

    qdot_approach: float = 2.2                    # rad/s — distance-independent forward joint-velocity demand (momentum build)
    acquire_squeeze: float = 0.16                 # N·m — grip to retain bilateral contact under the stronger push
    launch_vlo: float = 0.20                      # target-frame coin v_par launch interval (a band, not a point — anti-chatter)
    launch_vhi: float = 0.42
    coast_decel: float = 0.5                      # m/s² — assumed released-coast deceleration for the reachability guard
    impulse_budget: float = 0.6                   # m — bounded ∑ v_par·dt effort budget (mandatory exit)
    max_steps: int = 24                           # APPROACH horizon guard (mandatory safe exit)
    fn_min_safe: float = 0.05                     # N — below this on either tip ⇒ contact-loss risk ⇒ safe exit
    carry_steps: int = 0                          # C2.5 handoff duration between APPROACH and BRAKE (0 = direct handoff)
    carry_qref: float = 0.0                       # rad/s forward joint-velocity held during carry (0 = HELD_ZERO; >0 = HELD_LOW)
    carry_release: bool = False                   # True = PASSIVE_RELEASE (relax grip, coast free) — DIAGNOSTIC branch only


class HybridApproachController(TipReferencedController):
    """Frozen tip-transport scaffold + a prepended APPROACH_MOMENTUM_BUILD phase. `dtau_for_step` (inherited) calls the
    overridden `_base_targets`; the inherited `_servo` tracks the APPROACH's high `base_qref` exactly as it tracks the
    scaffold's. # Preconditions: a certified straddle snapshot. # Postconditions: monotone APPROACH→REGULATE→RELEASE (never
    regresses); |Δτ| ≤ slew unchanged; release stays R6-certificate-gated; with APPROACH disabled (`enabled=False`) reproduces
    the frozen scaffold bit-for-bit."""

    def __init__(self, snap: CradleSnapshot, params: TipTransportParams = TipTransportParams(),
                 approach: ApproachParams = ApproachParams(), cfg: Any = DELIVERY_CFG, enabled: bool = True) -> None:
        self.ap = approach
        self.enabled = bool(enabled)
        super().__init__(snap, params, cfg)

    def reset(self) -> None:
        super().reset()
        self.phase = APPROACH if self.enabled else REGULATE      # start in APPROACH (or the frozen scaffold if disabled)
        self._impulse = 0.0
        self._approach_steps = 0
        self._carry_remaining = 0
        self.approach_exit_step: "int | None" = None
        self.exit_reason: "str | None" = None
        self.approach_end: "dict | None" = None                  # (v_par, dtz) at APPROACH→CARRY/REGULATE
        self.brake_start: "dict | None" = None                   # (v_par, dtz) handed to the frozen brake

    def _carry_targets(self, rl: Any, t: int) -> "dict[str, Any]":
        """The candidate handoff phase (C2.5): HELD (grip retained, forward effort `carry_qref`) or PASSIVE_RELEASE (relax
        grip, coast free) for `carry_steps`, then monotone handoff to the frozen BRAKE."""
        e_par, dtz, v_par = self._live(rl)
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        if self._carry_remaining <= 0:                           # carry done → hand to the frozen servo/brake THIS step
            self.phase = REGULATE
            self.brake_start = {"v_par": round(v_par, 4), "dtz_mm": round(dtz * 1000, 1)}
            return super()._base_targets(rl, t)
        self._carry_remaining -= 1
        fwd_dir, sqz_dir = self._directions(rl, e_par, _coin_xy(rl))
        if self.ap.carry_release:                                # PASSIVE_RELEASE (diagnostic): relax the grip, coin coasts
            return {"dtz": dtz, "v_par": v_par, "fwd_dir": fwd_dir, "sqz_dir": sqz_dir, "in_release": True,
                    "base_qref": 0.0, "base_sqz": float(self._sqz), "both": both, "contact_risk": not both, "con": con}
        return {"dtz": dtz, "v_par": v_par, "fwd_dir": fwd_dir, "sqz_dir": sqz_dir, "in_release": False,
                "base_qref": float(self.ap.carry_qref), "base_sqz": float(self.p.squeeze_hold), "both": both,
                "contact_risk": not both, "con": con}

    def _exit_approach(self, dtz: float, v_par: float, both: bool, contact_risk: bool) -> "str | None":
        """First applicable monotone exit guard (safety ≻ launch ≻ reachability ≻ budget ≻ horizon); None ⇒ keep approaching.
        No future/K6/oracle info enters the guard — only measured causal state."""
        if contact_risk:
            return "SAFETY"
        if both and self.ap.launch_vlo <= v_par <= self.ap.launch_vhi:
            return "LAUNCH"
        d_remain = max(dtz - CENTER_TOL, 0.0)
        if v_par > 0.0 and v_par * v_par >= 2.0 * self.ap.coast_decel * d_remain:
            return "REACHABILITY"
        if self._impulse >= self.ap.impulse_budget:
            return "BUDGET"
        if self._approach_steps >= self.ap.max_steps:
            return "HORIZON"
        return None

    def _base_targets(self, rl: Any, t: int) -> "dict[str, Any]":
        if self.phase == CARRY:
            return self._carry_targets(rl, t)
        if self.phase != APPROACH:
            return super()._base_targets(rl, t)                  # REGULATE/HOLD/BRAKE/RELEASE = the frozen scaffold
        e_par, dtz, v_par = self._live(rl)
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        fn_min = min(float(con["left"]["fn"]) if con["left"] else 0.0, float(con["right"]["fn"]) if con["right"] else 0.0)
        contact_risk = (not both) or (fn_min < self.ap.fn_min_safe)
        self._impulse += max(v_par, 0.0) * float(self.snap.stack.control_dt)
        self._approach_steps += 1
        reason = self._exit_approach(dtz, v_par, both, contact_risk)
        if reason is not None:                                   # APPROACH exit → CARRY (if any) then the frozen brake
            self.approach_exit_step, self.exit_reason = int(t), reason
            self.approach_end = {"v_par": round(v_par, 4), "dtz_mm": round(dtz * 1000, 1)}
            if self.ap.carry_steps > 0:
                self.phase = CARRY
                self._carry_remaining = int(self.ap.carry_steps)
                return self._carry_targets(rl, t)
            self.phase = REGULATE
            self.brake_start = self.approach_end
            return super()._base_targets(rl, t)
        fwd_dir, sqz_dir = self._directions(rl, e_par, _coin_xy(rl))
        return {"dtz": dtz, "v_par": v_par, "fwd_dir": fwd_dir, "sqz_dir": sqz_dir, "in_release": False,
                "base_qref": float(self.ap.qdot_approach), "base_sqz": float(self.ap.acquire_squeeze), "both": both,
                "contact_risk": contact_risk, "con": con}
