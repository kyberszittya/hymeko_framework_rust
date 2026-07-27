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
from hymeko_rl.coin_delivery.theta_option.tip_transport import (
    REGULATE, RELEASE, TipReferencedController, TipTransportParams)

APPROACH = -1                                     # the momentum-build phase, BEFORE the scaffold's REGULATE(0)/RELEASE(1)
CARRY = -2                                        # the candidate handoff phase (held-carry or passive-release), before REGULATE
REACQUIRE = -3                                    # C2.8: re-acquire contact after a RELEASED_COAST, then hand to the frozen settle
MICRO = -4                                        # R3-B: bounded micro-transport (delta_forward over the settle) after re-acquire


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
    carry_squeeze: float = 0.14                   # N·m grip during held carry (C2.7 GUIDED_COAST = LOW ⇒ light contact, coasts)
    carry_release: bool = False                   # True = PASSIVE_RELEASE (relax grip, coast free) — DIAGNOSTIC branch only
    release_at_step: int = -1                     # C2.6 data-collection: force RELEASED_COAST at this step (−1 = off)
    coast_guard: bool = False                     # C2.6 deploy: RELEASE when the ROBUST predicted coast-landing ⊆ corridor
    guard_amin: float = 0.3                       # m/s² — LOW end of the observed released-coast deceleration (→ longest coast)
    guard_amax: float = 1.0                       # m/s² — HIGH end (→ shortest coast); interval NOT a point estimate (R5 lesson)
    corridor_hi: float = 0.020                    # m — the target corridor (the frozen 20 mm K6 zone)
    guard_margin: float = 0.004                   # m — uncertainty margin for the robust interval containment
    reacquire: bool = False                       # C2.8 R0/R1: after RELEASED_COAST, re-acquire contact when the coin slows in the corridor
    reacq_vclose: float = 0.16                    # m/s — start re-acquire only once the coin's closing velocity is BELOW this
    reacq_corridor_hi: float = 0.070              # m — coin must be within this reachable-contact corridor to re-acquire
    reacq_qref: float = 0.35                       # rad/s — gentle forward joint-velocity to catch the coasting coin
    reacq_squeeze_step: float = 0.02              # N·m/step — RAMPED squeeze (not stepped) — impulse-limited re-grip
    reacq_max_steps: int = 18                     # re-acquire horizon guard
    micro_transport: bool = False                 # R3-B: a MICRO-TRANSPORT (one bounded delta_forward channel) after re-acquire
    micro_forward: float = 0.0                    # rad/s bounded delta_forward added to the settle qref (0 = update-zero = C2.8)
    micro_vcap_qref: float = 0.9                  # rad/s hard cap on the nudged joint-velocity reference (low velocity cap)
    micro_impulse_budget: float = 0.04            # m — small ∑ v_par·dt budget for the micro-transport (mandatory exit)
    micro_max_steps: int = 12                     # short micro-transport horizon (mandatory exit)
    micro_brake_entry_hi: float = 0.018           # m — hand to the frozen settle once dtz ≤ this (brake-entry state)


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
        self.release_state: "dict | None" = None                 # (v_par, dtz, t) at a C2.6 coast-entry RELEASE
        self._reacq_sqz = self.p.squeeze_min
        self._reacq_steps = 0
        self.reacquire_start: "dict | None" = None               # coast state when re-acquire began
        self.reacquire_end: "dict | None" = None                 # {dtz, steps, success, first_contact_dv} at re-grip / timeout
        self._micro_impulse = 0.0
        self._micro_steps = 0

    def _micro_targets(self, rl: Any, t: int) -> "dict[str, Any]":
        """R3-B MICRO-TRANSPORT: the FROZEN settle targets + a bounded `delta_forward` on the joint-velocity reference, applied
        only in a legal transport state (bilateral contact, no reversal, above the brake-entry corridor, within impulse/horizon
        budget). `micro_forward = 0` ⇒ exactly the settle (update-zero identity). Exits to the frozen settle at brake-entry or
        on any budget/safety guard — it does NOT drive to K6."""
        bt = super()._base_targets(rl, t)                         # the frozen REGULATE/settle base (velocity_ref + squeeze)
        self._micro_steps += 1
        legal = (bt["both"] and not bt["contact_risk"] and bt["v_par"] >= -0.01 and bt["dtz"] > self.ap.micro_brake_entry_hi
                 and self._micro_impulse < self.ap.micro_impulse_budget and self._micro_steps <= self.ap.micro_max_steps)
        if not legal:                                            # brake-entry reached / budget spent / unsafe ⇒ frozen settle
            self.phase = REGULATE
            return bt
        self._micro_impulse += max(bt["v_par"], 0.0) * float(self.snap.stack.control_dt)
        return {**bt, "base_qref": float(min(bt["base_qref"] + self.ap.micro_forward, self.ap.micro_vcap_qref))}

    def _reacquire_targets(self, rl: Any, t: int) -> "dict[str, Any]":
        """C2.8 RE-ACQUIRE: gently catch the coasting coin (bounded forward joint-velocity) and RAMP the squeeze (impulse-
        limited) until bilateral contact re-forms, then hand to the frozen BRAKE/SETTLE. # Post: minimise first-contact
        impulse; do not command raw torque; monotone → REGULATE."""
        e_par, dtz, v_par = self._live(rl)
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        fn_min = min(float(con["left"]["fn"]) if con["left"] else 0.0, float(con["right"]["fn"]) if con["right"] else 0.0)
        self._reacq_sqz = min(self.p.squeeze_hold, self._reacq_sqz + self.ap.reacq_squeeze_step)
        self._reacq_steps += 1
        reacquired = both and fn_min > self.ap.fn_min_safe
        if reacquired or self._reacq_steps > self.ap.reacq_max_steps:
            self.reacquire_end = {"dtz_mm": round(dtz * 1000, 1), "steps": self._reacq_steps, "success": bool(reacquired),
                                  "v_par_at_regrip": round(v_par, 4)}
            if reacquired and self.ap.micro_transport:           # R3-B: bridge the last ~11 mm before the frozen settle
                self.phase = MICRO
                return self._micro_targets(rl, t)
            self.phase = REGULATE
            return super()._base_targets(rl, t)
        fwd_dir, sqz_dir = self._directions(rl, e_par, _coin_xy(rl))
        return {"dtz": dtz, "v_par": v_par, "fwd_dir": fwd_dir, "sqz_dir": sqz_dir, "in_release": False,
                "base_qref": float(self.ap.reacq_qref), "base_sqz": float(self._reacq_sqz), "both": both,
                "contact_risk": False, "con": con}

    def _guard_fires(self, v_par: float, dtz: float, both: bool) -> bool:
        """ROBUST coast-entry guard (R5 lesson: no online post-release friction estimate — use the observed deceleration
        INTERVAL). The passive stop distance is d_stop ∈ [v²/2·a_max, v²/2·a_min]; fire only when the WHOLE predicted landing
        interval lands in [−margin, corridor_hi] (least coast still reaches the zone; most coast does not overshoot). # Post:
        release only from bilateral contact with real forward momentum."""
        if (not both) or v_par <= 0.02:
            return False
        ds_min = v_par * v_par / (2.0 * self.ap.guard_amax)      # least coast (high deceleration)
        ds_max = v_par * v_par / (2.0 * self.ap.guard_amin)      # most coast (low deceleration)
        landing_far, landing_close = dtz - ds_min, dtz - ds_max
        m = self.ap.guard_margin
        return bool(landing_far <= self.ap.corridor_hi + m and landing_close >= -m)

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
                "base_qref": float(self.ap.carry_qref), "base_sqz": float(self.ap.carry_squeeze), "both": both,
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
        """Thin phase dispatcher — each hybrid mode owns its own target method; REGULATE/HOLD/BRAKE/RELEASE = frozen scaffold."""
        if self.phase == CARRY:
            return self._carry_targets(rl, t)
        if self.phase == REACQUIRE:
            return self._reacquire_targets(rl, t)
        if self.phase == MICRO:
            return self._micro_targets(rl, t)
        if self.phase == RELEASE:
            rt = self._maybe_reacquire(rl, t)
            if rt is not None:
                return rt
        if self.phase != APPROACH:
            return super()._base_targets(rl, t)
        return self._approach_targets(rl, t)

    def _maybe_reacquire(self, rl: Any, t: int) -> "dict | None":
        """C2.8 — from a RELEASED_COAST, enter RE-ACQUIRE once the coin slows (closing velocity) into the reachable corridor."""
        if not (self.ap.reacquire and self.reacquire_start is None):
            return None
        _e, dtz, v_par = self._live(rl)
        if v_par < self.ap.reacq_vclose and dtz <= self.ap.reacq_corridor_hi:
            self.phase = REACQUIRE
            self.reacquire_start = {"v_par": round(v_par, 4), "dtz_mm": round(dtz * 1000, 1), "t": int(t)}
            return self._reacquire_targets(rl, t)
        return None

    def _approach_exit(self, rl: Any, t: int, v_par: float, dtz: float) -> "dict[str, Any]":
        """APPROACH exit → CARRY (if configured) then the frozen brake."""
        self.approach_exit_step, self.exit_reason = int(t), self.exit_reason
        self.approach_end = {"v_par": round(v_par, 4), "dtz_mm": round(dtz * 1000, 1)}
        if self.ap.carry_steps > 0:
            self.phase = CARRY
            self._carry_remaining = int(self.ap.carry_steps)
            return self._carry_targets(rl, t)
        self.phase = REGULATE
        self.brake_start = self.approach_end
        return super()._base_targets(rl, t)

    def _approach_targets(self, rl: Any, t: int) -> "dict[str, Any]":
        """APPROACH_MOMENTUM_BUILD: distance-independent forward joint-velocity + acquisition squeeze until a causal exit
        (or a C2.6 coast-entry RELEASE) fires."""
        e_par, dtz, v_par = self._live(rl)
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        fn_min = min(float(con["left"]["fn"]) if con["left"] else 0.0, float(con["right"]["fn"]) if con["right"] else 0.0)
        contact_risk = (not both) or (fn_min < self.ap.fn_min_safe)
        if (self.ap.release_at_step >= 0 and t >= self.ap.release_at_step) or \
                (self.ap.coast_guard and self._guard_fires(v_par, dtz, both)):
            self.phase = RELEASE                                 # C2.6 coast-entry RELEASE (launch guard, ≠ the settle cert)
            self.release_state = {"v_par": round(v_par, 4), "dtz_mm": round(dtz * 1000, 1), "t": int(t)}
            return super()._base_targets(rl, t)
        self._impulse += max(v_par, 0.0) * float(self.snap.stack.control_dt)
        self._approach_steps += 1
        self.exit_reason = self._exit_approach(dtz, v_par, both, contact_risk)
        if self.exit_reason is not None:
            return self._approach_exit(rl, t, v_par, dtz)
        fwd_dir, sqz_dir = self._directions(rl, e_par, _coin_xy(rl))
        return {"dtz": dtz, "v_par": v_par, "fwd_dir": fwd_dir, "sqz_dir": sqz_dir, "in_release": False,
                "base_qref": float(self.ap.qdot_approach), "base_sqz": float(self.ap.acquire_squeeze), "both": both,
                "contact_risk": contact_risk, "con": con}
