"""COIN-DELIVERY-ACTOR-1 — cooperative fingertip push/plow delivery actors + clean instrumentation, attribution, and an
independent valid-delivery monitor (NO clamp-drag, NO RL, NO canonical-arm/actuator change).

CLAMP-ORACLE-0 (Case A) showed the canonical-arm clamp failure is kinematic, NOT a need for force actuation — but it does
NOT make clamp transport the required delivery mechanism. This module tests whether cleanly reconstructed cooperative
PUSH/PLOW actors deliver the cylindrical coin WITHOUT strict bilateral clamp transport, on the current clean harness
(6-DoF cooperative action a=[midpoint_x, midpoint_y, aperture, squeeze, differential, rotate]; delivery zone at (0,0.16)).

Reuses the env's per-step fingertip/body progress attribution and `coin_grip_control.normal_contact_forces`; adds the
per-fingertip L/R split, the 5-component attribution vector, the strict dwell-verified delivery monitor, and the mechanism
classifier. Attribution is IMPULSE-directed where possible; components that are approximations are labelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hymeko_rl.experiments.coin_delivery1 import _dir_to_zone
from hymeko_rl.train.coin_grip_control import normal_contact_forces

_EPS = 1e-9
_ZONE = np.array([0.0, 0.16])


class DeliveryActor(str, Enum):
    A0_SYM_PUSH = "A0_symmetric_push"        # both tips push symmetrically toward the zone (light, non-clamping)
    A1_VPLOW = "A1_v_plow"                    # tips angled into a V that plows the coin forward
    A2_ASYM_PUSH = "A2_asymmetric_push"       # differential/guided push (steer the coin)
    A3_SETUP_PUSH = "A3_setup_push"           # phase FSM: position behind the coin, then push
    A4_RECOVERY = "A4_release_recontact_push"  # pulse-release to reset contact + re-push (recovery)
    A5_NEUTRAL = "A5_neutral"                 # contact-preserving no-push control (the neutral baseline)


@dataclass(frozen=True)
class ActorParams:
    aperture: float = -0.2       # slightly open (avoid a sustained clamp → handoff termination)
    squeeze: float = 0.4         # light contact
    differential: float = 0.0
    rotate: float = 0.0
    v_plow_rotate: float = 0.5   # V-plow tip angle
    v_plow_squeeze: float = 0.3
    asym_differential: float = 0.6
    pulse_period: int = 4        # A4: release every pulse_period-th step


def actor_action(inner, t: int, actor: DeliveryActor, p: ActorParams = ActorParams()) -> np.ndarray:
    """Deterministic 6-DoF cooperative action for a push/plow actor. Push actors keep contact LIGHT and non-clamping so
    the coin is nudged toward the zone rather than clamped-and-carried (which triggers the 4-step handoff termination)."""
    d, n = _dir_to_zone(inner)
    if actor is DeliveryActor.A5_NEUTRAL:
        return np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], np.float32)            # hold contact, no translation
    if actor is DeliveryActor.A0_SYM_PUSH:
        return np.array([d[0], d[1], p.aperture, p.squeeze, 0.0, 0.0], np.float32)
    if actor is DeliveryActor.A1_VPLOW:
        return np.array([d[0], d[1], -0.1, p.v_plow_squeeze, 0.0, p.v_plow_rotate], np.float32)
    if actor is DeliveryActor.A2_ASYM_PUSH:
        return np.array([d[0], d[1], -0.3, 0.35, p.asym_differential, 0.0], np.float32)
    if actor is DeliveryActor.A3_SETUP_PUSH:
        far = n > inner._zone_half * 1.5
        sq = 0.8 if far else 0.3
        return np.array([d[0], d[1], 0.0, sq, 0.0, 0.0], np.float32)
    # A4 recovery: release the contact every pulse_period step (reset the both-contact streak), else push
    sq = p.squeeze if (t % p.pulse_period != p.pulse_period - 1) else -0.7
    return np.array([d[0], d[1], p.aperture, sq, 0.0, 0.0], np.float32)


# ── attribution ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Attribution:
    """The 5-component progress-attribution vector α = [L, R, body, support, free], each a FRACTION of the total
    toward-zone progress. L/R are IMPULSE-weighted (per-fingertip normal force). support ≈ 0 by construction (a
    slide-joint planar table RESISTS, never propels — reported for completeness). free = progress with no contact
    (momentum/coasting) — an APPROXIMATION (it cannot separate coasting from an unmodeled contribution)."""
    alpha_L: float
    alpha_R: float
    alpha_body: float
    alpha_support: float
    alpha_free: float
    total_progress: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def fingertip_fraction(self) -> float:
        return self.alpha_L + self.alpha_R


def _attribute_step(dprog: float, fl: float, fr: float, body: bool, acc: dict) -> None:
    """Attribute one step's toward-zone progress ``dprog`` to L/R (impulse-weighted) / body / free."""
    if dprog <= 0:
        return
    both = fl > 1e-4 and fr > 1e-4
    if both:
        acc["L"] += dprog * fl / (fl + fr)
        acc["R"] += dprog * fr / (fl + fr)
    elif fl > 1e-4:
        acc["L"] += dprog
    elif fr > 1e-4:
        acc["R"] += dprog
    elif body:
        acc["body"] += dprog
    else:
        acc["free"] += dprog


def _finalize_attribution(acc: dict) -> Attribution:
    total = acc["L"] + acc["R"] + acc["body"] + acc["free"] + _EPS
    return Attribution(round(acc["L"] / total, 3), round(acc["R"] / total, 3), round(acc["body"] / total, 3),
                       0.0, round(acc["free"] / total, 3), round(total, 4))


# ── mechanism classification ─────────────────────────────────────────────────────────────────────────────────────
class Mechanism(str, Enum):
    SYM_PUSH = "symmetric_cooperative_push"
    VPLOW = "v_plow"
    ASYM_PUSH = "asymmetric_guided_push"
    SETUP_PUSH = "setup_push"
    RECOVERY_PUSH = "recovery_assisted_push"
    ONE_FINGER_BULLDOZE = "one_finger_bulldoze"
    BODY_SHOVE = "body_shove"
    FREE_MOTION = "accidental_free_motion"
    CONTACT_STALL = "contact_stall"
    INVALID_INITIAL = "invalid_initial_success"
    UNCLASSIFIED = "unclassified"


# thresholds for the strict monitor + mechanism classification
_BODY_SHOVE_MAX = 0.20       # α_body above this = a body shove, not clean
_ONE_FINGER_MAX = 0.15       # min(α_L, α_R)/fingertip below this while pushing = a one-finger bulldoze
_DWELL_STEPS = 6             # consecutive in-zone steps required
_SETTLE_VEL = 0.15          # coin speed at delivery must be below this
_SETTLE_OMEGA = 2.0         # coin angular speed
_PROGRESS_MIN = 0.02        # meaningful progress floor


def classify_mechanism(actor: DeliveryActor, att: Attribution, *, made_progress: bool, initial_success: bool,
                        both_frac: float) -> Mechanism:
    """Classify the episode's dominant delivery mechanism (§mechanism classification)."""
    if initial_success:
        return Mechanism.INVALID_INITIAL
    if not made_progress:
        return Mechanism.CONTACT_STALL
    if att.alpha_body > _BODY_SHOVE_MAX:
        return Mechanism.BODY_SHOVE
    if att.fingertip_fraction < 0.5 and att.alpha_free > 0.5:
        return Mechanism.FREE_MOTION
    ff = att.fingertip_fraction + _EPS
    if min(att.alpha_L, att.alpha_R) / ff < _ONE_FINGER_MAX:
        return Mechanism.ONE_FINGER_BULLDOZE                       # one tip does ~all the work
    return {DeliveryActor.A0_SYM_PUSH: Mechanism.SYM_PUSH, DeliveryActor.A1_VPLOW: Mechanism.VPLOW,
            DeliveryActor.A2_ASYM_PUSH: Mechanism.ASYM_PUSH, DeliveryActor.A3_SETUP_PUSH: Mechanism.SETUP_PUSH,
            DeliveryActor.A4_RECOVERY: Mechanism.RECOVERY_PUSH}.get(actor, Mechanism.UNCLASSIFIED)


# ── rollout + valid-delivery monitor ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DeliveryResult:
    seed: int
    actor: str
    strict_delivery: bool        # the 9-condition valid-delivery monitor
    loose_in_zone: bool          # momentary in_zone ever (the old loose signal, for comparison)
    initial_success: bool
    progress: float
    min_dtz: float
    dwell: int
    settle_vel: float
    attribution: dict
    fingertip_fraction: float
    body_shove: bool
    mechanism: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def rollout_delivery(env, seed: int, actor: DeliveryActor, p: ActorParams = ActorParams(),
                     *, max_steps: int = 60, scramble=None) -> DeliveryResult:
    """Roll one actor from a fresh reset; accumulate impulse-directed attribution, dwell, settle, and apply the strict
    valid-delivery monitor + mechanism classification. History-independent; exact reset by seed."""
    env.reset(seed=int(seed))
    inner = env._env
    m0 = inner._planar_metrics
    initial_success = bool(m0.in_zone)
    start_dtz = float(m0.disk_to_zone)
    prev_dtz = start_dtz
    acc = {"L": 0.0, "R": 0.0, "body": 0.0, "free": 0.0}
    log = _roll_loop(env, inner, actor, p, max_steps, scramble, acc, prev_dtz, start_dtz)
    att = _finalize_attribution(acc)
    made_progress = log["progress"] >= _PROGRESS_MIN
    mech = classify_mechanism(actor, att, made_progress=made_progress, initial_success=initial_success,
                              both_frac=log["both_frac"])
    strict = _valid_delivery(initial_success, log, att, mech)
    return DeliveryResult(seed, actor.value, strict, log["loose"], initial_success, round(log["progress"], 4),
                          round(log["min_dtz"], 4), log["dwell"], round(log["settle_vel"], 4), att.as_dict(),
                          att.fingertip_fraction, att.alpha_body > _BODY_SHOVE_MAX, mech.value)


def _roll_loop(env, inner, actor, p, max_steps, scramble, acc, prev_dtz, start_dtz) -> dict:
    min_dtz, loose, dwell, best_dwell = prev_dtz, False, 0, 0
    both_steps, settle_vel, prev_body_steps = 0, 1.0, int(inner._arm_body_steps)
    for t in range(max_steps):
        a = np.clip(actor_action(inner, t, actor, p), -1, 1).astype(np.float32)
        if scramble is not None:
            a = scramble(a, t)
        _o, _r, term, trunc, info = env.step(a)
        mm = inner._planar_metrics
        dtz = float(mm.disk_to_zone)
        fl, fr = normal_contact_forces(inner)
        body = int(inner._arm_body_steps) > prev_body_steps         # arm-body↔coin contact occurred THIS step
        prev_body_steps = int(inner._arm_body_steps)
        _attribute_step(prev_dtz - dtz, fl, fr, body, acc)
        both_steps += int(bool(mm.left_contact and mm.right_contact))
        if bool(mm.in_zone):
            loose = True
            dwell += 1
            best_dwell = max(best_dwell, dwell)
            settle_vel = float(np.linalg.norm(mm.disk_vel))
        else:
            dwell = 0
        min_dtz = min(min_dtz, dtz)
        prev_dtz = dtz
        if term or trunc:
            break
    return {"progress": start_dtz - min_dtz, "min_dtz": min_dtz, "loose": loose, "dwell": best_dwell,
            "settle_vel": settle_vel, "both_frac": both_steps / max(1, max_steps)}


@dataclass(frozen=True)
class DeliveryReward:
    """A delivery-ALIGNED reward (separate from the monitor). Potential-based progress + a ONE-TIME graded delivery
    bonus + a bounded progress-gated contact-quality term − body-shove − stagnation. NO contact annuity (no per-step
    both-contact reward that a hold-without-progress can farm), and it is NEVER reused as monitor evidence. Unlike the
    v16 cooperative_push reward it does NOT penalise contact loss — so it does not fight intermittent-contact delivery."""
    w_progress: float = 20.0
    w_delivery: float = 40.0     # one-time, on first zone entry
    w_contact_quality: float = 2.0   # bounded, ONLY while progressing with both tips
    w_body_shove: float = 8.0
    w_stagnation: float = 0.5


def delivery_step_reward(m, d_dist: float, delivered_now: bool, body: bool, cfg: DeliveryReward = DeliveryReward()) -> float:
    """Per-step delivery-aligned reward. ``d_dist`` = prev_dtz − cur_dtz (positive = progress toward the zone).
    ``delivered_now`` = the one-time zone-entry event. ``body`` = arm-body↔coin contact this step."""
    both = bool(m.left_contact and m.right_contact)
    r = cfg.w_progress * d_dist
    if delivered_now:
        r += cfg.w_delivery
    if d_dist > 1e-4 and both:
        r += cfg.w_contact_quality * d_dist          # progress-gated (no idle-hold annuity)
    if body:
        r -= cfg.w_body_shove * 0.1
    if abs(d_dist) < 1e-4:
        r -= cfg.w_stagnation * 0.1                  # small per-step stagnation pressure
    return float(r)


def _valid_delivery(initial_success: bool, log: dict, att: Attribution, mech: Mechanism) -> bool:
    """The strict 9-condition valid-delivery monitor (independent of any training reward). Requires: not initially
    successful · zone entry · dwell ≥ K · low settle velocity · fingertip-attributed progress · body-shove below
    threshold · a clean mechanism (not bulldoze/shove/free/stall/invalid)."""
    clean = mech in (Mechanism.SYM_PUSH, Mechanism.VPLOW, Mechanism.ASYM_PUSH, Mechanism.SETUP_PUSH,
                     Mechanism.RECOVERY_PUSH)
    return bool(not initial_success and log["loose"] and log["dwell"] >= _DWELL_STEPS
                and log["settle_vel"] <= _SETTLE_VEL and att.fingertip_fraction >= 0.6
                and att.alpha_body <= _BODY_SHOVE_MAX and clean)
