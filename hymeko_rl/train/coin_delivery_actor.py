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


# ── the single canonical delivery rollout + its public result ────────────────────────────────────────────────────
@dataclass(frozen=True)
class RolloutStep:
    """One public rollout step: the action taken and the resulting PUBLIC planar metrics. Downstream code (scripted
    actors, BC/SAC eval, deterministic replay) reads this instead of the env's private ``_planar_metrics``."""
    action: tuple            # the 6-DoF cooperative action actually applied (post-clip / post-scramble)
    disk_to_zone: float
    disk_pos: tuple          # coin centre (x, y)
    disk_vel_norm: float
    left_contact: bool
    right_contact: bool
    fl: float                # left fingertip normal force (impulse-attribution source)
    fr: float                # right fingertip normal force
    body_contact: bool       # arm-body↔coin contact occurred THIS step
    in_zone: bool
    reward: float
    terminated: bool
    truncated: bool
    info: dict               # raw env info (env-specific telemetry; empty dict if the env emits none)


@dataclass(frozen=True)
class RolloutTrace:
    """The public result of one canonical rollout — the single source scripted/BC/SAC eval + replay consume.
    Derived summaries reproduce the historical ``_roll_loop`` semantics exactly (dwell = longest in-zone run;
    ``settle_vel`` = last in-zone step's coin speed, else 1.0; ``both_frac`` over the full step BUDGET)."""
    steps: list
    initial_success: bool
    start_dtz: float
    max_steps: int

    @property
    def loose(self) -> bool:
        return any(s.in_zone for s in self.steps)

    @property
    def min_dtz(self) -> float:
        return min([self.start_dtz, *(s.disk_to_zone for s in self.steps)])

    @property
    def progress(self) -> float:
        return self.start_dtz - self.min_dtz

    @property
    def best_dwell(self) -> int:
        best = run = 0
        for s in self.steps:
            run = run + 1 if s.in_zone else 0
            best = max(best, run)
        return best

    @property
    def settle_vel(self) -> float:
        vel = 1.0
        for s in self.steps:
            if s.in_zone:
                vel = s.disk_vel_norm
        return vel

    @property
    def both_frac(self) -> float:
        both = sum(int(s.left_contact and s.right_contact) for s in self.steps)
        return both / max(1, self.max_steps)

    @property
    def terminated(self) -> bool:
        return bool(self.steps and (self.steps[-1].terminated or self.steps[-1].truncated))


def _planar_env(env):
    """Resolve the underlying planar MuJoCo env (the metrics source) for either a raw ``ContactFormationEnv``
    (``._env``) or a wrapper exposing ``.inner`` (e.g. ``CoinDeliveryTrainEnv``). # Preconditions one must exist."""
    inner = getattr(env, "inner", None)
    return inner if inner is not None else env._env


def rollout(env, action_fn, *, max_steps: int = 60, scramble=None) -> RolloutTrace:
    """THE canonical delivery rollout. Steps ``env`` under ``action_fn(inner, t, obs) -> 6-DoF action`` (a scripted
    actor reads ``inner``/``t`` and ignores ``obs``; a learned policy reads ``obs``) and records the public per-step
    trace. All scripted-actor / BC / SAC evaluation and deterministic replay funnel through here — no experiment-level
    module re-implements the step/metric/termination loop or reads ``_planar_metrics``.

    # Preconditions the caller has already reset/restored ``env`` to the desired start state; ``action_fn`` returns
    a finite 6-vector. # Postconditions returns a :class:`RolloutTrace` whose summaries are history-independent."""
    inner = _planar_env(env)
    m0 = inner.planar_metrics
    obs = getattr(env, "_last_obs", None)                       # learned-policy obs; a scripted actor ignores it
    prev_body = int(inner.arm_body_steps)
    steps: list = []
    for t in range(max_steps):
        a = np.clip(action_fn(inner, t, obs), -1, 1).astype(np.float32)
        if scramble is not None:
            a = scramble(a, t)
        obs, r, term, trunc, info = env.step(a)
        mm = inner.planar_metrics
        fl, fr = normal_contact_forces(inner)
        body = int(inner.arm_body_steps) > prev_body            # arm-body↔coin contact occurred THIS step
        prev_body = int(inner.arm_body_steps)
        steps.append(RolloutStep(
            tuple(float(x) for x in a), float(mm.disk_to_zone),
            (float(mm.disk_pos[0]), float(mm.disk_pos[1])), float(np.linalg.norm(mm.disk_vel)),
            bool(mm.left_contact), bool(mm.right_contact), float(fl), float(fr), bool(body),
            bool(mm.in_zone), float(r), bool(term), bool(trunc), dict(info) if info else {}))
        if term or trunc:
            break
    return RolloutTrace(steps, bool(m0.in_zone), float(m0.disk_to_zone), int(max_steps))


def _attribution_from_trace(trace: RolloutTrace) -> Attribution:
    """Walk a public :class:`RolloutTrace` and accumulate the impulse-directed L/R/body/free progress attribution
    (identical accumulation to the historical inline loop, now over the public step record)."""
    acc = {"L": 0.0, "R": 0.0, "body": 0.0, "free": 0.0}
    prev = trace.start_dtz
    for s in trace.steps:
        _attribute_step(prev - s.disk_to_zone, s.fl, s.fr, s.body_contact, acc)
        prev = s.disk_to_zone
    return _finalize_attribution(acc)


def rollout_delivery(env, seed: int, actor: DeliveryActor, p: ActorParams = ActorParams(),
                     *, max_steps: int = 60, scramble=None) -> DeliveryResult:
    """Roll one scripted actor from a fresh reset through the canonical :func:`rollout`, then apply impulse-directed
    attribution + the strict valid-delivery monitor + mechanism classification. History-independent; reset by seed."""
    env.reset(seed=int(seed))
    trace = rollout(env, lambda inner, t, _obs: actor_action(inner, t, actor, p),
                    max_steps=max_steps, scramble=scramble)
    att = _attribution_from_trace(trace)
    mech = classify_mechanism(actor, att, made_progress=trace.progress >= _PROGRESS_MIN,
                              initial_success=trace.initial_success, both_frac=trace.both_frac)
    strict = _valid_delivery(trace, att, mech)
    return DeliveryResult(seed, actor.value, strict, trace.loose, trace.initial_success, round(trace.progress, 4),
                          round(trace.min_dtz, 4), trace.best_dwell, round(trace.settle_vel, 4), att.as_dict(),
                          att.fingertip_fraction, att.alpha_body > _BODY_SHOVE_MAX, mech.value)


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


def _valid_delivery(trace: "RolloutTrace", att: Attribution, mech: Mechanism) -> bool:
    """The strict 9-condition valid-delivery monitor (independent of any training reward). Requires: not initially
    successful · zone entry · dwell ≥ K · low settle velocity · fingertip-attributed progress · body-shove below
    threshold · a clean mechanism (not bulldoze/shove/free/stall/invalid). Reads the PUBLIC rollout trace."""
    clean = mech in (Mechanism.SYM_PUSH, Mechanism.VPLOW, Mechanism.ASYM_PUSH, Mechanism.SETUP_PUSH,
                     Mechanism.RECOVERY_PUSH)
    return bool(not trace.initial_success and trace.loose and trace.best_dwell >= _DWELL_STEPS
                and trace.settle_vel <= _SETTLE_VEL and att.fingertip_fraction >= 0.6
                and att.alpha_body <= _BODY_SHOVE_MAX and clean)
