"""The reward as a declarative term spec — the in-memory form of the
``meta_reward.hymeko`` vocabulary (the reward half of the agent description).

A :class:`RewardSpec` is an ordered list of ``(term_kind, weight)`` pairs; the scalar
reward is ``Σ weight · term(state)``. Each term kind maps to an extractor (Strategy) over
the live env state. :meth:`RewardSpec.from_hymeko` reads the terms + weights straight from a
``.hymeko`` task profile's ``reward_spec``, so a new reward needs only a new ``.hymeko`` —
the env's ``step`` no longer hard-codes ``-dist``.

Term kinds mirror ``data/robotics/meta_reward.hymeko``. Only the reaching task's terms are
implemented; the rest are a registry entry away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from hymeko_rl.env._profile import read_bundle
from hymeko_rl.env.safety import CLEAN_SAFETY

if TYPE_CHECKING:
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

# The env is duck-typed: a term reads only the attributes it needs (reach_thresh, _last_safety,
# _planar_metrics, …), so terms serve both ArmReachEnv and PlanarGraspEnv. Hence `Any` here.
RewardTerm = Callable[[Any, float, np.ndarray], float]


# ── reward-term extractors (Strategy) ────────────────────────────────────────
# Each returns the *unweighted* term value given the env, the EE-to-target distance, and
# the applied action; RewardSpec.evaluate scales by the declared weight.
def _term_reach_distance(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -dist


def _term_success_bonus(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return 1.0 if dist < env.reach_thresh else 0.0


def _term_action_cost(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    a = np.asarray(action, dtype=np.float64)
    return -float(a @ a)   # -‖action‖²


# ── safety / configuration terms (read the env's last SafetyState) ───────────
# The weighted magnitude (the bounded terminal penalty) lives in the .hymeko `weight`; here the
# unweighted term is just the indicator/shape. Fall back to a clean state so the terms are inert
# (0) on an env that has not computed a safety state.
def _term_ground_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).ground_contact else 0.0


def _term_self_collision_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).self_collision else 0.0


def _term_joint_limit_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    # -(1 - margin)²: 0 mid-range, rising smoothly to -1 at a joint limit.
    margin = getattr(env, "_last_safety", CLEAN_SAFETY).joint_margin
    return -((1.0 - margin) ** 2)


def _term_below_ground_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).below_ground else 0.0


# ── planar grasping terms (read the env's PlanarGraspMetrics; 0 on a non-grasp env) ──────────
# The dense pull (-‖disk - zone‖) reuses `reach_distance` by passing disk_to_zone as the distance.
def _term_both_contact(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    m = getattr(env, "_planar_metrics", None)
    return 1.0 if (m is not None and m.left_contact and m.right_contact) else 0.0


def _term_in_zone(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    m = getattr(env, "_planar_metrics", None)
    return 1.0 if (m is not None and m.in_zone) else 0.0


def _term_grasp_deliver(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Grasp-GATED success: +1 only when the coin is in the zone AND it was genuinely grasped at some point
    (``env._ever_grasped``, the latch set when both fingertips contact the coin). A *knock* (coin shoved into
    the zone without a two-finger grasp) earns **0** — closing the degenerate shortcut the bare ``in_zone``
    bonus rewarded (2026-06-27 diagnostic: 94% of deliveries were knocks). 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return 1.0 if (m.in_zone and bool(getattr(env, "_ever_grasped", False))) else 0.0


def _term_terminal_deliver(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """ONE-SHOT terminal delivery bonus: +1 exactly on the step that COMPLETES the held-in-zone dwell (the
    delivery moment, when ``_success`` reaches ``success_steps``), 0 on every other step. This is the anti-FARMING
    term: the dense per-step contact rewards (both/fingertouch/approach) form an *annuity* the policy collects by
    grasping-and-holding without delivering — and since the episode ends on success, delivering *early* forgoes
    the remaining annuity, so farming wins. A one-shot bonus paid only for *completing* delivery (not for holding)
    is what tips the optimum toward carry-to-zone. Fires once per completion (rising edge of the dwell), so it is
    an event bonus, not another annuity. 0 on a non-planar env (no ``_planar_metrics``).

    # Preconditions ``env`` exposes ``_planar_metrics`` (with ``in_zone``), ``_success`` (int), ``success_steps``.
    # Postconditions in ``{0.0, 1.0}``; 1.0 only on the exact step ``_success`` transitions to ``success_steps``."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    success_steps = int(getattr(env, "success_steps", 1))
    success = int(getattr(env, "_success", 0))   # counter BEFORE this step's update (reward is evaluated first)
    return 1.0 if (m.in_zone and success + 1 == success_steps) else 0.0


def _term_grasp_approach(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Dense approach shaping: -(mean of the two arms' fingertip→coin approach distances). Each arm's
    distance is the tip-dominant blend ``0.75·fingertip + 0.25·elbow`` (``PlanarGraspMetrics`` —
    shaping the true grasping point, not a body origin). Rises toward 0 as *both* arms close on the
    coin, giving PPO the gradient to convert 'near' into the two-finger contact the sparse
    ``both_contact`` cliff alone cannot bootstrap. 0 on a non-planar env (no ``_planar_metrics``)."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return -0.5 * (float(m.left_tip_dist) + float(m.right_tip_dist))


def _term_both_approach(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Coordination shaping: ``-max(left_tip_dist, right_tip_dist)`` — minus the *farther* arm's
    fingertip→coin distance. Unlike ``grasp_approach`` (the *mean*, where a close arm compensates for a
    far one and there is no pressure for simultaneity), this penalises the lagging arm, so the only way
    to raise it is to bring **both** fingertips close **at the same time** — the coordination gradient the
    compensable mean and the sparse ``both_contact`` cliff both lack. The two-arm ``coin_frictionloss``
    mechanism needs simultaneous force, so this targets the measured bottleneck (both_contact ≈ 0.02).
    0 on a non-planar env (no ``_planar_metrics``)."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return -max(float(m.left_tip_dist), float(m.right_tip_dist))


# ── Potential-based reward shaping (Ng, Harada & Russell 1999) ─────────────────
# F(s,s') = γ·Φ(s') − Φ(s) leaves the optimal policy INVARIANT (telescopes to Φ(terminal)−Φ(start), path-
# independent), so it CANNOT be farmed — unlike the per-step Φ annuity (in_zone/both) that the policy oscillates
# to milk. The dense guidance (progress toward zone / grasp) is preserved; the farming incentive is removed by
# construction. The previous potential is tracked per-episode on the env (cleared in reset()); the FIRST step of
# an episode returns 0 (initialises Φ(s0)). γ must match the training discount for exact policy-invariance.
_PBRS_GAMMA = 0.99


def _pbrs_shaping(env: "ArmReachEnv", potential_now: float, attr: str) -> float:
    """The potential-based shaping ``γΦ(s') − Φ(s)`` with ``Φ(s)`` tracked on the env under ``attr``.

    # Preconditions ``attr`` is cleared to ``None`` on ``env.reset`` (so each episode re-initialises Φ(s0)).
    # Postconditions returns 0.0 on the first call of an episode; else the exact PBRS shaping term."""
    prev = getattr(env, attr, None)
    setattr(env, attr, potential_now)
    return 0.0 if prev is None else _PBRS_GAMMA * potential_now - float(prev)


def _term_zone_progress(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """PBRS progress toward the zone, potential ``Φ = −disk_to_zone``. Rewards *net* progress (unfarmable — dipping
    in/out of the zone telescopes to zero), the policy-invariant replacement for the farmable ``in_zone`` annuity.
    0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return _pbrs_shaping(env, -float(m.disk_to_zone), "_pbrs_prev_zone")


def _term_grasp_progress(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """PBRS progress toward the grasp, potential ``Φ = −max(left_tip, right_tip)`` (both fingertips close at once).
    Unfarmable coordination gradient replacing the farmable per-step ``both_contact`` annuity. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return _pbrs_shaping(env, -max(float(m.left_tip_dist), float(m.right_tip_dist)), "_pbrs_prev_grasp")


def _term_conjunctive_progress(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """NONLINEAR **conjunctive** PBRS potential ``Φ = −max(disk_to_zone, left_tip, right_tip)`` — dominated by the
    WORST of {coin far from zone, either fingertip far from the coin}. Raising Φ forces reducing the *worst* one,
    so the policy must improve the grasp AND the carry TOGETHER; there is no credit for advancing one while
    neglecting another (the coupling the linear ``zone_progress + grasp_progress`` split lets the policy dodge by
    one-finger pushing — measured both_contact≈0). ``max`` is nonlinear, so this is a genuine conjunction, yet it
    is still a *potential over state* → ``γΦ(s')−Φ(s)`` telescopes → farm-proof (Ng-Harada-Russell). 0 on a
    non-planar env.

    # Postconditions the shaping is 0 on the first step of an episode; else ``γΦ(s')−Φ(s)`` with the max potential."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    phi = -max(float(m.disk_to_zone), float(m.left_tip_dist), float(m.right_tip_dist))
    return _pbrs_shaping(env, phi, "_pbrs_prev_conj")


def _term_settle(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Overshoot brake: penalise the coin's speed **only once it is inside the zone**, so the policy
    slows it to a stop there instead of pushing it straight through (the measured ep4 overshoot).
    The gate is ``dist < zone_half`` — braking during *approach* was measured to cause the opposite
    failure (undershoot, the coin stalling at the zone boundary), so the coin is left free to move
    until it is actually in the zone. ``dist`` is the disk→zone distance. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    zone_half = float(getattr(env, "_zone_half", 0.055))
    return -float(m.disk_speed) if dist < zone_half else 0.0


def _term_coin_pregrasp_still(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Gentle pre-grasp handling: ``-‖coin velocity‖`` until BOTH fingers have grasped the coin, then 0.

    Discourages flinging/knocking the coin across the table before a controlled grasp, while leaving a
    *slow* corral cheap (low speed → small penalty) and the post-grasp pull-to-zone phase entirely free.
    Speed- (not displacement-) based on purpose: the Galambos task *requires* corralling an out-of-band
    coin, so a displacement penalty would punish the task itself; a speed penalty only bites on ballistic
    knocks. Distinct from ``settle`` (zone-gated brake) and the removed ``action_cost`` (which penalised
    the ARM and froze it). 0 on a non-planar env or once ``env._ever_grasped``."""
    m = getattr(env, "_planar_metrics", None)
    if m is None or getattr(env, "_ever_grasped", True):
        return 0.0
    return -float(m.disk_speed)


# v_min: below this total arm joint speed (rad/s, summed over joints) the arm counts as stationary.
_ARM_STALL_VMIN = 1.0


def _term_arm_motion(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Anti-stall: penalise an idle arm so the policy keeps exploring instead of freezing (the
    failure that the removed ``action_cost`` term actively rewarded). 0 once the arm moves at
    ``_ARM_STALL_VMIN``; down to ``-_ARM_STALL_VMIN`` when fully frozen. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return -max(0.0, _ARM_STALL_VMIN - float(m.arm_speed))


def _term_arm_collision(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise the two arms colliding with each other (left-arm geom touching right-arm geom):
    -1 while they are in mutual contact, 0 otherwise. Keeps the fingers from crashing together
    instead of cooperating. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return -1.0 if m.arm_self_contact else 0.0


def _term_arm_body_collision(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise the two arms crashing into each other at a NON-fingertip body: ``-1`` while they are in mutual
    contact BUT the touch is not merely the two fingertips. Excludes the fingertip pair, so it does NOT fire on
    the coin-pinch (fingers close on opposite sides of the coin) — the principled 'hands collision' the 4-core
    note prescribed after whole-arm ``arm_collision`` at 2.0 killed grasping (0.615→0.0, 2026-06-28). This is the
    two-agent coordination constraint for the collaborative coin-toss: the arms must cooperate, not slam together.
    0 on a non-planar env. # Postconditions 0 during a real grasp (fingertips near the coin, arms not touching)."""
    m = getattr(env, "_planar_metrics", None)
    return -1.0 if (m is not None and m.arm_self_contact and not m.fingers_self_contact) else 0.0


def _term_finger_contact(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Reward EACH fingertip touching the coin: ``+1`` per contacting fingertip (0/1/2). Denser than
    ``both_contact`` (which pays only when BOTH touch simultaneously) — a graded pull toward the two-finger
    grasp that already rewards one fingertip on the coin before the second arrives, so the policy is not stuck
    on the all-or-nothing both-contact cliff. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    return (float(m.left_contact) + float(m.right_contact)) if m is not None else 0.0


def _term_fingers_collision(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise ONLY the two fingers crashing into each other (the fingertip-bearing distal links in mutual
    contact): -1 while they touch, else 0. Narrower than ``arm_collision`` (which fires on any left↔right body
    pair and, at weight 2.0, suppressed the whole approach — grasp-fraction 0.615→0.0, 2026-06-28). A clean
    coin-pinch keeps the fingers apart with the coin between them, so this does NOT fire during a real grasp; it
    bites only on a no-coin crash. 0 on a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    return -1.0 if getattr(m, "fingers_self_contact", False) else 0.0


def _term_arm_body_coin_contact(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise a NON-FINGERTIP arm link *pushing* the coin (v2 graded contact quality): ``-1`` per step while
    an arm-body↔coin contact is present AND **no** fingertip is on the coin — i.e. the body is moving the coin
    on its own (a shove), not merely grazing it during a fingertip grasp. This aligns the reward with the
    metric's progress-attribution grade (which credits body-only steps to ``body_progress``): a sustained
    body-only shove (an exploit) accrues a strong penalty; the clean two-fingertip grasp — where the hand
    incidentally touches the coin *while the fingertips grip* — pays nothing, so the penalty does not fight the
    only feasible grasp (cf. whole-arm ``arm_collision`` 2.0 which killed grasping, 0.615→0.0, 2026-06-28). 0
    on v1 / a non-planar env. # Postconditions ``<= 0``; identically 0 whenever contact legality is off or a
    fingertip is in contact."""
    m = getattr(env, "_planar_metrics", None)
    lg = getattr(m, "legality", None) if m is not None else None
    return -1.0 if (lg is not None and lg.arm_body_contact and not lg.fingertip_contact) else 0.0


# v2b calibration constants (2026-07-07): epsilon on body-only toward-zone speed to ignore numerical noise, and
# the negligible-body-progress threshold below which a delivery grades fingertip-dominant (matches the eval's
# _NEGLIGIBLE_BODY_EPS in exp_galambos_coord_ab).
_BODY_PROGRESS_EPS = 0.002
_CLEAN_BODY_PROGRESS = 0.005


def _term_body_progress_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise the coin being *pushed toward the zone* by a NON-fingertip arm link (v2b): the negative of the
    toward-zone coin speed during a body-ONLY contact (arm body on the coin, NO fingertip), above a small
    epsilon that ignores numerical noise. This is the PROGRESS-scaled replacement for the step-count
    ``arm_body_coin_contact`` penalty: an incidental distal-link graze that does not move the coin costs ~0,
    while a body shove that drives the coin at the target is penalised in proportion to how fast it moves it —
    the discriminating signal is body-only PROGRESS, not contact-step count (which mis-hit the scripted
    demonstrator's incidental graze, calibration 2026-07-07). 0 on v1 / when a fingertip is in contact.
    # Postconditions ``<= 0``; identically 0 when contact legality is off or any fingertip touches the coin."""
    m = getattr(env, "_planar_metrics", None)
    lg = getattr(m, "legality", None) if m is not None else None
    if lg is None or not lg.arm_body_contact or lg.fingertip_contact:
        return 0.0
    to_zone = np.array([env._zone_x - float(m.disk_pos[0]), env._zone_y - float(m.disk_pos[1])], np.float64)
    d = float(np.hypot(to_zone[0], to_zone[1]))
    if d < 1e-6:
        return 0.0
    vel_to_zone = float(np.dot(m.disk_vel, to_zone / d))       # + = coin moving toward the zone
    return -max(0.0, vel_to_zone - _BODY_PROGRESS_EPS)         # penalise ONLY positive body-driven progress


def _term_terminal_deliver_graded(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Contact-quality-gated terminal delivery bonus (v2b): like ``terminal_deliver`` (+1 on the step that
    COMPLETES the held-in-zone dwell), but SCALED by HOW the coin was delivered, read from the env's accumulated
    fingertip vs body-only progress:
      - ``+1.0`` — fingertip-dominant (body-only progress <= ``_CLEAN_BODY_PROGRESS``): the clean delivery;
      - ``+0.2`` — body-assisted (some body help, fingertips still moved most of the coin);
      - ``-0.5`` — body-driven exploit (the body moved most of the coin): a PENALTY, not a bonus.
    So a body-shove delivery cannot earn the clean delivery reward (calibration 2026-07-07). ``raw`` delivery is
    still logged as a metric; only the *reward* is gated. 0 off the completing step / a non-planar env."""
    m = getattr(env, "_planar_metrics", None)
    if m is None:
        return 0.0
    success_steps = int(getattr(env, "success_steps", 1))
    success = int(getattr(env, "_success", 0))                 # counter BEFORE this step's update
    if not (m.in_zone and success + 1 == success_steps):       # only on the delivery-completing step
        return 0.0
    bp = float(getattr(env, "_body_progress", 0.0))
    fp = float(getattr(env, "_fingertip_progress", 0.0))
    if bp <= _CLEAN_BODY_PROGRESS:
        return 1.0                                             # fingertip-dominant → full bonus
    if bp > fp:
        return -0.5                                            # body-driven exploit → penalty
    return 0.2                                                 # body-assisted → reduced bonus


def _term_out_of_bounds(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Penalise knocking the disk out of the workspace (a death): -1 on the step the disk leaves the
    table, 0 otherwise. Death only terminating left over-pushing unpunished. 0 on a non-planar env."""
    return -1.0 if getattr(env, "_disk_out", False) else 0.0


def _term_center_bonus(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Graded precision bonus: rises from 0 at the zone edge to 1 at the exact centre, so the
    policy is rewarded for *centring* the coin (the precision the sparse ``in_zone`` does not
    grade), within the zone-half tolerance. ``dist`` is the disk→zone distance. 0 on a non-planar
    env (no ``_zone_half``)."""
    zone_half = float(getattr(env, "_zone_half", 0.0))
    if zone_half <= 0.0 or dist >= zone_half:
        return 0.0
    return 1.0 - dist / zone_half


# ── generic goal-reaching + fast-and-smooth terms ────────────────────────────
def _term_goal_progress(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Dense locomotion driver: ``+(prev_dist - dist)`` — how much closer to the goal this step got. A
    scale-free per-step forward signal (telescopes to total distance closed), far stronger for learning a
    gait than the flat ``-dist``. Reads ``env._prev_dist`` (the distance at the previous step); 0 on the
    first step or if the env does not track it."""
    prev = getattr(env, "_prev_dist", None)
    return 0.0 if prev is None else float(prev - dist)


def _term_time_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Constant per-step cost: reaching the goal in fewer steps incurs less total penalty → go FAST.
    -1 every step; the .hymeko ``weight`` sets the time pressure."""
    return -1.0


def _actuated_qvel(env: Any) -> np.ndarray | None:
    dofs = getattr(env, "_act_dofs", None)
    if dofs is None or len(dofs) == 0:
        return None
    v: np.ndarray = np.asarray(env.data.qvel, dtype=np.float64)[dofs]
    return v


def _term_joint_velocity(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Smoothness: ``-Σ q̇²`` over the actuated joints — penalise thrashing the joints fast. 0 if the env
    does not expose ``_act_dofs``."""
    v = _actuated_qvel(env)
    return 0.0 if v is None else -float(v @ v)


def _term_joint_acceleration(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    """Smoothness/jerk: ``-Σ (Δq̇)²`` over the actuated joints — penalise jerky motion (large velocity
    change between steps). The **bounded** discrete-acceleration proxy: instantaneous ``qacc`` spikes to
    millions for a position servo chasing a jumped target, so the step-over-step velocity change is used
    instead. Reads the env's ``_prev_act_qvel`` (the actuated joints' velocity at the previous step); 0
    until the env has taken one step or if ``_act_dofs`` is absent."""
    v = _actuated_qvel(env)
    prev = getattr(env, "_prev_act_qvel", None)
    if v is None or prev is None:
        return 0.0
    dv = v - prev
    return -float(dv @ dv)


# ── standing / balance terms (read the torso pose; 0 on an env with no torso) ─
# These appear only in STAND_REWARD (quadruped_env). They are getattr-guarded so a spec that includes one on
# a non-quadruped env contributes 0 rather than raising — the same duck-typed contract as the pick terms.
def _term_upright(env: Any, dist: float, action: np.ndarray) -> float:
    """Torso levelness: ``cos`` between the torso local ``+z`` and world ``+z`` (1 = level, 0 = on its side,
    <0 = inverted). Reads ``env._torso_uprightness()``; 0 on an env with no torso."""
    fn = getattr(env, "_torso_uprightness", None)
    return float(fn()) if callable(fn) else 0.0


def _term_torso_height(env: Any, dist: float, action: np.ndarray) -> float:
    """Height keeping: ``-|z_torso - stand_height|`` — penalise deviating from the nominal standing height
    (crouching or collapsing). Reads ``env.torso`` / ``env.data.xpos`` and ``env._stand_height`` (the rest-pose
    torso height); 0 if the env declares no standing height."""
    h = getattr(env, "_stand_height", None)
    torso = getattr(env, "torso", None)
    if h is None or torso is None:
        return 0.0
    return -abs(float(env.data.xpos[torso, 2]) - float(h))


def _term_alive(env: Any, dist: float, action: np.ndarray) -> float:
    """Survival bonus: ``+1`` every step the episode is still alive. With fall-termination, a policy that keeps
    the torso upright accrues more total reward than one that falls early → surviving is strictly preferred.

    WARNING: unconditional — a collapsed-but-not-inverted robot still collects it while ``flip_cos`` is
    permissive, so it does NOT distinguish standing from crouching. Prefer :func:`_term_standing` (gated on the
    success predicate) for the stand task; ``alive`` is kept for envs that genuinely want raw survival."""
    return 1.0


def _term_standing(env: Any, dist: float, action: np.ndarray) -> float:
    """The success predicate AS a reward: ``+1`` iff the torso is upright (``> stand_cos``) AND at its nominal
    height (``|z − stand_height| < stand_height_tol``) — i.e. the exact ``DwellMetric('standing')`` condition the
    stand task is graded on. This makes the reward's argmax the success state, so crouching/collapsing/flopping
    (which the unconditional ``alive`` bonus rewarded) score 0 here. Reads ``env._standing`` (cached in ``step``
    just before the reward is evaluated); 0 on an env that declares no standing predicate.

    # Preconditions ``env._standing`` reflects the CURRENT post-physics pose (see the step() reorder).
    # Postconditions returns ``1.0`` exactly at the graded success condition, else ``0.0`` (reward ≡ metric)."""
    return 1.0 if getattr(env, "_standing", False) else 0.0


def _term_stand_still(env: Any, dist: float, action: np.ndarray) -> float:
    """Stay-put: ``-|v_x|`` — a standing robot should hold its ground, not drift/walk off. Reads ``env._vx``
    (the base-agnostic forward speed); 0 if absent."""
    v = getattr(env, "_vx", None)
    return -abs(float(v)) if v is not None else 0.0


# ── pick-and-place terms (read the env's PickMetrics; 0 on a non-pick env) ───
@dataclass(frozen=True)
class PickMetrics:
    """The pick-and-place state the reward terms read (cached by ``PickPlaceEnv.step``)."""
    approach: float          # ‖tool - object‖
    left: float              # left-finger contact indicator (0/1)
    right: float             # right-finger contact indicator (0/1)
    lifted: float            # object height above the surface
    lift_thresh: float       # the lift-success threshold
    to_target: float         # ‖object - target‖ (xy)
    reached: bool            # lifted AND within the place radius
    approach_contact: bool   # arm/gripper hit a surface while NOT over the object
    pre_grasp_disturb: float = 0.0   # object xy displacement before it is ever grasped (no nudging)


CLEAN_PICK = PickMetrics(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, False, 0.0)


def _pm(env: Any) -> PickMetrics:
    return getattr(env, "_pick_metrics", None) or CLEAN_PICK


def _term_pick_approach(env: Any, dist: float, action: np.ndarray) -> float:
    return -_pm(env).approach


def _term_pick_contact(env: Any, dist: float, action: np.ndarray) -> float:
    m = _pm(env)
    return m.left + m.right


def _term_pick_lift(env: Any, dist: float, action: np.ndarray) -> float:
    m = _pm(env)
    return min(m.lifted, m.lift_thresh)


def _term_pick_place_distance(env: Any, dist: float, action: np.ndarray) -> float:
    m = _pm(env)
    return -m.to_target if m.lifted > 0.02 else 0.0


def _term_pick_place_bonus(env: Any, dist: float, action: np.ndarray) -> float:
    return 1.0 if _pm(env).reached else 0.0


def _term_pick_approach_penalty(env: Any, dist: float, action: np.ndarray) -> float:
    return -1.0 if _pm(env).approach_contact else 0.0


def _term_pick_disturbance(env: Any, dist: float, action: np.ndarray) -> float:
    return -_pm(env).pre_grasp_disturb


# kind -> extractor. Defaults match meta_reward.hymeko.
_REWARD_TERMS: dict[str, RewardTerm] = {
    "pick_approach": _term_pick_approach,
    "pick_contact": _term_pick_contact,
    "pick_lift": _term_pick_lift,
    "pick_place_distance": _term_pick_place_distance,
    "pick_place_bonus": _term_pick_place_bonus,
    "pick_approach_penalty": _term_pick_approach_penalty,
    "pick_disturbance": _term_pick_disturbance,
    "goal_progress": _term_goal_progress,
    "time_penalty": _term_time_penalty,
    "joint_velocity": _term_joint_velocity,
    "joint_acceleration": _term_joint_acceleration,
    "upright": _term_upright,
    "torso_height": _term_torso_height,
    "alive": _term_alive,
    "standing": _term_standing,
    "stand_still": _term_stand_still,
    "reach_distance": _term_reach_distance,
    "success_bonus": _term_success_bonus,
    "action_cost": _term_action_cost,
    "ground_penalty": _term_ground_penalty,
    "self_collision_penalty": _term_self_collision_penalty,
    "joint_limit_penalty": _term_joint_limit_penalty,
    "below_ground_penalty": _term_below_ground_penalty,
    "both_contact": _term_both_contact,
    "in_zone": _term_in_zone,
    "grasp_deliver": _term_grasp_deliver,
    "terminal_deliver": _term_terminal_deliver,
    "grasp_approach": _term_grasp_approach,
    "both_approach": _term_both_approach,
    "zone_progress": _term_zone_progress,       # PBRS: policy-invariant progress toward the zone
    "grasp_progress": _term_grasp_progress,     # PBRS: policy-invariant progress toward the grasp
    "conjunctive_progress": _term_conjunctive_progress,   # PBRS: NONLINEAR conjunction (grasp AND carry together)
    "settle": _term_settle,
    "coin_pregrasp_still": _term_coin_pregrasp_still,
    "arm_motion": _term_arm_motion,
    "center_bonus": _term_center_bonus,
    "arm_collision": _term_arm_collision,
    "arm_body_collision": _term_arm_body_collision,
    "arm_body_coin_contact": _term_arm_body_coin_contact,   # v2 graded: penalise a non-fingertip arm link on the coin
    "body_progress_penalty": _term_body_progress_penalty,   # v2b: penalise body-only PROGRESS (not contact count)
    "terminal_deliver_graded": _term_terminal_deliver_graded,  # v2b: terminal bonus gated by contact-quality grade
    "finger_contact": _term_finger_contact,
    "fingers_collision": _term_fingers_collision,
    "out_of_bounds": _term_out_of_bounds,
}


@dataclass(frozen=True)
class RewardSpec:
    """An ordered tuple of ``(term_kind, weight)`` → the scalar reward ``Σ weight·term``.

    # Preconditions Non-empty; every kind is in :data:`_REWARD_TERMS`.
    # Postconditions ``evaluate`` returns a finite float; an all-zero-weight spec yields 0.
    """

    terms: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("RewardSpec must declare at least one term")
        unknown = [k for k, _ in self.terms if k not in _REWARD_TERMS]
        if unknown:
            raise ValueError(
                f"unknown reward term(s) {unknown}; known: {sorted(_REWARD_TERMS)}")

    def evaluate(self, env: Any, dist: float, action: np.ndarray) -> float:
        """Scalar reward from the live state: ``Σ weight · term(env, dist, action)``. ``env`` is
        duck-typed (ArmReachEnv or PlanarGraspEnv) — each term reads only what it needs."""
        return float(sum(w * _REWARD_TERMS[k](env, dist, action) for k, w in self.terms))

    @classmethod
    def from_hymeko(cls, profile_path: str | Path) -> "RewardSpec":
        """Build the spec from a ``.hymeko`` task profile's ``reward_spec`` bundle."""
        return cls(terms=read_reward_terms(profile_path))


def read_reward_terms(profile_path: str | Path) -> tuple[tuple[str, float], ...]:
    """Read a profile's ``reward_spec`` → ordered ``(term_kind, weight)`` pairs.

    The weight is the bundle arc's numeric annotation — ``@grasp_reward { (+ approach 4.0, …); }`` —
    so the hyperedge defines each term's weight, not the term node. For backward compatibility a term
    with no arc weight falls back to its body ``weight`` field, then to ``1.0``. See
    :func:`hymeko_rl.env._profile.read_bundle` for the (narrow, B-003-bridge) parse.

    # Errors ``FileNotFoundError``; ``ValueError`` (no/!1 reward_spec, undeclared member).
    """
    out: list[tuple[str, float]] = []
    for _name, kind, body, arc_weight in read_bundle(profile_path, "reward_spec"):
        if arc_weight is not None:
            weight = arc_weight
        else:
            match = re.search(r"weight\s+(-?[\d.]+)", body)
            weight = float(match.group(1)) if match else 1.0
        out.append((kind, weight))
    return tuple(out)


# The reaching task's default reward: dense negative distance to the goal (weight 1.0) —
# identical to the env's former procedural `-dist`.
REACH_REWARD = RewardSpec((("reach_distance", 1.0),))
