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
    "grasp_approach": _term_grasp_approach,
    "settle": _term_settle,
    "coin_pregrasp_still": _term_coin_pregrasp_still,
    "arm_motion": _term_arm_motion,
    "center_bonus": _term_center_bonus,
    "arm_collision": _term_arm_collision,
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
