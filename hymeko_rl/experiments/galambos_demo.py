"""Scripted Galambos demonstrator — to give behaviour cloning a starting policy past the two-arm grasp
hard-exploration wall (pure PPO never delivers; see reports/2026-06-24-galambos-hyperedge-ab.md).

Foundation (this file): :func:`planar_2link_ik`, analytic IK for one planar 2-link arm. The corral→pinch→pull
controller that drives both arms (built on this) is the next layer.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import mujoco
import numpy as np

from hymeko_rl.control.controller_spec import ControllerSpec


def planar_2link_ik(base_xy: tuple[float, float], l1: float, l2: float,
                    target_xy: tuple[float, float], *, elbow_up: bool = True) -> tuple[float, float]:
    """Joint targets ``(j1, j2)`` putting a planar 2-link arm's fingertip at ``target_xy``.

    Convention matches :func:`hymeko_rl.env.planar_grasp_env.make_planar_arms_mjcf`: both joints are ``+Z``
    hinges in the table plane and at ``(j1, j2) = (0, 0)`` the arm points ``+Y`` (each capsule ``fromto`` runs
    ``0 → +Y``). Forward kinematics of the tip relative to the base is therefore
    ``(-l1 sin j1 - l2 sin(j1+j2),  l1 cos j1 + l2 cos(j1+j2))``; this inverts it (links point at standard
    angle ``joint + π/2``, hence the ``-π/2`` offset on the shoulder).

    ``elbow_up`` selects which of the two elbow solutions (both reach the same tip).

    # Preconditions ``l1, l2 > 0``.
    # Postconditions the returned ``(j1, j2)`` place the tip at ``target_xy`` when it is within
      ``[|l1-l2|, l1+l2]`` of the base; an out-of-reach target is clamped to the nearest reachable point.
    # Errors ``ValueError`` if ``l1 <= 0`` or ``l2 <= 0``.
    """
    if l1 <= 0.0 or l2 <= 0.0:
        raise ValueError(f"link lengths must be positive; got l1={l1}, l2={l2}")
    dx = target_xy[0] - base_xy[0]
    dy = target_xy[1] - base_xy[1]
    r = math.hypot(dx, dy)
    r = min(max(r, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)        # clamp to the reachable annulus
    cos_q2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    q2 = math.acos(max(-1.0, min(1.0, cos_q2)))
    if not elbow_up:
        q2 = -q2
    # `r` may have been clamped; aim the shoulder along the (clamped) reach using the unit target direction.
    phi = math.atan2(dy, dx)
    q1_std = phi - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1_std - math.pi / 2.0, q2


@dataclass(frozen=True)
class _ArmKin:
    """Per-arm kinematics + action wiring extracted from the model (robust across robot=None / from_hymeko)."""

    side: str
    base_xy: tuple[float, float]
    l1: float
    l2: float
    tip_site: int
    a1: int           # action index of the shoulder actuator
    a2: int           # action index of the elbow actuator


def _extract_arms(model: Any) -> dict[str, _ArmKin]:
    """Find the two planar arms (base→link1→link2→tip) and their action indices from a stepped model.

    Relies only on the body-name ``_left``/``_right`` suffix (the same convention ``PlanarGraspEnv`` uses) and
    the body tree, so it works for the hand-authored and the emitted robot alike.

    # Postconditions returns ``{"left": _ArmKin, "right": _ArmKin}``.
    # Errors ``ValueError`` if either arm cannot be resolved.
    """
    act_of_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    arms: dict[str, _ArmKin] = {}
    for base in range(model.nbody):
        if int(model.body_parentid[base]) != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, base) or ""
        side = "left" if name.endswith("_left") else "right" if name.endswith("_right") else None
        if side is None:
            continue
        children = [b for b in range(model.nbody) if int(model.body_parentid[b]) == base]
        if not children:
            continue
        link1 = children[0]
        grandkids = [b for b in range(model.nbody) if int(model.body_parentid[b]) == link1]
        if not grandkids:
            continue
        link2 = grandkids[0]
        j_sh = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == link1]
        j_el = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == link2]
        if not j_sh or not j_el:
            continue
        tips = [s for s in range(model.nsite) if int(model.site_bodyid[s]) == link2]
        tip = tips[0] if tips else -1
        l1 = float(model.body_pos[link2][1])
        l2 = float(model.site_pos[tip][1]) if tip >= 0 else l1
        base_xy = (float(model.body_pos[base][0]), float(model.body_pos[base][1]))
        arms[side] = _ArmKin(side, base_xy, l1, l2, tip, act_of_joint[int(j_sh[0])], act_of_joint[int(j_el[0])])
    if "left" not in arms or "right" not in arms:
        raise ValueError(f"could not resolve both arms; found {sorted(arms)}")
    return arms


class GalambosDemonstrator:
    """Scripted corral→pinch→pull controller for the two planar arms — a demonstrator for behaviour cloning.

    Phases (per episode): **corral** (open the fingertips on either x-side of the coin), **pinch** (squeeze in
    until both fingers contact the coin), **pull** (translate the pinch toward the zone, dragging the coin). Each
    phase sets an (x, y) fingertip target per arm; :func:`planar_2link_ik` turns it into shoulder/elbow targets
    for the position servos. Reverts pull→pinch if contact is lost.

    # Preconditions ``env`` is a stepped :class:`~hymeko_rl.env.planar_grasp_env.PlanarGraspEnv`.
    """

    def __init__(self, env: Any, *, d_open: float | None = None, pull_step: float = 0.006,
                 corral_tol: float = 0.05) -> None:
        self._arms = _extract_arms(env.model)
        # PINCH-then-CARRY: each arm on its own x-side (the natural geometry — bases at ±x). Corral wide, pinch
        # to contact, then clamp HARD (gap well inside the coin → high normal force → finger-coin friction beats
        # table friction) and carry slowly to the zone. A loose grip friction-slips when dragged perpendicular.
        coin_r = float(getattr(env._env, "disk_radius", 0.02))
        self.d_touch = coin_r + 0.005                # pinch-phase gap: just reach contact
        self.d_clamp = max(0.004, coin_r - 0.010)    # carry-phase gap: hard clamp (penetrate for high friction)
        self.d_open = (coin_r + 0.065) if d_open is None else d_open
        self.pull_step = pull_step
        self.corral_tol = corral_tol
        self._phase = "corral"
        self._carry = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self._phase = "corral"

    def _tip_xy(self, env: Any, arm: _ArmKin) -> np.ndarray:
        return np.asarray(env.data.site_xpos[arm.tip_site][:2], dtype=np.float64)

    def _targets(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        """Fingertip (x, y) targets for the left and right arms under the current phase (and advance phase)."""
        coin = np.asarray(env._planar_metrics.disk_pos[:2], dtype=np.float64)
        zone = np.array([env._zone_x, env._zone_y], dtype=np.float64)
        contact = bool(env._planar_metrics.left_contact and env._planar_metrics.right_contact)
        ox = np.array([1.0, 0.0])
        if self._phase == "corral":
            t_l, t_r = coin - self.d_open * ox, coin + self.d_open * ox
            near_l = float(np.linalg.norm(self._tip_xy(env, self._arms["left"]) - t_l)) < self.corral_tol
            near_r = float(np.linalg.norm(self._tip_xy(env, self._arms["right"]) - t_r)) < self.corral_tol
            if near_l and near_r:
                self._phase = "pinch"
        elif self._phase == "pinch":
            t_l, t_r = coin - self.d_touch * ox, coin + self.d_touch * ox
            if contact:
                self._phase, self._carry = "carry", coin.copy()
        else:                                                   # carry: hard clamp + creep toward the zone
            if not contact:                                     # lost it — re-pinch where the coin is now
                self._phase = "pinch"
                t_l, t_r = coin - self.d_touch * ox, coin + self.d_touch * ox
            else:
                delta = zone - self._carry
                dist = float(np.linalg.norm(delta))
                if dist > 1e-9:
                    self._carry = self._carry + delta / dist * min(self.pull_step, dist)
                t_l, t_r = self._carry - self.d_clamp * ox, self._carry + self.d_clamp * ox
        return t_l, t_r

    def action(self, env: Any) -> np.ndarray:
        """The 4-dim position-servo command (shoulder/elbow targets for both arms) for the current state."""
        t_l, t_r = self._targets(env)
        return _ik_action(((self._arms["left"], t_l), (self._arms["right"], t_r)), env)


def _ik_action(pairs: "Iterable[tuple[_ArmKin, np.ndarray]]", env: Any) -> np.ndarray:
    """Map per-arm fingertip targets to the position-servo command via per-arm IK.

    Shared by every scripted controller (pinch-carry and push controller). Elbows bend outward (away from the
    partner arm) so the arms don't collide mid-table.
    """
    ctrl = np.zeros(int(env.n_actions), dtype=np.float32)
    for arm, tgt in pairs:
        j1, j2 = planar_2link_ik(arm.base_xy, arm.l1, arm.l2, (float(tgt[0]), float(tgt[1])),
                                 elbow_up=(arm.side == "right"))
        ctrl[arm.a1] = j1
        ctrl[arm.a2] = j2
    return np.clip(ctrl, env.action_space.low, env.action_space.high)


# ── Push (V-plow) demonstrator — fingertip-only physics (plan: docs/plans/2026-07-05-galambos-… ) ──────────


@dataclass(frozen=True)
class PushConfig:
    """Geometry/gait knobs of the V-plow. All lengths in metres, angles in radians.

    # Invariants ``press_max < contact_dist`` (targets never reach the coin centre); ``v_half_angle`` in
      ``(0, π/2)`` (slots stay behind the coin); ``brake_dist > 0``.
    """

    contact_dist: float          # coin_r + tip_r: tip-centre distance at which tip touches the coin
    v_half_angle: float = 0.70   # ±40° V-slots around the push ray's back point
    press_max: float = 0.012     # plow press depth (servo error = force); measured: 0.012 → 0.80 vs 0.006 → 0.54
    swing_margin: float = 0.030  # swing: orbit radius clearance above contact_dist
    max_arc: float = 0.50        # swing: max angular step of the orbit per control step
    front_avoid: float = 1.05    # swing: never orbit within ±60° of the push direction (don't bulldoze)
    slot_tol: float = 0.020      # swing→plow when both tips this close to their slots
    brake_dist: float = 0.060    # plow→brake inside this coin→zone distance; press scales ∝ distance

    @classmethod
    def from_params(cls, params: dict[str, float], *, contact_dist: float) -> "PushConfig":
        """Build from a declarative parameter dict (a controller profile's merged scalar fields);
        ``contact_dist`` stays scene-derived (coin_r + tip_r), never declared.

        # Errors ``ValueError`` on a parameter the config does not define (a .hymeko typo must fail loud).
        """
        known = {f.name for f in fields(cls)} - {"contact_dist"}
        unknown = sorted(params.keys() - known)
        if unknown:
            raise ValueError(f"unknown controller param(s) {unknown}; expected a subset of {sorted(known)}")
        return cls(contact_dist=contact_dist, **params)


def _ang(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def _on_circle(center: np.ndarray, radius: float, theta: float) -> np.ndarray:
    return center + radius * np.array([math.cos(theta), math.sin(theta)])


def fan_offsets(k: int, half_angle: float) -> np.ndarray:
    """The ``k`` fan angles around the back direction — the discrete rotation-group orbit that generates the
    slots (``k=2`` → the classic V at ``±half_angle``; ``k=1`` → a single pusher dead behind).

    # Preconditions ``k >= 1``. # Postconditions shape ``(k,)``, symmetric about 0, descending.
    """
    assert k >= 1
    return np.zeros(1) if k == 1 else np.linspace(half_angle, -half_angle, k)


def push_slots(coin: np.ndarray, zone: np.ndarray, press: float, cfg: PushConfig,
                   k: int = 2) -> "np.ndarray | None":
    """The ``k`` slot fingertip targets behind ``coin`` w.r.t. ``zone``, pressed ``press`` inside contact:
    slot ``i`` is the back direction rotated by :func:`fan_offsets` ``[i]`` (rotation-group indexing — no
    left/right naming).

    Returns shape ``(k, 2)`` (row ``i`` = slot ``i``, ccw-most first), or ``None`` when ``coin ≈ zone`` (no
    push ray; caller holds position).

    # Preconditions ``0 <= press < cfg.contact_dist``; ``k >= 1``.  # Postconditions every slot is strictly
      behind the coin: ``dot(slot - coin, zone - coin) < 0`` (holds while ``k·fan spacing`` stays within
      ``±v_half_angle < π/2``).
    """
    assert 0.0 <= press < cfg.contact_dist
    ray = zone - coin
    dist = float(np.linalg.norm(ray))
    if dist < 1e-6:
        return None
    theta_back = _ang(-ray / dist)
    r_slot = cfg.contact_dist - press
    return np.stack([_on_circle(coin, r_slot, theta_back + off)
                     for off in fan_offsets(k, cfg.v_half_angle)])


def orbit_step(tip: np.ndarray, coin: np.ndarray, slot: np.ndarray, theta_front: float,
               cfg: PushConfig) -> np.ndarray:
    """One closed-loop swing waypoint: orbit the tip around the coin toward ``slot``, never crossing the
    ``±front_avoid`` sector around ``theta_front`` (so the swing cannot bulldoze the coin off the ray); once
    angularly aligned with the slot, descend radially onto it.

    # Postconditions the returned waypoint is either the ``slot`` itself (aligned within ``align_arc``) or a
      point at radius ``contact_dist + swing_margin`` from ``coin``.
    """
    two_pi = 2.0 * math.pi
    align_arc = 0.20                            # within ~11° of the slot: stop orbiting, descend onto it
    theta_slot = _ang(slot - coin)
    theta_tip = _ang(tip - coin)
    span_ccw = (theta_slot - theta_tip) % two_pi
    span_cw = two_pi - span_ccw
    if min(span_ccw, span_cw) < align_arc:
        return slot

    def crosses_front(span: float, ccw: bool) -> bool:
        off = ((theta_front - theta_tip) % two_pi) if ccw else ((theta_tip - theta_front) % two_pi)
        return off < span + cfg.front_avoid or off > two_pi - cfg.front_avoid

    ccw = not crosses_front(span_ccw, True) if crosses_front(span_cw, False) != crosses_front(span_ccw, True) \
        else span_ccw <= span_cw
    span = span_ccw if ccw else span_cw
    step = min(cfg.max_arc, span) * (1.0 if ccw else -1.0)
    return _on_circle(coin, cfg.contact_dist + cfg.swing_margin, theta_tip + step)


@dataclass(frozen=True)
class PushObs:
    """Data-oriented controller observation — every env read happens ONCE here, per step.

    Downstream (guards, target laws) is pure and env-free, hence unit-testable without MuJoCo. Arms are
    INDEXED (``tips[i]`` = arm ``i``, base-x order), never named — the k-arm generalisation.

    # Invariants ``tips`` has shape ``(k, 2)``; ``u`` is the unit coin→zone ray, or ``None`` iff
      ``dist < 1e-6``.
    """

    coin: np.ndarray
    zone: np.ndarray
    tips: np.ndarray
    dist: float
    u: "np.ndarray | None"

    @classmethod
    def from_env(cls, env: Any, arms: "tuple[_ArmKin, ...]") -> "PushObs":
        coin = np.asarray(env._planar_metrics.disk_pos[:2], dtype=np.float64)
        zone = np.array([env._zone_x, env._zone_y], dtype=np.float64)
        tips = np.stack([np.asarray(env.data.site_xpos[a.tip_site][:2], dtype=np.float64) for a in arms])
        dist = float(np.linalg.norm(zone - coin))
        return cls(coin=coin, zone=zone, tips=tips, dist=dist,
                   u=(zone - coin) / dist if dist >= 1e-6 else None)


@dataclass(frozen=True)
class PushEvent:
    """One FSM transition (observability: probes read ``demo.events`` instead of re-deriving phases).
    ``margin`` is the firing guard's robustness ρ (> 0 by construction — how decisively the jump fired)."""

    step: int
    src: str
    dst: str
    reason: str
    margin: float = float("nan")


def press_depth(dist: float, cfg: PushConfig) -> float:
    """Continuous brake law: full press beyond ``brake_dist``, scaling linearly to 0 at the zone centre."""
    return cfg.press_max * min(1.0, dist / cfg.brake_dist)


def assign_slots(tips: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Arm→slot permutation minimising total tip travel (brute force over ``k!``).

    # Preconditions ``tips``/``slots`` are ``(k, 2)`` with ``k <= 6``. # Postconditions a permutation of
      ``range(k)``: arm ``i`` takes slot ``assign[i]``.
    """
    k = len(tips)
    assert k == len(slots) and k <= 6
    best = min(itertools.permutations(range(k)),
               key=lambda p: sum(float(np.linalg.norm(tips[i] - slots[p[i]])) for i in range(k)))
    return np.asarray(best, dtype=np.intp)


# ── Named guard/law behaviours bound by the .hymeko FSM (immutable dispatch tables, not mutable state) ────────
# Guards are STL-ROBUSTNESS monitors (the hymeko_monitor semantics): each returns a margin ρ, satisfied iff
# ρ > 0; conjunction = min, disjunction = max of the per-arm atom margins. The FSM fires on ρ > 0 — identical
# verdicts to the former booleans — and the margin itself is data (how close to a jump; shaping/curriculum).
# A law maps (obs, slots, assign) → (k, 2) targets.

_ZONE_EPS = 1e-6                              # the coin≈zone degeneracy threshold (matches PushObs.u)


def _guard_slots_reached(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                         cfg: PushConfig) -> float:
    # Gate tolerance includes press_max: the servo hovers with a steady-state error of ~the press depth,
    # so a bare slot_tol never fires (measured 2026-07-05: 33/33 failures stuck in SWING, 0.80 → 0.38).
    if slots is None or assign is None:
        return -math.inf
    return min((cfg.slot_tol + cfg.press_max) - float(np.linalg.norm(obs.tips[i] - slots[assign[i]]))
               for i in range(len(obs.tips)))


def _guard_formation_broken(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                    cfg: PushConfig) -> float:
    if obs.u is None:
        return -math.inf
    return max(float(np.dot(obs.tips[i] - obs.coin, obs.u)) - 0.5 * cfg.contact_dist
               for i in range(len(obs.tips)))


def _guard_coin_at_zone(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                        cfg: PushConfig) -> float:
    return _ZONE_EPS - obs.dist


def _guard_coin_left_zone(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                          cfg: PushConfig) -> float:
    return obs.dist - _ZONE_EPS


def _law_approach_orbit(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                     cfg: PushConfig) -> np.ndarray:
    if slots is None or assign is None or obs.u is None:
        return obs.tips
    theta_front = _ang(obs.u)
    return np.stack([orbit_step(obs.tips[i], obs.coin, slots[assign[i]], theta_front, cfg)
                     for i in range(len(obs.tips))])


def _law_pressed_slots(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                       cfg: PushConfig) -> np.ndarray:
    if slots is None or assign is None:
        return obs.tips
    return np.asarray(slots[assign])


def _law_freeze(obs: PushObs, slots: "np.ndarray | None", assign: "np.ndarray | None",
                cfg: PushConfig) -> np.ndarray:
    return obs.tips


_BehaviourFn = Callable[[PushObs, "np.ndarray | None", "np.ndarray | None", PushConfig], Any]
GUARDS: dict[str, _BehaviourFn] = {
    "slots_reached": _guard_slots_reached, "formation_broken": _guard_formation_broken,
    "coin_at_zone": _guard_coin_at_zone, "coin_left_zone": _guard_coin_left_zone,
}
LAWS: dict[str, _BehaviourFn] = {
    "approach_orbit": _law_approach_orbit, "pressed_slots": _law_pressed_slots, "freeze": _law_freeze,
}

#: The declarative controller profile the push controller runs by default (repo-root-relative, like the task
#: profiles). The FSM topology and the gait/geometry parameters live THERE, not in this module.
PUSH_PROFILE = "data/robotics/galambos_push.hymeko"


class PushDemonstrator:
    """Closed-loop push controller: k fingertips form a slot fan behind the coin along the coin→zone ray
    (:func:`push_slots` — rotation-group indexing) and plow it toward the zone; heading is re-aimed
    from the *current* coin every step; press (and thus speed) scales down inside ``brake_dist`` so the
    coin settles in the zone and dwell accrues.

    Replaces the pinch-carry :class:`GalambosDemonstrator` under fingertip-only physics, where a two-point
    hard clamp on a free cylinder is unstable (measured drop-loop failure, 68–80% of episodes, 2026-07-05
    probe). Same ``reset()`` / ``action(env)`` contract — a drop-in BC teacher.

    **The FSM is declared in** ``data/robotics/galambos_push.hymeko`` (phases, transitions, parameters)
    and walked via :class:`~hymeko_rl.control.controller_spec.ControllerSpec`; this class only binds the
    named guards/laws (:data:`GUARDS` / :data:`LAWS`) — the substrate discipline (reward-in-hymeko, now
    controller-in-hymeko). Step pipeline (dataflow): env → :class:`PushObs` → declared transitions
    (first firing guard wins) → slot re-assignment on every phase change → the phase's target law →
    :func:`_ik_action`. Transitions are recorded on :attr:`events`.

    # Preconditions ``env`` is a stepped :class:`~hymeko_rl.env.planar_grasp_env.PlanarGraspEnv`; the
      spec's laws/events all resolve in :data:`LAWS` / :data:`GUARDS`.
    # Invariants whenever slots exist, ``self._assign`` is a valid permutation before any target law runs.
    """

    def __init__(self, env: Any, *, cfg: PushConfig | None = None,
                 spec: ControllerSpec | None = None, profile: "str | Path" = PUSH_PROFILE,
                 laws: "dict[str, _BehaviourFn] | None" = None) -> None:
        self._arms = tuple(sorted(_extract_arms(env.model).values(), key=lambda a: a.base_xy[0]))
        self.spec = spec if spec is not None else ControllerSpec.from_hymeko(profile)
        self._laws = {**LAWS, **(laws or {})}   # strategy injection: e.g. a LEARNED per-mode law (hybrid RL)
        unbound = ({p.law for p in self.spec.phases} - self._laws.keys(),
                   {e for p in self.spec.phases for e, _ in p.transitions} - GUARDS.keys())
        if any(unbound):
            raise ValueError(f"controller profile binds unknown law(s)/event(s): {unbound}")
        coin_r = float(getattr(env._env, "disk_radius", 0.035))
        tip_r = 0.014                          # fingertip sphere size (make_planar_arms_mjcf / with_fingertip_sites)
        self.cfg = cfg if cfg is not None else PushConfig.from_params(self.spec.params,
                                                                          contact_dist=coin_r + tip_r)
        self._phase = self.spec.initial
        self._assign: "np.ndarray | None" = None
        self._step = 0
        self.events: list[PushEvent] = []
        self.last_margins: dict[str, float] = {}   # this step's guard robustness (monitor observability)
        self.last_targets: np.ndarray = np.zeros((len(self._arms), 2))

    def reset(self) -> None:
        self._phase = self.spec.initial
        self._assign = None
        self._step = 0
        self.events = []
        self.last_margins = {}
        self.last_targets = np.zeros((len(self._arms), 2))

    def _decide(self, obs: PushObs) -> np.ndarray:
        """obs → ``(k, 2)`` fingertip targets, advancing the declared FSM.

        Slots are the PRESSED ones throughout (swing descends already pressing, the gate measures against
        them): the unpressed variant regressed delivery 0.80 → 0.38 (2026-07-05 sweep) because the descent
        stopped short of the coin and the gate starved."""
        slots = (push_slots(obs.coin, obs.zone, press_depth(obs.dist, self.cfg), self.cfg,
                                k=len(self._arms)) if obs.u is not None else None)
        if slots is not None and self._assign is None:
            self._assign = assign_slots(obs.tips, slots)
        margins: dict[str, float] = {}         # evaluated guard robustness this step (monitor data)

        def fires(e: str) -> bool:
            margins[e] = float(GUARDS[e](obs, slots, self._assign, self.cfg))
            return margins[e] > 0.0

        nxt, event = self.spec.step(self._phase, fires)
        self.last_margins = margins
        if nxt != self._phase:
            self.events.append(PushEvent(self._step, self._phase, nxt, event, margin=margins[event]))
            if slots is not None:              # re-form the formation: fresh permutation BEFORE the target law
                self._assign = assign_slots(obs.tips, slots)
            self._phase = nxt
        law = self._laws[self.spec.phase(self._phase).law]
        return np.asarray(law(obs, slots, self._assign, self.cfg))

    def action(self, env: Any) -> np.ndarray:
        """The position-servo command for the current state (same contract as the pinch-carry class).
        The decided fingertip targets are exposed as :attr:`last_targets` (per-mode imitation data)."""
        targets = self._decide(PushObs.from_env(env, self._arms))
        self.last_targets = targets
        self._step += 1
        return _ik_action(zip(self._arms, targets), env)


class ArbitratedPushDemonstrator(PushDemonstrator):
    """:class:`PushDemonstrator` + a central safety arbiter over the JOINT state (coin, zone, both fingertips,
    both contacts, **coin velocity**). After the FSM proposes the candidate fingertip-target pair, the arbiter
    vetoes it only when the joint state says the plow works AGAINST delivery, substituting a safe fallback:

      * coin **receding** from the zone while pressing (``disk_vel·u < -v_eps``) → back the press off to
        ``backoff × press`` (stop adding energy in the wrong direction; the per-step re-aim recovers heading);
      * coin already advancing **fast near the zone** (``dist < brake_dist`` and ``disk_vel·u > v_cap``) → hold
        the formation at zero press (let the coin decelerate and settle instead of overshooting and rolling out
        — the dominant dwell failure).

    Otherwise the baseline action passes through unchanged, so the arbiter can only intervene on
    counter-productive steps — a no-degrade shield, not a new controller. Optimized parameters only (``v_cap``,
    ``v_eps``, ``backoff``): no raw-action learning, no new architecture. Same ``reset()``/``action(env)``
    contract, so it is a drop-in for the baseline everywhere the push controller is used.

    # Preconditions ``env.privileged`` metrics expose ``disk_vel`` (:class:`PlanarGraspMetrics`).
    # Postconditions the returned target pair equals the baseline's whenever no veto condition holds.
    """

    def __init__(self, env: Any, *, v_cap: float = 0.10, v_eps: float = 0.01, backoff: float = 0.35,
                 **kw: Any) -> None:
        super().__init__(env, **kw)
        self.v_cap = float(v_cap)
        self.v_eps = float(v_eps)
        self.backoff = float(backoff)

    def _arbitrate(self, env: Any, obs: PushObs,
                   targets: np.ndarray) -> "tuple[np.ndarray, str]":
        """Central evaluator: veto ``targets`` from the joint state; return ``(safe_targets, veto_reason)``
        (``veto_reason == ""`` means the baseline target passed through)."""
        pressing = self.spec.phase(self._phase).law == "pressed_slots"
        if not pressing or obs.u is None or self._assign is None:
            return targets, ""                                   # shield only the pressing phases
        v_toward = float(np.dot(np.asarray(env._planar_metrics.disk_vel, dtype=np.float64), obs.u))
        if obs.dist < self.cfg.brake_dist and v_toward > self.v_cap:      # overshoot risk near the zone → hold
            hold = push_slots(obs.coin, obs.zone, 0.0, self.cfg, k=len(self._arms))
            return (np.asarray(hold[self._assign]) if hold is not None else targets), "hold_overshoot"
        if v_toward < -self.v_eps:                                        # receding → back the press off
            back = push_slots(obs.coin, obs.zone, press_depth(obs.dist, self.cfg) * self.backoff,
                              self.cfg, k=len(self._arms))
            return (np.asarray(back[self._assign]) if back is not None else targets), "backoff_recede"
        return targets, ""

    def action(self, env: Any) -> np.ndarray:
        obs = PushObs.from_env(env, self._arms)
        targets = self._decide(obs)                              # advance the FSM, get the candidate pair
        targets, _veto = self._arbitrate(env, obs, targets)      # central safety arbiter over the joint state
        self.last_targets = targets
        self._step += 1
        return _ik_action(zip(self._arms, targets), env)


@dataclass(frozen=True)
class PushControllerParams:
    """High-level tunables for the phase push controller.

    This is the only learning-facing interface for the guarded controller. A neural component may predict these
    scalars, but it must not emit raw joint actions.
    """

    contact_offset: float = 0.0          # metres, added to contact_dist before slot generation
    push_gain: float = 1.0               # multiplier on press depth
    direction_correction: float = 0.0    # radians, rotates the push ray used for slot placement
    brake_threshold: float = 0.060       # metres, PUSH -> BRAKE threshold
    release_threshold: float = 0.010     # metres, BRAKE -> DONE threshold

    @classmethod
    def from_vector(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> "PushControllerParams":
        """Build bounded controller params from a 5-vector.

        # Errors ``ValueError`` if the vector is not length 5. # Postconditions every value is clipped to the
        conservative envelope used by :class:`PhasePushController`.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (5,):
            raise ValueError(f"controller parameter vector must have shape (5,), got {arr.shape}")
        return cls(
            contact_offset=float(np.clip(arr[0], -0.010, 0.010)),
            push_gain=float(np.clip(arr[1], 0.50, 1.50)),
            direction_correction=float(np.clip(arr[2], -0.35, 0.35)),
            brake_threshold=float(np.clip(arr[3], 0.030, 0.120)),
            release_threshold=float(np.clip(arr[4], 0.004, 0.025)),
        )


def _rot2(v: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


class PhasePushController:
    """Five-phase guarded push controller: APPROACH, CONTACT, PUSH, BRAKE, DONE.

    The controller converts high-level parameters into fingertip targets and then IK joint commands. The safety
    arbiter compares the parameterized candidate against the safe scripted candidate and falls back whenever the
    candidate violates target-directed pushing constraints. This is a controller-param learning interface, not a
    raw-action policy interface.
    """

    PHASES = ("APPROACH", "CONTACT", "PUSH", "BRAKE", "DONE")

    def __init__(self, env: Any, *, params: PushControllerParams | None = None,
                 safe_params: PushControllerParams | None = None,
                 v_cap: float = 0.10, v_eps: float = 0.01) -> None:
        self._arms = tuple(sorted(_extract_arms(env.model).values(), key=lambda a: a.base_xy[0]))
        coin_r = float(getattr(env._env, "disk_radius", 0.035))
        tip_r = 0.014
        base = PushConfig.from_params(ControllerSpec.from_hymeko(PUSH_PROFILE).params,
                                      contact_dist=coin_r + tip_r)
        self.base_cfg = base
        self.params = params if params is not None else PushControllerParams()
        self.safe_params = safe_params if safe_params is not None else PushControllerParams()
        self.v_cap = float(v_cap)
        self.v_eps = float(v_eps)
        self.phase = "APPROACH"
        self._assign: "np.ndarray | None" = None
        self._step = 0
        self.last_targets = np.zeros((len(self._arms), 2))
        self.last_safe_targets = np.zeros((len(self._arms), 2))
        self.last_veto = ""

    def reset(self) -> None:
        self.phase = "APPROACH"
        self._assign = None
        self._step = 0
        self.last_targets = np.zeros((len(self._arms), 2))
        self.last_safe_targets = np.zeros((len(self._arms), 2))
        self.last_veto = ""

    def set_params(self, params: PushControllerParams) -> None:
        self.params = params

    def set_param_vector(self, values: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self.params = PushControllerParams.from_vector(values)

    def _cfg(self, params: PushControllerParams) -> PushConfig:
        contact = max(0.010, self.base_cfg.contact_dist + params.contact_offset)
        return replace(self.base_cfg, contact_dist=contact, brake_dist=params.brake_threshold)

    def _corrected_obs(self, obs: PushObs, params: PushControllerParams) -> PushObs:
        if obs.u is None or abs(params.direction_correction) < 1e-12:
            return obs
        u = _rot2(obs.u, params.direction_correction)
        zone = obs.coin + u * max(obs.dist, 1e-6)
        return PushObs(coin=obs.coin, zone=zone, tips=obs.tips, dist=obs.dist, u=u)

    def _slots(self, obs: PushObs, params: PushControllerParams, phase: str) -> "np.ndarray | None":
        cfg = self._cfg(params)
        if obs.u is None:
            return None
        if phase == "APPROACH":
            press = 0.0
        elif phase == "CONTACT":
            press = min(0.35 * cfg.press_max, cfg.contact_dist - 1e-6)
        elif phase == "PUSH":
            press = press_depth(obs.dist, cfg) * params.push_gain
        elif phase == "BRAKE":
            scale = min(1.0, obs.dist / max(params.brake_threshold, 1e-6))
            press = cfg.press_max * params.push_gain * scale
        else:
            press = 0.0
        press = float(np.clip(press, 0.0, cfg.contact_dist - 1e-6))
        return push_slots(obs.coin, obs.zone, press, cfg, k=len(self._arms))

    def _targets_for(self, obs0: PushObs, params: PushControllerParams, phase: str) -> np.ndarray:
        obs = self._corrected_obs(obs0, params)
        cfg = self._cfg(params)
        slots = self._slots(obs, params, phase)
        if slots is not None and self._assign is None:
            self._assign = assign_slots(obs.tips, slots)
        if phase == "APPROACH":
            if slots is None:
                return obs.tips
            theta_front = _ang(obs.u) if obs.u is not None else 0.0
            assign = self._assign if self._assign is not None else assign_slots(obs.tips, slots)
            return np.stack([orbit_step(obs.tips[i], obs.coin, slots[assign[i]], theta_front, cfg)
                             for i in range(len(obs.tips))])
        if phase in ("CONTACT", "PUSH", "BRAKE"):
            if slots is None:
                return obs.tips
            assign = self._assign if self._assign is not None else assign_slots(obs.tips, slots)
            return np.asarray(slots[assign])
        return obs.tips

    def _advance_phase(self, env: Any, obs: PushObs) -> None:
        if obs.dist <= self.params.release_threshold or bool(getattr(env._planar_metrics, "in_zone", False)):
            self.phase = "DONE"
            return
        if self.phase == "DONE":
            if obs.dist > self.params.release_threshold:
                self.phase = "APPROACH"
            return
        slots = self._slots(obs, self.safe_params, "CONTACT")
        if slots is not None and self._assign is None:
            self._assign = assign_slots(obs.tips, slots)
        assign = self._assign
        reached = slots is not None and assign is not None and \
            all(float(np.linalg.norm(obs.tips[i] - slots[assign[i]]))
                < self.base_cfg.slot_tol + self.base_cfg.press_max for i in range(len(obs.tips)))
        contact = bool(env._planar_metrics.left_contact or env._planar_metrics.right_contact)
        both_contact = bool(env._planar_metrics.left_contact and env._planar_metrics.right_contact)
        if self.phase == "APPROACH" and reached:
            self.phase = "CONTACT"
        elif self.phase == "CONTACT" and (both_contact or contact):
            self.phase = "PUSH"
        elif self.phase == "PUSH" and obs.dist <= self.params.brake_threshold:
            self.phase = "BRAKE"
        elif self.phase in ("CONTACT", "PUSH", "BRAKE") and obs.u is not None:
            broken = max(float(np.dot(obs.tips[i] - obs.coin, obs.u)) - 0.5 * self.base_cfg.contact_dist
                         for i in range(len(obs.tips)))
            if broken > 0.0:
                self.phase = "APPROACH"
                self._assign = None

    def _behind_true_ray(self, obs: PushObs, targets: np.ndarray) -> bool:
        if obs.u is None:
            return True
        return all(float(np.dot(targets[i] - obs.coin, obs.u)) < 0.0 for i in range(len(targets)))

    def _arbitrate(self, env: Any, obs: PushObs, candidate: np.ndarray, safe: np.ndarray) -> tuple[np.ndarray, str]:
        if self.phase not in ("PUSH", "BRAKE"):
            return candidate, ""
        if not self._behind_true_ray(obs, candidate):
            return safe, "fallback_not_behind_coin"
        if obs.u is None:
            return safe, "fallback_no_push_ray"
        v_toward = float(np.dot(np.asarray(env._planar_metrics.disk_vel, dtype=np.float64), obs.u))
        if self.phase == "BRAKE" and v_toward > self.v_cap:
            return safe, "fallback_overshoot"
        if v_toward < -self.v_eps:
            return safe, "fallback_recede"
        return candidate, ""

    def action(self, env: Any) -> np.ndarray:
        obs = PushObs.from_env(env, self._arms)
        self._advance_phase(env, obs)
        safe = self._targets_for(obs, self.safe_params, self.phase)
        candidate = self._targets_for(obs, self.params, self.phase)
        targets, reason = self._arbitrate(env, obs, candidate, safe)
        self.last_safe_targets = safe
        self.last_targets = targets
        self.last_veto = reason
        self._step += 1
        return _ik_action(zip(self._arms, targets), env)


@dataclass(frozen=True)
class DeliveryFloorCriterion:
    """Acceptance rule for learned controller parameters: never accept a degrading controller."""

    scripted_delivery: float = 0.84
    tolerance: float = 0.03

    @property
    def floor(self) -> float:
        return round(self.scripted_delivery - self.tolerance, 10)

    def accepts(self, delivery: float, *, worst_seed_delivery: float | None = None) -> bool:
        if delivery < self.floor:
            return False
        return worst_seed_delivery is None or worst_seed_delivery >= self.floor
