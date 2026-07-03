"""Scripted Galambos demonstrator — to give behaviour cloning a starting policy past the two-arm grasp
hard-exploration wall (pure PPO never delivers; see reports/2026-06-24-galambos-hyperedge-ab.md).

Foundation (this file): :func:`planar_2link_ik`, analytic IK for one planar 2-link arm. The corral→pinch→pull
controller that drives both arms (built on this) is the next layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


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
        ctrl = np.zeros(int(env.n_actions), dtype=np.float32)
        for arm, tgt in ((self._arms["left"], t_l), (self._arms["right"], t_r)):
            # elbows bend outward (away from the partner arm) so the two arms don't collide mid-table.
            j1, j2 = planar_2link_ik(arm.base_xy, arm.l1, arm.l2, (float(tgt[0]), float(tgt[1])),
                                     elbow_up=(arm.side == "right"))
            ctrl[arm.a1] = j1
            ctrl[arm.a2] = j2
        return np.clip(ctrl, env.action_space.low, env.action_space.high)
