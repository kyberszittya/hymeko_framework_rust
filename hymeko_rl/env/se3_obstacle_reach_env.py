"""6D-1 — SE(3) obstacle reach: a contact-FREE obstacle between start and goal with topologically-disjoint route modes
(left / right / over / under). The decisive falsifiable test of whether the multimodal proposal helps when the working
solutions are genuinely distant, homotopy-separated routes — isolated from contact physics, friction, teacher
instability, retention, object rotation, and mesh/inertia error (all of which the O3 triangle would mix in).

Design choice (why the obstacle is VISUAL, contype=0): we test route SELECTION, not reactive collision avoidance. The
physics never blocks the arm, so it cannot fight the servo; instead the certificate GRADES the EE trajectory — an option
whose EE path enters the obstacle AABB is a collision (fail), one that routes around is certified. The proposal + search
must FIND the routing-around option. This keeps the route modes cleanly disjoint: a left-via path and a right-via path
are different homotopy classes and local jitter cannot cross between them.

# Preconditions inherited from SE3ReachEnv. # Invariants the obstacle is seated between the start-EE and goal-EE each
reset; the certificate adds EE-path collision-freedom to the pose gate.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.env.se3_reach_env import SE3ReachEnv

_OBSTACLE = ('<geom name="hk_obstacle" type="box" size="0.028 0.028 0.075" pos="0 0 0.3" '
             'contype="0" conaffinity="0" rgba="0.85 0.35 0.1 0.55"/>')   # VISUAL marker; collision graded geometrically


class SE3ObstacleReachEnv(SE3ReachEnv):
    """SE(3) pose reach with a contact-free obstacle between start and goal, graded on EE-path collision-freedom."""

    def __init__(self, *, obstacle_half: tuple[float, float, float] = (0.028, 0.028, 0.075),
                 ee_margin: float = 0.02, min_separation: float = 0.16, mjcf: str | None = None, **kwargs: Any) -> None:
        if mjcf is None:
            from hymeko_rl.env.arm_world import make_arm_mjcf
            mjcf = make_arm_mjcf(kwargs.get("control_mode", "position"))
        if "hk_obstacle" not in mjcf:
            mjcf = mjcf.replace("<worldbody>", "<worldbody>\n    " + _OBSTACLE, 1)   # reuse the geom-injection pattern
        super().__init__(mjcf=mjcf, **kwargs)
        self.obstacle_half = np.asarray(obstacle_half, np.float32)
        self.ee_margin = float(ee_margin)
        self.min_separation = float(min_separation)
        self._obstacle_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hk_obstacle")
        if self._obstacle_gid < 0:
            raise ValueError("obstacle geom not present after injection")
        self._start_ee = np.zeros(3, np.float32)

    # -- obstacle geometry ------------------------------------------------------
    def obstacle_center(self) -> np.ndarray:
        return np.asarray(self.model.geom_pos[self._obstacle_gid], np.float32)

    def ee_in_obstacle(self, ee_pos: np.ndarray) -> bool:
        """Is an EE position inside the obstacle box inflated by the EE margin (axis-aligned)."""
        d = np.abs(np.asarray(ee_pos, np.float32) - self.obstacle_center())
        return bool(np.all(d <= self.obstacle_half + self.ee_margin))

    def _seat_obstacle(self) -> None:
        """Place the obstacle at the start↔goal midpoint so the DIRECT EE path is blocked (mode routing is then
        required). Static-geom position is a model field; mj_forward re-derives its world pose."""
        mid = 0.5 * (self._start_ee + self._target)
        self.model.geom_pos[self._obstacle_gid] = mid.astype(np.float64)
        mujoco.mj_forward(self.model, self.data)

    def direct_path_blocked(self, samples: int = 12) -> bool:
        """Does the straight EE segment start→goal pass through the obstacle (so a detour is genuinely required)."""
        return any(self.ee_in_obstacle((1 - t) * self._start_ee + t * self._target)
                   for t in np.linspace(0.0, 1.0, samples))

    def route_feasible(self, direction: np.ndarray, offset: float = 0.14, samples: int = 16) -> bool:
        """Is the two-leg EE path start→via→goal (via = midpoint + offset·direction) collision-free — i.e. does this
        route family clear the obstacle geometrically. Used to LOG per-state route feasibility (§risk)."""
        via = 0.5 * (self._start_ee + self._target) + offset * np.asarray(direction, np.float32)
        pts = ([(1 - t) * self._start_ee + t * via for t in np.linspace(0, 1, samples // 2)]
               + [(1 - t) * via + t * self._target for t in np.linspace(0, 1, samples // 2)])
        return not any(self.ee_in_obstacle(p) for p in pts)

    def _start_config(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """An INDEPENDENT reachable start config whose EE clears ``min_separation`` from the goal EE — so an obstacle
        seated between them genuinely forces a detour (unlike the 6D-0 bounded-perturbation start). Capped reject
        sampling; falls back to the farthest-separated candidate seen."""
        best_q = self.np_random.uniform(lo, hi).astype(np.float32)
        best_ee = self._fk_pose(best_q)[0]
        best_sep = float(np.linalg.norm(best_ee - self._target))
        for _ in range(48):
            if best_sep >= self.min_separation:
                return best_q
            q = self.np_random.uniform(lo, hi).astype(np.float32)
            sep = float(np.linalg.norm(self._fk_pose(q)[0] - self._target))
            if sep > best_sep:
                best_sep, best_q = sep, q
        return best_q

    # -- reset: seat the obstacle once the start config AND goal pose are both set ----------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _obs, info = super().reset(seed=seed, options=options)   # sets goal pose (_reset_target) + start config (_start_config)
        self._start_ee = self._ee_pos().copy()             # the actual start-EE (arm now at the start config)
        self._seat_obstacle()
        info["obstacle_center"] = self.obstacle_center().copy()
        info["direct_blocked"] = self.direct_path_blocked()
        return self.node_features(), info
