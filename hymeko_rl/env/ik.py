"""Damped least-squares pose IK for the HyMeKo arms — a reusable solver.

Both ``arm_reach_env`` and ``pick_place_env`` carried their own inline single-step DLS-IK; the pick-place
6-DOF *pose* case drifted into degenerate poses because one Gauss-Newton step per env-frame cannot find the
narrow top-down configs (top-down tool orientation is only ~2% of config space — see
``reports/2026-06-22-pick-place-phase0.md``). This solver **iterates to convergence on a scratch
``MjData``**, so the returned joint config is a genuine pose solution, not a nudge.

Design: Strategy-free, stateless-per-solve. The live simulation ``MjData`` is never mutated.
"""
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

_DOWN = np.array([0.0, 0.0, -1.0])


class DampedPoseIK:
    """Iterative DLS IK to a tool-body Cartesian target (position, optionally tool +z → world −z).

    ``Δq = gain·Jᵀ(JJᵀ + λ²I)⁻¹ e`` per iteration, ``e`` = position error (and, when ``down``, the
    orientation error ``½·(ẑ_tool × −ẑ_world)``), integrated with joint-limit clamping on a scratch
    ``MjData`` until the position tolerance is met or ``iters`` is reached.

    # Preconditions
    - ``n_joints`` actuated DOFs are the **first** ``n_joints`` columns of the system Jacobian
      (true for the emitted arms: the arm joints precede grip/free joints in the kinematic tree).
    - ``ee_body`` is a valid body id.
    # Postconditions
    - Returns a length-``n_joints`` ``float32`` config within ``jnt_range`` for limited joints.
    - The ``MjModel`` and any live ``MjData`` are not mutated (an internal scratch ``MjData`` is used).
    """

    def __init__(self, model: mujoco.MjModel, ee_body: int, n_joints: int, *,
                 damping: float = 0.05, gain: float = 0.5) -> None:
        if not 0 < n_joints <= model.nv:
            raise ValueError(f"n_joints must be in 1..{model.nv}; got {n_joints}")
        if not 0 <= int(ee_body) < model.nbody:
            raise ValueError(f"ee_body {ee_body} out of range")
        self.model = model
        self.ee = int(ee_body)
        self.n = int(n_joints)
        self.damping = float(damping)
        self.gain = float(gain)
        self._scratch = mujoco.MjData(model)
        lo = np.full(self.n, -np.pi, dtype=np.float64)
        hi = np.full(self.n, np.pi, dtype=np.float64)
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j])
            if adr < self.n and bool(model.jnt_limited[j]):
                lo[adr], hi[adr] = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        self._lo, self._hi = lo, hi

    def solve(self, q_seed: np.ndarray, target_pos: np.ndarray, *, down: bool = True,
              iters: int = 80, pos_tol: float = 2e-3, rot_tol: float = 5e-2) -> np.ndarray:
        """Solve from ``q_seed`` (warm start) toward ``target_pos``. Returns the arm joint config.

        # Preconditions ``len(q_seed) >= n_joints``; ``target_pos`` is length-3.
        # Postconditions joint-limit-clamped config; convergence is best-effort (returns last iterate)."""
        d = self._scratch
        q = np.array(q_seed[: self.n], dtype=np.float64)
        target = np.asarray(target_pos, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        rerr = np.zeros(3)
        for _ in range(iters):
            d.qpos[:] = 0.0
            d.qpos[: self.n] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)            # populates cdof for mj_jac
            ee = np.array(d.xpos[self.ee])
            perr = target - ee
            mujoco.mj_jac(self.model, d, jacp, jacr, ee, self.ee)
            if down:
                zaxis = d.xmat[self.ee].reshape(3, 3)[:, 2]
                rerr = np.cross(zaxis, _DOWN)
                jac = np.vstack([jacp[:, : self.n], jacr[:, : self.n]])
                err = np.concatenate([perr, 0.5 * rerr])
            else:
                jac = jacp[:, : self.n]
                err = perr
            dq = self.gain * jac.T @ np.linalg.solve(
                jac @ jac.T + self.damping ** 2 * np.eye(jac.shape[0]), err)
            q = np.clip(q + dq, self._lo, self._hi)
            if float(np.linalg.norm(perr)) < pos_tol and (not down or float(np.linalg.norm(rerr)) < rot_tol):
                break
        return q.astype(np.float32)

    def solve_collision_free(self, q_seed: np.ndarray, target_pos: np.ndarray,
                             is_valid: Callable[[mujoco.MjData], bool], *, down: bool = True,
                             n_starts: int = 16, iters: int = 80, accept_pos: float = 5e-2) -> np.ndarray:
        """Multi-start DLS IK returning the first converged, **collision-free** config.

        A single DLS solve can converge to a self-colliding branch (the arm folded onto itself to point the
        tool down — physically blocked). This tries the warm seed plus a deterministic set of random seeds,
        forward-evaluates each solution on the scratch ``MjData`` (so contacts are populated), and accepts the
        first that reaches ``target_pos`` within ``accept_pos`` **and** passes ``is_valid`` (the caller's
        collision predicate). Deterministic: the random seeds use a fixed ``RandomState`` so the expert stays a
        single-valued function of state (clean BC demos).

        # Preconditions ``is_valid`` inspects a position-forwarded ``MjData`` and must not mutate it.
        # Postconditions returns a joint-limit-clamped config; falls back to the last iterate if none is clean.
        """
        rs = np.random.RandomState(0)
        starts = [np.asarray(q_seed[: self.n], dtype=np.float64)]
        starts += [rs.uniform(self._lo, self._hi) for _ in range(max(0, n_starts - 1))]
        d = self._scratch
        target = np.asarray(target_pos, dtype=np.float64)
        best, best_perr = starts[0].astype(np.float32), float("inf")   # closest-by-error, NOT a wild last iterate
        for s in starts:
            q = self.solve(s, target_pos, down=down, iters=iters)
            d.qpos[:] = 0.0
            d.qvel[:] = 0.0
            d.qpos[: self.n] = q
            mujoco.mj_forward(self.model, d)
            perr = float(np.linalg.norm(target - np.asarray(d.xpos[self.ee])))
            if perr < accept_pos and is_valid(d):
                return q
            if perr < best_perr:
                best, best_perr = q, perr
        return best
