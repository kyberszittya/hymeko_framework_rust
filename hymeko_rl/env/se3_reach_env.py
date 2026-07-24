"""6D-0 — SE(3) pose reach: the structural pose-reach variant of :class:`ArmReachEnv` (position **and** orientation).

Position-only reach vs pose reach is a STRUCTURAL difference (a different observation layout, a different reward, a
different success certificate), so this is a subclass — class-per-structural-variant (CLAUDE.md §6.5 #8) — not a runtime
flag on the base env. It reuses the base kinematics, safety, hypergraph and control space unchanged, overriding only the
three Template-Method hooks (`_reset_target`, `_reached`, `_extra_reward` / `_extra_step_info`) plus the SE(3) readers.

Reachability: the 4-DOF arm cannot reach arbitrary SO(3). Targets are therefore sampled as the forward kinematics of a
random joint config, so the full pose (position AND orientation) is reachable **by construction** — the same trick the
base env already uses for the position target, extended to the orientation.

# Preconditions inherited from ArmReachEnv (frame_skip ≥ 1, …); ``ang_thresh > 0``.
# Invariants observation is ``(n_vertices, 18)`` float32 (POSE_OBSERVATION); the certificate is
``pos_err < reach_thresh ∧ ang_err < ang_thresh``.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.env.observation import POSE_OBSERVATION, ObservationSpec


class SE3ReachEnv(ArmReachEnv):
    """End-effector POSE reaching (position + orientation) on the articulated arm."""

    def __init__(self, *, ang_thresh: float = 0.30, orient_weight: float = 0.3,
                 start_perturb: float = 0.30, obs_spec: ObservationSpec | None = None, **kwargs: Any) -> None:
        super().__init__(obs_spec=obs_spec if obs_spec is not None else POSE_OBSERVATION, **kwargs)
        if ang_thresh <= 0:
            raise ValueError("ang_thresh must be > 0")
        self.ang_thresh = float(ang_thresh)
        self.orient_weight = float(orient_weight)
        self.start_perturb = float(start_perturb)            # start = target config ± this (a closable 6-D pose error)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._target_q = np.zeros(self.model.nq, dtype=np.float32)

    # -- SE(3) readers (consumed by the POSE_OBSERVATION channels + the certificate) ----------
    def _ee_quat(self) -> np.ndarray:
        """EE body world orientation quaternion (w,x,y,z)."""
        return np.asarray(self.data.xquat[self._ee], dtype=np.float32)

    def target_quat(self) -> np.ndarray:
        return self._target_quat.copy()

    def orientation_error(self) -> np.ndarray:
        """Rotation vector taking the EE orientation to the target (``mju_subQuat(target, ee)``) — the actable
        orientation error, the orientation analogue of ``target − ee``. # Postconditions ``‖·‖`` is the geodesic
        angle in ``[0, π]`` (rotation-vector magnitude)."""
        res = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(res, self._target_quat.astype(np.float64), self._ee_quat().astype(np.float64))
        return res.astype(np.float32)

    def ee_angular_velocity(self) -> np.ndarray:
        """EE angular velocity (the rotational part of the spatial twist) = Jᵣ(q)·q̇."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self._ee_pos(), self._ee)
        n = self.n_actions
        return (jacr[:, :n] @ self.data.qvel[:n]).astype(np.float32)

    def ang_err(self) -> float:
        """Scalar geodesic orientation error (rad)."""
        return float(np.linalg.norm(self.orientation_error()))

    # -- FK-reachable pose target -------------------------------------------------------------
    def _fk_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(EE position, EE quaternion) at joint config ``q`` — no side effects (state restored)."""
        saved = self.data.qpos.copy()
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        pos, quat = self._ee_pos(), self._ee_quat()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return pos, quat

    def _sample_pose_target(self, lo: np.ndarray, hi: np.ndarray, *, attempts: int = 64,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A reachable POSE: the FK of a random joint config whose EE clears the robot column (``reach_min_radius``).
        Capped reject sampling (no infinite loop); falls back to the farthest candidate seen. Returns the achieving
        config too, so the start can be seated a bounded step from it.

        # Postconditions returns (pos(3), quat(4), q(nq)); the pose is reachable (``q`` achieves it exactly)."""
        best_q = self.np_random.uniform(lo, hi).astype(np.float32)
        best_p, best_quat = self._fk_pose(best_q)
        best_r = float(np.hypot(best_p[0], best_p[1]))
        if best_r >= self.reach_min_radius:
            return best_p, best_quat, best_q
        for _ in range(attempts - 1):
            q = self.np_random.uniform(lo, hi).astype(np.float32)
            p, quat = self._fk_pose(q)
            r = float(np.hypot(p[0], p[1]))
            if r >= self.reach_min_radius:
                return p, quat, q
            if r > best_r:
                best_r, best_p, best_quat, best_q = r, p, quat, q
        return best_p, best_quat, best_q

    # -- Template-Method overrides (position reach → pose reach) ------------------------------
    def _reset_target(self, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
        pos, quat, q = self._sample_pose_target(lo, hi)
        self._target = pos.copy()
        self._target_quat = quat.copy()
        self._target_q = q.copy()
        return {"target": self._target.copy(), "target_quat": self._target_quat.copy()}

    def _start_config(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Seat the start a bounded joint step from the target config (clipped to range) so the 6-D pose error is
        non-trivial but closable by the DLS-IK expert (a genuine pose reach, not an extreme reorientation)."""
        noise = self.np_random.uniform(-self.start_perturb, self.start_perturb, size=self._target_q.shape)
        return np.clip(self._target_q + noise.astype(np.float32), lo, hi)

    def _reached(self, dist: float) -> bool:
        """Pose certificate: EE within ``reach_thresh`` (position) AND within ``ang_thresh`` (orientation)."""
        return bool(dist < self.reach_thresh and self.ang_err() < self.ang_thresh)

    def _extra_reward(self, dist: float, ctrl: np.ndarray) -> float:
        """A weighted dense orientation-error penalty on top of the position reward (both are ‘closer = higher’)."""
        return float(-self.orient_weight * self.ang_err())

    def _extra_step_info(self) -> dict[str, Any]:
        return {"ang_err": self.ang_err()}

    # -- 6-DoF expert (position + orientation IK) --------------------------------------------
    @property
    def expert_action(self) -> np.ndarray:
        """Closed-loop demonstrator driving BOTH position and orientation: a damped-least-squares step on the stacked
        6-D error ``[target−ee ; rotvec(ee→target)]`` through the full geometric Jacobian ``[Jₚ ; Jᵣ]``. The desired
        joint step is realised in the env's control space by the base class's control-space mapping (torque / position /
        velocity). With 4 DoF the 6-D system is over-determined; DLS gives the least-squares joint step, which reaches
        the FK-sampled (hence reachable) pose."""
        n = self.n_actions
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self._ee_pos(), self._ee)
        jac = np.vstack([jacp[:, :n], jacr[:, :n]])                       # (6, n)
        err = np.concatenate([self._target - self._ee_pos(), self.orientation_error()]).astype(np.float64)  # (6,)
        dq = self.expert_gain * jac.T @ np.linalg.solve(jac @ jac.T + 0.05 ** 2 * np.eye(6), err)
        q, qd = self.data.qpos[:n], self.data.qvel[:n]
        q_des = q + dq
        if self.control_mode == "torque":
            a_des = (self._exp_kp * (q_des - q) - self._exp_kv * qd).astype(np.float64)
            ma = np.zeros(self.model.nv)
            mujoco.mj_mulM(self.model, self.data, ma, a_des)
            a = ma[:n] + self.data.qfrc_bias[:n]
        elif self.control_mode == "velocity":
            a = self._exp_vgain * dq
        else:  # position
            a = q_des
        return np.asarray(np.clip(a, self._ctrl_lo, self._ctrl_hi), dtype=np.float32)
