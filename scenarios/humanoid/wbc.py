"""Whole-Body Controller — contact-consistent task-space inverse dynamics for the humanoid.

The balance-residual action space (bounded joint-target offset around a fixed standing pose) cannot walk:
it keeps both feet planted, and the lateral CoM is rigidly pinned by the double-support closed chain (a
measured wall — `reports/2026-07-29-humanoid-walking-feasibility.md`). The principled action-space change
is a **whole-body controller**: solve, every tick, for the joint accelerations + contact wrenches that best
track a set of prioritised task accelerations (CoM, a body's Cartesian pose, posture) subject to the
floating-base dynamics and the stance-foot contact constraint, then recover the actuated torques by inverse
dynamics. A cost on a contact's wrench lets the controller **unload** a foot — the load-transfer primitive
walking is built from.

Formulation (no QP dependency — an equality-constrained least-squares solved via the KKT system):

    variables  x = [qacc (nv); lambda (6 per stance contact)]
    minimise   sum_i w_i || J_i qacc - a_i_des ||^2  + sum_c wf_c ||lambda_c||^2      (soft tasks + force cost)
    subject to  M[base] qacc - Jc[base]^T lambda = -h[base]      (unactuated floating-base dynamics)
                Jc qacc = 0                                       (stance foot does not accelerate)
    then        tau = M[act] qacc + h[act] - Jc[act]^T lambda     (actuated inverse dynamics)

`M` (`mj_fullM`), `h` (`qfrc_bias`), and the Jacobians come from MuJoCo. This is standard operational-space
/ whole-body control (Sentis/Righetti lineage) reduced to equalities (friction cones + torque limits are
handled by clipping, adequate on flat ground). Validated: stable standing and single-foot load transfer
(the double-support wall the quasi-static controllers could not pass). A full walking gait built on top
(`WalkGait`) is a research prototype — it takes several steps but is not yet a certified stable limit
cycle; see the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Task:
    """One weighted task-space acceleration objective ``w · ‖J q̈ − a_des‖²``.

    # Preconditions ``jac`` is ``(k, nv)``, ``acc_des`` is ``(k,)``, ``weight ≥ 0``.
    """

    jac: np.ndarray
    acc_des: np.ndarray
    weight: float


class WholeBodyController:
    """Contact-consistent task-space inverse dynamics for a MuJoCo floating-base model.

    # Preconditions the model has a floating base joint named ``base_joint`` (6 DOF) and ``act_dof`` lists
    the actuated velocity-DOF indices. # Postconditions ``solve`` returns actuated torques ``(nu,)`` that
    realise the weighted task accelerations under the given stance contacts, or the least-squares closest
    feasible torques. # Invariants stateless across calls (reads live ``data`` each tick).
    """

    def __init__(self, model, data, act_dof: "list[int]", base_joint: str = "base") -> None:
        self.m, self.d = model, data
        self.nv = int(model.nv)
        self.act_dof = list(act_dof)
        bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_joint)
        adr = int(model.jnt_dofadr[bj])
        self.base_dof = list(range(adr, adr + 6))

    # ---- Jacobian / task helpers ----
    def com_jacobian(self) -> np.ndarray:
        j = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, j, 1)
        return j

    def body_jacobian(self, body: int) -> "tuple[np.ndarray, np.ndarray]":
        jp = np.zeros((3, self.nv))
        jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.m, self.d, jp, jr, body)
        return jp, jr

    @staticmethod
    def orientation_error(r_cur: np.ndarray, r_des: np.ndarray) -> np.ndarray:
        """Small-angle orientation error ``½ vee(R_des R_curᵀ − …)`` (world-frame angular residual)."""
        re = r_des @ r_cur.T
        return 0.5 * np.array([re[2, 1] - re[1, 2], re[0, 2] - re[2, 0], re[1, 0] - re[0, 1]])

    def posture_task(self, q0j: np.ndarray, act_q: "list[int]", kp: float, kd: float,
                     weight: float) -> Task:
        """A weak joint-space posture regulator toward ``q0j`` (fills the task null space)."""
        jac = np.zeros((len(self.act_dof), self.nv))
        for i, dof in enumerate(self.act_dof):
            jac[i, dof] = 1.0
        qj = np.array([self.d.qpos[a] for a in act_q])
        qdj = np.array([self.d.qvel[dof] for dof in self.act_dof])
        return Task(jac, kp * (q0j - qj) - kd * qdj, weight)

    # ---- the QP-free KKT solve ----
    def solve(self, contacts: "list[int]", tasks: "list[Task]",
              force_cost: "tuple[slice, float] | None" = None, reg: float = 1e-4) -> np.ndarray:
        """Return actuated torques ``(nu,)``. ``contacts`` = stance-foot body ids (6D rigid contact each);
        ``force_cost`` = ``(slice into λ, weight)`` to penalise (unload) those contact wrenches.

        # Preconditions every ``Task.jac`` has ``nv`` columns. # Postconditions torques satisfy the
        floating-base dynamics with the contact wrenches the KKT system returns.
        """
        m, d, nv = self.m, self.d, self.nv
        mass = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, mass)
        h = np.asarray(d.qfrc_bias)
        jc = (np.vstack([np.vstack(self.body_jacobian(b)) for b in contacts])
              if contacts else np.zeros((0, nv)))
        ncl = jc.shape[0]
        nx = nv + ncl
        hmat = np.zeros((nx, nx))
        grad = np.zeros(nx)
        for t in tasks:
            hmat[:nv, :nv] += t.weight * (t.jac.T @ t.jac)
            grad[:nv] += t.weight * (t.jac.T @ t.acc_des)
        hmat[:nv, :nv] += reg * np.eye(nv)
        if ncl:
            hmat[nv:, nv:] += reg * np.eye(ncl)
        if force_cost is not None:
            sl, wf = force_cost
            idx = np.arange(nv + sl.start, nv + sl.stop)
            hmat[np.ix_(idx, idx)] += wf * np.eye(len(idx))
        # hard equality constraints: base dynamics (unactuated) + contact no-acceleration
        cb = np.zeros((6, nx))
        cb[:, :nv] = mass[self.base_dof]
        if ncl:
            cb[:, nv:] = -jc.T[self.base_dof]
        cc = np.zeros((ncl, nx))
        cc[:, :nv] = jc
        cmat = np.vstack([cb, cc])
        dvec = np.concatenate([-h[self.base_dof], np.zeros(ncl)])
        nc = cmat.shape[0]
        kkt = np.zeros((nx + nc, nx + nc))
        kkt[:nx, :nx] = hmat
        kkt[:nx, nx:] = cmat.T
        kkt[nx:, :nx] = cmat
        sol = np.linalg.solve(kkt + 1e-9 * np.eye(nx + nc), np.concatenate([grad, dvec]))
        qacc = sol[:nv]
        lam = sol[nv:nv + ncl]
        tau = (mass[np.ix_(self.act_dof, range(nv))] @ qacc + h[self.act_dof]
               - (jc.T[self.act_dof] @ lam if ncl else 0.0))
        return tau
