"""PMP optimal balance recovery transferred to the AIBO (embodiment-agnostic LIPM).

The Pontryagin optimum built for the humanoid (`hymeko_humanoid/scenarios/humanoid/pmp_recovery.py`)
is embodiment-agnostic: the sagittal/lateral balance of ANY legged body is a Linear Inverted
Pendulum. Here the same reduced LIPM/PMP core (Hamiltonian matrix → stable eigenspace → Riccati P →
optimal COM force f* = −r⁻¹BᵀP x) drives the 22-DOF quadruped. The optimal COM force (x AND y) is
applied through the quadruped's COM Jacobian on top of a leg PD-stand, **under the motion contract**
— a *postural* recovery (feet planted, no sprawl, no launch), unlike the retracted capture-widening.

(The LIPM/PMP math is duplicated from the humanoid module because the two live in isolated
worktrees; it is the same 2-state reduced model.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .locomotion_gait import SteeredTrotGait
from .motion_contract import JointVelocityGovernor


def riccati_from_hamiltonian(A: np.ndarray, B: np.ndarray, Q: np.ndarray, r: float) -> np.ndarray:
    """Algebraic Riccati P via the PMP Hamiltonian matrix's stable eigenspace (same as the humanoid)."""
    n = A.shape[0]
    ham = np.block([[A, -(B @ B.T) / r], [-Q, -A.T]])
    w, v = np.linalg.eig(ham)
    stable = v[:, np.argsort(w.real)[:n]]
    p = (stable[n:] @ np.linalg.inv(stable[:n])).real
    return 0.5 * (p + p.T)


@dataclass
class PMPQuadrupedRecovery:
    """PMP-optimal LIPM recovery for the AIBO: optimal COM force (x,y) via the COM Jacobian + PD-stand.

    # Preconditions env is a built QuadrupedGoalEnv. # Postconditions ``action(env)`` returns a
    normalised, motion-contract-governed action: a leg PD-stand plus the PMP-optimal COM-restoring
    force mapped to leg torques (τ = J_comᵀ f*), keeping the feet down (postural recovery).
    """

    z_com: float = 0.23              # quadruped standing COM height (lower than the humanoid -> faster LIPM)
    q_pos: float = 1.0
    q_vel: float = 0.1
    r: float = 0.02
    v_max: float = 8.0
    _K: np.ndarray = field(default=None, init=False, repr=False)
    _gait: SteeredTrotGait = field(default_factory=SteeredTrotGait, init=False, repr=False)
    _gov: JointVelocityGovernor = field(default=None, init=False, repr=False)
    _paws: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        w2 = 9.81 / self.z_com
        A = np.array([[0.0, 1.0], [w2, 0.0]])
        B = np.array([[0.0], [1.0]])
        P = riccati_from_hamiltonian(A, B, np.diag([self.q_pos, self.q_vel]), self.r)
        self._K = ((B.T @ P) / self.r).ravel()               # f* = −K [off, vel]  (per axis; x,y decouple)
        self._gov = JointVelocityGovernor(v_max=self.v_max)

    def optimal_force(self, off: float, vel: float) -> float:
        return float(-self._K @ np.array([off, vel]))

    def action(self, env) -> np.ndarray:
        import mujoco
        if not self._paws:
            self._paws = {k: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "paw_" + k)
                          for k in ("fl", "fr", "bl", "br")}
        d, m = env.data, env.model
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jacp, 1)
        jxy = jacp[0:2]                                       # COM (x,y) Jacobian
        com = d.subtree_com[1]
        support = 0.25 * sum(np.asarray(d.xpos[b][:2]) for b in self._paws.values())
        off = np.asarray(com[:2]) - support
        vel = jxy @ np.asarray(d.qvel)
        f = np.array([self.optimal_force(off[0], vel[0]), self.optimal_force(off[1], vel[1])])  # PMP-optimal (x,y)
        tau_full = jxy.T @ f                                  # optimal COM force -> joint torques
        a = self._gait.action(env, yaw_cmd=0.0, drive=0.0)   # leg PD-stand (feet planted)
        for i in range(m.nu):
            a[i] += float(tau_full[int(m.jnt_dofadr[m.actuator_trnid[i, 0]])]) / env.ctrl_range
        return self._gov.govern(env, np.clip(a, -1.0, 1.0)).astype(np.float32)
