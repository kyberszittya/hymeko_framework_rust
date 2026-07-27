"""Pontryagin (PMP) optimal balance recovery for the underactuated humanoid.

The sagittal balance is a Linear Inverted Pendulum (LIPM) — the underactuated inverted
pendulum at the heart of the floating humanoid. Pontryagin's Maximum Principle gives the
*optimal* recovery: for state x = [com_off, com_vel], dynamics ẋ = Ax + Bf (f = horizontal
COM force), and cost J = ½∫(xᵀQx + rf²)dt, the control Hamiltonian is

    H(x, λ, f) = ½(xᵀQx + r f²) + λᵀ(Ax + Bf)

PMP's necessary conditions: canonical equations ẋ = ∂H/∂λ, λ̇ = −∂H/∂x, and optimality
∂H/∂f = 0 ⟹ f* = −r⁻¹Bᵀλ. Eliminating f gives the **Hamiltonian (symplectic) system**

    d/dt [x; λ] = [[A, −B r⁻¹Bᵀ], [−Q, −Aᵀ]] [x; λ]     (the 2n×2n Hamiltonian matrix)

whose **stable eigenspace** yields λ = P x (the Riccati P), so the optimal feedback is
f* = −r⁻¹BᵀP x. This is the PMP→Riccati connection; we build P from the Hamiltonian matrix
directly (not a Riccati solver) to expose the Pontryagin structure, and apply the optimal COM
force to the full humanoid via the COM Jacobian (τ = J_comᵀ f* + gravity + posture + damping).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hamiltonian_matrix(A: np.ndarray, B: np.ndarray, Q: np.ndarray, r: float) -> np.ndarray:
    """The PMP Hamiltonian matrix [[A, −B r⁻¹Bᵀ], [−Q, −Aᵀ]] (state ⊕ costate dynamics)."""
    return np.block([[A, -(B @ B.T) / r], [-Q, -A.T]])


def riccati_from_hamiltonian(A: np.ndarray, B: np.ndarray, Q: np.ndarray, r: float) -> np.ndarray:
    """Solve the algebraic Riccati P via the Hamiltonian matrix's STABLE eigenspace (the PMP route).

    # Postconditions returns symmetric P >= 0 with λ = P x on the optimal (stable) manifold.
    """
    n = A.shape[0]
    w, v = np.linalg.eig(hamiltonian_matrix(A, B, Q, r))
    stable = v[:, np.argsort(w.real)[:n]]                     # n eigenvectors with Re(eig) < 0
    x1, x2 = stable[:n], stable[n:]
    p = (x2 @ np.linalg.inv(x1)).real
    return 0.5 * (p + p.T)                                    # symmetrize


@dataclass
class PMPBalanceRecovery:
    """PMP-optimal LIPM balance recovery, applied to the humanoid via the COM Jacobian.

    Builds the optimal COM-force feedback f* = −r⁻¹BᵀP x from the Hamiltonian matrix, then
    ``torque(env)`` maps it to actuated-joint torques. ``optimal_force`` / ``control_hamiltonian``
    expose the Pontryagin optimum for the reduced model.
    """

    z_com: float = 0.645             # LIPM pendulum height (standing COM)
    q_pos: float = 1.0               # state cost on COM offset
    q_vel: float = 0.1               # state cost on COM velocity
    r: float = 0.02                  # control (force) cost
    kp_pose: float = 25.0
    kd: float = 8.0
    tau_max: float = 150.0
    _P: np.ndarray = field(default=None, init=False, repr=False)
    _K: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        w2 = 9.81 / self.z_com
        self._A = np.array([[0.0, 1.0], [w2, 0.0]])          # inverted-pendulum: ẍ = ω²·x (+ force)
        self._B = np.array([[0.0], [1.0]])                    # horizontal COM force enters the velocity eqn
        self._Q = np.diag([self.q_pos, self.q_vel])
        self._P = riccati_from_hamiltonian(self._A, self._B, self._Q, self.r)
        self._K = ((self._B.T @ self._P) / self.r).ravel()    # f* = −K x   (1-D gain vector)

    def optimal_force(self, com_off: float, com_vel: float) -> float:
        """PMP-optimal horizontal COM force f* = −r⁻¹BᵀP x for the reduced LIPM."""
        return float(-self._K @ np.array([com_off, com_vel]))

    def control_hamiltonian(self, com_off: float, com_vel: float) -> float:
        """H(x, λ=Px, f*) along the optimal manifold — the Pontryagin optimum value."""
        x = np.array([com_off, com_vel])
        lam = self._P @ x
        f = self.optimal_force(com_off, com_vel)
        return float(0.5 * (x @ self._Q @ x + self.r * f * f) + lam @ (self._A @ x + self._B.flatten() * f))

    def torque(self, env) -> np.ndarray:
        import mujoco
        d, m = env.data, env.model
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jacp, 1)
        jx = jacp[0]                                          # sagittal (x) COM Jacobian row
        com = d.subtree_com[1]
        support_x = 0.5 * (float(d.xpos[env._fl][0]) + float(d.xpos[env._fr][0]))
        com_off = float(com[0]) - support_x
        com_vel = float(jx @ np.asarray(d.qvel))
        fx = self.optimal_force(com_off, com_vel)            # PMP-optimal COM force
        tau_com = jx * fx                                    # map force to joint torques (Jᵀ f)
        bias = np.array([float(d.qfrc_bias[dof]) for dof in env._act_dof])
        q_a = np.array([d.qpos[a] for a in env._act_qadr])
        qd_a = np.array([d.qvel[dof] for dof in env._act_dof])
        tau = bias + np.array([tau_com[dof] for dof in env._act_dof]) - self.kp_pose * (q_a - env._q0j) - self.kd * qd_a
        return np.clip(tau, -self.tau_max, self.tau_max)
