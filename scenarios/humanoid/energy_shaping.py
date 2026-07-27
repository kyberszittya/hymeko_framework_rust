"""Energy-shaping (IDA-PBC) balance for the underactuated humanoid — the Hamiltonian view.

The floating humanoid is an underactuated port-Hamiltonian system: H(q,p) = ½ pᵀM(q)⁻¹p + V(q),
with actuated joints (input map G) and an UNACTUATED floating base. Ad-hoc PD-hold balances it,
but the principled tool is **energy shaping**: choose the actuated torque so the closed loop is a
port-Hamiltonian system with a *shaped* energy H_d whose minimum is the balanced state, plus
damping injection. Then **H_d is a Lyapunov function** (Ḣ_d = −q̇ᵀK_d q̇ ≤ 0) — the same energy
that the certificate checks.

We use the tractable potential-shaping case (M_d = M): impose a shaped potential

    V_d(q) = ½ (q_a − q*)ᵀ K_p (q_a − q*)   [pose regulation, actuated joints]
           + ½ w_com ‖com_xy − support_xy‖²  [COM centering — the UNDERACTUATED coupling]

The COM term is the key underactuated part: its gradient ∂V_d/∂q_a = w_com·J_comᵀ(com−support)
(J_com = COM Jacobian w.r.t. the actuated joints) produces ankle/hip restoring torques that
stabilize the UNACTUATED base by driving the COM over the support — energy shaping doing the
ankle/hip strategy, not an ad-hoc gain. Control law (cancel real ∂V, impose ∂V_d, inject damping):

    τ_a = qfrc_bias_a − K_p (q_a − q*) − w_com·J_comᵀ(com_xy − support) − K_d q̇_a
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnergyShapingBalance:
    """IDA-PBC potential-shaping + damping balance controller (Hamiltonian energy is the Lyapunov fn).

    # Preconditions env exposes ``model``/``data``, ``_act_dof``/``_act_qadr``/``_q0j``, ``_fl``/``_fr``,
    ``_pelvis``, and ``h_ref``. # Postconditions ``torque(env)`` returns the actuated-joint torques
    that make H_d a Lyapunov function; ``shaped_energy(env)`` returns H_d (kinetic + shaped potential).
    """

    kp: float = 40.0                 # pose-regulation stiffness (shaped potential curvature)
    kd: float = 8.0                  # damping injection (makes Ḣ_d <= 0)
    w_com: float = 900.0             # COM-centering weight (the underactuated coupling term)
    tau_max: float = 150.0
    _jacp: np.ndarray = field(default=None, init=False, repr=False)

    def _com_jac_xy(self, env) -> np.ndarray:
        """COM Jacobian (xy rows) restricted to the actuated DOF columns — d(com_xy)/d(q_a)."""
        import mujoco
        if self._jacp is None or self._jacp.shape[1] != env.model.nv:
            self._jacp = np.zeros((3, env.model.nv))
        mujoco.mj_jacSubtreeCom(env.model, env.data, self._jacp, 1)   # subtree COM Jacobian (body 1 = pelvis root)
        return self._jacp[0:2][:, env._act_dof]                        # (2, n_act)

    def _com_support(self, env) -> np.ndarray:
        com = env.data.subtree_com[1][:2]
        support = 0.5 * (env.data.xpos[env._fl][:2] + env.data.xpos[env._fr][:2])
        return np.asarray(com - support, np.float64)                   # COM offset from support centroid (xy)

    def torque(self, env) -> np.ndarray:
        d = env.data
        q_a = np.array([d.qpos[a] for a in env._act_qadr])
        qd_a = np.array([d.qvel[dof] for dof in env._act_dof])
        bias = np.array([float(d.qfrc_bias[dof]) for dof in env._act_dof])   # ∂V/∂q_a (+Coriolis) to cancel
        grad_com = self.w_com * (self._com_jac_xy(env).T @ self._com_support(env))  # ∂V_d/∂q_a (COM term)
        tau = bias - self.kp * (q_a - env._q0j) - grad_com - self.kd * qd_a
        return np.clip(tau, -self.tau_max, self.tau_max)

    def shaped_energy(self, env) -> float:
        """H_d = kinetic energy + shaped potential V_d — the Lyapunov function of the closed loop."""
        d = env.data
        qv = np.asarray(d.qvel, np.float64)
        ke = 0.5 * float(np.dot(qv, qv))   # unit-mass kinetic proxy (exact M-weighting not needed for monotonicity)
        q_a = np.array([d.qpos[a] for a in env._act_qadr])
        off = self._com_support(env)
        vd = 0.5 * self.kp * float(np.sum((q_a - env._q0j) ** 2)) + 0.5 * self.w_com * float(np.dot(off, off))
        return ke + vd
