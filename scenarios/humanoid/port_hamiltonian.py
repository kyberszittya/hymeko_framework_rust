r"""Port-Hamiltonian (pH) description of the humanoid dynamics — the complete generate-and-control model.

This is the form the HyMeKo language is heading toward: a hypergraph model whose nodes carry pH roles
(energy_storage / dissipation / interconnection / ports) generates a Hamiltonian system that is then
controlled by passivity-based / energy-shaping methods. Here we DERIVE that pH system from the MuJoCo
humanoid and VERIFY its defining property numerically.

A rigid-body robot is an input-state-output port-Hamiltonian system on ``x = (q, p)`` with momentum
``p = M(q) q̇``:

    ⎡q̇⎤   ⎛⎡ 0   I⎤   ⎡0  0 ⎤⎞ ⎡∂H/∂q⎤   ⎡0⎤
    ⎢ ⎥ = ⎜⎢      ⎥ − ⎢     ⎥⎟ ⎢     ⎥ + ⎢ ⎥ τ ,     H(q,p) = ½ pᵀ M(q)⁻¹ p + V(q)
    ⎣ṗ⎦   ⎝⎣−I   0⎦   ⎣0  D ⎦⎠ ⎣∂H/∂p⎦   ⎣S⎦

- ``H`` (the **energy-storage**) = kinetic ½ q̇ᵀM q̇ plus gravitational potential ``V`` — the total energy;
- ``J = [[0, I], [−I, 0]]`` (the **interconnection**) is skew-symmetric → energy-conserving, symplectic;
- ``R = diag(0, D)`` (the **dissipation**) with ``D = Dᵀ ≥ 0`` (joint damping) → removes energy;
- ``g = [0; S]`` with collocated output ``y = Sᵀ q̇`` (the **ports**) → where actuation injects power.

Defining property (passivity), which this module checks along a MuJoCo trajectory:

    dH/dt = −q̇ᵀ D q̇ + yᵀ u ,   u = τ ,   y = Sᵀ q̇      (energy in only through the ports, out only via D)

The contact wrenches enter as an additional port ``Jcᵀ λ`` (energy exchanged with the ground); the CENTROIDAL
reduction (``centroidal_hamiltonian``) is the momentum-level pH — the same energy in the CoM linear + angular
momentum that ``centroidal_run`` optimises, with the contact forces as its ports.

# Preconditions: a MuJoCo model + data at a valid state. # Postconditions: ``energy_balance`` residual → 0
#   as the timestep → 0 for a conservative (τ, D, contacts consistent) trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PHStructure:
    """The port-Hamiltonian matrices at a state — the 'complete dynamical system description'."""

    H: float                         # total energy (Hamiltonian) = kinetic + potential
    M: np.ndarray                    # (nv, nv) mass matrix — the metric of the kinetic energy storage
    dHdq: np.ndarray                 # ∂H/∂q (generalised force from the energy: Coriolis-like + gravity)
    dHdp: np.ndarray                 # ∂H/∂p = q̇ (the velocity)
    D: np.ndarray                    # (nv, nv) dissipation (joint damping), symmetric PSD
    S: np.ndarray                    # (nv, nu) actuation/port map (selects actuated dofs)
    kinetic: float
    potential: float


class HumanoidPortHamiltonian:
    """Derive + verify the port-Hamiltonian form of the MuJoCo humanoid dynamics."""

    def __init__(self, model, data, mj) -> None:
        self._mj, self.model, self.data = mj, model, data
        self.nv = int(model.nv)
        self.nu = int(model.nu)
        self._act_dof = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(self.nu)]
        self.S = np.zeros((self.nv, self.nu))                # actuated-dof selector = the input ports
        for i, dof in enumerate(self._act_dof):
            self.S[dof, i] = float(model.actuator_gear[i, 0]) if model.actuator_gear[i, 0] else 1.0

    def mass_matrix(self) -> np.ndarray:
        M = np.zeros((self.nv, self.nv))
        self._mj.mj_fullM(self.model, self.data, M)          # mj 3.10 signature: (model, data, dst)
        return M

    def potential_energy(self) -> float:
        """Gravitational potential V(q) = −Σ mᵢ gᵀ rᵢ = Σ mᵢ g zᵢ (the energy-storage's potential part)."""
        g = float(-self.model.opt.gravity[2]) or 9.81
        return float(sum(self.model.body_mass[b] * g * self.data.xipos[b][2] for b in range(self.model.nbody)))

    def structure(self) -> PHStructure:
        """Extract the pH structure (H, M, ∂H/∂·, D, S) at the current state."""
        M = self.mass_matrix()
        qd = np.asarray(self.data.qvel)
        kinetic = 0.5 * float(qd @ M @ qd)
        potential = self.potential_energy()
        # ∂H/∂q along the motion is exactly the bias force (Coriolis/centrifugal + gravity) = qfrc_bias
        dHdq = np.asarray(self.data.qfrc_bias).copy()
        D = np.diag(np.asarray(self.model.dof_damping, dtype=float))
        return PHStructure(H=kinetic + potential, M=M, dHdq=dHdq, dHdp=qd.copy(), D=D, S=self.S,
                           kinetic=kinetic, potential=potential)

    def hamiltonian(self) -> float:
        s = self.structure()
        return s.H

    def power_ports(self, tau: np.ndarray) -> "tuple[float, float]":
        """(port power in yᵀu, dissipated power q̇ᵀDq̇) — the two terms of the passivity balance."""
        qd = np.asarray(self.data.qvel)
        u_gen = self.S @ np.asarray(tau)                     # generalised actuation force
        p_in = float(qd @ u_gen)                             # yᵀu with y = Sᵀ q̇
        p_diss = float(qd @ np.diag(np.asarray(self.model.dof_damping, dtype=float)) @ qd)
        return p_in, p_diss

    def energy_balance(self, tau: np.ndarray, steps: int = 200) -> "tuple[float, float, float]":
        """Simulate ``steps`` under constant ``tau``; return (ΔH, ∫(port−diss)dt, relative mismatch).

        Verifies the pH identity dH/dt = yᵀu − q̇ᵀDq̇: the measured energy change equals the net port power
        integrated over the trajectory. The mismatch is ``|ΔH − ∫port| / (|∫port| + ε·H₀)`` — small when no
        UNMODELLED port is active; a joint limit or ground contact is an additional port (constraint wrench)
        that shows up as a genuine mismatch (correct pH behaviour, not error)."""
        dt = float(self.model.opt.timestep)
        h0 = self.hamiltonian()
        integ = 0.0
        for _ in range(steps):
            p_in, p_diss = self.power_ports(tau)
            self.data.ctrl[: self.nu] = tau
            self._mj.mj_step(self.model, self.data)
            integ += (p_in - p_diss) * dt
        dH = self.hamiltonian() - h0
        mismatch = abs(dH - integ) / (abs(integ) + 1e-3 * abs(h0))
        return dH, integ, mismatch


def centroidal_hamiltonian(mass: float, com_z: float, p_lin: np.ndarray, ang_mom: float,
                           inertia: float, g: float = 9.81) -> float:
    r"""The CENTROIDAL port-Hamiltonian energy: the momentum-level reduction ``centroidal_run`` optimises.

    ``H_c = ½ |p|²/m + ½ L²/I + m g z`` — CoM translational + rotational kinetic + gravitational potential,
    with the **contact wrenches as the input ports** (ṗ = ΣF + mg, L̇ = Σ(rᵢ−r_com)×Fᵢ). The angular-momentum
    term ``½ L²/I`` is exactly the state the linear-only run planner omitted (the torso-pitch under-actuation);
    in the pH view controlling the pitch = regulating this port-energy channel.
    """
    return 0.5 * float(p_lin @ p_lin) / mass + 0.5 * ang_mom ** 2 / inertia + mass * g * com_z
