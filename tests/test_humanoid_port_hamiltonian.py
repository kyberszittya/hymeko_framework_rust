"""Port-Hamiltonian derivation — numeric (MuJoCo) passivity + symbolic (SymPy) system-equation contracts.

The pH form is only meaningful if its defining properties hold, so these are the contract tests:
- the interconnection J is skew-symmetric → energy is CONSERVED in free flight (numeric drift only);
- actuation is a PORT → ΔH equals the integrated port power ∫τᵀq̇ dt (passivity), absent unmodelled ports;
- the symbolic Hamiltonian/EOM match the mechanics, and IDA-PBC moves the energy minimum to the target.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from scenarios.humanoid.balance_env import _build
from scenarios.humanoid.port_hamiltonian import HumanoidPortHamiltonian
from scenarios.humanoid.symbolic_ph import (
    ida_pbc_potential_shaping,
    mechanical_ph,
    pendulum_ph,
    two_link_leg_ph,
)


def _ph():
    mj, m = _build("humanoid.hymeko")
    d = mj.MjData(m)
    return HumanoidPortHamiltonian(m, d, mj), mj, m, d


def test_ph_is_generated_from_the_hymeko_source() -> None:
    """generate → describe: the pH system is produced FROM a HyMeKo model file, with a role provenance."""
    from scenarios.humanoid.port_hamiltonian import HumanoidPortHamiltonian
    ph = HumanoidPortHamiltonian.from_hymeko("humanoid.hymeko")
    pr = ph.provenance()
    assert pr["source"] == "humanoid.hymeko"
    assert pr["revolute_joints"] >= 16 and pr["actuator_ports (nu)"] >= 16   # the .hymeko's joints → q + ports
    assert pr["total_mass_kg"] > 25.0                                        # geometry+inertias → the M metric's mass
    assert "M(q)" in pr["roles"]["elements + geometry (bodies, inertias)"]   # the storage-metric role is recorded
    ph.data.qpos[2] = 0.9
    ph._mj.mj_forward(ph.model, ph.data)
    assert ph.hamiltonian() > 0                                             # a valid Hamiltonian was generated


def test_mass_matrix_is_symmetric_positive_definite() -> None:
    """The kinetic-energy metric M(q) must be SPD — the storage is a genuine energy."""
    ph, _mj, _m, d = _ph()
    d.qpos[2] = 0.9
    ph._mj.mj_forward(ph.model, d)
    M = ph.mass_matrix()
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)


def test_energy_is_conserved_in_free_flight() -> None:
    """Skew-symmetric J → no energy source: in the air with τ = 0 the Hamiltonian is conserved (numeric drift)."""
    ph, _mj, _m, d = _ph()
    d.qpos[:] = 0
    d.qpos[2] = 3.0                                          # high up → no ground contact (isolated system)
    ph._mj.mj_forward(ph.model, d)
    h0 = ph.hamiltonian()
    dH, _integ, _mm = ph.energy_balance(np.zeros(ph.nu), steps=80)
    assert abs(dH) / h0 < 5e-3                               # relative drift is small (integrator, not a leak)


def test_actuation_is_a_passive_port() -> None:
    """ΔH equals the integrated port power ∫τᵀq̇ dt — the passivity identity — for a moderate (limit-free) torque."""
    ph, _mj, _m, d = _ph()
    d.qpos[:] = 0
    d.qpos[2] = 3.0
    ph._mj.mj_forward(ph.model, d)
    tau = np.zeros(ph.nu)
    tau[6] = 10.0
    dH, port, mismatch = ph.energy_balance(tau, steps=80)
    assert port > 0 and dH > 0                               # the port injects energy, H rises
    assert mismatch < 0.25                                   # ΔH ≈ ∫port (no unmodelled port active)


def test_symbolic_pendulum_hamiltonian_and_eom() -> None:
    """The SymPy pendulum pH reproduces H = ½p² + mgl(1−cosθ) and the canonical equations."""
    ph = pendulum_ph(m=1.0, ell=1.0, g=9.81, b=0.1)
    th, p0 = ph["q"][0], ph["p"][0]
    assert sp.simplify(ph["H"] - (sp.Rational(1, 2) * p0 ** 2 + 9.81 * (1 - sp.cos(th)))) == 0
    assert sp.simplify(ph["f"][0] - p0) == 0                 # q̇ = ∂H/∂p = p
    tau0 = ph["tau"][0]
    assert sp.simplify(ph["f"][1] - (-0.1 * p0 + tau0 - 9.81 * sp.sin(th))) == 0   # ṗ = −Dq̇ + τ − ∂V/∂q


def test_ida_pbc_moves_the_energy_minimum_to_the_target() -> None:
    """Potential-shaping IDA-PBC assigns V_d with a minimum at the target — the closed-loop equilibrium moves."""
    ph = pendulum_ph()
    th = ph["q"][0]
    target = sp.pi
    V_d = sp.Rational(1, 2) * 20 * (th - target) ** 2        # spring to the inverted (θ=π) equilibrium
    tau = ida_pbc_potential_shaping(ph, V_d)[0]
    # closed-loop potential gradient ∂V/∂q − Sτ must vanish at the target (it is the new equilibrium)
    V = 9.81 * (1 - sp.cos(th))
    closed = sp.diff(V, th) - tau
    assert abs(float(closed.subs(th, target))) < 1e-6
    assert abs(float(closed.subs(th, 0.0))) > 1e-3           # the old (hanging) equilibrium is no longer one


def test_two_link_leg_mass_matrix_is_symmetric() -> None:
    """The 2-link leg (the humanoid's sagittal building block) has a symmetric configuration-dependent M(q)."""
    leg = two_link_leg_ph()
    M = leg["M"]
    assert sp.simplify(M[0, 1] - M[1, 0]) == 0
    assert M[0, 0].has(sp.cos)                               # inertial coupling depends on the knee angle


def test_mechanical_ph_rejects_singular_mass_matrix() -> None:
    """A degenerate metric is not a valid energy storage — the derivation must fail, not silently produce junk."""
    q, qd = sp.symbols("q0 q1"), sp.symbols("q0d q1d")
    singular = sp.Matrix([[1, 1], [1, 1]])                   # rank-deficient → not invertible
    try:
        mechanical_ph(list(q), list(qd), singular, sp.Float(0))
        raise AssertionError("expected a failure on the singular mass matrix")
    except (ValueError, NonInvertibleMatrixError):
        pass
