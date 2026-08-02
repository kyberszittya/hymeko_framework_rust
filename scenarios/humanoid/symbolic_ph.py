r"""Symbolic port-Hamiltonian derivation (SymPy) — generate the complete system equations from a model.

Where ``port_hamiltonian`` extracts the pH structure *numerically* from MuJoCo, this derives it *symbolically*
with SymPy: given generalised coordinates ``q``, a mass matrix ``M(q)`` and a potential ``V(q)`` (which a
HyMeKo hypergraph model supplies via its energy_storage / interconnection / port roles), it produces

    H(q,p) = ½ pᵀ M(q)⁻¹ p + V(q)                                          (the Hamiltonian / energy storage)
    ⎡q̇⎤   ⎛⎡0  I⎤   ⎡0 0⎤⎞⎡∂H/∂q⎤   ⎡0⎤
    ⎢ ⎥ = ⎜⎢    ⎥ − ⎢   ⎥⎟⎢     ⎥ + ⎢ ⎥τ                                     (the pH equations of motion)
    ⎣ṗ⎦   ⎝⎣−I 0⎦   ⎣0 D⎦⎠⎣∂H/∂p⎦   ⎣S⎦

as closed-form SymPy expressions (→ LaTeX for the report/visualisation), plus the **IDA-PBC** energy-shaping
control law that assigns a desired energy ``H_d`` with a minimum at the target — the passivity-based
controller the balance/running work uses. The same machinery, with spatial operators, writes distributed
(PDE) port-Hamiltonians (e.g. a flexible link ``∂ₜ[q;p] = (J−R)δH`` with ``J`` a differential operator); a
1-D example is included as the PDE hook.

# Preconditions: ``M`` symmetric positive-definite (a valid kinetic metric); ``S`` full column rank.
# Postconditions: ``H`` matches ½q̇ᵀMq̇+V on the ``p = M q̇`` shell; the IDA-PBC ``V_d`` has its min at q*.
"""

from __future__ import annotations

import sympy as sp


def mechanical_ph(q: "list[sp.Symbol]", qdot: "list[sp.Symbol]", M: sp.Matrix, V: sp.Expr,
                  D: sp.Matrix | None = None, S: sp.Matrix | None = None) -> dict:
    r"""Derive the symbolic port-Hamiltonian form of a mechanical system.

    Returns a dict with the momenta ``p``, Hamiltonian ``H``, gradients ``∂H/∂q``/``∂H/∂p``, the structure
    matrices ``J``/``R``/``g``, and the pH vector field ``f = (J−R)∇H + gτ`` — all as SymPy expressions.
    """
    n = len(q)
    p = sp.Matrix([sp.Symbol(f"p_{i}", real=True) for i in range(n)])
    Minv = M.inv()
    H = sp.Rational(1, 2) * (p.T * Minv * p)[0, 0] + V
    dHdq = sp.Matrix([sp.diff(H, qi) for qi in q])
    dHdp = sp.Matrix([sp.diff(H, pi) for pi in p])
    In, Z = sp.eye(n), sp.zeros(n)
    J = sp.Matrix(sp.BlockMatrix([[Z, In], [-In, Z]]))       # skew-symmetric interconnection
    Dm = D if D is not None else sp.zeros(n)
    R = sp.Matrix(sp.BlockMatrix([[Z, Z], [Z, Dm]]))         # dissipation (joint damping)
    Sm = S if S is not None else In
    g = sp.Matrix(sp.BlockMatrix([[sp.zeros(n, Sm.shape[1])], [Sm]]))
    tau = sp.Matrix([sp.Symbol(f"tau_{i}", real=True) for i in range(Sm.shape[1])])
    grad = sp.Matrix.vstack(dHdq, dHdp)
    f = (J - R) * grad + g * tau                             # ẋ = (J−R)∇H + gτ
    return {"q": q, "p": list(p), "tau": list(tau), "H": sp.simplify(H), "M": M, "Minv": Minv,
            "dHdq": dHdq, "dHdp": dHdp, "J": J, "R": R, "g": g, "f": sp.simplify(f),
            "qddot": sp.simplify(f[n:, 0])}


def ida_pbc_potential_shaping(ph: dict, V_d: sp.Expr) -> sp.Matrix:
    r"""Potential-shaping IDA-PBC control law: assign the closed-loop potential ``V_d`` (min at the target).

    For ``M_d = M`` the matching condition reduces to ``S τ = ∂V/∂q − ∂V_d/∂q`` (shape gravity into a
    spring toward q*). Returns ``τ = S⁺ (∂V/∂q − ∂V_d/∂q)`` — the passivity-based controller; the closed loop
    is again port-Hamiltonian with energy ``H_d = ½pᵀM⁻¹p + V_d`` as its Lyapunov function.
    """
    q = ph["q"]
    V = ph["H"] - sp.Rational(1, 2) * (sp.Matrix(ph["p"]).T * ph["Minv"] * sp.Matrix(ph["p"]))[0, 0]
    dV = sp.Matrix([sp.diff(V, qi) for qi in q])
    dVd = sp.Matrix([sp.diff(V_d, qi) for qi in q])
    S = ph["g"][len(q):, :]
    return sp.simplify(S.pinv() * (dV - dVd))


# ---- canonical examples: the humanoid's building blocks ------------------------------------------------

def pendulum_ph(m: float = 1.0, ell: float = 1.0, g: float = 9.81, b: float = 0.1) -> dict:
    """A damped, actuated pendulum — the simplest underactuated pH (and the leaf of the leg chain)."""
    th, thd = sp.symbols("theta thetadot", real=True)
    M = sp.Matrix([[sp.Float(m) * sp.Float(ell) ** 2]])
    V = sp.Float(m) * sp.Float(g) * sp.Float(ell) * (1 - sp.cos(th))
    return mechanical_ph([th], [thd], M, V, D=sp.Matrix([[sp.Float(b)]]), S=sp.Matrix([[1]]))


def two_link_leg_ph(m1: float = 3.0, m2: float = 2.0, l1: float = 0.4, l2: float = 0.4,
                    g: float = 9.81) -> dict:
    """A planar 2-link leg (hip + knee) — the humanoid's sagittal building block, actuated at both joints."""
    q1, q2, d1, d2 = sp.symbols("q1 q2 q1dot q2dot", real=True)
    lc1, lc2 = l1 / 2, l2 / 2
    I1, I2 = m1 * l1 ** 2 / 12, m2 * l2 ** 2 / 12
    a = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2)
    bb = m2 * l1 * lc2
    d = I2 + m2 * lc2 ** 2
    M = sp.Matrix([[a + 2 * bb * sp.cos(q2), d + bb * sp.cos(q2)],
                   [d + bb * sp.cos(q2), d]])
    V = (m1 * g * lc1 + m2 * g * l1) * sp.sin(q1) + m2 * g * lc2 * sp.sin(q1 + q2)
    return mechanical_ph([q1, q2], [d1, d2], M, V, D=sp.eye(2) * sp.Float(0.05), S=sp.eye(2))


def flexible_link_pde() -> dict:
    r"""PDE port-Hamiltonian hook — a 1-D vibrating string/flexible link on a spatial domain.

    ``H = ½ ∫ (p²/ρ + T (∂ₓw)²) dx`` with the boundary-controlled pH ``∂ₜ[w; p] = J δH``, ``J`` the skew
    differential operator ``[[0, 1],[−1, 0]]`` acting through ``∂ₓ``. Returned symbolically to show the same
    energy/interconnection template extends from ODE robots to distributed (PDE) systems — the HyMeKo goal.
    """
    x, t, rho, T = sp.symbols("x t rho T", positive=True)
    w = sp.Function("w")(x, t)
    p = sp.Function("p")(x, t)
    dens = sp.Rational(1, 2) * (p ** 2 / rho + T * sp.diff(w, x) ** 2)     # energy density
    # δH/δw = −T ∂ₓₓ w ,  δH/δp = p/ρ ;  ∂ₜ w = δH/δp ,  ∂ₜ p = −δH/δw = T ∂ₓₓ w  → wave equation
    wave = sp.Eq(sp.diff(w, t, 2), (T / rho) * sp.diff(w, x, 2))
    return {"energy_density": dens, "dHdw": -T * sp.diff(w, x, 2), "dHdp": p / rho, "pde": wave}
