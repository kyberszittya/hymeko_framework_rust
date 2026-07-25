"""COOPERATIVE launch — A2 grasp-matrix resultant-force allocation (pure math; the full pinned E0 loop is exercised by the
bimanual_curriculum_e0 benchmark). The grasp allocation is the load-bearing new piece: given two contact points it must
produce two contact forces whose resultant is aimed along e_par at zero coin torque, inside the friction cone."""
import numpy as np

from hymeko_rl.coin_delivery.cooperative_launch import (
    CooperativeConfig, GraspAllocator, TwistAllocator, _grasp_allocation, _grasp_solve, _solve_contact_qp,
    forward_feasibility, internal_force_feasibility)


def _wrench(c, p_l, p_r, f_l, f_r):
    """Resultant [Fx, Fy, τ] on the coin (centre c) from world contact forces f_l, f_r at p_l, p_r."""
    net = f_l + f_r
    tau = sum(float((p - c)[0] * f[1] - (p - c)[1] * f[0]) for p, f in ((p_l, f_l), (p_r, f_r)))
    return np.array([net[0], net[1], tau])


def test_grasp_symmetric_backside_is_target_directed():
    """Two contacts symmetric about e_par on the coin's back side ⇒ resultant along +e_par, zero cross, zero torque."""
    c, r = np.zeros(2), 0.02
    e_par = np.array([1.0, 0.0])
    th = np.deg2rad(30)
    back = -np.cos(th) * e_par
    p_l = c + r * (back + np.sin(th) * np.array([0.0, 1.0]))     # +y side
    p_r = c + r * (back - np.sin(th) * np.array([0.0, 1.0]))     # -y side
    f_l, f_r = _grasp_allocation(c, p_l, p_r, e_par, mu=0.5, lam=0.02)
    w = _wrench(c, p_l, p_r, f_l, f_r)
    assert w[0] > 0                                              # net push toward the zone (+e_par)
    assert abs(w[1]) < 1e-3 * max(abs(w[0]), 1e-6)              # ~zero cross-track resultant
    assert abs(w[2]) < 1e-4                                      # ~zero coin torque (no spin)


def test_grasp_respects_friction_cone():
    """Each returned world force decomposes to |Ft| ≤ μ·Fn with Fn ≥ 0 (no pulling, tangential bounded)."""
    c = np.zeros(2)
    p_l, p_r = np.array([-0.018, 0.012]), np.array([-0.015, -0.010])   # asymmetric on purpose
    mu = 0.3
    f_l, f_r = _grasp_allocation(c, p_l, p_r, np.array([1.0, 0.0]), mu=mu, lam=0.02)
    for p, f in ((p_l, f_l), (p_r, f_r)):
        n = (c - p) / (np.linalg.norm(c - p) + 1e-12)
        tang = np.array([-n[1], n[0]])
        fn, ft = float(f @ n), float(f @ tang)
        assert fn >= -1e-9                                       # Fn ≥ 0
        assert abs(ft) <= mu * fn + 1e-6                         # |Ft| ≤ μ·Fn


def test_grasp_zero_when_no_geometry():
    """Degenerate: coincident contact at the centre ⇒ finite, bounded (regularised, no NaN)."""
    c = np.zeros(2)
    f_l, f_r = _grasp_allocation(c, np.array([1e-9, 0.0]), np.array([0.0, 1e-9]), np.array([1.0, 0.0]), mu=0.5, lam=0.05)
    assert np.all(np.isfinite(f_l)) and np.all(np.isfinite(f_r))


def test_allocators_construct_and_are_callable():
    cfg = CooperativeConfig()
    assert callable(TwistAllocator(cfg)) and callable(GraspAllocator(cfg))
    assert cfg.grasp_lam > 0                                     # A2 damping present


# --- grasp-feasibility oracle: the launch-wrench feasibility that gates the acquisition (E2B semantic audit) ---
E_PAR = np.array([1.0, 0.0])                                     # zone is toward +x
C = np.zeros(2)


def test_far_side_pair_is_launch_feasible():
    """Both tips on the FAR (−e_par) side ⇒ normals point toward +e_par ⇒ the forward direction is inside the friction
    cone (coeff > 0, feasible). The bounded magnitude scales with the per-contact realizable normal force."""
    p_l, p_r = np.array([-0.04, 0.012]), np.array([-0.04, -0.012])
    fa = forward_feasibility(C, p_l, p_r, E_PAR, mu=0.5)
    assert fa["feasible"] and fa["coeff"] > 0.5 and all(a > 0 for a in fa["per_contact"])
    assert forward_feasibility(C, p_l, p_r, E_PAR, mu=0.5, fn_max=(2.0, 2.0))["bounded_forward_force"] > fa["bounded_forward_force"]


def test_zone_side_pair_is_infeasible_for_forward_push():
    """Both tips on the ZONE-facing (+e_par) side ⇒ normals point −e_par ⇒ NO forward-pushing direction (coeff = 0). This
    is the case where a grasp allocator MUST return ~zero — an honest refusal, verified against the feasibility bound."""
    p_l, p_r = np.array([0.04, 0.012]), np.array([0.04, -0.012])
    fa = forward_feasibility(C, p_l, p_r, E_PAR, mu=0.3)
    assert not fa["feasible"] and fa["coeff"] < 1e-6            # geometrically infeasible
    f_l, f_r, diag = _grasp_solve(C, p_l, p_r, E_PAR, mu=0.3, lam=0.05)
    assert np.linalg.norm(f_l) < 1e-6 and np.linalg.norm(f_r) < 1e-6   # allocator agrees: zero
    assert diag["forward_force"] < 1e-6


def test_mixed_pair_feasibility_set_by_cone_geometry_not_sign():
    """A MIXED pair (one far-side, one zone-side) is decided by the friction cone, not a hardcoded 'both far-side' rule:
    the far-side contact alone makes the forward direction feasible."""
    p_far, p_zone = np.array([-0.04, 0.0]), np.array([0.03, 0.02])
    fa = forward_feasibility(C, p_far, p_zone, E_PAR, mu=0.5)
    assert fa["feasible"] and fa["per_contact"][0] > 0          # the far-side contact carries it


def test_internal_force_cradle_certificate():
    """The definitive cradle certificate: straddling contacts (opposite sides) admit a cone-admissible internal force
    with same-sign normals (Gf=0 while both press) → cradle feasible; same-side contacts require opposite-sign normals →
    infeasible. This supersedes the n_L·n_R prior and is what decides controller-work vs embodiment-rethink."""
    c = np.zeros(2)
    straddle = internal_force_feasibility(c, np.array([-0.04, 0.0]), np.array([0.04, 0.0]), mu=0.5)
    assert straddle["feasible"] and straddle["cone_admissible"] and straddle["fn_same_sign"]
    same = internal_force_feasibility(c, np.array([-0.04, 0.012]), np.array([-0.04, -0.012]), mu=0.5)
    assert not same["feasible"] and not same["cone_admissible"]  # same-side → grip needs |Ft| ≫ μFn (cone-infeasible)


def test_straddle_geometry_null_preload_feasibility():
    """The straddle prerequisite for a null-preload cradle: two contact normals oppose (n_L·n_R < 0) iff a net-zero squeeze
    with both tips pressing is geometrically possible. Same-side normals (dot > 0) add and cannot cancel."""
    from hymeko_rl.coin_delivery.cooperative_launch import _unit2
    c = np.zeros(2)
    # opposite sides: left tip at −x, right tip at +x → normals (+x) and (−x) oppose
    n_l = _unit2(c - np.array([-0.04, 0.0]))
    n_r = _unit2(c - np.array([0.04, 0.0]))
    assert float(n_l @ n_r) < 0                                 # straddle → null preload feasible
    # same side: both tips at −x → both normals +x, add
    n_l2 = _unit2(c - np.array([-0.04, 0.01]))
    n_r2 = _unit2(c - np.array([-0.04, -0.01]))
    assert float(n_l2 @ n_r2) > 0                               # same side → cannot cancel


def test_contact_qp_respects_hard_inequality_and_box():
    """The E3a v3 QP: min ½ΔqᵀHΔq + gᵀΔq s.t. C·Δq ≥ d ∧ |Δq| ≤ box. The unconstrained optimum would violate the
    inequality; the active-set solver must return a Δq that satisfies the hard constraint AND the box."""
    hess = np.eye(4)
    grad = np.array([1.0, 1.0, 0.0, 0.0])                        # unconstrained opt = −grad = (−1,−1,0,0)
    c_ineq = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])         # Δq0 ≥ 0.2, Δq1 ≥ 0.2 (opposes the unconstrained pull)
    d_ineq = np.array([0.2, 0.2])
    dq = _solve_contact_qp(hess, grad, c_ineq, d_ineq, dq_max=0.5)
    assert np.all(c_ineq @ dq >= d_ineq - 1e-6)                 # hard inequality satisfied
    assert np.all(np.abs(dq) <= 0.5 + 1e-9)                     # box satisfied


def test_contact_qp_unconstrained_when_feasible():
    """When the unconstrained optimum already satisfies the constraints, the QP returns it (clipped to the box)."""
    hess, grad = np.eye(4), np.array([-0.1, -0.1, 0.0, 0.0])    # opt = (0.1,0.1,0,0), already ≥ 0
    dq = _solve_contact_qp(hess, grad, np.array([[1.0, 0, 0, 0]]), np.array([0.0]), dq_max=0.5)
    assert np.allclose(dq, [0.1, 0.1, 0.0, 0.0], atol=1e-6)


def test_forward_feasibility_consistent_with_allocation():
    """The honest-refusal invariant: coeff = 0 ⟺ the grasp solve's realized forward force is ~zero; coeff > 0 ⟹ a directed
    wrench solve produces positive forward force (no false refusal)."""
    for p_l, p_r, mu in (([-0.04, 0.012], [-0.04, -0.012], 0.5), ([0.04, 0.012], [0.04, -0.012], 0.3)):
        fa = forward_feasibility(C, np.array(p_l), np.array(p_r), E_PAR, mu)
        _fl, _fr, diag = _grasp_solve(C, np.array(p_l), np.array(p_r), E_PAR, mu, lam=0.05)
        if not fa["feasible"]:
            assert diag["forward_force"] < 1e-6
        else:
            assert diag["forward_force"] > 0
