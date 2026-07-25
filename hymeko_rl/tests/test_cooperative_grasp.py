"""COOPERATIVE launch — A2 grasp-matrix resultant-force allocation (pure math; the full pinned E0 loop is exercised by the
bimanual_curriculum_e0 benchmark). The grasp allocation is the load-bearing new piece: given two contact points it must
produce two contact forces whose resultant is aimed along e_par at zero coin torque, inside the friction cone."""
import numpy as np

from hymeko_rl.coin_delivery.cooperative_launch import (
    CooperativeConfig, GraspAllocator, TwistAllocator, _grasp_allocation, _grasp_solve, forward_authority)


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


def test_far_side_pair_has_positive_forward_authority():
    """Both tips on the FAR (−e_par) side ⇒ normals point toward +e_par ⇒ pressing can push the coin toward the zone."""
    p_l, p_r = np.array([-0.04, 0.012]), np.array([-0.04, -0.012])
    fa = forward_authority(C, p_l, p_r, E_PAR, mu=0.5)
    assert fa["A_parallel"] > 0.5 and all(a > 0 for a in fa["per_contact"])


def test_zone_side_pair_is_infeasible_for_forward_push():
    """Both tips on the ZONE-facing (+e_par) side ⇒ normals point −e_par ⇒ NO forward push (A∥ ≈ 0). This is the case
    where a grasp allocator MUST return ~zero — an honest refusal, verified against the physical bound."""
    p_l, p_r = np.array([0.04, 0.012]), np.array([0.04, -0.012])
    fa = forward_authority(C, p_l, p_r, E_PAR, mu=0.3)
    assert fa["A_parallel"] < 1e-6                               # infeasible
    f_l, f_r, diag = _grasp_solve(C, p_l, p_r, E_PAR, mu=0.3, lam=0.05)
    assert np.linalg.norm(f_l) < 1e-6 and np.linalg.norm(f_r) < 1e-6   # allocator agrees: zero
    assert diag["forward_force"] < 1e-6


def test_mixed_pair_feasibility_set_by_cone_geometry_not_sign():
    """A MIXED pair (one far-side, one zone-side) is decided by the friction cone, not a hardcoded 'both far-side' rule:
    the far-side contact alone provides positive forward authority."""
    p_far, p_zone = np.array([-0.04, 0.0]), np.array([0.03, 0.02])
    fa = forward_authority(C, p_far, p_zone, E_PAR, mu=0.5)
    assert fa["A_parallel"] > 0 and fa["per_contact"][0] > 0     # the far-side contact carries it


def test_forward_authority_consistent_with_allocation():
    """The honest-refusal invariant: A∥ ≤ 0 ⟺ the grasp solve's realized forward force is ~zero; A∥ > 0 ⟹ a directed
    wrench solve produces positive forward force (no false refusal)."""
    for p_l, p_r, mu in (([-0.04, 0.012], [-0.04, -0.012], 0.5), ([0.04, 0.012], [0.04, -0.012], 0.3)):
        fa = forward_authority(C, np.array(p_l), np.array(p_r), E_PAR, mu)
        _fl, _fr, diag = _grasp_solve(C, np.array(p_l), np.array(p_r), E_PAR, mu, lam=0.05)
        if fa["A_parallel"] < 1e-6:
            assert diag["forward_force"] < 1e-6
        else:
            assert diag["forward_force"] > 0
