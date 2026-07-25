"""COOPERATIVE launch — A2 grasp-matrix resultant-force allocation (pure math; the full pinned E0 loop is exercised by the
bimanual_curriculum_e0 benchmark). The grasp allocation is the load-bearing new piece: given two contact points it must
produce two contact forces whose resultant is aimed along e_par at zero coin torque, inside the friction cone."""
import numpy as np

from hymeko_rl.coin_delivery.cooperative_launch import CooperativeConfig, GraspAllocator, TwistAllocator, _grasp_allocation


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
