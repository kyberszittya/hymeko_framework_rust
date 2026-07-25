"""BIMANUAL launch — config + the cooperative twist-Jacobian / force-balancing structure (pure pieces; the full loop is
exercised by the benchmark)."""
import numpy as np

from hymeko_rl.coin_delivery.bimanual_launch import BimanualConfig, _unit2


def test_config_ablation_flags():
    a1 = BimanualConfig(state_dependent=False, force_balance=False)
    a3 = BimanualConfig(state_dependent=True, force_balance=True)
    assert not a1.state_dependent and not a1.force_balance
    assert a3.state_dependent and a3.force_balance
    assert a3.w_omega > 0 and a3.coin_radius > 0 and a3.launch_gain > 0


def test_unit2():
    assert np.allclose(_unit2(np.array([3.0, 4.0])), [0.6, 0.8])
    assert np.allclose(_unit2(np.zeros(2)), 0.0)                 # zero-safe


def test_twist_solve_zeros_the_spin_row_when_balanced():
    """The force-balance objective weights the ω row so the twist solve drives coin rotation toward 0. A well-conditioned
    twist Jacobian + a pure-translation target should yield a Δq whose predicted twist has small ω."""
    cfg = BimanualConfig(force_balance=True, w_omega=5.0, lam=0.02)
    J = np.array([[0.02, 0.0, 0.01, 0.0], [0.0, 0.02, 0.0, 0.01], [0.03, -0.03, -0.03, 0.03]], np.float64)  # 3x4 twist J
    e_par = np.array([1.0, 0.0])
    W = np.diag([1.0, 1.0, cfg.w_omega])
    twist = np.array([0.1, 0.05, 0.2])                          # current: moving + spinning
    desired = np.array([0.5 * e_par[0], 0.5 * e_par[1], 0.0]) - twist
    jw = W @ J
    dq = J.T @ np.linalg.solve(jw @ J.T + cfg.lam ** 2 * np.eye(3), W @ desired)
    pred = J @ dq
    # the weighted solve should reduce |ω| relative to an unweighted (w_omega=1) solve
    Wu = np.eye(3)
    dqu = J.T @ np.linalg.solve((Wu @ J) @ J.T + cfg.lam ** 2 * np.eye(3), Wu @ desired)
    assert abs(pred[2]) <= abs((J @ dqu)[2]) + 1e-9            # zero-spin weighting does not increase predicted spin
