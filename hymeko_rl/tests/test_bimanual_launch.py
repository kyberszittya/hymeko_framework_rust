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


def test_twist_solve_is_finite_and_balance_changes_command():
    """The cooperative twist solve dq = Jᵀ(WJ Jᵀ + λ²I)⁻¹ W(desired) is finite (4-vector), and weighting the ω row
    (force balancing) changes the command vs not weighting it — i.e. the zero-spin objective is actually active."""
    J = np.array([[0.02, 0.0, 0.01, 0.0], [0.0, 0.02, 0.0, 0.01], [0.03, -0.03, -0.03, 0.03]], np.float64)  # 3x4 twist J
    twist = np.array([0.1, 0.05, 0.2])                          # current: moving + spinning
    desired = np.array([0.5, 0.0, 0.0]) - twist                 # want +x translation, zero spin

    def solve(w_omega, lam=0.02):
        W = np.diag([1.0, 1.0, w_omega])
        return J.T @ np.linalg.solve((W @ J) @ J.T + lam ** 2 * np.eye(3), W @ desired)
    dq_balanced, dq_unweighted = solve(5.0), solve(0.0)
    assert dq_balanced.shape == (4,) and np.all(np.isfinite(dq_balanced))
    assert not np.allclose(dq_balanced, dq_unweighted)          # the ω-weighting is load-bearing on the command
