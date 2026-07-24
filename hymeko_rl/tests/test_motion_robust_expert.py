"""C1 motion-robust coin expert — pure-piece tests (the DLS contact-Jacobian solver + config). The full closed-loop
controller is exercised by the governed re-measurement (needs MuJoCo + pi_0)."""
import numpy as np

from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, _jac_solve


def test_jac_solve_produces_desired_coin_velocity():
    # a well-conditioned 2x4 contact Jacobian: the DLS action should reproduce v_des through J (up to small damping)
    J = np.array([[0.02, 0.0, -0.01, 0.005], [0.0, 0.02, 0.005, -0.01]], np.float32)
    v_des = np.array([0.1, -0.05])
    a = _jac_solve(J, v_des, lam=0.01)
    assert np.allclose(J @ a, v_des, atol=0.02)          # action realises the requested coin velocity
    assert a.shape == (4,)


def test_jac_solve_damps_a_degenerate_jacobian():
    J = np.zeros((2, 4), np.float32)                     # no contact ⇒ degenerate; DLS must return finite (not blow up)
    a = _jac_solve(J, np.array([0.1, 0.1]), lam=0.05)
    assert np.all(np.isfinite(a)) and np.allclose(a, 0.0)


def test_controller_config_is_physical():
    c = CarryControllerConfig()
    assert 0 < c.v_transport < 0.5 and c.brake_k > 0 and c.acquire_mag > 0   # low transport speed, positive braking
    assert c.release_band > 0 and c.replan_every >= 1
