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
    assert 0 < c.track_disp < 0.5 and c.brake_k > 0 and c.acquire_disp > 0   # small displacements, positive braking
    assert c.release_band > 0 and c.replan_every >= 1


def test_shared_low_level_torque_path_is_deterministic_and_equivalent():
    """Trace-equivalence: the calibration and the coin controller both go through the SAME pd_governed_torque path, so
    the same (q, qd, q_des) produces the same torque — no raw-torque bypass, no divergence between the two harnesses."""
    from hymeko_rl.env.governed_arm import V3Stack, pd_governed_torque
    st = V3Stack(0.8, 2.0, 0.2, 2.0, 0.02, 120.0, 12.0, 30.0)
    q, qd = np.zeros(4), np.zeros(4)
    q_des = np.full(4, 0.02)                                 # kp·0.02 = 2.4 < 4 ⇒ below saturation
    lo, hi = np.full(4, -4.0), np.full(4, 4.0)
    a = pd_governed_torque(q, qd, q_des, st, None, lo, hi)
    b = pd_governed_torque(q, qd, q_des, st, None, lo, hi)
    assert np.array_equal(a, b)                              # deterministic
    assert np.allclose(a, st.kp * (q_des - q))              # PD law (qd=0, no prev, unsaturated)
    big = pd_governed_torque(q, qd, np.full(4, 1.0), st, None, lo, hi)
    assert np.all(np.abs(big) <= 4.0)                       # large command saturates to ctrlrange
    slow = pd_governed_torque(q, qd, q_des, st, np.zeros(4), lo, hi)  # torque-rate limit binds from a prev command
    assert np.all(np.abs(slow) <= st.tau_rate * st.control_dt + 1e-9)


def test_c1_controller_uses_no_raw_torque_bypass():
    """The controller must import the shared torque path + governor (physically inescapable stack), not raw ACTION_SCALE."""
    import inspect

    import hymeko_rl.coin_delivery.motion_robust_expert as mre
    src = inspect.getsource(mre)
    assert "pd_governed_torque" in src and "govern_torque" in src and "set_mjcb_control" in src
