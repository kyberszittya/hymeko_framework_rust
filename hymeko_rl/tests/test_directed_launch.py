"""TARGET_DIRECTED_LAUNCH — target frame + directed contact-Jacobian solve (pure pieces; the full loop is exercised by
the benchmark)."""
import numpy as np

from hymeko_rl.coin_delivery.directed_launch import DirectedLaunchConfig, _directed_delta


def test_target_frame_orthonormal_solve():
    e_par = np.array([1.0, 0.0], np.float32)                     # zone straight ahead (+x)
    e_cross = np.array([0.0, 1.0], np.float32)
    J = np.array([[0.02, 0.0, -0.01, 0.005], [0.0, 0.02, 0.005, -0.01]], np.float32)   # well-conditioned
    coin_v = np.array([0.1, 0.15], np.float32)                   # some cross-track drift (v_cross = 0.15)
    ddir, v_par, v_cross = _directed_delta(J, e_par, e_cross, coin_v, v_target=0.5, cfg=DirectedLaunchConfig())
    assert np.isclose(v_par, 0.1) and np.isclose(v_cross, 0.15)  # frame decomposition
    assert ddir.shape == (4,) and np.isclose(np.linalg.norm(ddir), 1.0, atol=1e-5)   # unit joint direction


def test_cross_suppression_changes_the_command():
    e_par, e_cross = np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)
    J = np.array([[0.02, 0.0, -0.01, 0.005], [0.0, 0.02, 0.005, -0.01]], np.float32)
    coin_v = np.array([0.1, 0.3], np.float32)                    # large cross-track
    off = _directed_delta(J, e_par, e_cross, coin_v, 0.5, DirectedLaunchConfig(enable_cross_suppress=False))[0]
    on = _directed_delta(J, e_par, e_cross, coin_v, 0.5, DirectedLaunchConfig(enable_cross_suppress=True, k_cross=4.0))[0]
    assert not np.allclose(off, on)                             # the cross term changes the directed command (well-cond J)


def test_config_stage_flags():
    c = DirectedLaunchConfig()
    assert c.enable_directed and c.enable_cross_suppress and not c.enable_contact_select   # L2 by default
    assert c.coin_radius > 0 and 0 < c.k_cross
