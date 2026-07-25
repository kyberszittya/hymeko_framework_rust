"""FORCE_SLIP_SEMANTICS_V1 controller — config + coast-model launch target (pure pieces; the full loop is exercised by
the ablation experiment)."""
import numpy as np

from hymeko_rl.coin_delivery.force_slip_carry import ForceSlipConfig, _launch_velocity


def test_launch_velocity_follows_the_coast_model():
    # to stop AT the target after coasting d under Coulomb μ, launch at sqrt(2 μ g d)
    mu, d = 0.15, 0.10
    v = _launch_velocity(mu, d, cap=2.0)
    assert np.isclose(v, np.sqrt(2 * mu * 9.81 * d), atol=1e-6)
    assert _launch_velocity(mu, 100.0, cap=0.8) == 0.8           # capped (never a projectile)
    assert _launch_velocity(mu, 0.0, cap=2.0) == 0.0             # already there ⇒ no launch


def test_config_stage_flags_and_physical_bounds():
    c = ForceSlipConfig()
    assert c.enable_preload and c.enable_target_velocity and c.enable_slip_aware and c.enable_impulse_gate \
        and c.enable_predictive_brake                             # full S5 by default
    assert c.preload_lo < c.preload_hi and c.preload_hi < 17.0    # a normal-force band, not the 17 N spike
    assert 0 < c.v_target_cap <= 2.0 and c.slip_thresh > 0
    assert c.accel_gain > 0.1                                     # driven, not the under-driving 0.05 (measured)


def test_controller_uses_shared_stack_and_coast_model():
    import inspect

    import hymeko_rl.coin_delivery.force_slip_carry as fs
    src = inspect.getsource(fs)
    assert "pd_governed_torque" in src and "govern_torque" in src   # shared governed stack, no raw-torque bypass
    assert "_launch_velocity" in src                                # launch target from the coast model, not K6
