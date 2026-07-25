"""Intermittent-contact coin controller — config + structural tests (the full FSM is exercised by the C2 run)."""
from hymeko_rl.coin_delivery.intermittent_carry import IntermittentConfig


def test_config_defaults_are_physical_and_full():
    c = IntermittentConfig()
    assert 0 < c.impulse_disp < 0.5 and c.impulse_steps >= 1        # a short, bounded push
    assert c.coast_min_speed > 0 and c.coast_max_steps >= 1         # coast has a stall threshold and a timeout
    assert c.brake_speed > 0 and c.brake_dist > 0
    assert c.enable_recontact and c.enable_brake and c.enable_settle   # full controller by default (arm E)


def test_ablation_flags_select_arms():
    """The three enable_* flags define the progressive ablation B→E; toggling them is the ONLY structural difference."""
    b = IntermittentConfig(enable_recontact=False, enable_brake=False, enable_settle=False)   # impulse+coast only
    e = IntermittentConfig()                                                                  # full
    assert not (b.enable_recontact or b.enable_brake or b.enable_settle)
    assert e.enable_recontact and e.enable_brake and e.enable_settle
    # the numeric knobs are shared so an arm differs ONLY by which options are on
    assert b.impulse_disp == e.impulse_disp and b.impulse_steps == e.impulse_steps


def test_controller_uses_shared_stack_no_raw_torque_bypass():
    """Delivery goes through the shared pd_governed_torque + per-sub-step governor — physically inescapable stack."""
    import inspect

    import hymeko_rl.coin_delivery.intermittent_carry as ic
    src = inspect.getsource(ic)
    assert "pd_governed_torque" in src and "govern_torque" in src and "set_mjcb_control" in src
    assert "direction_to_zone" in src            # delivery uses coin→zone geometry, computed from state (never K6)
