"""COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1 analysis-primitive tests (pure, deterministic; no env)."""
from hymeko_rl.coin_delivery.coin_contract_audit import (
    braking_eligibility_sweep,
    dwell_k_success,
    recertify,
    success_ladder,
    touched_ever_vs_current,
)


def _step(dtz, spd, lf=True, rf=False):
    return {"disk_to_zone": dtz, "disk_speed": spd, "left_fingertip": lf, "right_fingertip": rf,
            "arm_body_contact": False, "arm_body_impulse": 0.0, "force_left": 1.0, "force_right": 1.0,
            "body_progress": 0.0, "ever_grasped": True}


def _traj(k_hold, *, dtz=0.015, spd=0.02, touch=True):
    approach = [_step(0.3, 0.3, lf=touch) for _ in range(3)]      # touched, far
    hold = [_step(dtz, spd, lf=touch) for _ in range(k_hold)]     # centered+settled hold
    return approach + hold


def test_recertify_dwell_k_monotone():
    steps = _traj(8)                                              # 8-step centered+settled hold
    assert recertify(steps, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=6)[0]      # K6 delivered
    assert recertify(steps, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=10)[0] is False  # K10 not held
    assert recertify(steps, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=3)[0]      # K3 easier


def test_recertify_center_tol_gates_containment():
    steps = _traj(8, dtz=0.03)                                    # coin center at 0.03 (not fully contained)
    assert recertify(steps, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=6)[0] is False
    assert recertify(steps, 0.02, center_tol=0.04, settle_vel=0.06, dwell_req=6)[0]      # looser tol delivers


def test_recertify_requires_footprints_disjoint():
    steps = _traj(8)
    assert recertify(steps, 0.0, center_tol=0.02, settle_vel=0.06, dwell_req=6)[0] is False  # clearance 0 ⇒ never


def test_success_ladder_grades_beyond_k6():
    steps = _traj(4)                                             # only 4-step hold: K3 yes, K6 no
    lad = success_ladder(steps)
    assert lad["target_entry"] and lad["one_step_in_zone"] and lad["k3_dwell"] and not lad["k6_dwell"]
    assert lad["max_held_dwell"] == 4 and lad["ever_touched"]


def test_dwell_k_success_matches_hold():
    steps = _traj(6)
    assert dwell_k_success(steps, 6) and dwell_k_success(steps, 3) and not dwell_k_success(steps, 10)


def test_touched_ever_vs_current_detects_release():
    steps = [_step(0.3, 0.3, lf=True)] + [_step(0.015, 0.02, lf=False, rf=False) for _ in range(6)]  # touch then release
    tv = touched_ever_vs_current(steps)
    assert tv["delivered_touched_ever"] and tv["current_contact_through_dwell"] is False   # certifier passes on touched-ever


def test_braking_eligibility_target_directed_signed():
    # target-directed (radial>0) vs target-away (radial<0) must be separated; abs() would wrongly count the retreating one
    rows = [{"pi0_radial_vel": 0.10, "support": {"n_safe_beneficial": 2}},   # approaching fast + support (eligible)
            {"pi0_radial_vel": -0.10, "support": {"n_safe_beneficial": 3}},  # RETREATING fast — NOT braking-eligible
            {"pi0_radial_vel": 0.01, "support": {"n_safe_beneficial": 1}}]   # slow toward — false intervention
    r = braking_eligibility_sweep(rows, [0.05])["v_excess=0.05"]
    assert r["n_target_directed"] == 1 and r["n_target_away"] == 1           # signed split, not abs
    assert r["support_over_target_directed"] == 1.0 and r["false_interventions_on_slow_states"] == 1


def test_recertify_detects_certification_on_final_step():
    # K6 dwell completes exactly on the final post-step — must be detected (terminal state not omitted)
    steps = [_step(0.3, 0.3)] + [_step(0.015, 0.02) for _ in range(6)]       # 6 consecutive centered+settled = last step
    delivered, cert_idx, _ = recertify(steps, 0.02, center_tol=0.02, settle_vel=0.06, dwell_req=6)
    assert delivered and cert_idx == len(steps) - 1


def test_decompose_reward_sums_to_scalar():
    from hymeko_rl.coin_delivery.coin_contract_audit import decompose_reward
    from hymeko_rl.coin_delivery.coin_rl_env import CANONICAL_REWARD_FILE, CoinRL4Dof
    from hymeko_rl.env.reward import RewardSpec
    rl = CoinRL4Dof(horizon=10); rl.reset(1011)
    spec = RewardSpec.from_hymeko(CANONICAL_REWARD_FILE)
    import numpy as np
    for _ in range(5):
        rl.step(np.zeros(4, np.float32))
        scalar = float(spec.evaluate(rl.inner, rl._dtz(), rl.inner.data.ctrl))
        comps = decompose_reward(rl.inner, rl._dtz(), rl.inner.data.ctrl, spec.terms)
        assert abs(sum(comps.values()) - scalar) < 1e-6
