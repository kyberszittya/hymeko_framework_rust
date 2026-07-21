"""STRICT_MONITOR_CONTRACT gate (2026-07-22) — under the corrected physical-contact contract:
1. the production certifier agrees with the independent raw-MuJoCo strict oracle on a diagnostic rollout;
2. arm-link contact is NOT an automatic failure (the redefined `clean`);
3. an excessive body-only-driven sweep with no grasp IS flagged as unclean.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.delivery_certificate import CertStep, DeliveryCertifier, DeliveryThresholds
from hymeko_rl.coin_delivery.raw_strict_oracle import raw_cert_step


def _diag_state():
    from hymeko_rl.coin_delivery.fixed_position import CoinInitialState, TABLE_Z
    return CoinInitialState(coin_position=(0.030, 0.230, TABLE_Z),
                            target_position=(-0.021588, 0.145066, 0.004), zone_half=0.04)


def test_certifier_matches_raw_oracle_on_rollout():
    """Stream a composed rollout under corrected physics; the certifier (cached planar_metrics) and the raw-MuJoCo
    oracle must agree step-by-step on disk_to_zone / disk_speed / contacts and on the final strict verdict."""
    import torch

    from hymeko_rl.coin_delivery.fixed_position import apply_initial_state
    from hymeko_rl.coin_delivery.fixed_position_replay import build_actors
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    from hymeko_rl.train.sac import build_sac
    env, cf = neutral_env(prefix_steps=0, geom="POINT"); inner = cf._env
    apply_initial_state(env, cf, _diag_state(), seed_hint=0)
    appr, _ = build_actors("P4_E_APPROACH_HANDOFF")
    tr, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    tr.load_state_dict(torch.load("experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt", weights_only=True)); tr.eval()
    tfn = _greedy_action_fn(tr)
    cert_prod = DeliveryCertifier(initial_clearance=_clearance(inner))
    cert_raw = DeliveryCertifier(initial_clearance=_clearance(inner))
    max_dz = max_v = 0.0
    # phase 1: E-approach
    for _k in range(160):
        sp = _cert_step(inner, cf); sr = raw_cert_step(inner, cf)
        max_dz = max(max_dz, abs(sp.disk_to_zone - sr.disk_to_zone)); max_v = max(max_v, abs(sp.disk_speed - sr.disk_speed))
        assert sp.left_fingertip == sr.left_fingertip and sp.right_fingertip == sr.right_fingertip
        cert_prod.update(sp); cert_raw.update(sr)
        m = inner._planar_metrics
        if m.left_contact and m.right_contact:
            break
        a = appr.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64); cf._t = 0; cf._both_hist = []
    env._suffix_t = 0; env._prev_dtz = env._dtz(); env._prev_both = env._both(); o = cf._obs(np.zeros(4, np.float32))
    for _t in range(200):
        sp = _cert_step(inner, cf); sr = raw_cert_step(inner, cf)
        max_dz = max(max_dz, abs(sp.disk_to_zone - sr.disk_to_zone)); max_v = max(max_v, abs(sp.disk_speed - sr.disk_speed))
        cert_prod.update(sp); cert_raw.update(sr)
        if cert_prod.delivery_certified:
            break
        o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
    assert max_dz < 1e-3, f"certifier vs oracle disk_to_zone disagree by {max_dz}"
    assert max_v < 0.02, f"certifier vs oracle disk_speed disagree by {max_v}"
    assert cert_prod.delivery_certified == cert_raw.delivery_certified, "strict verdict disagreement"


def test_arm_link_contact_is_not_auto_failure():
    """A centered/settled step WITH arm-link contact (and a grasp) must still accumulate delivery dwell (clean)."""
    c = DeliveryCertifier(initial_clearance=0.05)
    s = CertStep(disk_to_zone=0.005, disk_speed=0.01, left_fingertip=True, right_fingertip=True,
                 arm_body_contact=True, arm_body_impulse=5.0, body_progress=0.0, ever_grasped=True)
    for _ in range(DeliveryThresholds().dwell_req):
        c.update(s)
    assert c.delivery_certified, "arm-link contact wrongly voided a clean, grasped, centered delivery"


def test_excessive_body_sweep_with_no_grasp_is_unclean():
    """A large body-only-driven sweep with NO grasp is a bulldoze → unclean → no delivery."""
    c = DeliveryCertifier(initial_clearance=0.05)
    s = CertStep(disk_to_zone=0.005, disk_speed=0.01, left_fingertip=False, right_fingertip=False,
                 arm_body_contact=True, arm_body_impulse=5.0,
                 body_progress=DeliveryThresholds().body_sweep_max + 0.02, ever_grasped=False)
    for _ in range(DeliveryThresholds().dwell_req + 4):
        c.update(s)
    assert not c.delivery_certified, "an excessive no-grasp body sweep was wrongly certified"


def test_fingertip_attribution_and_robot_touch_preserved():
    c = DeliveryCertifier(initial_clearance=0.05)
    s = CertStep(disk_to_zone=0.005, disk_speed=0.01, left_fingertip=True, right_fingertip=False,
                 arm_body_contact=False, arm_body_impulse=0.0, body_progress=0.0, ever_grasped=False)
    for _ in range(DeliveryThresholds().dwell_req):
        c.update(s)
    assert c.robot_touched and c.delivery_certified
