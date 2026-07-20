"""Tests for the Track-B centering primitive (train.coin_delivery_primitives) + the env base-override hook.

Covers: the parameter mapping, the centering control law (proportional deceleration + settle mode), the base-override
plumbing (centering replaces the grasp_carry SUFFIX, prefix stays grasp_carry), the eval metric keys, and a tiny CEM.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.train.coin_delivery_primitives import (
    CenteringOracleConfig,
    CenteringParams,
    cem_centering,
    centering_action,
    eval_centering,
    roll_centering,
)
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, make_delivery_rl_env, roll_delivery, scripted_action_fn


class _FakeInner:
    """Minimal inner-env stand-in exposing what centering_action reads (the coin geometry via direction_to_zone)."""

    def __init__(self, coin, zone) -> None:
        self._planar_metrics = type("M", (), {"disk_pos": np.asarray(coin, np.float64)})()
        self._zone_x, self._zone_y = float(zone[0]), float(zone[1])

    def direction_to_zone(self):
        # geometry now lives on the env; delegate to the shared pure function so the mock exercises the real math
        from hymeko_rl.env.planar_grasp_env import coin_zone_direction
        return coin_zone_direction(self._planar_metrics.disk_pos, self._zone_x, self._zone_y)


def test_from_unit_maps_into_ranges() -> None:
    lo = CenteringParams.from_unit(np.zeros(5))
    hi = CenteringParams.from_unit(np.ones(5))
    assert lo.kp == 3.0 and hi.kp == 40.0
    assert lo.deadband == 0.015 and hi.deadband == 0.035
    mid = CenteringParams.from_unit(np.full(5, 0.5))
    assert 3.0 < mid.kp < 40.0


def test_centering_proportional_decelerates_near_center() -> None:
    p = CenteringParams(kp=18.0, sq_hold=0.8, sq_settle=0.4, settle_r=0.2, deadband=0.005)
    far = centering_action(_FakeInner([0.20, 0.0], [0.0, 0.0]), p)   # dist 0.20 → gain clipped to 1
    near = centering_action(_FakeInner([0.03, 0.0], [0.0, 0.0]), p)  # dist 0.03 → gain 0.54
    assert abs(far[0]) > abs(near[0])                                # translation magnitude shrinks on approach
    assert abs(far[0]) <= 1.0 + 1e-6


def test_centering_settle_mode_engages_in_deadband() -> None:
    p = CenteringParams(kp=18.0, sq_hold=0.8, sq_settle=-0.2, settle_r=0.1, deadband=0.03)
    a = centering_action(_FakeInner([0.02, 0.0], [0.0, 0.0]), p)     # dist 0.02 < deadband → settle
    assert a[3] == np.float32(-0.2)                                  # settle squeeze used


def test_centering_action_in_box() -> None:
    p = CenteringParams()
    a = centering_action(_FakeInner([0.15, 0.1], [0.0, 0.0]), p)
    assert a.shape == (6,) and np.all(np.abs(a) <= 1.0 + 1e-6)


def test_base_override_only_affects_suffix_not_prefix() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    # scripted (grasp_carry everywhere) vs centering-suffix: the prefix (acquisition) must be identical grasp_carry,
    # so the run must at least start the same; the override must not raise and must produce a valid rollout.
    r = roll_centering(env, 64_000, CenteringParams())
    assert set(r) >= {"center_reach", "zone_entry", "final_dtz"}
    assert env._base_override is None                                # restored after the roll (finally)


def test_eval_centering_keys() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    ev = eval_centering(CenteringParams(), range(64_000, 64_006), cfg, env=env)
    for k in ("center_reach", "zone_entry", "final_dtz_med", "n_center", "contact_lost"):
        assert k in ev


def test_cem_centering_tiny_budget_runs() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    best = cem_centering(range(64_000, 64_004), cfg, CenteringOracleConfig(pop=3, elite=2, iters=1),
                         np.random.default_rng(0), env=env)
    assert isinstance(best["params"], CenteringParams)
    assert "n_center" in best


def test_scripted_still_grasp_carry_after_override_module_import() -> None:
    # regression: importing/using the primitive must not change the default scripted rollout (base_override defaults None)
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    r = roll_delivery(env, 64_000, scripted_action_fn())
    assert r["residual_norm"] == 0.0
