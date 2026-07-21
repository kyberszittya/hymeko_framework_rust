"""Tests for the fixed-initial-position Coin Delivery replay (schema, apply/extract bit-identity, fail-loud gates,
deterministic-problem vs exact-state equivalence)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.fixed_position import (TABLE_Z, CoinInitialState, InvalidInitialState,
                                                    apply_initial_state, extract_initial_state)
from hymeko_rl.coin_delivery.fixed_position_replay import analyze_reachability, assert_replayable

_ZONE = (-0.021588, 0.145066)


@pytest.fixture(scope="module")
def env_cf():
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    return neutral_env(prefix_steps=0, geom="POINT")


def _state(cx=-0.115, cy=0.115, **kw):
    return CoinInitialState(coin_position=(cx, cy, TABLE_Z),
                            target_position=(_ZONE[0], _ZONE[1], 0.004), zone_half=0.04, **kw)


# ── schema validation (fail loud, §2) ────────────────────────────────────────────────────────────────────────────────
def test_valid_state_roundtrips_through_dict() -> None:
    st = _state()
    st.validate()
    assert CoinInitialState.from_dict(st.to_dict()).problem_hash() == st.problem_hash()


def test_reject_wrong_arm_dim() -> None:
    with pytest.raises(InvalidInitialState, match="arm_qpos"):
        CoinInitialState(arm_qpos=(0.0, 0.0, 0.0)).validate()


def test_reject_non_finite() -> None:
    with pytest.raises(InvalidInitialState, match="non-finite"):
        _state(cx=float("nan")).validate()


def test_reject_non_planar_quaternion() -> None:
    r2 = float(np.sqrt(0.5))                                          # a unit quaternion about the x-axis (tips out of plane)
    with pytest.raises(InvalidInitialState, match="yaw-only"):
        _state(coin_quaternion=(r2, r2, 0.0, 0.0)).validate()


def test_reject_coin_off_table_plane() -> None:
    with pytest.raises(InvalidInitialState, match="table plane"):
        CoinInitialState(coin_position=(0.0, 0.1, TABLE_Z + 0.05)).validate()


def test_reject_control_dt_mismatch() -> None:
    with pytest.raises(InvalidInitialState, match="control_dt"):
        _state(control_dt=0.02).validate()


def test_yaw_quaternion_roundtrip() -> None:
    st = _state(coin_quaternion=(np.cos(0.3), 0.0, 0.0, np.sin(0.3)))
    assert abs(st.coin_yaw - 0.6) < 1e-9


# ── extract / apply bit-identity + problem hash (§1A/§5) ──────────────────────────────────────────────────────────────
def test_extract_apply_is_bit_identical(env_cf) -> None:
    env, cf = env_cf
    env.set_stage(0)
    env.reset(seed=1011)
    st = extract_initial_state(env, cf)
    qpos0, qvel0, zone0 = cf._env.data.qpos.copy(), cf._env.data.qvel.copy(), (cf._env._zone_x, cf._env._zone_y)
    apply_initial_state(env, cf, st, seed_hint=0)
    assert np.abs(cf._env.data.qpos - qpos0).max() == 0.0
    assert np.abs(cf._env.data.qvel - qvel0).max() == 0.0
    assert abs(cf._env._zone_x - zone0[0]) + abs(cf._env._zone_y - zone0[1]) == 0.0


def test_problem_hash_is_deterministic_and_pose_sensitive(env_cf) -> None:
    a = _state(cx=-0.115, cy=0.115)
    assert a.problem_hash() == _state(cx=-0.115, cy=0.115).problem_hash()
    assert a.problem_hash() != _state(cx=-0.114, cy=0.115).problem_hash()


def test_seed_1011_problem_matches_release(env_cf) -> None:
    """The seed-1011 canonical problem is stable: neutral arm, the known coin, clearance ≈ +0.079."""
    env, cf = env_cf
    env.set_stage(0)
    env.reset(seed=1011)
    st = extract_initial_state(env, cf)
    assert st.arm_qpos == (0.0, 0.0, 0.0, 0.0)
    assert np.allclose(st.coin_position[:2], (-0.158877, 0.121579), atol=1e-5)
    r = analyze_reachability(env, cf)
    assert abs(r.signed_clearance - 0.079284) < 1e-4
    assert not (r.initial_contact["left"] or r.initial_contact["right"])


# ── fail-loud replay gates (§2) ──────────────────────────────────────────────────────────────────────────────────────
def test_assert_replayable_rejects_embodiment_mismatch(env_cf) -> None:
    env, cf = env_cf
    st = _state()
    apply_initial_state(env, cf, st, seed_hint=0)
    st_wrong = CoinInitialState.from_dict({**st.to_dict(), "embodiment": "RING"})
    with pytest.raises(InvalidInitialState, match="embodiment"):
        assert_replayable(env, cf, st_wrong, embodiment="POINT", neutral_start=True, require_checkpoints=False)


def test_assert_replayable_rejects_target_overlap(env_cf) -> None:
    env, cf = env_cf
    st = CoinInitialState(coin_position=(0.0, 0.11, TABLE_Z), target_position=(0.0, 0.10, 0.004), zone_half=0.04)
    apply_initial_state(env, cf, st, seed_hint=0)
    with pytest.raises(InvalidInitialState, match="overlaps the target"):
        assert_replayable(env, cf, st, embodiment="POINT", neutral_start=True, require_checkpoints=False)


def test_reachability_reports_geometry(env_cf) -> None:
    env, cf = env_cf
    st = _state()
    apply_initial_state(env, cf, st, seed_hint=0)
    r = analyze_reachability(env, cf)
    assert r.signed_clearance > 0 and r.coin_reachable and r.collision_free
    assert set(r.fingertip_distances) == {"left", "right"}


# ── deterministic-problem vs exact-state equivalence (§5) — a short 1-rep rollout must be bit-identical ───────────────
def test_seed_and_exact_rollout_are_bit_identical(env_cf) -> None:
    env, cf = env_cf
    from hymeko_rl.coin_delivery.fixed_position_replay import build_actors, composed_rollout
    approach, tfn = build_actors("P4_E_APPROACH_HANDOFF")
    env.set_stage(0)
    env.reset(seed=1011)
    seed_tr = composed_rollout(env, cf, approach, tfn, grasp_hold=1, contact_window=20)
    env.set_stage(0)
    env.reset(seed=1011)
    st = extract_initial_state(env, cf)
    apply_initial_state(env, cf, st, seed_hint=0)
    exact_tr = composed_rollout(env, cf, approach, tfn, grasp_hold=1, contact_window=20)
    assert seed_tr.trajectory_hash() == exact_tr.trajectory_hash()
    assert seed_tr.strict_delivered == exact_tr.strict_delivered
