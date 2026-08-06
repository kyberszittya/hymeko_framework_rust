"""Tests for COIN-TRANSPORT-1 (object-centric clamp-and-translate transport action space).

Covers: the object-centric action mapping (midpoint ⊥ clamp ⊥ differential), fingertip-target roundtrip, the scramble
controls, the transport primitive families / FSM, handoff snapshot exactness (restore is byte-exact by hash), and the
gate rollout structure — the integrity fixtures the campaign requires (§17/§18).
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hymeko_rl.env.planar_snapshot import restore_planar
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode, make_acq_env
from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.train.coin_transport import (
    Scramble,
    TransportFamily,
    TransportParams,
    TransportPrimitive,
    TState,
    _hash_state,
    eval_transport,
    extract_handoffs,
    fingertip_targets,
    obj_to_env,
    roll_transport,
    scramble_action,
)


def _acq() -> AcqParams:
    d = json.loads(open("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json").read())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


# ── object-centric action mapping ────────────────────────────────────────────────────────────────────────────────────
def test_obj_to_env_axis_positions() -> None:
    a = obj_to_env(np.array([0.3, -0.4]), 0.5, -0.2, 0.9, rotate=0.1)
    assert a[0] == np.float32(0.3) and a[1] == np.float32(-0.4)   # midpoint
    assert a[2] == np.float32(0.5)                                 # aperture
    assert a[3] == np.float32(0.9)                                 # squeeze (clamp)
    assert a[4] == np.float32(-0.2)                                # differential
    assert a[5] == np.float32(0.1)                                 # rotate


def test_obj_to_env_clips() -> None:
    a = obj_to_env(np.array([5.0, -5.0]), 2.0, -3.0, 9.0)
    assert np.all(np.abs(a) <= 1.0 + 1e-6)


def test_fingertip_roundtrip() -> None:
    c = np.array([0.1, 0.2]); d = np.array([0.03, -0.01])
    pL, pR = fingertip_targets(c, d)
    assert np.allclose((pL + pR) / 2, c)                          # midpoint recovered
    assert np.allclose(pL - pR, d)                                # differential recovered


# ── scramble controls ────────────────────────────────────────────────────────────────────────────────────────────────
def test_scramble_s0_identity() -> None:
    a = np.array([0.3, -0.4, 0.5, 0.9, -0.2, 0.1], np.float32)
    assert np.allclose(scramble_action(a, Scramble.S0_CORRECT, np.random.default_rng(0)), a)


def test_scramble_aperture_sign() -> None:
    a = np.array([0.3, -0.4, 0.5, 0.9, -0.2, 0.1], np.float32)
    s = scramble_action(a, Scramble.S2_APERTURE_SIGN, np.random.default_rng(0))
    assert s[2] == np.float32(-0.5)                                # aperture inverted, magnitude matched


def test_scramble_differential_sign() -> None:
    a = np.array([0.3, -0.4, 0.5, 0.9, -0.2, 0.1], np.float32)
    s = scramble_action(a, Scramble.S3_DIFFERENTIAL, np.random.default_rng(0))
    assert s[4] == np.float32(0.2)                                 # differential inverted


def test_scramble_random_matched_magnitude() -> None:
    a = np.array([0.3, -0.4, 0.0, 0.0, 0.0, 0.0], np.float32)
    s = scramble_action(a, Scramble.S5_RANDOM, np.random.default_rng(0))
    assert abs(np.linalg.norm(s[:2]) - np.linalg.norm(a[:2])) < 1e-5   # midpoint amplitude preserved


# ── transport primitive families / FSM ───────────────────────────────────────────────────────────────────────────────
def test_t0_is_grasp_carry_shape() -> None:
    env = make_acq_env()
    env.reset(seed=64_010)
    prim = TransportPrimitive(TransportFamily.T0_GRASP_CARRY, TransportParams())
    a = prim.action(env._env, np.zeros(41, np.float32))
    assert a.shape == (6,) and a[3] == np.float32(0.8)            # grasp_carry squeeze


def test_t4_fsm_starts_in_clamp_verify() -> None:
    prim = TransportPrimitive(TransportFamily.T4_STAGED, TransportParams(h_clearance=3))
    assert prim.state == TState.HANDOFF
    prim.reset()
    assert prim.state == TState.HANDOFF


def test_t5_recovery_latch() -> None:
    prim = TransportPrimitive(TransportFamily.T5_RECOVERY, TransportParams())
    assert prim.recovered is False


def test_transport_params_from_unit_ranges() -> None:
    lo = TransportParams.from_unit(np.zeros(5))
    hi = TransportParams.from_unit(np.ones(5))
    assert lo.v_translate == 0.3 and abs(hi.v_translate - 1.0) < 1e-9
    assert lo.a_target == 0.3 and abs(hi.a_target - 1.0) < 1e-9


# ── handoff snapshot exactness (integrity §17) ───────────────────────────────────────────────────────────────────────
def test_handoff_extraction_and_restore_is_exact() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    assert len(H) >= 1
    h = H[0]
    # restoring the snapshot reproduces the exact state hash (byte-exact restore)
    restore_planar(env._env, h.snap)
    assert _hash_state(env._env) == h.state_hash


def test_roll_transport_gate_structure() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    r = roll_transport(env, H[0], TransportFamily.T1_RIGID, TransportParams(), horizon=40)
    for k in ("g0", "g1", "g2", "g3_zone", "g4_center", "g5_release", "highest_gate", "progress"):
        assert k in r


def test_eval_transport_keys() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    ev = eval_transport(env, H, TransportFamily.T0_GRASP_CARRY, TransportParams(), horizon=40)
    for k in ("zone_entry", "center_reach", "clamp_retention", "transport_progress"):
        assert k in ev
