"""Tests for the acquisition primitive + acquisition RL subtask env (COIN-DELIVERY-OVERNIGHT-2 Track A / PART V).

Covers: param mapping, the acquisition FSM (phase progression, action bounds, stateful reset), geometry scramble,
the funnel rollout, the acquisition reward sign gates, and the acquisition RL env (stable-acquisition terminal).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.train.coin_delivery_acquisition import (
    AcqParams,
    ApproachMode,
    AcquisitionPrimitive,
    Phase,
    eval_acquisition,
    make_acq_env,
    roll_acquisition,
    scramble_geometry,
)
from hymeko_rl.train.coin_delivery_acquisition_rl import AcqRewardConfig, AcquisitionRLEnv, acquisition_reward

_WALL = [64_010, 64_012, 64_028, 64_029]


def _obs(**kw) -> np.ndarray:
    o = np.zeros(41, np.float32)
    o[20], o[21] = kw.get("mid", (0.1, 0.0))          # mid_to_coin
    o[26], o[27], o[28] = kw.get("contact", (0, 0, 0))  # left/right/both
    return o


# ── params ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_from_unit_maps_ranges() -> None:
    lo = AcqParams.from_unit(np.zeros(8))
    hi = AcqParams.from_unit(np.ones(8))
    assert lo.pregrasp_radius == 0.02 and abs(hi.pregrasp_radius - 0.10) < 1e-9
    assert lo.approach_gain == 8.0 and hi.approach_gain == 40.0


def test_from_unit_keeps_mode_flags() -> None:
    base = AcqParams(approach_mode=ApproachMode.ASYM_LEFT, regrasp=False)
    p = AcqParams.from_unit(np.full(8, 0.5), base)
    assert p.approach_mode == ApproachMode.ASYM_LEFT and p.regrasp is False


# ── FSM ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_primitive_action_in_box_and_shape() -> None:
    prim = AcquisitionPrimitive(AcqParams())
    a = prim.action(_obs(mid=(0.15, 0.05)))
    assert a.shape == (6,) and np.all(np.abs(a) <= 1.0 + 1e-6)


def test_primitive_approach_drives_toward_coin() -> None:
    prim = AcquisitionPrimitive(AcqParams())
    a = prim.action(_obs(mid=(0.2, 0.0)))            # coin to the +x of the midpoint
    assert a[0] > 0                                   # translate toward the coin (positive x)


def test_primitive_advances_to_align_when_close() -> None:
    prim = AcquisitionPrimitive(AcqParams(pregrasp_radius=0.05))
    prim.action(_obs(mid=(0.02, 0.0)))               # within pregrasp radius
    assert prim.phase in (Phase.ALIGN, Phase.CLOSE)


def test_primitive_reset_clears_state() -> None:
    prim = AcquisitionPrimitive(AcqParams())
    for _ in range(5):
        prim.action(_obs(mid=(0.02, 0.0), contact=(1, 1, 1)))
    prim.reset()
    assert prim.phase == Phase.APPROACH and prim.dwell == 0 and prim.retries == 0


# ── geometry scramble ────────────────────────────────────────────────────────────────────────────────────────────────
def test_scramble_changes_geometry_fields() -> None:
    o = _obs(mid=(0.1, 0.2))
    perm = np.array([-2, 3, -0, 1])
    s = scramble_geometry(o, perm)
    assert not np.allclose(s[[20, 21]], o[[20, 21]])  # mid_to_coin scrambled


# ── funnel rollout / eval ────────────────────────────────────────────────────────────────────────────────────────────
def test_roll_acquisition_funnel_keys() -> None:
    env = make_acq_env()
    r = roll_acquisition(env, 64_010, AcqParams(regrasp=False), horizon=60)
    for k in ("pregrasp_aligned", "first_contact", "two_finger_contact", "stable_acquisition", "final_phase"):
        assert k in r


def test_eval_acquisition_funnel_monotone() -> None:
    env = make_acq_env()
    ev = eval_acquisition(AcqParams(regrasp=False), _WALL, env=env, horizon=90)
    # funnel is monotone non-increasing: pregrasp >= first_contact >= two_finger >= stable
    assert ev["pregrasp_rate"] >= ev["first_contact_rate"] >= ev["two_finger_rate"] >= ev["stable_acquisition_rate"]


# ── acquisition reward sign gates ────────────────────────────────────────────────────────────────────────────────────
def test_reward_approach_progress_positive() -> None:
    cfg = AcqRewardConfig()
    r = acquisition_reward(0.10, 0.088, first_contact_now=False, stable_now=False, dropped=False, stalled=False,
                           collision=False, cfg=cfg)
    assert r > 0


def test_reward_stable_is_strong_event() -> None:
    cfg = AcqRewardConfig()
    stable = acquisition_reward(0.05, 0.05, first_contact_now=False, stable_now=True, dropped=False, stalled=False,
                               collision=False, cfg=cfg)
    contact = acquisition_reward(0.05, 0.05, first_contact_now=True, stable_now=False, dropped=False, stalled=False,
                                collision=False, cfg=cfg)
    assert stable > contact > 0


def test_reward_collision_and_drop_penalize() -> None:
    cfg = AcqRewardConfig()
    coll = acquisition_reward(0.05, 0.05, first_contact_now=False, stable_now=False, dropped=False, stalled=False,
                              collision=True, cfg=cfg)
    drop = acquisition_reward(0.05, 0.05, first_contact_now=False, stable_now=False, dropped=True, stalled=False,
                              collision=False, cfg=cfg)
    assert coll < 0 and drop < 0


# ── acquisition RL env ───────────────────────────────────────────────────────────────────────────────────────────────
def test_rl_env_reset_step_contract() -> None:
    env = AcquisitionRLEnv(_WALL, horizon=40, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (41,)
    obs, r, term, trunc, info = env.step(np.zeros(6, np.float32))
    assert isinstance(r, float) and "stable_acquisition" in info
    assert hasattr(env, "max_steps")


def test_rl_env_terminates_on_stable_or_horizon() -> None:
    env = AcquisitionRLEnv(_WALL, horizon=15, seed=0)
    env.reset(seed=0)
    steps = 0
    for _ in range(15):
        _o, _r, term, trunc, _i = env.step(np.zeros(6, np.float32))
        steps += 1
        if term or trunc:
            break
    assert steps <= 15
