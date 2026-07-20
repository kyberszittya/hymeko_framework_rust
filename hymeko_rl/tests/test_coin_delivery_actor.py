"""Tests for COIN-DELIVERY-ACTOR-1 (cooperative push/plow delivery actors + monitor + attribution + reward).

Covers: the actor action shapes, the impulse attribution L/R split, the strict valid-delivery monitor (rejects initial
success / requires dwell+fingertip-attribution / rejects body shove), the mechanism classifier (one-finger bulldoze,
body shove), and the delivery-aligned reward ordering R(clean) > R(partial) > R(stall) and clean > body-shove.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.experiments.pedc_selection import _env
from hymeko_rl.train.coin_delivery_actor import (
    Attribution,
    DeliveryActor,
    DeliveryReward,
    Mechanism,
    _attribute_step,
    _finalize_attribution,
    _valid_delivery,
    classify_mechanism,
    delivery_step_reward,
    rollout_delivery,
)


class _M:  # minimal metrics stub for the reward
    def __init__(self, left, right):
        self.left_contact, self.right_contact = left, right


# --- actors -----------------------------------------------------------------------------------------------------
def test_all_actors_emit_6dof_bounded_actions() -> None:
    env = _env()
    env.reset(seed=64_000)
    inner = env._env
    for a in DeliveryActor:
        from hymeko_rl.train.coin_delivery_actor import actor_action
        act = actor_action(inner, 0, a)
        assert act.shape == (6,) and np.all(np.abs(act) <= 1.0 + 1e-6)


# --- attribution: impulse-weighted L/R split --------------------------------------------------------------------
def test_attribution_splits_LR_by_impulse() -> None:
    acc = {"L": 0.0, "R": 0.0, "body": 0.0, "free": 0.0}
    _attribute_step(0.10, fl=3.0, fr=1.0, body=False, acc=acc)   # both contact → 75/25 split
    att = _finalize_attribution(acc)
    assert abs(att.alpha_L - 0.75) < 1e-6 and abs(att.alpha_R - 0.25) < 1e-6
    assert att.fingertip_fraction == 1.0 and att.alpha_support == 0.0


def test_attribution_body_and_free() -> None:
    acc = {"L": 0.0, "R": 0.0, "body": 0.0, "free": 0.0}
    _attribute_step(0.05, fl=0.0, fr=0.0, body=True, acc=acc)     # body-only push
    _attribute_step(0.05, fl=0.0, fr=0.0, body=False, acc=acc)    # no contact → free
    att = _finalize_attribution(acc)
    assert abs(att.alpha_body - 0.5) < 1e-6 and abs(att.alpha_free - 0.5) < 1e-6


# --- mechanism classifier ---------------------------------------------------------------------------------------
def test_classifier_one_finger_bulldoze() -> None:
    att = Attribution(alpha_L=0.95, alpha_R=0.02, alpha_body=0.0, alpha_support=0.0, alpha_free=0.03, total_progress=0.1)
    m = classify_mechanism(DeliveryActor.A0_SYM_PUSH, att, made_progress=True, initial_success=False, both_frac=0.2)
    assert m is Mechanism.ONE_FINGER_BULLDOZE                     # one tip does ~all the work → not clean cooperative


def test_classifier_body_shove_and_initial() -> None:
    shove = Attribution(0.1, 0.1, 0.7, 0.0, 0.1, 0.1)
    assert classify_mechanism(DeliveryActor.A0_SYM_PUSH, shove, made_progress=True, initial_success=False, both_frac=0.5) is Mechanism.BODY_SHOVE
    clean = Attribution(0.4, 0.4, 0.0, 0.0, 0.2, 0.1)
    assert classify_mechanism(DeliveryActor.A0_SYM_PUSH, clean, made_progress=True, initial_success=True, both_frac=0.5) is Mechanism.INVALID_INITIAL


# --- strict valid-delivery monitor ------------------------------------------------------------------------------
def _trace(initial=False, loose=True, dwell=8, vel=0.05):
    """A minimal public-trace stand-in exposing exactly the fields the strict monitor reads."""
    import types
    return types.SimpleNamespace(initial_success=initial, loose=loose, best_dwell=dwell, settle_vel=vel)


def test_monitor_requires_dwell_and_fingertip_and_rejects_initial() -> None:
    clean_att = Attribution(0.4, 0.4, 0.0, 0.0, 0.2, 0.1)
    assert _valid_delivery(_trace(), clean_att, Mechanism.SYM_PUSH) is True
    assert _valid_delivery(_trace(initial=True), clean_att, Mechanism.SYM_PUSH) is False   # initial success rejected
    assert _valid_delivery(_trace(dwell=2), clean_att, Mechanism.SYM_PUSH) is False         # dwell too short
    assert _valid_delivery(_trace(vel=0.9), clean_att, Mechanism.SYM_PUSH) is False         # not settled
    low_ft = Attribution(0.1, 0.05, 0.0, 0.0, 0.85, 0.1)
    assert _valid_delivery(_trace(), low_ft, Mechanism.FREE_MOTION) is False                # not fingertip-attributed


def test_monitor_rejects_body_shove_mechanism() -> None:
    shove = Attribution(0.2, 0.2, 0.5, 0.0, 0.1, 0.1)
    assert _valid_delivery(_trace(), shove, Mechanism.BODY_SHOVE) is False


# --- delivery-aligned reward ordering ---------------------------------------------------------------------------
def test_reward_ordering_clean_gt_partial_gt_stall() -> None:
    cfg = DeliveryReward()
    m = _M(True, True)
    # clean delivery: progress + one-time delivery bonus
    r_clean = sum(delivery_step_reward(m, 0.01, t == 0, False, cfg) for t in range(5))
    r_partial = sum(delivery_step_reward(m, 0.01, False, False, cfg) for _ in range(5))     # progress, no delivery
    r_stall = sum(delivery_step_reward(m, 0.0, False, False, cfg) for _ in range(5))         # no progress
    assert r_clean > r_partial > r_stall


def test_reward_clean_gt_body_shove() -> None:
    cfg = DeliveryReward()
    m = _M(True, True)
    r_clean = sum(delivery_step_reward(m, 0.01, t == 0, False, cfg) for t in range(5))
    r_shove = sum(delivery_step_reward(_M(False, False), 0.01, False, True, cfg) for _ in range(5))
    assert r_clean > r_shove


# --- integration: a rollout produces a classified result with attribution ---------------------------------------
def test_rollout_produces_classified_result() -> None:
    env = _env()
    r = rollout_delivery(env, 64_000, DeliveryActor.A1_VPLOW)
    assert isinstance(r.strict_delivery, bool) and 0.0 <= r.fingertip_fraction <= 1.0
    assert r.mechanism in {m.value for m in Mechanism}
    for k in ("alpha_L", "alpha_R", "alpha_body", "alpha_support", "alpha_free"):
        assert k in r.attribution
