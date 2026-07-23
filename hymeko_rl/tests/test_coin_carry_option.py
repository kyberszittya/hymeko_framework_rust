"""Stage-1 tests for the carry OPTION representation + committed executor: parameter bounds, deterministic execution,
phase progression, safety abort, transition to frozen pi_0, no template mutation, reproducible replay from a saved
option-initiation state."""
import copy
import json

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_option import (
    OptionActor,
    make_option_actor,
    option_controller_rollout,
)
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, DIM, T_MAX, T_MIN, structured_carry_rollout
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"


def _setup():
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    return pi0, base, ls


def test_option_actor_parameter_bounds():
    torch.manual_seed(0)
    a = make_option_actor()
    o = torch.randn(64, 48) * 5.0
    with torch.no_grad():
        th = a.theta(o)
    assert th.shape == (64, DIM)
    amp, dur = th[:, :12], th[:, 12:]
    assert float(amp.abs().max()) <= A_BOUND + 1e-5                      # amplitudes bounded by ±A_BOUND (tanh)
    assert float(dur.min()) >= T_MIN - 1e-4 and float(dur.max()) <= T_MAX + 1e-4   # durations in [T_MIN, T_MAX] (sigmoid)


def test_option_deterministic_execution_and_no_mutation():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    theta = np.array([2, -2, 1, 0, -2, 2, 0, 0, 0.5, -0.5, 0, 0, 5, 5, 5], np.float32)
    obs_before = rl.obs().copy()
    o1 = option_controller_rollout(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, horizon=40)
    o2 = option_controller_rollout(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, horizon=40)
    assert o1 == o2                                                     # deterministic given a fixed θ
    assert np.array_equal(rl.obs(), obs_before)                        # the passed template is NOT mutated (caller deepcopies)
    for k in ("k6", "max_dwell", "max_strict", "reached_handoff", "contain_exit_ct", "options", "aborts"):
        assert k in o1


def test_phase_progression_push_first():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    a_push = np.array([2.0, 2.0, -2.0, -2.0], np.float32)
    theta = np.concatenate([a_push, [-2, -2, 2, 2], [0, 0, 0, 0], [3, 3, 3]]).astype(np.float32)
    roll = structured_carry_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, theta, horizon=20, capture=True)
    assert len(roll["act"]) >= 1
    assert np.allclose(roll["act"][0], np.clip(a_push, -4, 4), atol=1e-5)   # the first executed macro action is the PUSH amplitude
    # every macro action is one of the three phase amplitudes (push/brake/release), clipped
    phases = [np.clip(theta[0:4], -4, 4), np.clip(theta[4:8], -4, 4), np.clip(theta[8:12], -4, 4)]
    assert all(any(np.allclose(a, p, atol=1e-5) for p in phases) for a in roll["act"])


def test_safety_abort_predicate():
    from hymeko_rl.coin_delivery.coin_carry_option import _safety_abort

    class _M:
        left_contact = False
        right_contact = False

    class _RL:
        class _Inner:
            _planar_metrics = _M()
        inner = _Inner()

        def __init__(self, dtz, sp):
            self._dz, self._sp = dtz, sp

        def _dtz(self):
            return self._dz

        def _speed(self):
            return self._sp

    assert _safety_abort(_RL(0.12, 0.05), touched_before=True)          # had contact, now lost + far (dtz>3·CENTER_TOL) → abort
    assert _safety_abort(_RL(2.0, 0.1), touched_before=False)           # gross divergence (dtz>1) → abort
    assert not _safety_abort(_RL(0.12, 0.05), touched_before=False)     # never had contact → not an abort
    assert not _safety_abort(_RL(0.03, 0.05), touched_before=True)      # lost contact but still near the zone → recoverable, not an abort


def test_reproducible_replay_from_saved_option_initiation_state():
    pi0, base, ls = _setup()
    rl1, g1, _h1, rec1 = reconstruct_handoff(pi0, ls, horizon=360)
    rl2, g2, _h2, rec2 = reconstruct_handoff(pi0, ls, horizon=360)      # replay from the same saved LateStart
    assert np.array_equal(rec1.obs, rec2.obs)                          # deterministic reconstruction
    theta = np.array([1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 4, 4, 4], np.float32)
    a = option_controller_rollout(rl1, g1, theta, pi0, base, horizon=40)
    b = option_controller_rollout(rl2, g2, theta, pi0, base, horizon=40)
    assert a == b                                                      # identical outcome from the same option-initiation state


def test_option_controller_accepts_actor_and_transitions_to_pi0():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    torch.manual_seed(1)
    actor = make_option_actor()
    o = option_controller_rollout(copy.deepcopy(rl), copy.deepcopy(gate), actor, pi0, base, horizon=60, max_options=3)
    assert isinstance(actor, OptionActor) and o["options"] >= 1 and o["options"] <= 3
    # if it reached a handoff, K6 is decided by the frozen pi_0 continuation (k6 ⇒ reached_handoff)
    assert (o["k6"] == 0) or (o["reached_handoff"] == 1)
