"""Central safety arbiter on the push controller (ArbitratedPushDemonstrator) — veto-logic unit tests.

These check the arbiter's three joint-state decisions (pass-through / hold-on-overshoot / back-off-on-recede);
they do NOT assert a delivery improvement (the measured delivery was flat vs baseline — 0.854 → 0.854).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.galambos_demo import (
    ArbitratedPushDemonstrator, DeliveryFloorCriterion, PhasePushController,
    PushControllerParams, PushObs, _ik_action, push_slots,
)


def _pressing_demo() -> tuple[PlanarGraspEnv, ArbitratedPushDemonstrator, PushObs, np.ndarray]:
    """Drive the controller until it is in a pressing phase (law == pressed_slots), STEPPING the env each
    tick so the FSM actually progresses; return the env, demo, current obs, and the baseline candidate."""
    env = PlanarGraspEnv(robot=None, max_steps=200, difficulty=0.3)
    env.reset(seed=1)
    demo = ArbitratedPushDemonstrator(env, v_cap=0.05, v_eps=0.01, backoff=0.3)
    obs = PushObs.from_env(env, demo._arms)
    base = np.asarray(demo._decide(obs))
    for _ in range(200):
        if demo.spec.phase(demo._phase).law == "pressed_slots" and demo._assign is not None:
            break
        env.step(_ik_action(zip(demo._arms, base), env))   # advance physics with the decided action
        obs = PushObs.from_env(env, demo._arms)
        base = np.asarray(demo._decide(obs))               # advances the FSM at the new state
    assert demo.spec.phase(demo._phase).law == "pressed_slots", "setup never reached a pressing phase"
    return env, demo, obs, base


def test_arbiter_passthrough_when_coin_still() -> None:
    env, demo, obs, base = _pressing_demo()
    env._planar_metrics = replace(env._planar_metrics, disk_vel=np.zeros(2, dtype=np.float32))
    out, reason = demo._arbitrate(env, obs, base)
    assert reason == "" and np.allclose(out, base)         # near-zero coin velocity → baseline passes through


def test_arbiter_backs_off_when_coin_recedes() -> None:
    env, demo, obs, base = _pressing_demo()
    assert obs.u is not None
    env._planar_metrics = replace(env._planar_metrics, disk_vel=(-0.10 * obs.u).astype(np.float32))
    out, reason = demo._arbitrate(env, obs, base)
    assert reason == "backoff_recede"                      # coin moving away from the zone → press backed off


def test_arbiter_holds_on_overshoot_near_zone() -> None:
    env, demo, _obs, base = _pressing_demo()
    coin = np.array([0.0, 0.10])
    zone = np.array([0.0, 0.115])                          # dist 0.015 < brake_dist
    u = (zone - coin) / float(np.linalg.norm(zone - coin))
    tips = np.zeros((len(demo._arms), 2))
    near = PushObs(coin=coin, zone=zone, tips=tips, dist=float(np.linalg.norm(zone - coin)), u=u)
    env._planar_metrics = replace(env._planar_metrics, disk_vel=(0.10 * u).astype(np.float32))
    out, reason = demo._arbitrate(env, near, base)
    assert reason == "hold_overshoot"                      # fast toward zone near it → hold at zero press
    hold = push_slots(coin, zone, 0.0, demo.cfg, k=len(demo._arms))
    assert hold is not None and np.allclose(out, hold[demo._assign])


def test_arbiter_action_contract_matches_baseline_shape() -> None:
    # Drop-in contract: action(env) returns a bounded n_actions command, same as the baseline controller.
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=3)
    demo = ArbitratedPushDemonstrator(env)
    a = demo.action(env)
    assert a.shape == (env.n_actions,)
    assert bool(np.all(a >= env.action_space.low - 1e-5) and np.all(a <= env.action_space.high + 1e-5))


def test_phase_controller_params_are_bounded_five_scalars() -> None:
    params = PushControllerParams.from_vector([-10.0, 99.0, 3.0, 10.0, -1.0])
    assert params.contact_offset == -0.010
    assert params.push_gain == 1.50
    assert params.direction_correction == 0.35
    assert params.brake_threshold == 0.120
    assert params.release_threshold == 0.004
    try:
        PushControllerParams.from_vector([0.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("learned controller interface must be exactly five high-level parameters")


def test_phase_controller_action_shape_and_phase_names() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)
    env.reset(seed=4)
    ctrl = PhasePushController(env)
    assert ctrl.PHASES == ("APPROACH", "CONTACT", "PUSH", "BRAKE", "DONE")
    ctrl.set_param_vector([0.0, 1.0, 0.0, 0.06, 0.01])
    action = ctrl.action(env)
    assert action.shape == (env.n_actions,)
    assert bool(np.all(action >= env.action_space.low - 1e-5) and np.all(action <= env.action_space.high + 1e-5))
    assert not hasattr(ctrl, "action_mean")                    # no raw neural-policy/action interface


def test_phase_controller_arbiter_falls_back_when_candidate_recedes() -> None:
    env = PlanarGraspEnv(robot=None, max_steps=80, difficulty=0.3)
    env.reset(seed=5)
    ctrl = PhasePushController(env, params=PushControllerParams(push_gain=1.4), v_eps=0.01)
    obs = PushObs.from_env(env, ctrl._arms)
    ctrl.phase = "PUSH"
    safe = ctrl._targets_for(obs, ctrl.safe_params, ctrl.phase)
    candidate = ctrl._targets_for(obs, ctrl.params, ctrl.phase)
    assert obs.u is not None
    from dataclasses import replace
    env._planar_metrics = replace(env._planar_metrics, disk_vel=(-0.10 * obs.u).astype(np.float32))
    out, reason = ctrl._arbitrate(env, obs, candidate, safe)
    assert reason == "fallback_recede"
    assert np.allclose(out, safe)


def test_delivery_floor_criterion_is_scripted_minus_three_points() -> None:
    crit = DeliveryFloorCriterion(scripted_delivery=0.84, tolerance=0.03)
    assert crit.floor == 0.81
    assert crit.accepts(0.82, worst_seed_delivery=0.81)
    assert not crit.accepts(0.80)
    assert not crit.accepts(0.84, worst_seed_delivery=0.79)
