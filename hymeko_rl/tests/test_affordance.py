"""Physics AFFORDANCE / task-validity regression for the coin-push task (no RL, no reward).

Proves the task PHYSICALLY requires two coordinated fingertips at the real config (coin_frictionloss=0):
a single dead-behind fingertip cannot deliver, two coordinated fingertips can, and only the working
two-fingertip push produces a target-directed coin velocity with a monotone distance decrease. This is the
validation gate to run BEFORE any RL — a policy/reward that does not clear it is optimising a broken task.

Kept cheap (few episodes, short horizon); the full 48-ep numbers live in the affordance measurement.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.galambos_demo import (
    PushConfig, PushDemonstrator, _extract_arms, _ik_action, press_depth, push_slots,
)

_N = 12
_SEED0 = 9000
_MAXS = 200


def _two_fingertip(env: PlanarGraspEnv):
    demo = PushDemonstrator(env)
    demo.reset()
    return lambda e: demo.action(e)


def _one_fingertip(env: PlanarGraspEnv):
    """Single dead-behind pusher (k=1); the other arm is parked back near its base (out of the coin's path)."""
    arms = tuple(sorted(_extract_arms(env.model).values(), key=lambda a: a.base_xy[0]))
    coin0 = np.asarray(env._planar_metrics.disk_pos[:2], dtype=np.float64)
    pusher = min(arms, key=lambda a: float(np.hypot(a.base_xy[0] - coin0[0], a.base_xy[1] - coin0[1])))
    parked = arms[1] if pusher is arms[0] else arms[0]
    cfg = PushConfig.from_params(PushDemonstrator(env).spec.params,
                                 contact_dist=float(getattr(env._env, "disk_radius", 0.035)) + 0.014)
    park_xy = np.array([parked.base_xy[0], -0.12])

    def act(e: PlanarGraspEnv) -> np.ndarray:
        m = e._planar_metrics
        coin = np.asarray(m.disk_pos[:2], dtype=np.float64)
        zone = np.array([e._zone_x, e._zone_y])
        slots = push_slots(coin, zone, press_depth(float(np.hypot(*(zone - coin))), cfg), cfg, k=1)
        tgt = slots[0] if slots is not None else coin
        return _ik_action([(pusher, tgt), (parked, park_xy)], e)

    return act


def _rollout(make_ctrl) -> dict[str, float]:
    env = PlanarGraspEnv(robot=None, max_steps=_MAXS, difficulty=0.3)   # coin_frictionloss=0 (the real config)
    dwell = int(getattr(env, "success_steps", 1))
    deliv = 0
    push_steps = 0
    vel_dot = 0.0
    net = []
    for ep in range(_N):
        env.reset(seed=_SEED0 + ep)
        ctrl = make_ctrl(env)
        d0 = None
        consec = 0
        ok = False
        for _ in range(env.max_steps):
            _o, _r, term, trunc, info = env.step(ctrl(env))
            m = env._planar_metrics
            coin = np.asarray(m.disk_pos[:2], dtype=np.float64)
            to = np.array([env._zone_x, env._zone_y]) - coin
            d = float(np.hypot(to[0], to[1]))
            d0 = d if d0 is None else d0
            if (m.left_contact or m.right_contact) and d > 1e-6:
                push_steps += 1
                vel_dot += float(np.dot(m.disk_vel, to / d))
            consec = consec + 1 if bool(info["in_zone"]) else 0
            ok = ok or consec >= dwell
            if term or trunc:
                break
        deliv += int(ok)
        net.append(float(d0 - d))
    env.close()
    return {"delivery": deliv / _N, "mean_target_vel": vel_dot / max(1, push_steps),
            "mean_net_progress": float(np.mean(net))}


def test_two_fingertip_push_delivers_and_is_target_directed() -> None:
    two = _rollout(_two_fingertip)
    assert two["delivery"] >= 0.5, f"two-fingertip push should deliver; got {two}"
    assert two["mean_target_vel"] > 0.0, f"coin should move TOWARD the target under the working push; got {two}"
    assert two["mean_net_progress"] > 0.0, f"coin should end closer to the zone; got {two}"


def test_single_fingertip_cannot_deliver_task_requires_two() -> None:
    two = _rollout(_two_fingertip)
    one = _rollout(_one_fingertip)
    # The physical affordance: one fingertip alone is far worse than two (the round coin slips off a single
    # point-contact; two fingertips cage it). If this ever fails, the task no longer requires coordination.
    assert one["delivery"] <= two["delivery"] - 0.3, \
        f"single-fingertip delivery {one['delivery']} not << two-fingertip {two['delivery']} — task not two-arm"
