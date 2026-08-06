"""Spring-hop-to-goal — the spring-legged AIBO reaches a forward goal by repeated upright hops.

Certifies: it reaches a designated forward goal; it stays upright every hop (never flips); it makes
monotone forward progress; a farther goal takes more hops; and the forward push stays motor-limited
(the horizontal drive is a real ≤5 N·m actuator, not an injected velocity — while the vertical lift
is the passive spring, covered by the spring_leg tests).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.spring_hop_gait import SpringHopGait
from scenarios.aibo.spring_leg import LEGS, SpringLegSpec, build_spring_legged


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                           max_steps=200)
    return build_spring_legged(env._mjcf, SpringLegSpec(100.0, 0.0, 0.05))


def test_reaches_forward_goal(model: mujoco.MjModel) -> None:
    r = SpringHopGait(model, goal_distance=0.6).run()
    assert r.reached
    assert not r.fell
    assert 0 < r.n_hops <= SpringHopGait(model).max_hops


def test_stays_upright_every_hop(model: mujoco.MjModel) -> None:
    r = SpringHopGait(model, goal_distance=0.6).run()
    assert r.min_upright > 0.8            # the torso stays upright the whole run (no flip / no fall)


def test_makes_monotone_forward_progress(model: mujoco.MjModel) -> None:
    r = SpringHopGait(model, goal_distance=0.6).run()
    xs = r.hop_x
    assert len(xs) >= 2
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:]))   # never moves backward hop-to-hop
    assert xs[-1] > xs[0]                                    # net forward


def test_farther_goal_takes_more_hops(model: mujoco.MjModel) -> None:
    near = SpringHopGait(model, goal_distance=0.4).run()
    far = SpringHopGait(model, goal_distance=0.8).run()
    assert far.reached and near.reached
    assert far.n_hops > near.n_hops


def test_forward_push_stays_motor_limited(model: mujoco.MjModel) -> None:
    gait = SpringHopGait(model, goal_distance=0.6, hip_tau_cap=5.0, motor_tau_cap=8.0)
    hip_dofs = {int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                       f"hip_flex_{leg}")]) for leg in LEGS}
    peak_push, peak_any = 0.0, 0.0

    def hook(d, phase, hop):
        nonlocal peak_push, peak_any
        for dof in hip_dofs:
            t = abs(float(d.qfrc_applied[dof]))
            peak_any = max(peak_any, t)
            if phase == "launch":
                peak_push = max(peak_push, t)

    gait.run(on_step=hook)
    assert peak_push <= 5.0 + 1e-6       # the forward drive never exceeds hip_tau_cap
    assert peak_any <= 8.0 + 1e-6        # NO leg actuator torque exceeds the realistic motor cap


def test_catch_preserves_forward_momentum(model: mujoco.MjModel) -> None:
    # the catch must NOT brake to a stand between strides — the forward velocity stays alive so the
    # gait flows forward with its momentum (the fix for "stops to stabilise instead of continuing").
    bv = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base")])
    catch_vx = []

    def hook(d, phase, hop):
        if phase == "catch":
            catch_vx.append(float(d.qvel[bv]))

    SpringHopGait(model, goal_distance=1.2).run(on_step=hook)
    catch_vx = np.array(catch_vx)
    assert catch_vx.mean() > 0.3              # net forward momentum carried through the catch
    assert (catch_vx > 0.05).mean() > 0.7     # forward-moving in the large majority of catch steps


def test_goal_distance_must_be_positive(model: mujoco.MjModel) -> None:
    with pytest.raises(ValueError):
        SpringHopGait(model, goal_distance=0.0)


def test_run_is_deterministic(model: mujoco.MjModel) -> None:
    a = SpringHopGait(model, goal_distance=0.6).run()
    b = SpringHopGait(model, goal_distance=0.6).run()
    assert a.n_hops == b.n_hops and a.forward == b.forward
