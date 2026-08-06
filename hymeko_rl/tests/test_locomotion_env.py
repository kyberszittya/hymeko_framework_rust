"""Contract tests for the fast-dynamics locomotion envs (`hymeko_rl/env/locomotion_env.py`) + scripted
experts (`locomotion_experts.py`). These lock the *trainer contract* and the physics-stability regressions
(the velocity-servo vehicle fix, fall termination) — NOT the scripted experts' locomotion performance, which
is a measured baseline the RL improves on. Fast: envs are cheap (0.04–0.18 ms/step).

Run: pytest -p no:randomly hymeko_rl/tests/test_locomotion_env.py"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hymeko_rl.env.locomotion_env import (
    SUBSTRATES,
    make_f1tenth,
    make_race_car,
    make_vehicle,
)
from hymeko_rl.env.locomotion_experts import CpgGait, DiffDrivePursuit

# (substrate, expected obs vertices, expected action dim)
_DIMS = [("cheetah", 7, 6), ("humanoid", 13, 12), ("vehicle", 6, 4)]


@pytest.mark.parametrize("name,n_vtx,n_act", _DIMS)
def test_construction_and_spaces(name: str, n_vtx: int, n_act: int) -> None:
    env = SUBSTRATES[name](max_steps=50)
    assert env.observation_space.shape == (n_vtx, 2)
    assert env.action_space.shape == (n_act,)
    assert env.n_actions == n_act
    obs, info = env.reset(seed=0)
    assert obs.shape == (n_vtx, 2) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and "dist" in info


@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_reset_is_seed_deterministic(name: str) -> None:
    o1, _ = SUBSTRATES[name](max_steps=50).reset(seed=7)
    o2, _ = SUBSTRATES[name](max_steps=50).reset(seed=7)
    o3, _ = SUBSTRATES[name](max_steps=50).reset(seed=8)
    assert np.array_equal(o1, o2)              # same seed → identical obs
    assert not np.array_equal(o1, o3)          # different seed → different init noise


@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_step_contract_5tuple(name: str) -> None:
    env = SUBSTRATES[name](max_steps=50)
    env.reset(seed=0)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, r, term, trunc, info = out
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    assert isinstance(r, float) and np.isfinite(r)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert {"dist", "vx", "upright", "x", "step"} <= set(info)


@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_expert_action_valid(name: str) -> None:
    env = SUBSTRATES[name](max_steps=50)
    env.reset(seed=0)
    for _ in range(10):
        a = env.expert_action
        assert a.shape == (env.n_actions,) and a.dtype == np.float32
        assert np.all(np.abs(a) <= 1.0 + 1e-6)   # normalised to [-1, 1]
        env.step(a)


@pytest.mark.parametrize("name", ["cheetah", "humanoid"])
def test_legged_fall_terminates(name: str) -> None:
    """A legged runner driven by random torques must fall (terminate) within the horizon — the fall predicate
    (torso below fall_height OR tipped past flip_cos) is live, so a policy cannot farm reward while collapsed."""
    env = SUBSTRATES[name](max_steps=400)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    terminated = False
    for _ in range(400):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, env.n_actions).astype(np.float32))
        if terminated or truncated:
            break
    assert terminated, "random-torque legged runner should fall (terminate), not survive the full horizon"


def test_vehicle_stays_upright_under_pursuit() -> None:
    """Regression for the velocity-servo fix: torque wheels spun to runaway speed and the reaction torque
    flipped the chassis (upright 1.0 → 0.16 by step 37). With velocity servos the pursuit expert holds the
    chassis upright."""
    env = make_vehicle(max_steps=300)
    env.reset(seed=0)
    uprights = []
    for _ in range(300):
        _, _, term, trunc, info = env.step(env.expert_action)
        uprights.append(info["upright"])
        if term or trunc:
            break
    assert min(uprights) > 0.5, f"vehicle rolled (min upright {min(uprights):.3f}) — velocity-servo regression"


def test_vehicle_makes_forward_progress_and_reaches_waypoint() -> None:
    env = make_vehicle(max_steps=1500)
    env.reset(seed=0)
    x0 = float(env.data.xpos[env.torso, 0])
    for _ in range(1500):
        _, _, term, trunc, _ = env.step(env.expert_action)
        if term or trunc:
            break
    assert env._wp >= 1, "pursuit expert should reach at least the first waypoint"
    assert abs(float(env.data.xpos[env.torso, 0]) - x0) > 0.5, "vehicle should move from spawn"


@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_privileged_state_finite_1d(name: str) -> None:
    env = SUBSTRATES[name](max_steps=50)
    env.reset(seed=0)
    p = env.privileged_state()
    assert p.ndim == 1 and p.dtype == np.float32 and np.isfinite(p).all()


def test_cpg_gait_deterministic_and_bounded() -> None:
    gait = CpgGait.alternating(6, freq=2.0, amp=0.8)

    class _Stub:
        _step = 12
        _dt = 0.002
        frame_skip = 5

    a1, a2 = gait.action(_Stub()), gait.action(_Stub())
    assert np.array_equal(a1, a2) and a1.shape == (6,)
    assert np.all(np.abs(a1) <= 1.0 + 1e-6)


def test_diffdrive_pursuit_steers_toward_waypoint() -> None:
    """Regression for the wheel-side sign bug (vehicle drove to −y for a +y waypoint). A +heading error is a
    CCW turn toward a +y waypoint; on the robot_4wh geometry that needs the −y-side wheels {fl,rl}={1,3} to
    drive FASTER than the +y-side wheels {fr,rr}={0,2}."""
    pursuit = DiffDrivePursuit()

    class _Stub:
        def heading_error(self) -> float:
            return 0.5   # waypoint CCW (to +y)

        def dist_to_goal(self) -> float:
            return 5.0

    a = pursuit.action(_Stub())
    minus_y_side = a[1] + a[3]     # fl + rl → speed up to turn CCW
    plus_y_side = a[0] + a[2]      # fr + rr → slow down
    assert minus_y_side > plus_y_side, "CCW turn toward +y waypoint must drive the −y-side wheels faster"


def test_race_car_reaches_high_speed_upright() -> None:
    """The full-scale race car (long wheelbase, low CG, big wheels) accelerates past 180 km/h and stays
    upright — unlike robot_4wh, which wheelies/flips above ~5 m/s."""
    env = make_race_car(top_speed_kmh=220.0, max_steps=2500)
    env.reset(seed=0)
    dt = env.frame_skip * env._dt
    prev = env.data.xpos[env.torso, :2].copy()
    top = 0.0
    for _ in range(2500):
        _, _, term, trunc, _ = env.step(env.expert_action)
        cur = env.data.xpos[env.torso, :2].copy()
        top = max(top, float(np.hypot(*(cur - prev)) / dt))
        prev = cur
        if term or trunc:
            break
    assert env._torso_uprightness() > 0.5, "race car should stay upright at speed"
    assert top * 3.6 > 180.0, f"race car should exceed 180 km/h (got {top * 3.6:.0f})"


def test_f1tenth_ackermann_control_interface() -> None:
    """F1TENTH exposes the 2-D [throttle, steer] Ackermann interface; the plant rewrite leaves 4 actuators
    (2 rear velocity + 2 front steer), the free-roll front-wheel motors stripped."""
    env = make_f1tenth(max_steps=50)
    assert env.action_space.shape == (2,) and env.n_actions == 2
    assert env.model.nu == 4
    a = env.expert_action
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0 + 1e-6)


def test_f1tenth_numerically_stable_full_lap() -> None:
    """Regression for the light-wheel QACC blowup (390 m/s / NaN from a stiff servo on 0.2 kg wheels). With
    armature on every wheel dof + a drive-torque limit, the F1TENTH stays finite + upright and tracks most of
    its gentle Ackermann lap."""
    env = make_f1tenth(max_steps=4000)
    env.reset(seed=0)
    minup = 1.0
    for _ in range(4000):
        _, _, term, trunc, info = env.step(env.expert_action)
        minup = min(minup, info["upright"])
        assert np.isfinite(env.data.qacc).all(), "F1TENTH QACC blew up (light-wheel instability regression)"
        if term or trunc:
            break
    assert minup > 0.5, f"F1TENTH flipped (min upright {minup:.2f})"
    assert env._wp >= len(env._track) // 2, f"F1TENTH should track most of the lap ({env._wp}/{len(env._track) - 1})"


@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_step_latency_budget(name: str) -> None:
    """Performance gate (§3): median env.step wall under budget. Legged (348-d humanoid) is the worst case."""
    env = SUBSTRATES[name](max_steps=200)
    env.reset(seed=0)
    a = env.action_space.sample()
    for _ in range(10):        # warm up
        env.step(a)
    ts = []
    for _ in range(50):
        t = time.perf_counter()
        env.step(a)
        ts.append(time.perf_counter() - t)
    median_ms = float(np.median(ts) * 1e3)
    assert median_ms < 5.0, f"{name} step median {median_ms:.2f} ms exceeds 5 ms budget"
