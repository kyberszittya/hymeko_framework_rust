"""Goal-reaching quadruped env: explicit base-mode handling, standing, goal/flip termination, the
distance/time/smoothness reward wiring, and a perf budget.

Complements the cross-scenario checks in ``test_scenario_sanity.py`` with the quadruped-specific contract:
the world attachment declared in ``quadruped.hymeko`` is promoted correctly (``free`` -> ``<freejoint>``,
``fixed`` -> welded), the robot holds a stance, reaching the goal terminates, and the reward includes the
fast-and-smooth terms.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hymeko_rl.env.arm_world import emit_arm_mjcf, strip_actuators
from hymeko_rl.env.quadruped_env import GOAL_REWARD, QuadrupedGoalEnv, set_base_mode

_QUAD = "data/robotics/quadruped.hymeko"


# ── explicit, declared base-mode promotion (not auto-injected) ────────────────
def test_set_base_mode_free_makes_freejoint_and_strips_motor() -> None:
    out = set_base_mode(emit_arm_mjcf(_QUAD, name="quad"), "base", "free")
    assert '<freejoint name="base"/>' in out
    assert 'joint="base"' not in out            # base motor stripped (free base is passive)
    assert '<joint name="base"' not in out


def test_set_base_mode_fixed_removes_base_joint() -> None:
    out = set_base_mode(emit_arm_mjcf(_QUAD, name="quad"), "base", "fixed")
    assert '<joint name="base"' not in out and '<freejoint' not in out


def test_set_base_mode_missing_joint_raises() -> None:
    plain = strip_actuators(emit_arm_mjcf(_QUAD, name="quad"), ["base"])
    plain = plain.replace('<joint name="base"', '<joint name="other"')
    with pytest.raises(ValueError, match="base"):
        set_base_mode(plain, "base", "free")


def test_free_and_fixed_dof_counts() -> None:
    free, fixed = QuadrupedGoalEnv(base="free"), QuadrupedGoalEnv(base="fixed")
    assert int(free.model.njnt) == int(fixed.model.njnt) + 1     # free base adds one (free) joint
    assert int(free.model.nu) == int(fixed.model.nu) == 8        # 8 leg motors either way


# ── reward wiring: distance + time + smoothness terms present and finite ──────
def test_goal_reward_terms_present() -> None:
    kinds = [k for k, _ in GOAL_REWARD.terms]
    for required in ("goal_progress", "success_bonus", "time_penalty",
                     "joint_velocity", "joint_acceleration"):
        assert required in kinds, f"goal reward missing {required}"


def test_reward_finite_and_bounded() -> None:
    """The jerk term uses Δq̇ (bounded), so the reward never explodes even under random full-scale action."""
    env = QuadrupedGoalEnv(base="free", max_steps=80)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(80):
        _o, r, term, trunc, _i = env.step(rng.uniform(-1, 1, env.n_actions).astype(np.float32))
        rewards.append(r)
        if term or trunc:
            break
    assert np.isfinite(rewards).all()
    assert min(rewards) > -1e4, f"reward exploded: min {min(rewards)}"


def test_goal_geometry_and_obs() -> None:
    """The goal sits ``goal_distance`` ahead in +x and the torso vertex carries [dx_to_goal, vx]."""
    env = QuadrupedGoalEnv(base="free", goal_distance=2.0)
    obs, info = env.reset(seed=0)
    assert info["dist"] == pytest.approx(2.0, abs=0.1)
    assert float(obs[0, 0]) == pytest.approx(2.0, abs=0.1)   # torso vertex dx_to_goal ≈ goal_distance


# ── physical sanity: stands, goal/flip terminate ─────────────────────────────
def test_stands_under_gravity() -> None:
    """Zero control: the damped legs hold the torso up (do not collapse to the floor)."""
    env = QuadrupedGoalEnv(base="free", max_steps=200)
    env.reset(seed=0)
    rest_z = float(env.data.xpos[env.torso, 2])
    z = rest_z
    for _ in range(150):
        _o, _r, term, trunc, _i = env.step(np.zeros(env.n_actions, dtype=np.float32))
        z = float(env.data.xpos[env.torso, 2])
        if term or trunc:
            break
    assert z > rest_z - 0.15, f"torso collapsed to {z:.3f} (rest {rest_z:.3f})"


def test_reaching_goal_terminates() -> None:
    """Teleporting the torso onto the goal makes the next step report reached + terminate."""
    env = QuadrupedGoalEnv(base="free")
    env.reset(seed=0)
    env.data.qpos[0] = env.goal[0]            # free-joint x = goal x
    env.data.qpos[1] = env.goal[1]
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    _o, _r, term, _trunc, info = env.step(np.zeros(env.n_actions, dtype=np.float32))
    assert info["reached"] and term


def test_flip_terminates() -> None:
    """A trivially strict flip cutoff terminates once the torso tips."""
    env = QuadrupedGoalEnv(base="free", flip_cos=0.99)
    env.reset(seed=0)
    saw_term = any(env.step(env.action_space.high.astype(np.float32))[2] for _ in range(200))
    assert saw_term, "strict flip_cos never triggered termination"


def test_fixed_base_never_terminates_on_flip() -> None:
    """A welded torso cannot tip, so the flip guard is inert in fixed mode."""
    env = QuadrupedGoalEnv(base="fixed", flip_cos=0.99, max_steps=50)
    env.reset(seed=0)
    terms = [env.step(env.action_space.high.astype(np.float32))[2] for _ in range(50)]
    assert not any(terms)


# ── performance: wall-time budget + peak Python allocation (§3) ────────────────
def test_step_throughput_and_memory() -> None:
    """Median over 5 measured rounds of 200 steps stays under a wall budget; tracked allocation stays tiny
    (well under the 16 GB RSS cap). Single-shot timing is diagnostic only (§3 benchmark stability)."""
    import tracemalloc
    env = QuadrupedGoalEnv(base="free", max_steps=200)
    rng = np.random.default_rng(0)
    times = []
    tracemalloc.start()
    for _ in range(5):
        env.reset(seed=0)
        t = time.perf_counter()
        for _ in range(200):
            env.step(rng.uniform(-1, 1, env.n_actions).astype(np.float32))
        times.append(time.perf_counter() - t)
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    times.sort()
    assert times[2] < 2.0, f"200-step median {times[2]:.3f}s over budget"
    assert peak_mb < 256.0, f"tracked peak {peak_mb:.1f} MB unexpectedly large"
