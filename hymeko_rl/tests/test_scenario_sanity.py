"""Per-scenario geometry + motion sanity checks (cart-pole / coin-grasp / quadruped).

One parametrized contract over every HSiKAN testbed env, asserting the HyMeKo-emitted robot is (a) *well
generated* — a valid MuJoCo model with the expected body/joint/actuator/hypergraph structure, finite rest
state — and (b) *capable of moving* — driving the actuators changes the per-vertex observation and the state
stays finite. Catches a broken emit (wrong DOF count, NaN inertia, dead actuators, bolted base) before any
training run is queued on it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv


@dataclass(frozen=True)
class ScenarioSpec:
    """A scenario under sanity test: how to build it + its expected emitted geometry."""

    name: str
    make_env: Callable[[], object]
    n_vertices: int          # kinematic-hypergraph vertices (= emitted links)
    n_actions: int           # actuated DOF after base/passive stripping
    obs_feat: int            # per-vertex observation width

    def build(self) -> object:
        return self.make_env()


SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec("cartpole", lambda: InvertedPendulumEnv(), n_vertices=2, n_actions=1, obs_feat=2),
    ScenarioSpec("coingrasp", lambda: PlanarGraspEnv.from_hymeko(max_steps=60), n_vertices=6,
                 n_actions=4, obs_feat=8),
    ScenarioSpec("quadruped", lambda: QuadrupedGoalEnv(base="free", max_steps=60), n_vertices=33,
                 n_actions=12, obs_feat=2),   # faithful 22-DOF Aibo: 12 RL-controlled legs, expressive held
]
_IDS = [s.name for s in SCENARIOS]


@pytest.mark.parametrize("spec", SCENARIOS, ids=_IDS)
def test_geometry_well_formed(spec: ScenarioSpec) -> None:
    """The emitted robot is a structurally valid model with the expected DOF + hypergraph counts."""
    env = spec.build()
    model = env.model                                            # type: ignore[attr-defined]
    assert model.nbody > 1, "no bodies emitted"
    assert int(model.nu) == spec.n_actions, f"actuator count {model.nu} != {spec.n_actions}"
    assert env.hg.n_vertices == spec.n_vertices                  # type: ignore[attr-defined]
    # rest state and inertia are finite (no NaN mass/inertia from a degenerate geom)
    assert np.isfinite(env.model.body_mass).all()               # type: ignore[attr-defined]
    assert float(env.model.body_mass.sum()) > 0.0               # type: ignore[attr-defined]


@pytest.mark.parametrize("spec", SCENARIOS, ids=_IDS)
def test_observation_contract(spec: ScenarioSpec) -> None:
    """``reset`` yields a finite ``(n_vertices, obs_feat)`` observation and a matching action space."""
    env = spec.build()
    obs, _info = env.reset(seed=0)                               # type: ignore[attr-defined]
    assert obs.shape == (spec.n_vertices, spec.obs_feat)
    assert np.isfinite(obs).all()
    assert env.action_space.shape == (spec.n_actions,)          # type: ignore[attr-defined]


@pytest.mark.parametrize("spec", SCENARIOS, ids=_IDS)
def test_capable_of_moving(spec: ScenarioSpec) -> None:
    """Driving the actuators changes the per-vertex state — the robot is not dead/bolted, and stays finite.

    Applies a sustained actuation (the action-space bound) and asserts the observation moves measurably while
    remaining finite over the horizon."""
    env = spec.build()
    obs0, _ = env.reset(seed=0)                                 # type: ignore[attr-defined]
    drive = np.asarray(env.action_space.high, dtype=np.float32)  # type: ignore[attr-defined]
    drive = np.where(np.isfinite(drive), drive, 1.0).astype(np.float32)
    obs = obs0
    moved = 0.0
    for _ in range(25):
        obs, _r, terminated, truncated, _info = env.step(drive)  # type: ignore[attr-defined]
        assert np.isfinite(obs).all(), "non-finite state under actuation (sim blew up)"
        moved = max(moved, float(np.abs(obs - obs0).max()))
        if terminated or truncated:
            break
    assert moved > 1e-3, f"{spec.name}: actuation produced no motion (dead actuators / bolted base?)"


@pytest.mark.parametrize("spec", SCENARIOS, ids=_IDS)
def test_reset_deterministic(spec: ScenarioSpec) -> None:
    """A fixed seed reproduces the reset state exactly (no reliance on system entropy)."""
    a, _ = spec.build().reset(seed=7)                           # type: ignore[attr-defined]
    b, _ = spec.build().reset(seed=7)                           # type: ignore[attr-defined]
    assert np.array_equal(a, b)
