"""Stage 1: a HyMeKo *scenario* describes the whole RL problem, and builds the same env as the Python config.

The gate for "HyMeKo as the base of RL scenario description": the env built from ``pick_place_scenario.hymeko``
is behaviorally identical (spawn, dynamics, obs, reward) to the hand-configured ``fanuc_pick_env`` — the same
parity discipline that guards the declarative reward (``test_pick_place_task``).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.scenario import ScenarioSpec
from hymeko_rl.render_pick_place import fanuc_pick_env

_SCENARIO = "data/robotics/pick_place_scenario.hymeko"


def test_scenario_spec_reads_fields_and_imports() -> None:
    """The scenario model parses into the verified FANUC config: scene scalars/vectors + robot + reward."""
    spec = ScenarioSpec.from_hymeko(_SCENARIO)
    assert spec.robot.endswith("arm_gripper_fanuc_import.hymeko")
    assert spec.reward_profile is not None and spec.reward_profile.endswith("pick_place_task.hymeko")
    assert spec.obj_radius == (0.28, 0.40) and spec.target_xy == (0.34, 0.0)
    assert spec.arm_home == (0.0, 1.0, 0.8, 0.0, 0.8, 0.0)
    assert spec.max_steps == 620 and spec.lift_thresh == 0.035 and spec.place_radius == 0.075


def test_scenario_env_matches_python_config() -> None:
    """The scenario-built env reproduces ``fanuc_pick_env`` step-for-step on a fixed seed + action sequence.

    Reward parity is transitive: the scenario wires the declarative reward (pick_place_task), which equals the
    env's procedural reward (guarded by test_pick_place_task) — so matching the procedural ``fanuc_pick_env``
    here proves the scenario bundles scene AND reward correctly.
    """
    scn = ScenarioSpec.from_hymeko(_SCENARIO).build(max_steps=60)
    ref = fanuc_pick_env(max_steps=60)
    obs_s, _ = scn.reset(seed=3)
    obs_r, _ = ref.reset(seed=3)                       # same seed → identical spawn iff the cfg matches
    assert obs_s.shape == obs_r.shape == (9, 10)
    assert np.allclose(obs_s, obs_r), "scenario spawn/obs differs from the Python config"
    rng = np.random.default_rng(7)
    for _ in range(60):
        a = rng.uniform(ref._a_lo, ref._a_hi).astype(np.float32)
        o_s, r_s, t_s, tr_s, _ = scn.step(a)
        o_r, r_r, t_r, tr_r, _ = ref.step(a)
        assert np.allclose(o_s, o_r), "scenario dynamics/obs diverged from the Python config"
        assert abs(r_s - r_r) < 1e-5, f"reward mismatch: scenario={r_s} python={r_r}"
        assert t_s == t_r and tr_s == tr_r
        if t_s or tr_s:
            break
