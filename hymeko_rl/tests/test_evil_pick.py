"""Tests for the parametric 'evil' pick-and-place environment generator."""
from __future__ import annotations

import numpy as np

from hymeko_rl.evil_pick import evil_config, make_evil_env


def test_evil_config_monotonically_harder() -> None:
    """Higher difficulty → object pushed out, heavier, smaller, more slippery (deterministically harder)."""
    rng = np.random.default_rng(0)
    easy = evil_config(0.0, rng)
    rng = np.random.default_rng(0)                       # same draw → compare the difficulty effect only
    evil = evil_config(1.0, rng)
    assert evil["obj_radius"][0] > easy["obj_radius"][0]          # farther out
    assert evil["box_mass"] > easy["box_mass"]                    # heavier
    assert evil["box_half"] < easy["box_half"]                    # smaller
    assert evil["finger_friction"] < easy["finger_friction"]      # more slippery
    assert evil["obj_angle"][1] > easy["obj_angle"][1]            # wider


def test_evil_config_clamps_difficulty() -> None:
    rng = np.random.default_rng(0)
    lo = evil_config(-5.0, rng)
    rng = np.random.default_rng(0)
    at0 = evil_config(0.0, rng)
    assert lo["obj_radius"] == at0["obj_radius"]                  # clamped to 0


def test_make_evil_env_builds_and_steps() -> None:
    """A generated evil env constructs and runs the gym contract (9-vertex hypergraph, (9,8) obs)."""
    env = make_evil_env(0.7, seed=3, max_steps=5)
    obs, info = env.reset(seed=3)
    assert env.hg.n_vertices == 9 and obs.shape == (9, 10)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (9, 10) and np.isfinite(r)
    assert "approach_contact" in info
