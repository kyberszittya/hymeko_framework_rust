"""HYMEKO_SCENE_RUNTIME sentinels (2026-07-22): prove the canonical scene geometry (target zone, coin spawn,
workspace bounds, success criterion, disk radius) is LOAD-BEARING from ``galambos_env.hymeko`` — an edit to a scene
value propagates through ``EnvSpec.from_hymeko`` into the compiled runtime (a MjModel geom size / a runtime scene
attribute) — and that the CANONICAL env (``make_coin_env``) reads the scene from the spec rather than silently
falling back to the Python ``DEFAULT_ENV``.

Each propagation sentinel writes a temp copy of ``galambos_env.hymeko`` (same directory, so the relative
``@"meta_env.hymeko"`` import resolves) with ONE scene field mutated, reloads via ``from_hymeko``, builds the env,
and asserts the runtime carries the mutated value.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.env_factory import make_coin_env
from hymeko_rl.env.env_spec import DEFAULT_ENV, EnvSpec
from hymeko_rl.env.planar_grasp_env import _PLANAR_ENV, PlanarGraspEnv

SRC = Path(_PLANAR_ENV)
TMP = SRC.parent / "_scene_sentinel_tmp.hymeko"   # same dir → relative @"meta_env.hymeko" import resolves


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if TMP.exists():
        TMP.unlink()


def _mutate(pattern: str, repl: str) -> Path:
    text = SRC.read_text(encoding="utf-8")
    new = re.sub(pattern, repl, text, count=1)
    assert new != text, f"scene sentinel could not mutate via {pattern!r}"
    TMP.write_text(new, encoding="utf-8")
    return TMP


def _coin_geom_radius(env: PlanarGraspEnv) -> float:
    import mujoco
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    return float(env.model.geom_size[gid][0])


def test_disk_radius_is_load_bearing_into_mjmodel():
    spec = EnvSpec.from_hymeko(str(_mutate(r"radius 0\.02;", "radius 0.031;")))
    assert abs(spec.disk_radius - 0.031) < 1e-9
    env = PlanarGraspEnv(env=spec, robot_source="hymeko_spec")
    assert abs(_coin_geom_radius(env) - 0.031) < 1e-6, "coin geom radius did not track the scene spec"


def test_success_steps_is_load_bearing():
    spec = EnvSpec.from_hymeko(str(_mutate(r"steps 5\.0", "steps 9.0")))
    env = PlanarGraspEnv(env=spec, robot_source="hymeko_spec")
    assert env.success_steps == 9, f"success_steps did not track the scene spec: {env.success_steps}"


def test_zone_half_is_load_bearing():
    spec = EnvSpec.from_hymeko(str(_mutate(r"half 0\.04;", "half 0.055;")))
    env = PlanarGraspEnv(env=spec, robot_source="hymeko_spec")
    assert abs(env._zone_half - 0.055) < 1e-9, f"zone half did not track the scene spec: {env._zone_half}"


def test_canonical_env_reads_scene_from_spec_not_silent_default():
    # make_coin_env must build its scene from galambos_env.hymeko via from_hymeko — NOT the Python DEFAULT_ENV.
    canon = make_coin_env(embodiment="POINT")._env
    from_spec = EnvSpec.from_hymeko(_PLANAR_ENV)
    assert canon == from_spec, "canonical scene must equal EnvSpec.from_hymeko(galambos_env.hymeko)"
    # It is a FRESH spec object read from disk, not the shared frozen DEFAULT_ENV singleton.
    assert canon is not DEFAULT_ENV, "canonical scene must not be the DEFAULT_ENV singleton (silent fallback)"


def test_scene_source_hymeko_spec_equals_from_hymeko():
    # The scene_source="hymeko_spec" construction path resolves to from_hymeko(galambos_env.hymeko).
    env = PlanarGraspEnv(scene_source="hymeko_spec", robot_source="hymeko_spec")
    assert env._env == EnvSpec.from_hymeko(_PLANAR_ENV)


def test_unknown_scene_source_hard_fails():
    with pytest.raises(ValueError, match="scene_source"):
        PlanarGraspEnv(scene_source="bogus", robot_source="hymeko_spec")


def test_scene_missing_required_term_hard_fails():
    # Drop the whole `@dsk: p.disk {...}` term → from_hymeko must raise (no silent default for `disk`).
    TMP.write_text(re.sub(r"@dsk:.*\n", "", SRC.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="dsk|disk"):
        EnvSpec.from_hymeko(str(TMP))


def test_stepping_the_mutated_scene_env_is_stable():
    # An outer-runtime smoke: the mutated-scene env resets and steps without error (the scene is genuinely compiled).
    spec = EnvSpec.from_hymeko(str(_mutate(r"radius 0\.02", "radius 0.025")))
    env = PlanarGraspEnv(env=spec, robot_source="hymeko_spec")
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
