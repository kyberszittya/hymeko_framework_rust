"""The declarative observation spec — channels read from a ``.hymeko`` profile drive
``node_features`` (no hand-coded column offsets).

Covers the pure spec/reader (no CLI) and an equivalence check that the spec-driven
observation is byte-identical to the former hand-coded layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.env.observation import (
    REACH_OBSERVATION,
    ObservationSpec,
    read_channels,
)

_REPO = Path(__file__).resolve().parents[2]
_OBS_PROFILE = _REPO / "data" / "robotics" / "arm_reach_observation.hymeko"


# ── spec (pure) ──────────────────────────────────────────────────────────────
def test_reach_obs_dim_is_eight() -> None:
    assert REACH_OBSERVATION.obs_dim == 8   # qpos(1)+qvel(1)+target(3)+ee_error(3)
    assert REACH_OBSERVATION.channels == (
        "joint_position", "joint_velocity", "target_position", "ee_error")


def test_obs_dim_follows_channels() -> None:
    assert ObservationSpec(("joint_position", "joint_velocity")).obs_dim == 2
    assert ObservationSpec(("target_position",)).obs_dim == 3


def test_unknown_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown observation channel"):
        ObservationSpec(("joint_position", "not_a_channel"))
    with pytest.raises(ValueError, match="at least one channel"):
        ObservationSpec(())


# ── reader (the .hymeko bridge) ──────────────────────────────────────────────
def test_read_channels_from_reach_profile() -> None:
    """The reader recovers the reaching profile's ordered channel kinds — so the
    .hymeko observation profile and the default Python spec agree."""
    assert read_channels(_OBS_PROFILE) == REACH_OBSERVATION.channels
    assert ObservationSpec.from_hymeko(_OBS_PROFILE).obs_dim == 8


def test_reader_rejects_missing_observation_space(tmp_path: Path) -> None:
    bad = tmp_path / "no_space.hymeko"
    bad.write_text("foo { @qpos: feat.joint_position { (+ j1); } }")
    with pytest.raises(ValueError, match="observation_space"):
        read_channels(bad)


def test_reader_preserves_declared_order(tmp_path: Path) -> None:
    """Channel order in the spec follows the observation_space arc, not the decl order."""
    prof = tmp_path / "reordered.hymeko"
    prof.write_text(
        "p {\n"
        "  @a: feat.joint_velocity { (+ j1); }\n"
        "  @b: glob.ee_error { (+ f); }\n"
        "  @c: feat.joint_position { (+ j1); }\n"
        "  @s: obs.observation_space { (+ c, + a, + b); }\n"
        "}\n")
    assert read_channels(prof) == ("joint_position", "joint_velocity", "ee_error")


# ── equivalence: spec-driven obs == former hand-coded layout ─────────────────
def test_assemble_matches_legacy_layout() -> None:
    """The spec-driven observation reproduces the old node_features columns exactly
    (per-joint qpos@0, qvel@1; broadcast target@2:5, ee_error@5:8)."""
    from hymeko_rl.env.arm_reach_env import ArmReachEnv   # 4-DOF arm_world (no CLI needed)

    env = ArmReachEnv(control_mode="torque")
    env.reset(seed=3)
    obs = env.node_features()
    assert obs.shape == (env.hg.n_vertices, 8)

    # reconstruct the legacy layout independently and compare.
    legacy = np.zeros((env.hg.n_vertices, 8), dtype=np.float32)
    for j in range(env.model.njnt):
        v = int(env._jnt_vtx[j])
        if 0 <= v < env.hg.n_vertices:
            legacy[v, 0] = env.data.qpos[j]
            legacy[v, 1] = env.data.qvel[j]
    legacy[:, 2:5] = env._target
    legacy[:, 5:8] = env._target - env._ee_pos()
    assert np.allclose(obs, legacy)
