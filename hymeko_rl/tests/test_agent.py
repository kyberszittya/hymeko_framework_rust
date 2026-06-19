"""Tests for AgentSpec — the MDP composed as data from the .hymeko profiles.

`from_hymeko` parity: the composed spec's dims/bounds must match an ArmReachEnv built from the same
sources (CLI-gated). Plus the frozen-dataclass invariants.
"""
from __future__ import annotations

import pytest

from hymeko_rl.agent import AgentSpec
from hymeko_rl.env.arm_world import _hymeko_cli

_REACH = "data/robotics/reach_arm.hymeko"
_OBS = "data/robotics/arm_reach_observation.hymeko"
_SAFE = "data/robotics/arm_reach_safe_task.hymeko"


def _cli() -> bool:
    try:
        _hymeko_cli()
        return True
    except FileNotFoundError:
        return False


requires_cli = pytest.mark.skipif(not _cli(), reason="hymeko CLI not built")


def test_agentspec_rejects_degenerate_fields() -> None:
    with pytest.raises(ValueError, match="obs_dim/action_dim/n_vertices"):
        AgentSpec(obs_dim=0, action_dim=4, action_low=-1.0, action_high=1.0, reward="x")
    with pytest.raises(ValueError, match="action_high"):
        AgentSpec(obs_dim=8, action_dim=4, action_low=1.0, action_high=1.0, reward="x")
    with pytest.raises(ValueError, match="n_vertices"):
        AgentSpec(obs_dim=8, action_dim=4, action_low=-1.0, action_high=1.0, reward="x",
                  n_vertices=0)


@requires_cli
def test_from_hymeko_matches_the_env() -> None:
    """AgentSpec.from_hymeko composes the same dims/bounds an ArmReachEnv builds from the sources."""
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

    spec = AgentSpec.from_hymeko(_REACH, obs_profile=_OBS, task_profile=_SAFE,
                                 control_mode="position")
    env = ArmReachEnv.from_hymeko(_REACH, obs_profile=_OBS, control_mode="position",
                                  ee_body="flange")
    assert spec.obs_dim == env.obs_spec.obs_dim
    assert spec.n_vertices == env.hg.n_vertices
    assert spec.action_dim == env.n_actions
    assert spec.action_low == pytest.approx(float(env.action_space.low.min()))
    assert spec.action_high == pytest.approx(float(env.action_space.high.max()))
    assert spec.reward == "arm_reach_safe_task"   # names the task profile
