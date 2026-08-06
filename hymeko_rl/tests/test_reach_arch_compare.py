"""The HSiKAN-vs-MLP comparison harness — structure only (the real run is the experiment).

Tiny config (few demos/epochs) so it stays a unit test; skips when the CLI isn't built.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.env.arm_world import _hymeko_cli
from hymeko_rl.experiments.reach_arch_compare import (
    _iqr,
    compare_backbones,
    emitted_arm_factory,
)


def _cli_available() -> bool:
    try:
        _hymeko_cli()
        return True
    except FileNotFoundError:
        return False


requires_cli = pytest.mark.skipif(not _cli_available(), reason="hymeko CLI not built")


def test_iqr() -> None:
    assert _iqr([1.0]) == 0.0            # degenerate → 0
    assert _iqr([1.0, 2.0, 3.0, 4.0, 5.0]) > 0.0


@requires_cli
def test_factory_builds_canonical_reach_arm() -> None:
    # default = the tractable canonical reach arm (4-DOF, position-controlled, BC-learnable).
    env = emitted_arm_factory()()
    assert env.n_actions == 4
    assert env.control_mode == "position"
    assert env.obs_spec.obs_dim == 8
    assert env.reward_spec.terms == (("reach_distance", 1.0),)
    # position actuators (ctrlrange = joint range ±π), retargeted from the emitted torque motors.
    assert bool(np.all(env.model.actuator_ctrllimited))
    assert env.action_space.high[0] == pytest.approx(3.1416, abs=1e-3)


@requires_cli
def test_position_emit_retargets_actuators_and_adds_damping() -> None:
    from pathlib import Path

    from hymeko_rl.env.arm_world import emit_arm_mjcf
    reach = Path(__file__).resolve().parents[2] / "data" / "robotics" / "reach_arm.hymeko"
    torque = emit_arm_mjcf(reach, control_mode="torque")
    position = emit_arm_mjcf(reach, control_mode="position")
    assert "<motor" in torque and "<position" not in torque
    assert "<position" in position and "<motor" not in position
    assert 'damping="2.0"' in position   # added for stable PD tracking


@requires_cli
def test_compare_returns_both_backbones() -> None:
    factory = emitted_arm_factory()
    res = compare_backbones(
        factory, seeds=[0], backbones=("hsikan", "mlp"),
        n_demos=4, n_epochs=2, hidden=16)
    assert set(res) == {"hsikan", "mlp"}
    for kind in ("hsikan", "mlp"):
        assert res[kind]["n_params"] > 0
        assert "reach_median" in res[kind]
        assert len(res[kind]["per_seed"]) == 1
