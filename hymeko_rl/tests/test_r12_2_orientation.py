"""R12.2-A — unit/integration tests for the non-invasive yaw-varied handoff adapter.

Guards the contracts R12.2 depends on: the adapter sets the planar yaw qpos[6]; it is BIT-IDENTICAL to the frozen
`_home_with_coin` at yaw=0 (so the frozen R11.6C/R11.7 pipeline is provably unperturbed); `object_yaw` tracks the
commanded placement; and a non-finite yaw is rejected. The rig build is a mujoco integration cost, done once per test
via a fixture (no global state).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.r12_orientation import home_with_coin_yaw, object_yaw
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

_XY = np.array([0.0, 0.30])          # a canonical in-reach coin placement


@pytest.fixture(scope="module")
def rig() -> Any:
    return _rig(object_spec=variant("O4-S").object_spec)   # box — the orientation-sensitive object


def test_home_with_coin_yaw_sets_qpos6(rig: Any) -> None:
    home, _ = home_with_coin_yaw(rig, _XY, 0.30)
    q6 = float(home.branch().inner.data.qpos[6])           # disk_rz hinge = planar yaw
    assert abs(q6 - 0.30) < 1e-6, f"qpos[6] not set to yaw: {q6}"


def test_yaw_zero_reproduces_frozen_home(rig: Any) -> None:
    """Regression: yaw=0 must be BIT-IDENTICAL to the frozen `_home_with_coin` — proves the adapter cannot perturb the
    yaw=0 pipeline that R11.6C/R11.7 results depend on."""
    q_new = home_with_coin_yaw(rig, _XY, 0.0)[0].branch().inner.data.qpos.copy()
    q_ref = Z._home_with_coin(rig, _XY)[0].branch().inner.data.qpos.copy()
    assert np.array_equal(q_new, q_ref), f"yaw=0 diverges from frozen home: {q_new - q_ref}"


def test_object_yaw_tracks_placement(rig: Any) -> None:
    """`object_yaw` must move ~1:1 with the commanded placement yaw (any constant geom-mount offset cancels in the
    difference)."""
    y0 = object_yaw(home_with_coin_yaw(rig, _XY, 0.0)[0])
    y5 = object_yaw(home_with_coin_yaw(rig, _XY, 0.50)[0])
    assert abs((y5 - y0) - 0.50) < 1e-3, f"object_yaw did not track placement: Δ={y5 - y0}"


def test_nonfinite_yaw_raises() -> None:
    """Failure case: a non-finite yaw is a caller bug, rejected before any state is touched (rig never accessed)."""
    with pytest.raises(ValueError, match="finite"):
        home_with_coin_yaw({}, _XY, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        home_with_coin_yaw({}, _XY, math.inf)
