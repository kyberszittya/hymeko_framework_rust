"""Tests for procedural terrain (`env/terrain.py`) + its env integration + the Qt simulator build. The Qt
smoke is guarded (offscreen platform); it skips cleanly where a Qt platform plugin is unavailable.

Run: pytest -p no:randomly hymeko_rl/tests/test_terrain.py"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hymeko_rl.env.locomotion_env import make_f1tenth, make_vehicle
from hymeko_rl.env.terrain import procedural_hfield


@pytest.mark.parametrize("kind", ["flat", "hills", "bumps", "ramps"])
def test_procedural_hfield_range_and_border(kind: str) -> None:
    h = procedural_hfield(kind, n=48, seed=0)
    assert h.shape == (48, 48) and h.dtype == np.float32
    assert 0.0 <= float(h.min()) and float(h.max()) <= 1.0
    # the skirt flattens the border ring to ~0 so a vehicle rolls in from flat ground
    assert float(h[0].max()) < 1e-5 and float(h[-1].max()) < 1e-5


def test_procedural_hfield_deterministic() -> None:
    assert np.array_equal(procedural_hfield("bumps", n=32, seed=3), procedural_hfield("bumps", n=32, seed=3))


def test_flat_vehicle_has_no_heightfield() -> None:
    assert make_vehicle(max_steps=10).model.nhfield == 0


@pytest.mark.parametrize("make", [make_vehicle, make_f1tenth])
def test_terrain_vehicle_builds_and_drives_finite(make) -> None:
    env = make(max_steps=300, terrain="bumps")
    assert env.model.nhfield == 1 and np.count_nonzero(env.model.hfield_data) > 0
    env.reset(seed=0)
    minup = 1.0
    for _ in range(300):
        _, _, term, trunc, info = env.step(env.expert_action)
        minup = min(minup, info["upright"])
        assert np.isfinite(env.data.qacc).all(), "terrain drive blew up"
        if term or trunc:
            break
    assert minup > 0.3, f"vehicle flipped on terrain (min upright {minup:.2f})"


def test_qt_simulator_headless_build() -> None:
    """The Qt window builds, ticks, and renders across vehicle/terrain without a display (offscreen platform)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from hymeko_rl.gui.vehicle_qt import VehicleSimWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        win = VehicleSimWindow()
    except Exception as exc:                     # no usable Qt platform plugin in this env
        pytest.skip(f"Qt platform unavailable: {exc}")
    try:
        for vehicle, terrain in [("f1tenth", "flat"), ("vehicle", "bumps")]:
            win._vehicle.setCurrentText(vehicle)
            win._terrain.setCurrentText(terrain)
            win._reload()
            for _ in range(5):
                win._tick()
            pm = win._view.pixmap()
            assert pm is not None and not pm.isNull()
    finally:
        win.close()
    _ = app
