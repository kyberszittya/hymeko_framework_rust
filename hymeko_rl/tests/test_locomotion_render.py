"""Tests for the locomotion render helper (`viz/locomotion_render.py`) + the live-sim track drawing
(`gui/vehicle_sim.py`). Rendering is exercised at tiny resolution / few frames to stay fast; the viewer
launch itself is not tested (needs mjpython + a display).

Run: pytest -p no:randomly hymeko_rl/tests/test_locomotion_render.py"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hymeko_rl.env.locomotion_env import make_f1tenth, make_vehicle
from hymeko_rl.gui import vehicle_sim
from hymeko_rl.viz import locomotion_render as lr


def test_decorated_render_model_matches_dof_and_adds_track() -> None:
    env = make_vehicle(max_steps=30)
    rmodel = lr._decorated_render_model(env, floor_size=20, texrepeat=20, show_track=True,
                                        marker_r=0.3, line_r=0.09)
    assert rmodel.nq == env.model.nq and rmodel.nv == env.model.nv     # visual-only decoration, same DOF
    names = [mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(rmodel.ngeom)]
    assert "hk_floor" in names and any(n and n.startswith("hk_wp") for n in names)   # checker floor + pylons
    assert "floor" not in names                                        # blank collision plane dropped


def test_track_markers_are_noncolliding() -> None:
    env = make_vehicle(max_steps=30)
    rmodel = lr._decorated_render_model(env, floor_size=20, texrepeat=20, show_track=True,
                                        marker_r=0.3, line_r=0.09)
    for g in range(rmodel.ngeom):
        nm = mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(("hk_wp", "hk_wl")):
            assert int(rmodel.geom_contype[g]) == 0 and int(rmodel.geom_conaffinity[g]) == 0


def test_beautified_video_writes_mp4(tmp_path: Path) -> None:
    env = make_vehicle(max_steps=12)
    out = tmp_path / "clip.mp4"
    p = lr.beautified_video(env, lambda e, o: e.expert_action, out, width=128, height=128, stride=4,
                            lookat=(0, 5, 0), dist=18, elev=-55, floor_size=20, texrepeat=20)
    assert p.exists() and p.stat().st_size > 0


def test_beautified_video_writes_gif(tmp_path: Path) -> None:
    env = make_f1tenth(max_steps=12)
    out = tmp_path / "clip.gif"
    p = lr.beautified_video(env, lambda e, o: e.expert_action, out, width=128, height=128, stride=4,
                            floor_size=20, texrepeat=20)
    assert p.exists() and p.suffix == ".gif" and p.stat().st_size > 0


def test_vehicle_sim_scripted_source_and_track_draw() -> None:
    env = make_f1tenth(max_steps=30)
    env.reset(seed=0)
    a = vehicle_sim._scripted_source()(env)
    assert a.shape == (env.n_actions,)
    scn = mujoco.MjvScene(env.model, maxgeom=200)
    vehicle_sim._draw_track(scn, env)
    assert scn.ngeom == len(env._track)                               # one pylon per waypoint
    # start pylon is green, the rest orange
    assert np.allclose(scn.geoms[0].rgba[:3], [0.15, 0.9, 0.25], atol=1e-3)
