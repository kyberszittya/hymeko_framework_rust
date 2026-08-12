"""viz.rollout_film — the canonical camera presets, the frame_hook Filmer, and the qpos-sequence renderer that replace
the per-file _cam/_Filmer/render-loop copies."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.env.object_spec import ObjectSpec, Shape
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.viz.rollout_film import CAMS, Filmer, render_qpos_seq, top_down


def _coin_model() -> Any:
    return PlanarGraspEnv(**ObjectSpec(shape=Shape.CYLINDER, radius=0.02).planar_env_kwargs()).model


def test_cams_presets_build_valid_cameras() -> None:
    for name, factory in CAMS.items():
        cam = factory()
        assert cam.distance > 0.0, f"{name} preset has non-positive distance"


def test_render_qpos_seq_renders_each_state() -> None:
    m = _coin_model()
    qseq = np.zeros((3, m.nq))
    frames = render_qpos_seq(m, qseq, top_down(), height=64, width=80)
    assert len(frames) == 3
    assert frames[0].shape == (64, 80, 3) and frames[0].dtype == np.uint8


class _Inner:
    def __init__(self, model: Any, data: Any) -> None:
        self.model, self.data = model, data

    def direction_to_zone(self) -> "tuple[np.ndarray, float]":
        return np.zeros(2), 0.05


class _FakeRl:
    """The slice of the rollout sim the Filmer reads (model + data + direction_to_zone)."""

    def __init__(self, model: Any, data: Any) -> None:
        self.inner = _Inner(model, data)


def test_filmer_captures_frames_and_dtz_per_step() -> None:
    m = _coin_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    film = Filmer(top_down(), phase="delivery", height=64, width=80)
    rl = _FakeRl(m, d)
    film(rl, 1)
    film(rl, 2)
    assert len(film.frames) == 2 == len(film.dtz_mm)
    assert film.frames[0].shape == (64, 80, 3)
    assert film.dtz_mm[0] == 50.0                     # 0.05 m → 50 mm
    film.close()
