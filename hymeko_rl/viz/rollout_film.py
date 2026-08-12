"""Canonical rollout filming for the coin-delivery sims — one home for the camera presets, the ``frame_hook`` filmer,
and the qpos-sequence renderer.

The ``video_*`` experiments and the demo each re-declared ``_cam()`` (×8, ~3 presets) and a ``_Filmer`` class (×3,
near-identical) plus ad-hoc ``mujoco.Renderer`` loops (audit 2026-08-12). This module holds those once; call sites
import ``Filmer`` / ``CAMS`` / ``render_qpos_seq`` instead of reimplementing them. Renders are deterministic, so a
migrated call site is byte-identical to its former local copy (the migration is parity-gated).
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

H, W = 480, 560                     # the coin-delivery clip size used across the video_* experiments


def _make_cam(distance: float, elevation: float, azimuth: float, lookat: "tuple[float, float, float]") -> Any:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    cam.lookat = np.array(lookat, np.float64)
    return cam


def top_down() -> Any:
    """Near-overhead view — the planar task reads best top-down (removes the depth illusion of the coin passing under an
    arm link; the coin only collides with fingertips + floor). Matches ``video_r11_6c_composition._cam``."""
    return _make_cam(1.02, -87.0, 90.0, (0.0, 0.09, 0.0))


def angled() -> Any:
    """The 3/4 view used by the reach/capture videos (``0.9 / -55 / 90``, lookat 0.08)."""
    return _make_cam(0.9, -55.0, 90.0, (0.0, 0.08, 0.0))


def side() -> Any:
    """A lower side view (``0.60 / -68 / 90``)."""
    return _make_cam(0.60, -68.0, 90.0, (0.0, 0.09, 0.0))


CAMS = {"top_down": top_down, "angled": angled, "side": side}


class Filmer:
    """A ``frame_hook(rl, t)`` that lazily builds a ``mujoco.Renderer`` and captures ``(frame, coin dtz mm)`` per
    governed rollout step — read-only, non-behavioural. Generalizes the per-file ``_Filmer`` variants (the optional
    ``phase`` tag some carried is kept). Pass an instance as ``frame_hook=`` to a rollout.

    # Preconditions: ``rl.inner`` exposes ``model``, ``data``, and ``direction_to_zone()``.
    # Postconditions: after N steps, ``len(frames) == len(dtz_mm) == N``; each frame is ``(height, width, 3)`` uint8."""

    def __init__(self, cam: Any, phase: "str | None" = None, height: int = H, width: int = W) -> None:
        self.cam, self.phase, self.height, self.width = cam, phase, height, width
        self.r: Any = None
        self.frames: list[Any] = []
        self.dtz_mm: list[float] = []

    def __call__(self, rl: Any, _t: int) -> None:
        if self.r is None:
            self.r = mujoco.Renderer(rl.inner.model, height=self.height, width=self.width)
        self.r.update_scene(rl.inner.data, camera=self.cam)
        self.frames.append(np.asarray(self.r.render(), np.uint8))
        self.dtz_mm.append(float(rl.inner.direction_to_zone()[1]) * 1000.0)

    def close(self) -> None:
        if self.r is not None:
            self.r.close()
            self.r = None


def render_qpos_seq(model: Any, qpos_seq: np.ndarray, cam: Any, height: int = H, width: int = W) -> "list[np.ndarray]":
    """Render a recorded ``qpos`` trajectory (``T×nq``) to frames offscreen — for replaying a real rollout's states.

    # Preconditions: ``qpos_seq`` rows match ``model.nq``.
    # Postconditions: returns ``len(qpos_seq)`` frames of shape ``(height, width, 3)`` uint8."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(np.asarray(renderer.render(), np.uint8))
    renderer.close()
    return frames
