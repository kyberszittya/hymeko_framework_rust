"""EXACT_ZERO_HOME honest delivery video — the coin pushed to strict K6 starting from q=[0,0,0,0].

Renders the full continuous chain the ``coin_zero_home_reach`` solver produces, with the codebase's own
``mujoco.Renderer`` + ``viz.rollout_overlay`` panels and the mandatory ``assert_trace_render_consistency`` gate:

    phase 1  REACH     : task-space 3-shell IK reach from the true zero home (coin untouched)
    phase 2  CAPTURE   : the READY-specific re-solved capture (bit-exact reproduction via the zero-theta roller)
    phase 3  DELIVERY  : the frozen downstream driving the coin to strict K6

First frame is exactly q=[0,0,0,0]. No snapshot teleport, no staging pose, no teacher. Writes an MP4 + GIF + a hashed
manifest. Run: ``python -m hymeko_rl.experiments.video_coin_zero_home``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.moving_precapture import R2_ALPHA
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_zero_home_reach import do_reach, solve_capture
from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, assert_trace_render_consistency, encode_clip, overlay_frames, summary_card)

OUT = Path("reports/2026-07-29-coin-zero-home-delivery/video")
CENTER_TOL_MM = 20.0
H, W = 420, 540


def _cam() -> Any:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 0.9, -55, 90
    cam.lookat = np.array([0.0, 0.08, 0.0])
    return cam


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


class _Filmer:
    """Lazily builds a ``mujoco.Renderer`` and captures (frame, coin-dtz) per governed step, read-only."""

    def __init__(self, cam: Any, phase: str) -> None:
        self.cam, self.phase, self.r = cam, phase, None
        self.frames: list = []
        self.dtz_mm: list = []

    def __call__(self, rl: Any, _t: int) -> None:
        if self.r is None:
            self.r = mujoco.Renderer(rl.inner.model, height=H, width=W)
        self.r.update_scene(rl.inner.data, camera=self.cam)
        self.frames.append(np.asarray(self.r.render(), np.uint8))
        self.dtz_mm.append(float(rl.inner.direction_to_zone()[1]) * 1000.0)

    def close(self) -> None:
        if self.r is not None:
            self.r.close()


def _overlay(frames: list, dtz_mm: np.ndarray, phases: list, end: str) -> list:
    n = len(frames)

    def status(t: int) -> str:
        tt = min(t, n - 1)
        return f"{phases[tt]}   step {tt+1}/{n}" if tt < n - 2 else f"DONE — {end}"

    def info(t: int) -> list:
        tt = min(t, n - 1)
        return [f"phase   {phases[tt]}", f"coin dtz {dtz_mm[tt]:7.1f} mm",
                "start   q = [0, 0, 0, 0]  (true zero home)", f"end     {end}"]

    panels = [StatusBar("EXACT ZERO HOME [0,0,0,0]  ->  push coin to strict K6", status),
              TimeSeriesPanel({"coin dtz (mm)": dtz_mm}, title="coin distance to target zone (20mm goal dashed)",
                              threshold=CENTER_TOL_MM, size=(300, 150)),
              InfoPanel(info)]
    return overlay_frames(frames, panels)


def render(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    cam = _cam()

    reach_film = _Filmer(cam, "REACH")
    r = do_reach(rig, cfg, frame_hook=reach_film)
    reach_film.close()
    if r is None:
        raise RuntimeError("no validated reach spiral")
    ready, coin, m_reach = r["ready"], r["coin"], r["metrics"]

    capres = solve_capture(rig, ready, 0)
    roller = tpo.TorquePathCaptureRoll(ready, rig["ref"], rig["stack"], capres.params, coin)
    cap_film = _Filmer(cam, "CAPTURE")
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32), frame_hook=cap_film)
    cap_film.close()
    snapshot = res["snapshot"]

    down = rig["down"]
    ctrl = HandoffResetTemporalController(snapshot, CloneActor(down.model, down.norm), down.r2_fn,
                                          ResidualBounds(alpha=R2_ALPHA))
    down_film = _Filmer(cam, "DELIVERY")
    dm = velocity_rollout(snapshot, ctrl, down.cfg, frame_hook=down_film)
    down_film.close()

    k6 = bool(delivery_success(dm, down.cfg))
    min_dtz = round(_min_dtz_mm(snapshot, dm), 2)
    frames = reach_film.frames + cap_film.frames + down_film.frames
    dtz_mm = np.asarray(reach_film.dtz_mm + cap_film.dtz_mm + down_film.dtz_mm, np.float64)
    phases = (["REACH"] * len(reach_film.frames) + ["CAPTURE"] * len(cap_film.frames)
              + ["DELIVERY"] * len(down_film.frames))
    end = f"STRICT K6  {min_dtz:.2f}mm" if k6 else f"{dtz_mm[-1]:.0f}mm"

    consistency = assert_trace_render_consistency(frames, dtz_mm, label="coin_zero_home")
    clip = _overlay(frames, dtz_mm, phases, end)
    card = summary_card((clip[0].width, clip[0].height), "EXACT ZERO HOME -> strict K6 (honest full chain)",
                        [("start q", "[0, 0, 0, 0]"), ("coin moved in reach", f"{m_reach['coin_moved_before_capture_mm']} mm"),
                         ("reach tip clearance", f"{m_reach['min_tip_coin_clr_mm']} mm"), ("min_dtz", f"{min_dtz:.2f} mm"),
                         ("K6 (strict)", "delivered" if k6 else "no")], hold=55)
    clip = clip + card

    mp4 = encode_clip(clip, out / "zero_home_to_k6.mp4", fps=20)
    gif = out / "zero_home_to_k6.gif"
    imageio.mimsave(gif, [np.asarray(im, np.uint8) for im in clip], duration=0.05)
    manifest = {
        "contract": "COIN_ZERO_HOME_DELIVERY_VIDEO", "purpose": "deterministic visualization only (selects nothing)",
        "start_state": "q=[0,0,0,0], qdot=0 (true zero home; both arms fully extended)",
        "chain": "EXACT_ZERO_HOME -> task-space 3-shell IK reach -> READY -> re-solved capture -> frozen downstream -> K6",
        "k6": k6, "min_dtz_mm": min_dtz, "reach_metrics": m_reach,
        "reach_frames": len(reach_film.frames), "capture_frames": len(cap_film.frames),
        "delivery_frames": len(down_film.frames), "n_frames": len(clip), "trace_consistency": consistency,
        "launch_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "mp4": str(mp4), "gif": str(gif), "gif_sha256_16": _sha(str(gif)),
        "rendering_command": "python -m hymeko_rl.experiments.video_coin_zero_home"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(f"  reach {len(reach_film.frames)} + capture {len(cap_film.frames)} + delivery {len(down_film.frames)} frames | "
          f"K6 {k6} min_dtz {min_dtz}mm | coin moved in reach {m_reach['coin_moved_before_capture_mm']}mm | {consistency}")
    print(f"  MP4 {mp4}\n  GIF {gif} (sha {manifest['gif_sha256_16']})\nZERO_HOME_VIDEO_DONE")
    return manifest


if __name__ == "__main__":
    render()
