"""Presentation-quality rendering for the locomotion substrates — a decorated chase-camera renderer so the
fast vehicles are actually *visible* (a car on a blank floor reads as frozen; a checker floor scrolling under
a tracking camera reads as motion) and the **track is drawn** (waypoint pylons + a racing line). Writes MP4
(compact, via imageio-ffmpeg) or GIF. Wraps the existing pieces (``scene_style.beautify_mjcf`` +
``eval.evaluate._write_gif``); it does not reimplement the MuJoCo render loop (§6.1).

Subtleties handled: the env's own blank collision-floor plane (``name="floor"``) is dropped from the *render*
model (physics runs on the env's untouched model) and the checker floor is pinned to that plane's z (no
coplanar gray patch); the offscreen framebuffer is enlarged for HD; track markers are visual-only
(``contype=0``) so they never perturb physics."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import mujoco
import numpy as np

from hymeko_rl.env.scene_style import SceneStyle, beautify_mjcf
from hymeko_rl.eval.evaluate import _write_gif

ActionFn = Callable[[Any, np.ndarray], np.ndarray]

# Studio floor: enough tile contrast that the ground visibly scrolls (the motion cue), on a light sky.
_FLOOR = dict(floor_rgb1=(0.30, 0.33, 0.39), floor_rgb2=(0.60, 0.63, 0.69))


def _floor_z(env: Any) -> float:
    m = re.search(r'<geom name="floor" type="plane" pos="[^"]*?([-0-9.]+)"', env._mjcf)
    return float(m.group(1)) if m else 0.0


def _track_markers_xml(track: np.ndarray, z: float, *, marker_r: float, line_r: float,
                       marker_every: int = 1) -> str:
    """Visual-only geoms drawing the track: the translucent capsule 'racing line' between consecutive samples
    (always), plus a pylon every ``marker_every`` samples (green start, orange rest) — sparse pylons keep a
    dense Bézier circuit from becoming a wall. ``contype=0 conaffinity=0`` → decoration only."""
    parts = []
    for i, (x, y) in enumerate(track):
        if i == 0 or i % max(1, marker_every) == 0:
            rgba = "0.15 0.9 0.25 1" if i == 0 else "1 0.55 0.05 1"
            parts.append(f'<geom name="hk_wp{i}" type="cylinder" size="{marker_r} {marker_r * 1.6}" '
                         f'pos="{x:g} {y:g} {z + marker_r * 1.6:g}" rgba="{rgba}" contype="0" conaffinity="0"/>')
        if i + 1 < len(track):
            x2, y2 = track[i + 1]
            parts.append(f'<geom name="hk_wl{i}" type="capsule" contype="0" conaffinity="0" '
                         f'fromto="{x:g} {y:g} {z + line_r:g} {x2:g} {y2:g} {z + line_r:g}" '
                         f'size="{line_r}" rgba="0.95 0.95 1 0.4"/>')
    return "".join(parts)


def _decorated_render_model(env: Any, *, floor_size: float, texrepeat: int, show_track: bool,
                            marker_r: float, line_r: float, marker_every: int = 1, shadowsize: int = 2048,
                            offsamples: int = 4) -> mujoco.MjModel:
    """A visually decorated ``MjModel`` (skybox + checker floor + lights + optional track markers) with the SAME
    DOF as ``env.model``; the blank collision plane is dropped and the checker floor pinned to its z."""
    zf = _floor_z(env)
    style = SceneStyle(floor_size=floor_size, floor_texrepeat=texrepeat, **_FLOOR)
    dec = beautify_mjcf(env._mjcf, style)
    if "hk_terrain" in env._mjcf:
        # Terrain IS the ground: keep the heightfield geom, drop the flat checker floor beautify added.
        dec = re.sub(r'<geom name="hk_floor"[^/]*/>', "", dec)
    else:
        dec = re.sub(r'<geom name="floor" type="plane"[^/]*/>', "", dec)      # blank collision square
        dec = re.sub(r'(<geom name="hk_floor"[^/]*?)/>', rf'\1 pos="0 0 {zf:.3f}"/>', dec)
    dec = dec.replace('shadowsize="4096"', f'shadowsize="{shadowsize}"').replace('offsamples="8"',
                                                                                 f'offsamples="{offsamples}"')
    if show_track and getattr(env, "_track", None) is not None:
        markers = _track_markers_xml(np.asarray(env._track), zf, marker_r=marker_r, line_r=line_r,
                                     marker_every=marker_every)
        dec = dec.replace("<worldbody>", "<worldbody>" + markers, 1)
    return mujoco.MjModel.from_xml_string(dec)


def _write_video(frames: list[np.ndarray], out_path: str | Path, fps: int) -> Path:
    """MP4 via imageio-ffmpeg (libx264, compact) when the path ends in ``.mp4``; else GIF."""
    out = Path(out_path)
    if out.suffix.lower() == ".mp4":
        imageio.mimwrite(str(out), frames, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        return out
    return _write_gif(frames, out, fps=fps)


def beautified_video(env: Any, action_fn: ActionFn, out_path: str | Path, *, seed: int = 0,
                     width: int = 960, height: int = 720, fps: int = 30, stride: int = 1,
                     dist: float = 15.0, elev: float = -20.0, azim: float = 110.0,
                     lookat: tuple[float, float, float] | None = None,
                     floor_size: float = 60.0, texrepeat: int = 60, show_track: bool = True,
                     marker_r: float = 0.4, line_r: float = 0.12, marker_every: int = 1) -> Path:
    """Render one episode of ``action_fn`` on ``env`` to an HD **MP4** (``.mp4``) or GIF, over a checker floor
    with the track drawn. Camera: a fixed overhead at ``lookat`` if given (frames the whole track), else a
    chase camera tracking ``env.torso``. ``stride`` subsamples rendered frames (render cost, not sim);
    ``marker_every`` sparsifies pylons on a dense track. # Postconditions one video; env dynamics untouched."""
    rmodel = _decorated_render_model(env, floor_size=floor_size, texrepeat=texrepeat, show_track=show_track,
                                     marker_r=marker_r, line_r=line_r, marker_every=marker_every)
    rmodel.vis.global_.offwidth = int(width)
    rmodel.vis.global_.offheight = int(height)
    rdata = mujoco.MjData(rmodel)
    cam = mujoco.MjvCamera()
    if lookat is not None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(env.torso)
    cam.distance, cam.elevation, cam.azimuth = float(dist), float(elev), float(azim)
    renderer = mujoco.Renderer(rmodel, height=int(height), width=int(width))
    frames: list[np.ndarray] = []
    env.reset(seed=seed)
    try:
        for k in range(int(env.max_steps)):
            _, _, terminated, truncated, _ = env.step(action_fn(env, None))
            if k % stride == 0:
                rdata.qpos[:] = env.data.qpos
                rdata.qvel[:] = env.data.qvel
                mujoco.mj_forward(rmodel, rdata)
                renderer.update_scene(rdata, camera=cam)
                frames.append(renderer.render().copy())
            if terminated or truncated:
                break
    finally:
        renderer.close()
    return _write_video(frames, out_path, fps=fps)


# Back-compat alias (GIF path).
def beautified_episode_gif(env: Any, action_fn: ActionFn, out_path: str | Path, **kw: Any) -> Path:
    return beautified_video(env, action_fn, out_path, **kw)
