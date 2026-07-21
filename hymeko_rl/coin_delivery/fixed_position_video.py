"""Real-time video export for the fixed-position Coin Delivery replay (§7).

Renders the traced two-phase rollout of a :class:`~hymeko_rl.coin_delivery.fixed_position.CoinInitialState` to an
mp4 at **100 FPS real time** (``control_dt = 0.01 s`` ⇒ one frame per control step ⇒ wall-clock == physical time),
overlaying the exact fixed start, initial coin coordinates, initial clearance, active policy, phase, elapsed physical
time and the live strict result. Uses the shared :func:`composed_rollout` (single-sourced rollout) via its
``on_step`` hook — the frames are the exact replayed states, not a re-simulation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_W, _H = 960, 720
_FPS = 100                       # 1/control_dt ⇒ real time


def _camera() -> Any:
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.10, 0.0]     # centre of the planar workspace
    cam.distance = 0.62
    cam.azimuth = 90.0
    cam.elevation = -89.0                # near top-down
    return cam


def _overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Draw stacked text lines (top-left) onto an RGB frame. Falls back to the raw frame if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    except OSError:
        font = big = ImageFont.load_default()
    y = 10
    for i, (text, color) in enumerate(lines):
        f = big if i == 0 else font
        draw.rectangle([6, y, 6 + int(len(text) * (13 if i == 0 else 10)) + 8, y + (30 if i == 0 else 24)],
                       fill=(0, 0, 0))
        draw.text((10, y + 2), text, fill=color, font=f)
        y += 32 if i == 0 else 26
    return np.asarray(img)


def render_policy(env: Any, cf: Any, state: Any, policy: str, out_path: str | Path, *,
                  reach: Any, label: str = "") -> dict[str, Any]:
    """Render one policy's rollout for ``state`` to ``out_path`` (mp4, 100 FPS). Returns a summary + the video path."""
    import imageio.v2 as imageio
    import mujoco

    from hymeko_rl.coin_delivery.fixed_position import CONTROL_DT, apply_initial_state
    from hymeko_rl.coin_delivery.fixed_position_replay import build_actors, composed_rollout
    apply_initial_state(env, cf, state, seed_hint=0)
    inner = cf._env
    inner.model.vis.global_.offwidth = _W                       # the default offscreen buffer is 640×480
    inner.model.vis.global_.offheight = _H
    renderer = mujoco.Renderer(inner.model, height=_H, width=_W)
    cam = _camera()
    approach, tfn = build_actors(policy)
    frames: list[np.ndarray] = []
    cx, cy = float(state.coin_position[0]), float(state.coin_position[1])
    green, white, cyan, red = (80, 255, 120), (240, 240, 240), (120, 220, 255), (255, 120, 120)

    def _grab(phase: str, step: int, delivered: bool) -> None:
        renderer.update_scene(inner.data, cam)
        f = renderer.render()
        t_s = step * CONTROL_DT
        m = inner._planar_metrics
        dtz = float(m.disk_to_zone)
        strict = "DELIVERED" if delivered else "…"
        lines = [(f"EXACT FIXED START — {label}", white),
                 (f"coin0 = ({cx:+.3f}, {cy:+.3f})  clearance0 = {reach.signed_clearance:+.4f} m", cyan),
                 (f"policy = {policy}", green if policy.startswith('P4') else red),
                 (f"phase = {phase}   coin->zone = {dtz:.3f} m", white),
                 (f"t = {t_s:5.2f} s (real)   strict = {strict}",
                  green if delivered else white)]
        frames.append(_overlay(f, lines))

    _grab("start", 0, False)
    tr = composed_rollout(env, cf, approach, tfn, grasp_hold=1, contact_window=20, policy=policy, on_step=_grab)
    # hold the final frame for ~0.5 s so the result is readable
    for _ in range(int(0.5 * _FPS)):
        renderer.update_scene(inner.data, cam)
        f = renderer.render()
        lines = [(f"EXACT FIXED START — {label}", white),
                 (f"coin0 = ({cx:+.3f}, {cy:+.3f})  clearance0 = {reach.signed_clearance:+.4f} m", cyan),
                 (f"policy = {policy}", green if policy.startswith('P4') else red),
                 (f"result: {'STRICT DELIVERED' if tr.strict_delivered else 'NOT DELIVERED (' + tr.failure_reason + ')'}",
                  green if tr.strict_delivered else red),
                 (f"completion = {tr.completion_time_s:.2f} s   steps = {tr.n_steps}", white)]
        frames.append(_overlay(f, lines))
    renderer.close()
    imageio.mimsave(str(out_path), frames, fps=_FPS, quality=8, macro_block_size=None)
    return {"policy": policy, "path": str(out_path), "n_frames": len(frames),
            "strict_delivered": tr.strict_delivered, "trajectory_hash": tr.trajectory_hash()}


def render_fixed_position_videos(state: Any, out_dir: str | Path, *, label: str = "user-selected") -> dict[str, Any]:
    """Export the two §7 videos: the P4 real-time delivery + the P0/P1/P4 controls (concatenated), both 100 FPS."""
    import hashlib

    import imageio.v2 as imageio

    from hymeko_rl.coin_delivery.fixed_position_replay import analyze_reachability
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env, cf = neutral_env(prefix_steps=0, geom="POINT")
    from hymeko_rl.coin_delivery.fixed_position import apply_initial_state
    apply_initial_state(env, cf, state, seed_hint=0)
    reach = analyze_reachability(env, cf)

    rt_path = out_dir / "coin_delivery_fixed_position_real_time.mp4"
    rt = render_policy(env, cf, state, "P4_E_APPROACH_HANDOFF", rt_path, reach=reach, label=label)

    # controls: P4, P1, P0 rendered and concatenated into one comparison clip
    ctrl_path = out_dir / "coin_delivery_fixed_position_controls.mp4"
    clips = []
    for pol in ("P4_E_APPROACH_HANDOFF", "P1_FROZEN_TRANSPORT", "P0_ZERO_ACTION"):
        tmp = out_dir / f"_ctrl_{pol}.mp4"
        render_policy(env, cf, state, pol, tmp, reach=reach, label=label)
        clips.extend(imageio.mimread(str(tmp), memtest=False))
        tmp.unlink()
    imageio.mimsave(str(ctrl_path), clips, fps=_FPS, quality=8, macro_block_size=None)

    def _h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {"real_time": {"path": str(rt_path), "sha256": _h(rt_path), "strict_delivered": rt["strict_delivered"]},
            "controls": {"path": str(ctrl_path), "sha256": _h(ctrl_path), "n_frames": len(clips)}}
