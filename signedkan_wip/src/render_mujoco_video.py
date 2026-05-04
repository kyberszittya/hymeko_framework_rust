"""Render a full annotated MuJoCo simulation video.

Each frame is captured at the requested fps, body positions are
projected to 2D and overlaid as orange dots + yellow labels, and the
HSiKAN/MuJoCo-comparison badge is drawn at the bottom.  A running
timestamp + per-mechanism agreement readout is shown top-left.
The result is encoded to MP4 (H.264) when imageio + ffmpeg are
available, with a fallback PNG sequence otherwise.

Usage:
    python -m signedkan_wip.src.render_mujoco_video \
        --mech 4dof --duration 4.0 --fps 30 \
        --out demo_video --width 640 --height 480
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from .annotate_mujoco_frames import (
    project_body_to_screen, get_view_matrix, annotate_frame,
)
from .mujoco_bridge import MuJoCoBridge


def render_video(mech: str, duration: float, fps: int,
                   width: int, height: int, out_dir: Path,
                   keep_frames: bool = False):
    if mech == "4dof":
        sim = MuJoCoBridge.canonical_4dof_arm()
    elif mech == "4bar":
        sim = MuJoCoBridge.canonical_4bar()
    else:
        raise ValueError(mech)

    m, d = sim.model, sim.data
    sim.reset()
    timestep = m.opt.timestep
    n_steps = int(duration / timestep)
    record_every = max(1, int((1.0 / fps) / timestep))

    out_dir.mkdir(parents=True, exist_ok=True)
    if keep_frames:
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.elevation = -25
    cam.azimuth = 45

    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                   for i in range(1, m.nbody)]   # skip world body
    n_bodies = len(body_names)

    frames: list[np.ndarray] = []
    print(f"  rendering {n_steps // record_every + 1} frames at "
          f"{width}x{height}, {fps}fps ...")
    t_start = time.perf_counter()
    for step in range(n_steps + 1):
        t = step * timestep
        for ai in range(m.nu):
            f = 0.5 + 0.2 * ai
            d.ctrl[ai] = 0.6 * np.sin(2 * np.pi * f * t)
        mujoco.mj_step(m, d)
        if step % record_every == 0:
            renderer.update_scene(d, camera=cam)
            img = Image.fromarray(renderer.render())
            view_mat = get_view_matrix(cam)
            body_pixels = [
                project_body_to_screen(m, d, bi + 1, None, view_mat,
                                          width, height)
                for bi in range(n_bodies)
            ]
            annotated = annotate_frame(img, body_pixels, body_names,
                                          t, n_bodies, width, height)
            frames.append(np.array(annotated))
            if keep_frames:
                fid = step // record_every
                annotated.save(frames_dir / f"frame_{fid:05d}.png")
    renderer.close()
    elapsed = time.perf_counter() - t_start
    print(f"  rendered {len(frames)} frames in {elapsed:.2f}s "
          f"({len(frames)/elapsed:.0f} fps render)")

    # Encode video
    try:
        import imageio.v2 as imageio
        mp4_path = out_dir / f"hsikan_mujoco_{mech}.mp4"
        imageio.mimsave(str(mp4_path), frames, fps=fps,
                          codec="libx264", quality=9,
                          macro_block_size=1)
        size_kb = mp4_path.stat().st_size // 1024
        print(f"  encoded H.264 MP4: {mp4_path} ({size_kb} KB, "
              f"{len(frames)/fps:.1f}s @ {fps}fps)")
    except Exception as e:
        print(f"  MP4 encoding failed ({type(e).__name__}: {e})")
        print(f"  Frames available at {frames_dir}/")
    # Also write a GIF for web use
    try:
        import imageio.v2 as imageio
        gif_path = out_dir / f"hsikan_mujoco_{mech}.gif"
        # GIF: subsample frames to keep file small
        gif_frames = frames[::max(1, fps // 10)]   # ~10 fps GIF
        imageio.mimsave(str(gif_path), gif_frames, fps=10,
                          loop=0)
        size_kb = gif_path.stat().st_size // 1024
        print(f"  encoded GIF: {gif_path} ({size_kb} KB, "
              f"{len(gif_frames)/10:.1f}s @ 10fps)")
    except Exception as e:
        print(f"  GIF encoding failed ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="4dof", choices=["4dof", "4bar"])
    ap.add_argument("--duration", type=float, default=4.0,
                    help="Simulation length in seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", type=Path, default=Path("demo_video"))
    ap.add_argument("--keep-frames", action="store_true",
                    help="Also keep individual PNG frames")
    args = ap.parse_args()

    print(f"=== MuJoCo demo video ===")
    print(f"  mechanism: {args.mech}")
    print(f"  duration: {args.duration}s @ {args.fps}fps "
          f"({args.duration*args.fps:.0f} frames)")
    print(f"  resolution: {args.width}x{args.height}")
    print(f"  output: {args.out}")
    render_video(args.mech, args.duration, args.fps,
                   args.width, args.height, args.out,
                   keep_frames=args.keep_frames)
    print("done.")


if __name__ == "__main__":
    main()
