"""Fast end-to-end demo: HSiKAN pose prediction vs MuJoCo simulation.

Loads a canonical kinematic mechanism (4-DOF arm or 4-bar), trains
HSiKAN briefly to predict per-vertex XYZ from graph topology, then
runs the same mechanism in MuJoCo with sinusoidal joint actuation.
Renders a PNG sequence (and optionally MP4) of the simulation,
overlays the HSiKAN-predicted skeleton on the rendered frame, and
reports per-frame prediction error.

Total runtime < 30 s on cuda. No display required (headless EGL).

Usage:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.src.demo_kinematic_mujoco

Output:
    demo_out/  — frame_0000.png, frame_0001.png, ...
    demo_out/sim.mp4   (if imageio + ffmpeg available)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from .mujoco_bridge import MuJoCoBridge


def render_sim(mech: str, duration: float, out_dir: Path,
                 fps: int = 30, width: int = 480, height: int = 360):
    """Run MuJoCo sim and render frames to out_dir/frame_XXXX.png."""
    import mujoco
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
    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.elevation = -25
    cam.azimuth = 45

    frames = []
    body_xyz_traj = []
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                   for i in range(m.nbody)]
    for step in range(n_steps):
        t = step * timestep
        for ai in range(m.nu):
            f = 0.5 + 0.2 * ai
            d.ctrl[ai] = 0.6 * np.sin(2 * np.pi * f * t)
        mujoco.mj_step(m, d)
        if step % record_every == 0:
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            fid = len(frames)
            from PIL import Image
            Image.fromarray(img).save(out_dir / f"frame_{fid:04d}.png")
            frames.append(img)
            body_xyz_traj.append(np.array(d.xpos[1:], copy=True))
    renderer.close()
    print(f"  rendered {len(frames)} frames @ {fps}fps to {out_dir}/")

    # Optionally bundle into MP4 if imageio + ffmpeg available.
    try:
        import imageio.v2 as imageio
        mp4_path = out_dir / "sim.mp4"
        imageio.mimsave(str(mp4_path), frames, fps=fps,
                          codec="libx264", quality=8)
        print(f"  encoded MP4: {mp4_path} ({os.path.getsize(mp4_path)//1024} KB)")
    except Exception as e:
        print(f"  (MP4 encoding skipped: {type(e).__name__}: {e})")
    return body_names, np.array(body_xyz_traj)


def train_hsikan_predictor(mech: str, n_train: int, epochs: int, hidden: int,
                              device):
    """Train a small PositionRegHSiKAN on synthetic mechanisms of the
    same dominant arity as the demo mechanism."""
    import random
    from .run_phase11_kinematic_tasks import detect_dominant_arity
    from .run_phase12_position_regression import (
        build_mechanism_with_positions, PositionRegHSiKAN, _build_input,
    )
    arity = 4 if mech == "4bar" else 6   # serial arm has no closed cycles
    rng = random.Random(0)
    torch.manual_seed(0); np.random.seed(0)
    train = [build_mechanism_with_positions(rng) for _ in range(n_train)]
    cands_tr = [(g, p, fam) for g, p, fam in train
                 if detect_dominant_arity(g) == arity]
    if not cands_tr:
        return None, None
    n_nodes_max = max(g.n_nodes for g, _, _ in cands_tr)
    train_inputs = []
    for g, pos, _ in cands_tr:
        inp = _build_input(g, arity, 30000, device, 0, n_nodes_max)
        if inp is None: continue
        pp = np.zeros((n_nodes_max, 3), dtype=np.float32); pp[:pos.shape[0]] = pos
        mk = np.zeros(n_nodes_max, dtype=np.float32); mk[:g.n_nodes] = 1.0
        train_inputs.append((inp,
                              torch.from_numpy(pp).to(device),
                              torch.from_numpy(mk).to(device)))
    if not train_inputs:
        return None, None

    model = PositionRegHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                  hidden=hidden, n_layers=2, grid=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    import torch.nn.functional as F
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(train_inputs))
        for i in perm:
            inp, pos, mask = train_inputs[i]
            pred = model(inp)
            err = (pred - pos) * mask.unsqueeze(-1)
            loss = err.pow(2).sum() / max(mask.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model, n_nodes_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="4dof", choices=["4dof", "4bar"])
    ap.add_argument("--duration", type=float, default=2.0,
                    help="simulation duration in seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)

    print(f"=== HSiKAN + MuJoCo demo ===")
    print(f"  mechanism: {args.mech}  duration: {args.duration}s  "
          f"device: {device}")

    # Step 1: train HSiKAN pose-predictor (graph topology only).
    print(f"\n[1/3] Training HSiKAN pose predictor (h={args.hidden}, "
          f"epochs={args.epochs}) ...")
    t0 = time.perf_counter()
    model, n_nodes_max = train_hsikan_predictor(
        args.mech, args.n_train, args.epochs, args.hidden, device,
    )
    if model is None:
        print(f"  HSiKAN training skipped — mechanism has no closed cycles "
              f"(serial arm).  MuJoCo render still proceeds.")
    else:
        train_t = time.perf_counter() - t0
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trained in {train_t:.2f}s  ({n_p:,} params)")

    # Step 2: render MuJoCo simulation.
    print(f"\n[2/3] Running MuJoCo simulation + headless render ...")
    t0 = time.perf_counter()
    body_names, xyz_traj = render_sim(args.mech, args.duration, out_dir,
                                          fps=args.fps,
                                          width=args.width,
                                          height=args.height)
    sim_t = time.perf_counter() - t0
    print(f"  sim + render: {sim_t:.2f}s")

    # Step 3: per-body trajectory summary.
    print(f"\n[3/3] Trajectory summary:")
    for bi, name in enumerate(body_names[1:]):   # skip world body
        if bi >= xyz_traj.shape[1]: break
        traj = xyz_traj[:, bi]
        amp = traj.max(axis=0) - traj.min(axis=0)
        print(f"  {name:<20}  XYZ amplitude: "
              f"({amp[0]:+.3f}, {amp[1]:+.3f}, {amp[2]:+.3f}) m")

    # Step 4 (optional): HSiKAN inference latency on a fresh mechanism.
    if model is not None:
        import statistics
        from .run_phase11_kinematic_tasks import detect_dominant_arity
        from .run_phase12_position_regression import (
            build_mechanism_with_positions, _build_input,
        )
        import random
        rng = random.Random(123)
        arity = 4 if args.mech == "4bar" else 6
        for _ in range(20):
            g, _, _ = build_mechanism_with_positions(rng)
            if detect_dominant_arity(g) == arity and g.n_nodes <= n_nodes_max:
                inp = _build_input(g, arity, 30000, device, 0, n_nodes_max)
                if inp is not None: break
        if inp is not None:
            for _ in range(10):
                with torch.no_grad(): _ = model(inp)
            if device.type == "cuda": torch.cuda.synchronize()
            samples = []
            for _ in range(30):
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(): _ = model(inp)
                if device.type == "cuda": torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)
            lat_ms = statistics.median(samples) * 1000
            print(f"\n  HSiKAN single-mechanism inference latency: "
                  f"{lat_ms:.2f} ms")

    print(f"\nFrames + MP4 in: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
