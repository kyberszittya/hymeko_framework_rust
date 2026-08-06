"""Runnable, interactive live simulation of the fast-dynamics vehicles / locomotion substrates — watch a
car (or robot) drive its track live in a MuJoCo window you can orbit, pan and zoom, with the waypoint track
drawn as pylons + a racing line.

macOS runs the interactive viewer on the main thread, so launch with **mjpython** (bundled in the venv):

    .venv/bin/mjpython -m hymeko_rl.gui.vehicle_sim --vehicle racecar
    .venv/bin/mjpython -m hymeko_rl.gui.vehicle_sim --vehicle f1tenth
    .venv/bin/mjpython -m hymeko_rl.gui.vehicle_sim --vehicle vehicle    # diff-drive
    .venv/bin/mjpython -m hymeko_rl.gui.vehicle_sim --vehicle cheetah    # legged (no track)

By default the scripted expert drives (the demonstrator you see in the MP4s). Pass ``--policy path.pt`` to
drive with a trained off-policy actor once the campaign has produced one. The physics run on the substrate
env (`hymeko_rl.env.locomotion_env`); the track pylons are drawn as viewer user-scene geoms (decoration only,
never perturbing physics). Reuses the substrate factories + experts; nothing is reimplemented (§6.1)."""
from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import mujoco
import mujoco.viewer
import numpy as np

from hymeko_rl.env.locomotion_env import SUBSTRATES

# Camera presets per substrate: (distance, elevation, azimuth). Loops get a higher, wider view.
_CAM = {
    "racecar": (24.0, -14.0, 132.0),
    "f1tenth": (16.0, -40.0, 90.0),
    "vehicle": (11.0, -40.0, 90.0),
    "cheetah": (6.0, -12.0, 90.0),
    "humanoid": (6.0, -12.0, 90.0),
}
_EYE3 = np.eye(3, dtype=np.float64).reshape(9)


def _scripted_source() -> Callable[[Any], np.ndarray]:
    return lambda env: env.expert_action


def _policy_source(path: str, env: Any) -> Callable[[Any], np.ndarray]:
    """A trained off-policy actor as a greedy action source (deterministic ``action_mean``). Built to match
    the env's flattened obs; loads a ``DeterministicActor`` state_dict saved by the campaign."""
    import torch

    from hymeko_rl.train.ddpg import build_offpolicy

    flat = int(np.prod(env.observation_space.shape))
    actor, _ = build_offpolicy("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                               action_scale=1.0, n_critics=1, hidden=64, device="cpu")
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()

    def act(env: Any) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(env.node_features()[None], dtype=torch.float32)
            return np.asarray(actor.action_mean(obs).squeeze(0).numpy(), dtype=np.float32)

    return act


def _draw_track(scn: mujoco.MjvScene, env: Any) -> None:
    """Populate the viewer's user scene with the track: a pylon at each waypoint (green start, orange rest).
    Called each frame (idempotent) so it survives viewer scene rebuilds."""
    track = getattr(env, "_track", None)
    if track is None:
        return
    zf = float(env.model.geom_pos[_floor_geom(env), 2]) if _floor_geom(env) is not None else 0.0
    r = 0.28 if env.cfg.name == "f1tenth" else (2.2 if env.cfg.name == "racecar" else 0.3)
    scn.ngeom = 0
    for i, (x, y) in enumerate(track):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        rgba = np.array([0.15, 0.9, 0.25, 1.0] if i == 0 else [1.0, 0.55, 0.05, 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([r, r, r * 1.6]), np.array([float(x), float(y), zf + r * 1.6]),
                            _EYE3, rgba)
        scn.ngeom += 1


def _floor_geom(env: Any) -> int | None:
    for gid in range(env.model.ngeom):
        if env.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            return gid
    return None


def run(vehicle: str, *, policy: str | None = None, seed: int = 0, realtime: bool = True) -> None:
    if vehicle not in SUBSTRATES:
        raise SystemExit(f"unknown vehicle {vehicle!r}; choose from {sorted(SUBSTRATES)}")
    env = SUBSTRATES[vehicle]()
    env.reset(seed=seed)
    action_source = _policy_source(policy, env) if policy else _scripted_source()
    dt = env.frame_skip * float(env.model.opt.timestep)
    dist, elev, azim = _CAM.get(vehicle, (10.0, -20.0, 90.0))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = dist, elev, azim
        print(f"[vehicle_sim] {vehicle}: live. Orbit/zoom with the mouse; close the window to stop.")
        step = 0
        while viewer.is_running():
            t0 = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(action_source(env))
            _draw_track(viewer.user_scn, env)
            viewer.sync()
            step += 1
            if terminated or truncated:
                env.reset(seed=seed + step)          # loop the demo continuously
            if realtime:
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Interactive live vehicle/locomotion simulation (run via mjpython).")
    ap.add_argument("--vehicle", default="racecar", choices=sorted(SUBSTRATES))
    ap.add_argument("--policy", default=None, help="optional trained actor .pt (else the scripted expert)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-realtime", action="store_true", help="run as fast as possible (no wall-clock pacing)")
    args = ap.parse_args(argv)
    run(args.vehicle, policy=args.policy, seed=args.seed, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
