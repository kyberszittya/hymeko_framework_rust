"""Visual simulation of the reaching rollout — render the policy driving the arm.

The first *watchable* output of the Kato POC: an action source (the scripted DLS-IK expert
or a behaviour-cloned policy) is stepped through :class:`ArmReachEnv` while an offscreen
``mujoco.Renderer`` captures frames, with the reach target drawn as a 3-D scene marker. The
frames are encoded by a small **encoder Strategy** — ``gif`` (Pillow, no extra dependency)
or ``mp4`` (imageio; a CORE.YAML §1 dependency, inert until approved + installed).

    python -m hymeko_rl.render_reach --source expert --encoder gif --out reports/reach
    python -m hymeko_rl.render_reach --source bc --encoder mp4 --out reports/reach  # needs imageio
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from hymeko_rl.bc import _make_policy, behaviour_clone, collect_demos
from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.env.arm_world import CONTROL_MODES, emit_arm_mjcf
from hymeko_rl.env.scene_style import SceneStyle, beautify_mjcf
from hymeko_rl.evaluate import _stamp_frames, now_stamp
from hymeko_rl.policy import ActorCritic

# The default robot to render: the 6-DOF anthropomorphic arm (the showcase morphology).
_DEFAULT_ROBOT = "data/robotics/anthropomorphic_arm.hymeko"

# An action source: (env, obs) -> action in the actuator ctrlrange.
ActionFn = Callable[[ArmReachEnv, np.ndarray], np.ndarray]

_TARGET_RGBA = np.array([0.9, 0.2, 0.2, 1.0], dtype=np.float32)   # red reach target
_TARGET_RADIUS = 0.025


@dataclass(frozen=True)
class CameraView:
    """A fixed orbit camera for the rollout (azimuth/elevation in degrees, distance in m)."""
    distance: float = 1.3
    elevation: float = -20.0
    azimuth: float = 45.0
    lookat_z: float = 0.25

    def to_mjv(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = self.distance
        cam.elevation = self.elevation
        cam.azimuth = self.azimuth
        cam.lookat = np.array([0.0, 0.0, self.lookat_z])
        return cam


def expert_source() -> ActionFn:
    """Render the scripted closed-loop demonstrator (always reaches — the reference motion)."""
    return lambda env, _obs: env.expert_action


def policy_source(ac: ActorCritic) -> ActionFn:
    """Render a learned policy's deterministic action mean."""
    @torch.no_grad()
    def fn(_env: ArmReachEnv, obs: np.ndarray) -> np.ndarray:
        a = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
        return np.asarray(a.squeeze(0).numpy(), dtype=np.float32)
    return fn


def _draw_target(scene: mujoco.MjvScene, target: np.ndarray) -> None:
    """Append a sphere marker at ``target`` to the rendered scene (no-op if geom buffer full).

    # Preconditions ``target`` is a length-3 world position.
    # Invariants never writes past ``scene.maxgeom`` (overflow is undefined behaviour).
    """
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([_TARGET_RADIUS, 0.0, 0.0]),
        np.asarray(target, dtype=np.float64),
        np.eye(3).flatten(), _TARGET_RGBA)
    scene.ngeom += 1


def render_rollout(env: ArmReachEnv, action_fn: ActionFn, *, seed: int = 0,
                   width: int = 480, height: int = 360,
                   camera: CameraView | None = None,
                   max_frames: int | None = None) -> list[np.ndarray]:
    """Step ``action_fn`` through one episode of ``env``, capturing one frame per env step.

    # Preconditions a ``mujoco.Renderer`` is constructible (an offscreen GL context exists).
    # Postconditions a non-empty list of ``(height, width, 3)`` uint8 frames; the reach target
    is drawn each frame; deterministic given ``seed`` and ``action_fn``.
    """
    cam = (camera or CameraView()).to_mjv()
    obs, info = env.reset(seed=seed)
    target = np.asarray(info["target"], dtype=np.float64)
    cap = env.max_steps if max_frames is None else min(env.max_steps, max_frames)
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        for _ in range(cap):
            renderer.update_scene(env.data, camera=cam)
            _draw_target(renderer.scene, target)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
            obs, _reward, terminated, truncated, _info = env.step(action_fn(env, obs))
            if terminated or truncated:
                # capture the final settled frame too, then stop.
                renderer.update_scene(env.data, camera=cam)
                _draw_target(renderer.scene, target)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8))
                break
    finally:
        renderer.close()
    return frames


# ── encoder Strategy ─────────────────────────────────────────────────────────
def _encode_gif(frames: list[np.ndarray], out: Path, fps: int) -> None:
    """Encode an animated GIF via Pillow (no extra dependency)."""
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=max(1, int(1000 / fps)), loop=0)


def _encode_mp4(frames: list[np.ndarray], out: Path, fps: int) -> None:
    """Encode an H.264 MP4 via imageio — a CORE.YAML §1 dependency, absent by default."""
    try:
        # optional dependency (pyproject `demo` group); guarded for base installs.
        import imageio.v2 as imageio
    except ImportError as err:
        raise RuntimeError(
            "mp4 encoding needs imageio + imageio-ffmpeg (pyproject `demo` group). "
            "Install with `uv sync --group demo`, then retry — or use --encoder gif "
            "(no dependency) meanwhile.") from err
    # imageio stubs type the frames list invariantly; ndarray frames are the intended input.
    imageio.mimsave(str(out), frames, fps=fps, codec="libx264", quality=8,  # type: ignore[arg-type]
                    macro_block_size=1)


_ENCODERS: dict[str, tuple[str, Callable[[list[np.ndarray], Path, int], None]]] = {
    "gif": (".gif", _encode_gif),
    "mp4": (".mp4", _encode_mp4),
}


def encode(frames: list[np.ndarray], out: str | Path, fps: int, kind: str,
           *, stamp: str | None = None) -> Path:
    """Encode ``frames`` to ``out`` using encoder ``kind`` (the file extension is forced to
    match). Returns the written path. ``stamp`` is a bottom-right provenance label drawn on every
    frame (``None`` auto-stamps the current time, ``""`` disables).

    # Preconditions ``frames`` non-empty; ``kind in _ENCODERS``.
    # Errors ``ValueError`` (empty frames / unknown kind); ``RuntimeError`` (mp4 without dep).
    """
    if not frames:
        raise ValueError("no frames to encode")
    if kind not in _ENCODERS:
        raise ValueError(f"unknown encoder {kind!r}; expected one of {sorted(_ENCODERS)}")
    frames = _stamp_frames(frames, now_stamp() if stamp is None else stamp)
    suffix, fn = _ENCODERS[kind]
    path = Path(out).with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fn(frames, path, fps)
    return path


def build_render_env(robot: str | None = _DEFAULT_ROBOT, *, control_mode: str = "position",
                     pretty: bool = True, ee_body: str = "tool",
                     style: SceneStyle | None = None,
                     reach_min_radius: float = 0.2) -> ArmReachEnv:
    """The env to render: an emitted ``.hymeko`` arm, optionally beautified.

    With ``robot=None`` falls back to the bare default :class:`ArmReachEnv` (the hand-authored
    4-DOF arm) so the original behaviour stays reachable. Otherwise emits the robot's MJCF, applies
    :func:`beautify_mjcf` when ``pretty`` (skybox / floor / lights / polish — visual only), and
    binds the end-effector ``ee_body``. Position control is the default: the DLS-IK expert tracks
    smoothly (the 6-DOF torque expert saturates --- 2026-06-19 emitted-arm-physics finding).

    # Preconditions ``control_mode in CONTROL_MODES``; the robot/CLI exist when ``robot`` is set.
    # Postconditions a renderable :class:`ArmReachEnv`; physics matches the bare arm plus a floor.
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be in {CONTROL_MODES}; got {control_mode!r}")
    if robot is None:
        return ArmReachEnv(reach_min_radius=reach_min_radius)
    mjcf = emit_arm_mjcf(robot, name="arm", control_mode=control_mode)
    if pretty:
        mjcf = beautify_mjcf(mjcf, style)
    return ArmReachEnv(mjcf=mjcf, control_mode=control_mode, ee_body=ee_body,
                       reach_min_radius=reach_min_radius)


def _bc_policy(env: ArmReachEnv, *, n_demos: int, n_epochs: int, hidden: int,
               seed: int) -> ActorCritic:
    """Train a quick HSiKAN behaviour-cloned policy for the rendered demo."""
    torch.manual_seed(seed)
    ac = _make_policy("hsikan", env, hidden)
    obs, acts = collect_demos(env, n_demos, seed)
    behaviour_clone(ac, obs, acts, n_epochs=n_epochs, seed=seed)
    return ac


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="expert", choices=["expert", "bc"])
    ap.add_argument("--encoder", default="gif", choices=sorted(_ENCODERS))
    ap.add_argument("--out", default="reports/2026-06-19-6dof-sim")
    ap.add_argument("--robot", default=_DEFAULT_ROBOT,
                    help="robot .hymeko to render; 'default' = the bare 4-DOF arm_world arm")
    ap.add_argument("--control", default="position", choices=sorted(CONTROL_MODES))
    ap.add_argument("--ee-body", default="tool", help="end-effector body name in the emitted arm")
    ap.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True,
                    help="apply scene beautification (skybox/floor/lights/polish)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demos", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args(argv)

    robot = None if a.robot == "default" else a.robot
    env = build_render_env(robot, control_mode=a.control, pretty=a.pretty, ee_body=a.ee_body)
    if a.source == "expert":
        action_fn = expert_source()
    else:
        action_fn = policy_source(_bc_policy(
            env, n_demos=a.demos, n_epochs=a.epochs, hidden=a.hidden, seed=a.seed))
    frames = render_rollout(env, action_fn, seed=a.seed, width=a.width, height=a.height)
    path = encode(frames, a.out, a.fps, a.encoder)
    print(f"wrote {path} ({len(frames)} frames, {a.source} source, {a.encoder})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
