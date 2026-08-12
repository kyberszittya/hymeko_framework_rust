"""Live interactive MuJoCo viewer for the real coin delivery — the native 3D robotics GUI (orbit / zoom / pan).

A single ``mujoco.viewer`` window shows the real trained delivery physics; keyboard controls cycle the object shape,
target scenario and strategy and run a real delivery, which plays back IN the 3D window (the compute runs on a worker
thread so the window stays live). Thin glue over the real pipeline (``delivery_viewer.record_delivery``) + the target
geometry (``dataset.scenario_by_id``); no physics/render reimplemented.

Following ``viz/viewer.py``: the loop logic (:func:`drive_delivery`) is pure and unit-tested with a fake handle; only
:func:`launch` needs a display. Run with **mjpython** (macOS main-thread requirement):

    PYTHONPATH=. .venv/bin/mjpython -m hymeko_rl.gui.delivery_live

Keys: SPACE run · N next shape · T next target · G next strategy · Q / Esc quit.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import scenario_by_id
from hymeko_rl.gui.delivery_viewer import SHAPES, STRATEGIES, DeployedPolicy, record_delivery

_KEY = {32: "run", 78: "shape", 84: "target", 71: "strategy", 81: "quit", 256: "quit"}   # GLFW: space/N/T/G/Q/Esc
_PLANE_Z = 0.02
_SHAPE_LABELS = {"O0": "coin", "O4-S": "square", "O6-T": "triangle", "O7-P": "pentagon",
                 "O8-H": "hexagon", "O3-E": "ellipse", "O9-K": "capsule"}


class Handle(Protocol):
    """The slice of ``mujoco.viewer.Handle`` the loop needs (a fake satisfies it in tests)."""

    def is_running(self) -> bool: ...
    def sync(self) -> None: ...
    @property
    def user_scn(self) -> Any: ...


@dataclass
class DemoState:
    """Interactive state; the key_callback flips the flags, the loop consumes them."""

    shape_i: int = 0
    strategy_i: int = 0
    sids: list[str] = field(default_factory=list)
    sid_i: int = 0
    run_requested: bool = False
    want_shape_change: bool = False
    quit: bool = False
    status: str = "SPACE run · N shape · T target · G strategy · Q quit"

    @property
    def shape_id(self) -> str:
        return SHAPES[self.shape_i]

    @property
    def strategy(self) -> str:
        return STRATEGIES[self.strategy_i]

    @property
    def sid(self) -> str:
        return self.sids[self.sid_i]


def make_key_callback(state: DemoState) -> Callable[[int], None]:
    def cb(keycode: int) -> None:
        action = _KEY.get(keycode)
        if action == "run":
            state.run_requested = True
        elif action == "shape":
            state.shape_i = (state.shape_i + 1) % len(SHAPES)
            state.want_shape_change = True                       # a new shape = a new model = relaunch
        elif action == "target":
            state.sid_i = (state.sid_i + 1) % len(state.sids)
            state.status = f"target: {state.sid}"
        elif action == "strategy":
            state.strategy_i = (state.strategy_i + 1) % len(STRATEGIES)
            state.status = f"strategy: {state.strategy}"
        elif action == "quit":
            state.quit = True
    return cb


def _target_xy(sid: str, zone: "tuple[float, float]") -> np.ndarray:
    t = getattr(scenario_by_id(sid), "target_xy", None)
    return np.asarray(t, np.float64) if t is not None else np.asarray(zone, np.float64)


def _mark_target(handle: Handle, xy: np.ndarray) -> None:
    scn = handle.user_scn
    scn.ngeom = 0
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0.0, 0.0]), np.array([xy[0], xy[1], _PLANE_Z]),
                            np.eye(3).flatten(), np.array([0.15, 0.7, 0.2, 0.55], np.float32))
        scn.ngeom += 1


def drive_delivery(handle: Handle, state: DemoState, policy: Any, model: Any, data: Any, zone: "tuple[float, float]",
                   *, realtime: bool = True, sleep_fn: Callable[[float], Any] = time.sleep,
                   compute: Callable[..., dict] = record_delivery) -> str:
    """Sync the viewer; on a run request compute the real delivery on a worker thread and replay its qpos into ``data``.
    Returns 'shape' (relaunch with the new shape's model) or 'quit'. # Invariants: touches sim state only via qpos replay
    + mj_forward; the worker only reads (record_delivery branches its own sim)."""
    dt = float(model.opt.timestep) * 5
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            result.update(compute(policy, state.shape_id, state.sid, state.strategy) or {})
        except Exception as e:                                   # noqa: BLE001
            result.update({"error": f"{type(e).__name__}: {e}"})
        result["_done"] = True

    worker: threading.Thread | None = None
    while handle.is_running() and not state.quit and not state.want_shape_change:
        _mark_target(handle, _target_xy(state.sid, zone))
        handle.sync()
        if state.run_requested and worker is None:
            state.run_requested = False
            result.clear()
            state.status = f"running: {_SHAPE_LABELS.get(state.shape_id, state.shape_id)} -> {state.sid} ..."
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
        if worker is not None and result.get("_done"):
            if "error" in result:
                state.status = str(result["error"])
            else:
                for q in result["qpos_seq"]:                     # replay the real physics states in the live 3D view
                    if not handle.is_running() or state.quit or state.want_shape_change:
                        break
                    data.qpos[: len(q)] = q
                    mujoco.mj_forward(model, data)
                    _mark_target(handle, _target_xy(state.sid, zone))
                    handle.sync()
                    if realtime:
                        sleep_fn(dt)
                state.status = f"K6={result.get('k6')}  dtz_end={result.get('dtz_end_mm')}mm  ({result.get('retrieval')})"
            worker = None
        sleep_fn(0.02)
    return "shape" if state.want_shape_change else "quit"


def viewer_model(shape_id: str) -> Any:
    """The MuJoCo model for the viewer window — the same shape/builder record_delivery uses (nq-aligned so its recorded
    qpos replays 1:1)."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    from hymeko_rl.coin_delivery.object_curriculum import variant
    return PlanarGraspEnv(**variant(shape_id).object_spec.planar_env_kwargs()).model


def launch() -> None:                                            # pragma: no cover — needs a display (mjpython)
    import mujoco.viewer

    from hymeko_rl.env.object_spec import ObjectSpec, Shape
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

    policy = DeployedPolicy()
    zenv = PlanarGraspEnv(**ObjectSpec(shape=Shape.CYLINDER, radius=0.02).planar_env_kwargs())
    zone = (float(zenv._zone_x), float(zenv._zone_y))
    state = DemoState(sids=list(policy.samples.keys()))
    while not state.quit:
        model = viewer_model(state.shape_id)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state.want_shape_change = False
        print(f"[{_SHAPE_LABELS.get(state.shape_id, state.shape_id)}] {state.status}", flush=True)
        with mujoco.viewer.launch_passive(model, data, key_callback=make_key_callback(state)) as handle:
            drive_delivery(handle, state, policy, model, data, zone)


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
