"""Native PySide6 desktop simulator for the fast-dynamics vehicles / locomotion substrates. A window with a
**live** MuJoCo view (offscreen `mujoco.Renderer` → `QImage`, stepped by a main-thread `QTimer`), a vehicle
selector, a **terrain** toggle (flat / rally heightfields), camera modes, live telemetry (speed, lap, tilt),
and play/pause/reset. The track is drawn (waypoint pylons + racing line) on a checker floor or over terrain.

Runs on **normal python** (unlike the mjpython viewer): the `Renderer` owns its GL context and we only blit
its numpy output into a `QImage`, so Qt + MuJoCo coexist on the main thread (the qt_sim precedent, 2026-07-14).

    ./.venv/bin/python -m hymeko_rl.gui.vehicle_qt

Reuses the substrate factories + scripted experts, `viz.locomotion_render` decoration, and `qt_sim._to_pixmap`;
nothing is re-implemented (§6.1). PySide6 is a user-approved §1 Mac-runner dep (2026-07-14)."""
from __future__ import annotations

import sys
from typing import Any

import mujoco
import numpy as np

try:
    from PySide6 import QtCore, QtWidgets
except ImportError as err:
    raise SystemExit("PySide6 is required: `uv pip install --python .venv/bin/python pyside6`") from err

from hymeko_rl.env.locomotion_env import SUBSTRATES, make_f1tenth, make_race_car, make_vehicle
from hymeko_rl.gui.qt_sim import _to_pixmap
from hymeko_rl.viz.locomotion_render import _decorated_render_model

_W, _H = 900, 620
# per-vehicle (floor_size, texrepeat, marker_r, line_r, chase(dist,elev,azim), overhead(lookat,dist,elev), marker_every)
_VIS = {
    "racecar": (620, 300, 6.0, 1.5, (30, -12, 132), ((300, 130, 0), 720, -66), 12),   # Bézier circuit
    "f1tenth": (80, 80, 0.28, 0.09, (15, -38, 90), ((0, 8, 0), 27, -55), 1),
    "vehicle": (60, 60, 0.30, 0.09, (11, -38, 90), ((0, 5, 0), 19, -55), 1),
    "cheetah": (40, 40, 0.3, 0.1, (6, -12, 90), ((3, 0, 0), 10, -35), 1),
    "humanoid": (40, 40, 0.3, 0.1, (6, -12, 90), ((0, 0, 1), 10, -35), 1),
}
_TERRAIN_VEHICLES = {"vehicle", "f1tenth"}
_TERRAINS = ["flat", "hills", "bumps", "ramps"]


def _build_env(vehicle: str, terrain: str) -> Any:
    """Construct a substrate env, with terrain for the wheeled vehicles that support it."""
    t = None if terrain == "flat" else terrain
    if vehicle == "vehicle":
        return make_vehicle(max_steps=100_000, terrain=t)
    if vehicle == "f1tenth":
        return make_f1tenth(max_steps=100_000, terrain=t)
    if vehicle == "racecar":
        return make_race_car(course="circuit", max_steps=100_000)     # flat Bézier GP circuit
    return SUBSTRATES[vehicle](max_steps=100_000)


class VehicleSimWindow(QtWidgets.QMainWindow):
    """Live vehicle simulator window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HyMeKo — vehicle / locomotion simulator")
        self._env: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._rmodel: mujoco.MjModel | None = None
        self._rdata: mujoco.MjData | None = None
        self._cam = mujoco.MjvCamera()
        self._prev_xy = np.zeros(2)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.addLayout(self._controls(), 0)
        self._view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumSize(_W, _H)
        self._view.setStyleSheet("background:#0f1416;border:1px solid #26343a;border-radius:8px;")
        root.addWidget(self._view, 1)
        self.setStyleSheet(
            "QWidget{background:#0f1416;color:#e7eef0;font-size:13px;}"
            "QPushButton{background:#3fb6a0;color:#06110e;border:0;border-radius:6px;padding:8px;font-weight:700;}"
            "QComboBox{background:#161e21;border:1px solid #26343a;border-radius:5px;padding:5px;}"
            "QGroupBox{border:1px solid #26343a;border-radius:8px;margin-top:8px;padding:10px;}"
            "QGroupBox::title{color:#8ea3ab;subcontrol-origin:margin;left:8px;}")

    def _controls(self) -> QtWidgets.QVBoxLayout:
        col = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("scene")
        form = QtWidgets.QFormLayout(box)
        self._vehicle = QtWidgets.QComboBox()
        self._vehicle.addItems(["racecar", "f1tenth", "vehicle", "cheetah", "humanoid"])
        self._vehicle.currentTextChanged.connect(self._reload)
        self._terrain = QtWidgets.QComboBox()
        self._terrain.addItems(_TERRAINS)
        self._terrain.currentTextChanged.connect(self._reload)
        self._camera = QtWidgets.QComboBox()
        self._camera.addItems(["chase", "overhead"])
        self._camera.currentTextChanged.connect(self._apply_camera)
        form.addRow("vehicle", self._vehicle)
        form.addRow("terrain", self._terrain)
        form.addRow("camera", self._camera)
        col.addWidget(box)
        for label, slot in [("▶  Play", self._play), ("‖  Pause", self._pause),
                            ("↺  Reset", self._reset)]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            col.addWidget(b)
        self._telemetry = QtWidgets.QLabel("—")
        self._telemetry.setStyleSheet("font-family:monospace;background:#161e21;border-radius:6px;padding:8px;")
        self._telemetry.setWordWrap(True)
        col.addWidget(self._telemetry)
        col.addStretch(1)
        return col

    # ---- lifecycle -----------------------------------------------------------------------------
    def _reload(self) -> None:
        """(Re)build the env + decorated render model for the current vehicle/terrain selection."""
        was_running = self._timer.isActive()
        self._timer.stop()
        vehicle = self._vehicle.currentText()
        terrain = self._terrain.currentText() if vehicle in _TERRAIN_VEHICLES else "flat"
        self._terrain.setEnabled(vehicle in _TERRAIN_VEHICLES)
        self._env = _build_env(vehicle, terrain)
        self._env.reset(seed=0)
        fs, tr, mr, lr, _, _, me = _VIS[vehicle]
        if self._renderer is not None:
            self._renderer.close()
        self._rmodel = _decorated_render_model(self._env, floor_size=fs, texrepeat=tr, show_track=True,
                                               marker_r=mr, line_r=lr, marker_every=me)
        self._rmodel.vis.global_.offwidth, self._rmodel.vis.global_.offheight = _W, _H
        self._rdata = mujoco.MjData(self._rmodel)
        self._renderer = mujoco.Renderer(self._rmodel, height=_H, width=_W)
        self._prev_xy = self._env.data.xpos[self._env.torso, :2].copy()
        self._apply_camera()
        self._render_once()
        if was_running:
            self._timer.start()

    def _apply_camera(self) -> None:
        vehicle = self._vehicle.currentText()
        _, _, _, _, chase, over, _ = _VIS[vehicle]
        if self._camera.currentText() == "overhead":
            lookat, dist, elev = over
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = lookat
            self._cam.distance, self._cam.elevation, self._cam.azimuth = dist, elev, 90
        else:
            d, e, a = chase
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = int(self._env.torso)
            self._cam.distance, self._cam.elevation, self._cam.azimuth = d, e, a
        if not self._timer.isActive():
            self._render_once()

    def _render_once(self) -> None:
        self._rdata.qpos[:] = self._env.data.qpos
        self._rdata.qvel[:] = self._env.data.qvel
        mujoco.mj_forward(self._rmodel, self._rdata)
        self._renderer.update_scene(self._rdata, camera=self._cam)
        self._view.setPixmap(_to_pixmap(self._renderer.render()))

    def _tick(self) -> None:
        _, _, terminated, truncated, info = self._env.step(self._env.expert_action)
        self._render_once()
        xy = self._env.data.xpos[self._env.torso, :2]
        dt = self._env.frame_skip * float(self._env.model.opt.timestep)
        speed = float(np.hypot(*(xy - self._prev_xy)) / dt)
        self._prev_xy = xy.copy()
        wp = getattr(self._env, "_wp", None)
        ntrack = len(getattr(self._env, "_track", [])) - 1
        lap = f"waypoint {wp}/{ntrack}\n" if wp is not None and ntrack > 0 else ""
        self._telemetry.setText(f"speed  {speed:6.2f} m/s\n       {speed * 3.6:6.1f} km/h\n"
                                f"{lap}tilt   {info['upright']:+.2f}\nstep   {info['step']}")
        if terminated or truncated:
            self._env.reset(seed=info["step"])

    def _play(self) -> None:
        self._timer.start()

    def _pause(self) -> None:
        self._timer.stop()

    def _reset(self) -> None:
        self._env.reset(seed=0)
        self._prev_xy = self._env.data.xpos[self._env.torso, :2].copy()
        self._render_once()

    def closeEvent(self, event: Any) -> None:
        self._timer.stop()
        if self._renderer is not None:
            self._renderer.close()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = VehicleSimWindow()
    win.show()
    win._play()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
