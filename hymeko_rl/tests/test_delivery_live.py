"""delivery_live loop logic — verified headless with a fake handle (the GL `launch` is display-only), mirroring how
viz/viewer.py unit-tests `drive_viewer`. No window, no real physics: a stub `compute` stands in for the delivery."""
from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.env.object_spec import ObjectSpec, Shape
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.gui.delivery_live import DemoState, drive_delivery, make_key_callback


def _coin_model() -> Any:
    return PlanarGraspEnv(**ObjectSpec(shape=Shape.CYLINDER, radius=0.02).planar_env_kwargs()).model


class _FakeHandle:
    """is_running / sync / user_scn — a real MjvScene backs user_scn so `_mark_target` runs for real."""

    def __init__(self, model: Any, alive: int = 400) -> None:
        self._scn = mujoco.MjvScene(model, maxgeom=100)
        self._alive = alive
        self.syncs = 0

    def is_running(self) -> bool:
        self._alive -= 1
        return self._alive > 0

    def sync(self) -> None:
        self.syncs += 1

    @property
    def user_scn(self) -> Any:
        return self._scn


def _state() -> DemoState:
    return DemoState(sids=["bank_c0_2", "bank_c1_+0.03_-0.02", "bank_c3_r5_a-15"])


def test_key_callback_sets_flags_and_cycles() -> None:
    st = _state()
    cb = make_key_callback(st)
    cb(32)
    assert st.run_requested                               # SPACE
    cb(84)
    assert st.sid_i == 1                                  # T -> next target
    cb(71)
    assert st.strategy_i == 1                             # G -> next strategy
    cb(81)
    assert st.quit                                        # Q


def test_drive_runs_compute_and_replays_into_data() -> None:
    model = _coin_model()
    data = mujoco.MjData(model)
    st = _state()
    st.run_requested = True
    calls: list[tuple[str, str, str]] = []
    qframe = np.linspace(-0.2, 0.2, model.nq)

    def stub(policy: Any, shape: str, sid: str, strategy: str) -> dict:
        calls.append((shape, sid, strategy))
        return {"qpos_seq": np.tile(qframe, (4, 1)), "k6": True, "dtz_end_mm": 12.3, "retrieval": "stub"}

    handle = _FakeHandle(model)
    drive_delivery(handle, st, policy=None, model=model, data=data, zone=(0.0, 0.16),
                   realtime=False, sleep_fn=lambda _dt: time.sleep(0.001), compute=stub)
    assert calls == [("O0", "bank_c0_2", st.strategy)]    # compute ran once for the current selection
    assert handle.syncs > 4                                # synced during idle + the 4 replay frames
    assert np.allclose(data.qpos, qframe)                  # the last real state was replayed into the viewer sim
    assert "K6=True" in st.status


def test_drive_quits_immediately_without_compute() -> None:
    model = _coin_model()
    st = _state()
    st.quit = True
    calls: list[Any] = []
    drive_delivery(_FakeHandle(model), st, policy=None, model=model, data=mujoco.MjData(model),
                   zone=(0.0, 0.16), realtime=False, sleep_fn=lambda _dt: None,
                   compute=lambda *a: calls.append(a) or {})
    assert calls == []                                     # Q pressed -> loop never computed
