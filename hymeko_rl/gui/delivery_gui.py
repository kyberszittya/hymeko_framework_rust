"""Interactive delivery GUI (tkinter) — REAL trained coin-delivery physics in a window.

Pick an object shape, click a target on the top-down map, pick a strategy, and RUN: the real pipeline
(``reconstruct_capture`` → deployed retrieval / teacher θ → ``rollout_primitive``) runs on a worker thread and the
rendered rollout plays back in the window. Thin glue only — it composes ``delivery_viewer`` (the real pipeline) and
``viz.rollout_film`` (the consolidated renderer); no physics/render is reimplemented here.

Run:  PYTHONPATH=. python -m hymeko_rl.gui.delivery_gui   (a plain window; no mjpython needed — offscreen render)
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from hymeko_rl.coin_delivery.delivery_bc.dataset import scenario_by_id
from hymeko_rl.env.object_spec import ObjectSpec, Shape
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.gui.delivery_viewer import SHAPES, STRATEGIES, DeployedPolicy, record_delivery
from hymeko_rl.viz.rollout_film import render_qpos_seq, top_down

_SHAPE_LABELS = {"O0": "coin", "O4-S": "square", "O6-T": "triangle", "O7-P": "pentagon",
                 "O8-H": "hexagon", "O3-E": "ellipse", "O9-K": "capsule"}
_CANVAS = 320                                   # target-map canvas size (px)
_XR, _YR = (-0.16, 0.16), (0.02, 0.28)          # world extents the map spans


def _zone_xy() -> "tuple[float, float]":
    env = PlanarGraspEnv(**ObjectSpec(shape=Shape.CYLINDER, radius=0.02).planar_env_kwargs())
    return float(env._zone_x), float(env._zone_y)


def _target_of_scenario(sid: str, zone: "tuple[float, float]") -> np.ndarray:
    t = getattr(scenario_by_id(sid), "target_xy", None)
    return np.asarray(t, np.float64) if t is not None else np.asarray(zone, np.float64)


class DeliveryGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("HyMeKo — delivery demo (real MuJoCo physics)")
        self.policy = DeployedPolicy()
        self.zone = _zone_xy()
        self.samples = self.policy.samples
        self.sids = list(self.samples.keys())
        self.targets = {sid: _target_of_scenario(sid, self.zone) for sid in self.sids}
        self.shape = tk.StringVar(value="O0")
        self.strategy = tk.StringVar(value=STRATEGIES[0])
        self.sid = self.sids[0]
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._frames: list[Any] = []
        self._imgtk: Any = None

        ctrl = ttk.Frame(root, padding=8)
        ctrl.grid(row=0, column=0, sticky="ns")
        ttk.Label(ctrl, text="object shape").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(ctrl, self.shape, "O0", *[s for s, _ in SHAPES],
                       command=lambda _e: self._redraw_map()).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(ctrl, text="strategy").grid(row=2, column=0, sticky="w")
        ttk.OptionMenu(ctrl, self.strategy, STRATEGIES[0], *STRATEGIES).grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(ctrl, text="target — click the map").grid(row=4, column=0, sticky="w")
        self.map = tk.Canvas(ctrl, width=_CANVAS, height=_CANVAS, bg="#f4f4f4", highlightthickness=1)
        self.map.grid(row=5, column=0, pady=4)
        self.map.bind("<Button-1>", self._on_click)
        self.run_btn = ttk.Button(ctrl, text="RUN ▶", command=self._on_run)
        self.run_btn.grid(row=6, column=0, sticky="ew", pady=6)
        self.status = ttk.Label(ctrl, text="", wraplength=_CANVAS, foreground="#333")
        self.status.grid(row=7, column=0, sticky="w")

        self.view = ttk.Label(root)                 # the rendered rollout playback
        self.view.grid(row=0, column=1, padx=8, pady=8)
        self._show_placeholder()
        self._redraw_map()
        self._set_status("ready — pick a shape + target, then RUN")

    # -- coordinate mapping (world ↔ canvas) ------------------------------------------------------------------------
    def _to_px(self, xy: np.ndarray) -> "tuple[float, float]":
        px = (xy[0] - _XR[0]) / (_XR[1] - _XR[0]) * _CANVAS
        py = (1.0 - (xy[1] - _YR[0]) / (_YR[1] - _YR[0])) * _CANVAS
        return px, py

    def _from_px(self, px: float, py: float) -> np.ndarray:
        x = _XR[0] + px / _CANVAS * (_XR[1] - _XR[0])
        y = _YR[0] + (1.0 - py / _CANVAS) * (_YR[1] - _YR[0])
        return np.array([x, y])

    # -- map --------------------------------------------------------------------------------------------------------
    def _redraw_map(self) -> None:
        self.map.delete("all")
        for sid, t in self.targets.items():
            px, py = self._to_px(t)
            sel = sid == self.sid
            self.map.create_oval(px - 4, py - 4, px + 4, py + 4,
                                 fill="#1a7d2f" if sel else "#9fd6a8", outline="#0a5d1f" if sel else "#7ab585")
        tpx, tpy = self._to_px(self.targets[self.sid])
        self.map.create_text(tpx, tpy - 12, text="target", fill="#0a5d1f", font=("TkDefaultFont", 8))

    def _on_click(self, event: Any) -> None:
        w = self._from_px(event.x, event.y)
        self.sid = min(self.sids, key=lambda s: float(np.linalg.norm(self.targets[s] - w)))
        self._redraw_map()
        self._set_status(f"target: {self.sid}")

    # -- run (worker thread) ----------------------------------------------------------------------------------------
    def _on_run(self, *_a: Any) -> None:
        self.run_btn.config(state="disabled")
        self._set_status(f"running real physics: {_SHAPE_LABELS.get(self.shape.get(), self.shape.get())} → {self.sid} …")
        threading.Thread(target=self._worker, args=(self.shape.get(), self.sid, self.strategy.get()),
                         daemon=True).start()
        self.root.after(80, self._poll)

    def _worker(self, shape: str, sid: str, strategy: str) -> None:
        try:
            res = record_delivery(self.policy, shape, sid, strategy)
            if "error" not in res:
                res["pil"] = [ImageTk.PhotoImage(Image.fromarray(f))
                              for f in render_qpos_seq(res["model"], res["qpos_seq"], top_down())]
            self._q.put(res)
        except Exception as e:                       # noqa: BLE001 — surface, never crash the UI thread
            self._q.put({"error": f"{type(e).__name__}: {e}"})

    def _poll(self) -> None:
        try:
            res = self._q.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll)
            return
        self.run_btn.config(state="normal")
        if "error" in res:
            self._set_status(res["error"])
            return
        self._frames = res["pil"]
        self._set_status(f"{res['retrieval']}  →  K6={res['k6']}  dtz_end={res['dtz_end_mm']}mm  "
                         f"({len(self._frames)} steps)")
        self._play(0)

    def _play(self, i: int) -> None:
        if i < len(self._frames):
            self._imgtk = self._frames[i]
            self.view.config(image=self._imgtk)
            self.root.after(40, lambda: self._play(i + 1))

    # -- misc -------------------------------------------------------------------------------------------------------
    def _show_placeholder(self) -> None:
        self._imgtk = ImageTk.PhotoImage(Image.new("RGB", (560, 480), "#2b2b2b"))
        self.view.config(image=self._imgtk)

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)


def main() -> None:
    root = tk.Tk()
    DeliveryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
