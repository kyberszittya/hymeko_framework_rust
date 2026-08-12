"""Live MuJoCo viewer demo of the REAL trained coin-delivery physics.

Thin glue over the library: for a chosen object shape, target scenario, and strategy it runs the *real* pipeline —
``reconstruct_capture`` (exact-zero reach → certified capture) → the deployed retrieval θ (or the scenario's teacher θ)
→ ``rollout_primitive`` (the governed Δτ transport) — records the real per-step physical state, and replays it in a live
``mujoco.viewer`` window (orbit / zoom / pan). No reimplementation, no kinematics: every frame is a real MuJoCo state
produced by the frozen dynamics contract, and the K6 verdict is the frozen monitor's.

macOS needs the viewer on the main thread — run with **mjpython**:

    PYTHONPATH=. .venv/bin/mjpython -m hymeko_rl.gui.delivery_viewer

Keys (in the terminal that launched it): the loop prompts for shape / scenario / strategy and RUNs each real delivery.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    bc_context, fresh_rig, reconstruct_capture, scenario_by_id)
from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer, clip_theta
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

_FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
# Deployable target scenarios (the coin's frozen retrieval covers these) — each id encodes the relocated target.
SCENARIOS = ["bank_c0_2", "bank_c1_+0.03_-0.02", "bank_c2_+0.025_-0.015", "bank_c3_r5_a-15", "bank_c3_r9_a-30"]
SHAPES = ["O0", "O4-S", "O6-T", "O7-P", "O8-H", "O3-E", "O9-K"]      # coin / square / triangle / penta / hexa / ellipse / capsule
STRATEGIES = ["TD3 (deployed retrieval)", "teacher (CEM θ)"]


class DeployedPolicy:
    """The frozen descriptor→θ retrieval table (the deployed coin policy) + the teacher θ per scenario."""

    def __init__(self) -> None:
        fp = json.loads(_FROZEN.read_text())
        x = np.asarray(fp["table"]["X"], np.float64)
        self.std = Standardizer.fit(x)
        self.xs = self.std.transform(x)
        self.ids: list[str] = fp["table"]["scenario_ids"]
        self.theta = np.asarray(fp["table"]["theta"], np.float64)
        self.samples = {s.scenario_id: s for s in _load_dataset(_DATASET)}

    def theta_for(self, sid: str, strategy: str) -> "tuple[np.ndarray, str]":
        smp = self.samples[sid]
        if strategy.startswith("teacher"):
            return np.asarray(smp.theta, np.float64), f"teacher θ ({sid})"
        i = int(np.argmin(np.linalg.norm(self.xs - self.std.transform(np.asarray(smp.x, np.float64)), axis=1)))
        return self.theta[i], f"retrieved {self.ids[i]}"


def record_delivery(policy: DeployedPolicy, shape: str, sid: str, strategy: str) -> dict[str, Any]:
    """Run the REAL pipeline and record the per-step full ``qpos`` (real physics states) for playback.

    # Postconditions: returns {model, qpos_seq (T×nq), k6, dtz_end_mm, retrieval, seed} — or {error} if no grasp."""
    cfg, conf, obj = bc_context()
    rig = fresh_rig() if shape == "O0" else _rig(object_spec=variant(shape).object_spec)
    smp = policy.samples[sid]
    rc = reconstruct_capture(rig, cfg, conf, obj, scenario_by_id(sid), smp.seed)
    snap = rc.result.outcome.snapshot if rc is not None else None
    if snap is None:                              # the round family does not certify a straddle (the R11.7 cert wall)
        return {"error": f"{shape}: no certified grasp — this shape does not certify a straddle here"}
    theta, retrieval = policy.theta_for(sid, strategy)
    qseq: list[np.ndarray] = []
    model_box: list[Any] = []

    def hook(rl: Any, _t: int) -> None:
        if not model_box:
            model_box.append(rl.inner.model)
        qseq.append(rl.inner.data.qpos.copy())

    m = rollout_primitive(snap, clip_theta(theta), CLOSED_LOOP_CFG, frame_hook=hook)
    return {"model": model_box[0], "qpos_seq": np.asarray(qseq), "k6": bool(delivery_success(m, CLOSED_LOOP_CFG)),
            "dtz_end_mm": round(float(m["dtz_end"]) * 1000, 1), "retrieval": retrieval, "seed": smp.seed}


def render_gif(model: Any, qpos_seq: np.ndarray, path: Path, fps: int = 30) -> None:
    """Offscreen render of a recorded real-physics trajectory to a GIF (no display needed) — reuses the codebase's
    ``mujoco.Renderer`` + top-down camera. The canonical headless visualization (same as the video_* experiments)."""
    from PIL import Image

    from hymeko_rl.experiments.video_r11_6c_composition import _cam
    from hymeko_rl.viz.rollout_overlay import encode_clip
    renderer = mujoco.Renderer(model, height=480, width=560)
    data = mujoco.MjData(model)
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=_cam())
        frames.append(Image.fromarray(np.asarray(renderer.render(), np.uint8)))
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_clip(frames, path, fps=fps)


def _play(model: Any, qpos_seq: np.ndarray, fps: float = 30.0) -> None:
    """Replay a recorded real-physics trajectory (per-step qpos) in a live viewer window."""
    data = mujoco.MjData(model)
    dt = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for q in qpos_seq:
            if not viewer.is_running():
                return
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
        for _ in range(int(fps * 1.5)):                    # hold the final K6 pose
            if not viewer.is_running():
                return
            viewer.sync()
            time.sleep(dt)


def main() -> None:
    policy = DeployedPolicy()
    shape, sid, strat = "O0", SCENARIOS[1], STRATEGIES[0]
    print("HyMeKo delivery viewer — REAL physics (mujoco.viewer).  Ctrl-C to quit.")
    while True:
        print(f"\nshape={shape}  scenario(target)={sid}  strategy={strat}")
        print(f"  s=shape ({'/'.join(SHAPES)})  t=scenario  g=strategy  ENTER=run  q=quit")
        cmd = input("> ").strip().lower()
        if cmd == "q":
            return
        if cmd == "s":
            shape = SHAPES[(SHAPES.index(shape) + 1) % len(SHAPES)]
            continue
        if cmd == "t":
            sid = SCENARIOS[(SCENARIOS.index(sid) + 1) % len(SCENARIOS)]
            continue
        if cmd == "g":
            strat = STRATEGIES[(STRATEGIES.index(strat) + 1) % len(STRATEGIES)]
            continue
        res = record_delivery(policy, shape, sid, strat)
        if "error" in res:
            print(f"  {res['error']}")
            continue
        print(f"  {res['retrieval']}  →  K6={res['k6']}  dtz_end={res['dtz_end_mm']}mm  "
              f"({len(res['qpos_seq'])} real steps) — opening viewer…")
        _play(res["model"], res["qpos_seq"])


def render_headless(out: Path = Path("reports/2026-08-12-delivery-viewer")) -> None:
    """Render the two real K6 deliveries (coin + triangle-with-teacher) to GIFs — no display; canonical offscreen viz."""
    policy = DeployedPolicy()
    for shape, sid, strat, name in (("O0", "bank_c1_+0.03_-0.02", "TD3 (deployed retrieval)", "coin_td3"),
                                    ("O6-T", "bank_c1_+0.03_-0.02", "teacher (CEM θ)", "triangle_teacher")):
        res = record_delivery(policy, shape, sid, strat)
        if "error" in res:
            print(f"  {name}: {res['error']}", flush=True)
            continue
        p = out / f"{name}_K6-{res['k6']}_{res['dtz_end_mm']}mm.gif"
        render_gif(res["model"], res["qpos_seq"], p)
        print(f"  {name}: {res['retrieval']} K6={res['k6']} dtz={res['dtz_end_mm']}mm -> {p}", flush=True)


if __name__ == "__main__":
    import sys
    if "--render" in sys.argv:
        render_headless()
    else:
        main()
