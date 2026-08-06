"""R11.6C-D demo video — teacher-free exact-zero coin delivery with verified hybrid control.

Renders the deployed R11.6C decision honestly: from a certified capture handoff, the retrieval policy selects ONE stored
theta (no CEM, no rollout-oracle, no teacher query at run time) and the pure transport primitive drives the coin to
strict K6. Shows a success montage (canonical + relocated targets) and the localized r9 far-angle undershoot with the
SAME visualization, plus overlays: coin distance-to-zone, retrieved bank element, the runtime no-teacher guarantees, and
the final certificate. Reuses the codebase's ``mujoco.Renderer`` + ``viz.rollout_overlay``. Selects nothing; viz only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer, clip_theta
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset
from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, encode_clip, overlay_frames, summary_card)

FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
OUT = Path("reports/2026-08-06-r11-6c-demo-video")
CENTER_TOL_MM = 20.0
H, W = 460, 560
SUCCESS = ["bank_c0_2", "bank_c1_+0.03_-0.02", "bank_c2_+0.025_-0.015", "bank_c3_r5_a-15"]
FAILURE = "bank_c3_r9_a-30"


def _cam() -> Any:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 0.9, -55, 90
    cam.lookat = np.array([0.0, 0.08, 0.0])
    return cam


class _Filmer:
    """Captures (frame, coin dtz mm) per governed delivery step from a ``mujoco.Renderer`` (read-only)."""

    def __init__(self, cam: Any) -> None:
        self.cam, self.r, self.frames, self.dtz_mm = cam, None, [], []

    def __call__(self, rl: Any, _t: int) -> None:
        if self.r is None:
            self.r = mujoco.Renderer(rl.inner.model, height=H, width=W)
        self.r.update_scene(rl.inner.data, camera=self.cam)
        self.frames.append(np.asarray(self.r.render(), np.uint8))
        self.dtz_mm.append(float(rl.inner.direction_to_zone()[1]) * 1000.0)

    def close(self) -> None:
        if self.r is not None:
            self.r.close()


def _nearest(x: np.ndarray, Xs: np.ndarray, ids: "list[str]", theta: list, std: Standardizer) -> "tuple[str, np.ndarray]":
    i = int(np.argmin(np.linalg.norm(Xs - std.transform(x), axis=1)))
    return ids[i], np.asarray(theta[i], np.float64)


def deliver_clip(sid: str, smp: Any, fp: dict, Xs: np.ndarray, std: Standardizer, cfg: Any, conf: Any, obj: Any) -> dict:
    """Render one scenario's teacher-free retrieval delivery; return frames + overlay facts."""
    rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(sid), smp.seed)
    if rc is None:
        return {"sid": sid, "error": "no_certified_grasp"}
    snap = rc.result.outcome.snapshot
    bank_id, theta = _nearest(np.asarray(smp.x, np.float64), Xs, fp["table"]["scenario_ids"], fp["table"]["theta"], std)
    film = _Filmer(_cam())
    m = rollout_primitive(snap, clip_theta(theta), CLOSED_LOOP_CFG, frame_hook=film)
    film.close()
    k6 = bool(delivery_success(m, CLOSED_LOOP_CFG))
    dtz = np.asarray(film.dtz_mm, np.float64)
    return {"sid": sid, "bank_id": bank_id, "frames": film.frames, "dtz_mm": dtz, "k6": k6,
            "final_dtz": round(float(m["dtz_end"]) * 1000, 2), "n": len(film.frames)}


def _overlay(clip: dict) -> list:
    dtz, n, sid = clip["dtz_mm"], clip["n"], clip["sid"]
    end = f"STRICT K6  {clip['final_dtz']:.1f}mm" if clip["k6"] else f"undershoot {clip['final_dtz']:.0f}mm"

    def status(t: int) -> str:
        return f"TRANSPORT   step {min(t, n - 1) + 1}/{n}" if t < n - 2 else f"DONE — {end}"

    def info(_t: int) -> list:
        return [f"scenario   {sid}", f"retrieved  {clip['bank_id']}", "runtime    no delivery CEM",
                "           no rollout oracle", "           no teacher query", f"result     {end}"]

    panels = [StatusBar("TEACHER-FREE RETRIEVAL DELIVERY  ->  strict K6", status),
              TimeSeriesPanel({"coin dtz (mm)": dtz}, title="coin distance to target zone (20mm goal dashed)",
                              threshold=CENTER_TOL_MM, size=(320, 150)), InfoPanel(info)]
    return overlay_frames(clip["frames"], panels)


def render(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fp = json.loads(FROZEN.read_text())
    X = np.asarray(fp["table"]["X"], np.float64)
    std = Standardizer.fit(X)
    Xs = std.transform(X)
    cfg, conf, obj = bc_context()
    smps = {s.scenario_id: s for s in _load_dataset(B1_DATASET)}
    clip: list = []
    facts: list = []
    for sid in [*SUCCESS, FAILURE]:
        c = deliver_clip(sid, smps[sid], fp, Xs, std, cfg, conf, obj)
        if "error" in c:
            print(f"  {sid}: {c['error']}", flush=True)
            continue
        clip += _overlay(c)
        facts.append({"sid": sid, "bank_id": c["bank_id"], "k6": c["k6"], "final_dtz": c["final_dtz"]})
        print(f"  {sid:22s} retrieved {c['bank_id']:22s} K6={c['k6']} dtz={c['final_dtz']}mm ({c['n']} frames)", flush=True)
    size = (clip[0].width, clip[0].height)
    n_k6 = sum(1 for f in facts if f["k6"])
    card = summary_card(size, "Teacher-Free Exact-Zero Coin Delivery with Verified Hybrid Control",
                        [("runtime", "no CEM / no oracle / no teacher"), ("deliveries shown", f"{n_k6} K6 + 1 localized limit"),
                         ("delivery decision", "top-1 retrieval of a stored theta"),
                         ("next", "object-conditioned pushing")], hold=70)
    clip = clip + card
    mp4 = encode_clip(clip, out / "r11_6c_demo.mp4", fps=20)
    gif = out / "r11_6c_demo.gif"
    imageio.mimsave(gif, [np.asarray(im, np.uint8) for im in clip], duration=0.05)
    manifest = {"contract": "R11_6C_DEMO_VIDEO", "purpose": "deterministic visualization only (selects nothing)",
                "facts": facts, "n_frames": len(clip), "mp4": str(mp4), "gif": str(gif)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(f"  MP4 {mp4}\n  GIF {gif}\nR11_6C_DEMO_VIDEO_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    render()
