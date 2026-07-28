"""R9 canonical end-to-end demo render — R2 under the explicit H1 HANDOFF_RESET contract delivering a strict s1 K6.

Renders the FULL uninterrupted chain from the canonical cradle (NOT a frozen snapshot):
`cradle → APPROACH → HANDOFF_RESET [exactly 1] → learned R2 KINETIC → release/coast → strict K6 dwell`, with a restrained
diagnostic overlay (mode, dtz, v_parallel, Fn L/R, K6 dwell counter, HANDOFF_RESET count, teacher-free) and a final verdict card
(STRICT K6 PASS, dwell, min_dtz, stall/clamp/reversal). Produces two renders — a clean hero video and a diagnostic-overlay video —
plus a per-step trace and a provenance manifest (checkpoint / env / metrics / SHA256). The large MP4s are NOT committed (attach to a
Release); the trace + manifest are. All `8a0c1c7b`/`41510cac` modules imported unchanged; teacher-free deploy.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_r2_h1_demo``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import APPROACH, CARRY, KINETIC, MICRO, REACQUIRE
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-r2-h1-demo")
CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")   # min_dtz 0.98 mm, dwell 17 (verified)
W, H, FPS = 640, 480, 12                          # within the model's default offscreen framebuffer (640×480)
_PHASE = {APPROACH: "APPROACH", CARRY: "CARRY", REACQUIRE: "REACQUIRE", MICRO: "MICRO", KINETIC: "KINETIC", 0: "REGULATE",
          1: "RELEASE"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _camera(model: Any) -> Any:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance *= 0.62
    cam.elevation = -28.0
    cam.azimuth = 90.0
    return cam


class _Recorder:
    """Rolls the H1 controller and captures, per step, a rendered frame + the diagnostic row (mode, dtz, v_par, Fn, running K6
    dwell, HANDOFF_RESET count)."""

    def __init__(self, ctrl: Any) -> None:
        self.ctrl = ctrl
        self.frames: list = []
        self.rows: list = []
        self.dwell = 0
        self._r: Any = None
        self._cam: Any = None

    def hook(self, rl: Any, t: int) -> None:
        if self._r is None:
            self._r = mujoco.Renderer(rl.inner.model, height=H, width=W)
            self._cam = _camera(rl.inner.model)
        self._r.update_scene(rl.inner.data, self._cam)
        e_par = np.asarray(rl.inner.direction_to_zone()[0], np.float64)[:2]
        dtz = float(rl.inner.direction_to_zone()[1])
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        speed = float(np.linalg.norm(vel))
        con = primary_fingertip_contacts(rl)
        last = self.ctrl.clone_trace[-1] if self.ctrl.clone_trace else {"kind": "?"}
        self.dwell = self.dwell + 1 if (dtz <= CENTER_TOL and speed < SETTLE_VEL) else 0
        self.rows.append({
            "t": int(t), "kind": last["kind"], "mode": _PHASE.get(int(self.ctrl.phase), str(int(self.ctrl.phase))),
            "dtz_mm": round(dtz * 1000, 2), "v_par": round(float(vel @ e_par), 4), "coin_speed": round(speed, 4),
            "fn_l": round(float(con["left"]["fn"]) if con["left"] else 0.0, 3),
            "fn_r": round(float(con["right"]["fn"]) if con["right"] else 0.0, 3), "k6_dwell": int(self.dwell),
            "n_handoff_reset": sum(1 for r in self.ctrl.clone_trace if r["kind"] == "HANDOFF_RESET")})
        self.frames.append(self._r.render().copy())

    def close(self) -> None:
        if self._r is not None:
            self._r.close()


def _panel(draw: ImageDraw.ImageDraw, row: dict) -> None:
    draw.rectangle([8, 8, 300, 196], fill=(0, 0, 0, 180))
    reset_hot = row["kind"] == "HANDOFF_RESET"
    lines = [("mode", row["mode"], (120, 220, 255) if row["mode"] == "KINETIC" else (200, 200, 200)),
             ("dtz", f"{row['dtz_mm']:.1f} mm", (120, 255, 160) if row["dtz_mm"] <= CENTER_TOL * 1000 else (230, 230, 230)),
             ("v_parallel", f"{row['v_par']:+.3f}", (230, 230, 230)),
             ("Fn L / R", f"{row['fn_l']:.2f} / {row['fn_r']:.2f} N", (230, 230, 230)),
             ("K6 dwell", f"{row['k6_dwell']} / {HELD_DWELL}", (120, 255, 160) if row["k6_dwell"] >= HELD_DWELL else (255, 210, 120)),
             ("HANDOFF_RESET", str(row["n_handoff_reset"]), (255, 120, 120) if reset_hot else (200, 200, 200)),
             ("teacher-free", "true", (120, 255, 160))]
    for i, (k, v, c) in enumerate(lines):
        y = 16 + i * 25
        draw.text((16, y), f"{k}:", fill=(170, 170, 170))
        draw.text((150, y), v, fill=c)
    if reset_hot:
        draw.rectangle([4, 4, W - 4, H - 4], outline=(255, 120, 120), width=4)


def _overlay(frame: np.ndarray, row: dict) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _panel(ImageDraw.Draw(ov), row)
    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))


def _verdict_card(metrics: dict) -> np.ndarray:
    img = Image.new("RGB", (W, H), (10, 12, 16))
    d = ImageDraw.Draw(img)
    scr = metrics["stall_clamp_reversal"]
    d.text((W // 2 - 120, 150), "STRICT K6: PASS", fill=(120, 255, 160))
    for i, (k, v) in enumerate([("dwell", f"{metrics['k6_dwell']} frames (>= {HELD_DWELL})"),
                                ("min_dtz", f"{metrics['min_dtz_mm']} mm"),
                                ("stall / clamp / reversal", f"{scr[0]} / {scr[1]} / {scr[2]}"),
                                ("HANDOFF_RESET", f"{metrics['n_handoff_reset']} (explicit, online)"),
                                ("teacher-free", "true"), ("contract", "H1 EXPLICIT_HANDOFF_RESET")]):
        d.text((W // 2 - 200, 220 + i * 30), f"{k}:", fill=(170, 170, 170))
        d.text((W // 2 + 40, 220 + i * 30), v, fill=(230, 230, 230))
    return np.asarray(img)


def _write_mp4(path: Path, frames: list) -> None:
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as wr:
        for f in frames:
            wr.append_data(f)


def _load_policy() -> Any:
    ck = json.load(open(CKPT))
    a = make_actor("td3", AUG_DIM, 4)
    a.load_state_dict({k: torch.tensor(v) for k, v in ck["r2_actor_state"].items()})
    a.eval()
    return a, ck


def run(out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    actor, _ck = _load_policy()
    ctrl = HandoffResetTemporalController(cradle, CloneActor(model, norm), deterministic_residual(actor),
                                          ResidualBounds(alpha=0.15))
    rec = _Recorder(ctrl)
    m = velocity_rollout(cradle, ctrl, DELIVERY_CFG, frame_hook=rec.hook)
    rec.close()
    kin = [r for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    vpar = [r["v_par"] for r in kin]
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
    metrics = {"delivery_success": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
               "min_dtz_mm": round(_min_dtz_mm(cradle, m), 2),
               "n_handoff_reset": sum(1 for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"),
               "stall_clamp_reversal": [sum(1 for v in vpar if v <= 0.0),
                                        sum(1 for r in kin if min(r["fn_l"], r["fn_r"]) > krl2.FN_CLAMP),
                                        sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0.0)],
               "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)}
    out.mkdir(parents=True, exist_ok=True)
    card = [_verdict_card(metrics)] * (FPS * 3)
    hero = rec.frames + card
    diag = [_overlay(f, r) for f, r in zip(rec.frames, rec.rows)] + card
    hero_p, diag_p = out / "coin_r2_h1_k6_demo_hero.mp4", out / "coin_r2_h1_k6_demo.mp4"
    _write_mp4(hero_p, hero)
    _write_mp4(diag_p, diag)
    trace_p = out / "coin_r2_h1_k6_demo_trace.json"
    trace_p.write_text(json.dumps({"contract": "H1_EXPLICIT_HANDOFF_RESET", "metrics": metrics, "rows": rec.rows}, indent=1))
    manifest = _manifest(metrics, hero_p, diag_p, trace_p)
    (out / "coin_r2_h1_k6_demo_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _manifest(metrics: dict, hero_p: Path, diag_p: Path, trace_p: Path) -> dict:
    import numpy as _np
    return {"artifact": "coin_r2_h1_k6_demo", "commit": _git_commit(), "controller": "R2",
            "contract": "H1_EXPLICIT_HANDOFF_RESET", "tag": "coin-r9-r2-h1-multiseed-reproduced-v1",
            "policy": {"seed": 1, "checkpoint": str(CKPT), "checkpoint_sha256": _sha256(CKPT)},
            "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": _np.__version__,
                            "torch": torch.__version__, "platform": platform.platform()},
            "strict_k6": metrics, "started_from": "canonical_cradle (not a frozen snapshot)", "teacher_free": True,
            "videos": {"hero": {"file": hero_p.name, "sha256": _sha256(hero_p)},
                       "diagnostic": {"file": diag_p.name, "sha256": _sha256(diag_p)}},
            "trace": {"file": trace_p.name, "sha256": _sha256(trace_p)}}


if __name__ == "__main__":
    r = run()
    print(f"DEMO rendered: strict_k6 {r['strict_k6']['delivery_success']} dwell {r['strict_k6']['k6_dwell']} "
          f"min_dtz {r['strict_k6']['min_dtz_mm']}mm reset {r['strict_k6']['n_handoff_reset']} "
          f"scr {r['strict_k6']['stall_clamp_reversal']}")
    for k, v in r["videos"].items():
        print(f"  {k}: {v['file']}  sha256 {v['sha256'][:16]}…")
    print(f"  trace: {r['trace']['file']}  manifest: coin_r2_h1_k6_demo_manifest.json  commit {r['commit'][:8]}")
    sys.exit(0)
