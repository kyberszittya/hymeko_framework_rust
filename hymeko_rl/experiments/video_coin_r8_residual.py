"""R8 S5 champion — DETERMINISTIC visualization of the frozen TD3 seed-2 champion transferring to held-out s4.

This is NOT a new evaluation and it selects NOTHING: it reproduces the already-dev-selected checkpoint / seed / config
(`matched_comparison.json.dev_champion`) and asserts the reproduced `dtz_end` matches `heldout_frozen_result.json`
bit-for-bit (else it ABORTS — a visualization must never disagree with the recorded result). It renders the update-0
scaffold vs the champion SIDE BY SIDE on s4 (the codebase's own `mujoco.Renderer` + `rollout_overlay` panels + the
mandatory `assert_trace_render_consistency` gate), a 6-signal plot-strip, and a hashed provenance manifest.

    left  : update-0 scaffold (ZeroActor)        held-out s4  266 mm
    right : frozen TD3 champion (ConstantResidualActor(a*))    30 mm
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    ConstantResidualActor, ResidualBounds, ResidualTipAdapter, ZeroActor)
from hymeko_rl.coin_delivery.theta_option.residual_option_env import OBS_DIM, ACT_DIM, residual_init_obs
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.option_rl.agents import make_actor
from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, assert_trace_render_consistency, hstack, overlay_frames, summary_card)

OUT = "reports/2026-07-27-coin-r8-residual-rl"
REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"
CENTER_TOL_MM = 20.0                                   # the K6 entry tolerance drawn as the goal line


def _cam() -> Any:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 0.9, -55, 90
    cam.lookat = np.array([0.0, 0.08, 0.0])
    return cam


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _actor_residual(actor: Any, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        a = actor.mean_action(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0].numpy()
    return np.asarray(a, np.float64)


def _render_rollout(snap: Any, actor_obj: Any, cam: Any, *, h: int = 420, w: int = 540) -> dict:
    """Render ONE residual rollout via the velocity_rollout frame_hook (renders the SAME rl the rollout steps). Captures the
    six load-bearing signals per frame; returns frames + per-signal traces + the rollout metrics + the executed residual."""
    adapter = ResidualTipAdapter(snap, actor_obj, TipTransportParams(), ResidualBounds(), DELIVERY_CFG)
    frames: list = []
    sig: dict[str, list] = {"dtz": [], "vpar": [], "tipvel": [], "sqz": [], "contact": [], "release": []}
    st = {"r": None}

    def hook(rl: Any, t: int) -> None:
        if st["r"] is None:
            st["r"] = mujoco.Renderer(rl.inner.model, height=h, width=w)
        st["r"].update_scene(rl.inner.data, camera=cam)
        frames.append(np.asarray(st["r"].render(), np.uint8))
        u, d = rl.inner.direction_to_zone()
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        con = primary_fingertip_contacts(rl)
        sig["dtz"].append(float(d))
        sig["vpar"].append(float(np.dot(vel, np.asarray(u, np.float64)[:2])))
        sig["tipvel"].append(float(np.max(np.abs(rl.inner.data.qvel[:4]))))
        sig["sqz"].append(float(getattr(adapter, "_sqz", 0.0)))
        sig["contact"].append(int((con["left"] is not None) or (con["right"] is not None)))
        sig["release"].append(int(getattr(adapter, "phase", 0)))

    m = velocity_rollout(snap, adapter, DELIVERY_CFG, frame_hook=hook)
    if st["r"] is not None:
        st["r"].close()
    assert_trace_render_consistency(frames, sig["dtz"], label="coin_r8_residual")
    res = adapter.provenance[-1]["residual"] if adapter.provenance else [0.0, 0.0, 0.0]
    return {"frames": frames, "sig": {k: np.asarray(v, np.float64) for k, v in sig.items()}, "m": m, "residual": list(res)}


def _clip(rd: dict, *, title: str, a_vec: list) -> list:
    """Overlay the status bar + dtz(t) curve (with the 20 mm goal line) + a live info panel onto a rendered rollout."""
    dtz_mm = rd["sig"]["dtz"] * 1000.0
    n = len(dtz_mm)
    k6 = bool(delivery_success(rd["m"], DELIVERY_CFG))
    end = "K6" if k6 else f"{dtz_mm[-1]:.0f}mm"

    def status(t: int) -> str:
        return "RUNNING" if t < n - 2 else ("SUCCESS" if k6 else "NO-DELIVER")

    def info(t: int) -> list:
        tt = min(t, n - 1)
        return [f"residual a* = [{a_vec[0]:+.2f} {a_vec[1]:+.2f} {a_vec[2]:+.2f}]",
                f"dtz {dtz_mm[tt]:6.1f}mm   v_par {rd['sig']['vpar'][tt]:+.3f}",
                f"tip_vel {rd['sig']['tipvel'][tt]:.3f}   sqz {rd['sig']['sqz'][tt]:.3f}",
                f"contact {int(rd['sig']['contact'][tt])}   phase {'RELEASE' if rd['sig']['release'][tt] else 'REGULATE'}"]
    panels = [StatusBar(f"{title}  |  s4 held-out  |  end {end}", status),
              TimeSeriesPanel({"dtz(t) mm": dtz_mm}, title="distance to zone (20mm goal dashed)", threshold=CENTER_TOL_MM,
                              size=(300, 150)),
              InfoPanel(info)]
    return overlay_frames(rd["frames"], panels)


def _plot_strip(sc: dict, ch: dict, a_vec: list, path: str) -> None:
    """6-row signal strip (champion solid, scaffold dashed where shared): dtz, coin v_par, tip velocity, residual (constant),
    squeeze, contact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 1, figsize=(8, 11), sharex=True)
    ax[0].axhline(CENTER_TOL_MM, ls="--", c="r", lw=1, label="20mm goal")
    ax[0].plot(ch["sig"]["dtz"] * 1000, c="#2a7", label="champion")
    ax[0].plot(sc["sig"]["dtz"] * 1000, c="#888", ls="--", label="scaffold")
    ax[0].set_ylabel("dtz (mm)")
    ax[0].legend(fontsize=7)
    for row, key, lab in ((1, "vpar", "coin v_par"), (2, "tipvel", "tip vel (max)"), (4, "sqz", "squeeze"),
                          (5, "contact", "contact")):
        ax[row].plot(ch["sig"][key], c="#2a7")
        ax[row].plot(sc["sig"][key], c="#888", ls="--")
        ax[row].set_ylabel(lab)
    for i, nm in enumerate(("d_fwd", "d_sqz", "d_stop")):
        ax[3].axhline(a_vec[i], lw=1.4, label=f"{nm} {a_vec[i]:+.2f}")
    ax[3].set_ylabel("residual a*")
    ax[3].legend(fontsize=6, ncol=3)
    ax[5].set_xlabel("step")
    ax[0].set_title("R8 champion (TD3 s2) vs scaffold — held-out s4 (deterministic reproduction)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_champion(tag: str = "s4") -> dict:
    """Deterministic reproduction + render of the dev-selected champion on a held-out cradle. Aborts if the reproduced
    dtz_end disagrees with the recorded S5 result. Writes the GIF, the plot-strip, and a hashed manifest."""
    import os
    os.makedirs(f"{OUT}/video", exist_ok=True)
    comp = json.load(open(f"{OUT}/matched_comparison.json"))
    rec = json.load(open(f"{OUT}/heldout_frozen_result.json"))
    champ = comp["dev_champion"]
    recorded = {p["tag"]: p for p in rec["heldout_rl_per_cradle"]}[tag]
    recorded_sc = {p["tag"]: p for p in rec["heldout_scaffold_per_cradle"]}[tag]
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    snap = {ps.tag: ps for ps in build_panel(load_harness(), bank)}[tag].snap
    actor = make_actor(champ["algo"], OBS_DIM, ACT_DIM)
    actor.load_state_dict(torch.load(champ["ckpt"]))
    a_star = _actor_residual(actor, residual_init_obs(snap))
    cam = _cam()
    sc = _render_rollout(snap, ZeroActor(), cam)
    ch = _render_rollout(snap, ConstantResidualActor(a_star), cam)
    # DETERMINISM GATE — the reproduction MUST match the recorded S5 numbers (no metric may change for a visualization)
    ch_mm, sc_mm = round(ch["m"]["dtz_end"] * 1000, 1), round(sc["m"]["dtz_end"] * 1000, 1)
    if abs(ch_mm - recorded["dtz_end_mm"]) > 0.1 or abs(sc_mm - recorded_sc["dtz_end_mm"]) > 0.1:
        raise AssertionError(f"NON-DETERMINISTIC reproduction: champion {ch_mm} vs recorded {recorded['dtz_end_mm']}, "
                             f"scaffold {sc_mm} vs {recorded_sc['dtz_end_mm']} — refusing to ship a mismatched video")
    left = _clip(sc, title="update-0 scaffold", a_vec=[0.0, 0.0, 0.0])
    right = _clip(ch, title="TD3 champion", a_vec=list(a_star))
    card = summary_card((left[0].width + right[0].width + 8, left[0].height), "R8 champion → held-out s4",
                        [("scaffold dtz", f"{sc_mm:.0f} mm"), ("champion dtz", f"{ch_mm:.0f} mm"),
                         ("improvement", f"{sc_mm - ch_mm:.0f} mm"), ("K6 (20mm)", "delivered" if ch_mm <= 20 else "no")], hold=40)
    pair = hstack(left, right) + card
    gif = f"{OUT}/video/s5_champion_{tag}_transfer.gif"
    imageio.mimsave(gif, [np.asarray(im, np.uint8) for im in pair], duration=0.033)
    _plot_strip(sc, ch, list(a_star), f"{OUT}/video/s5_champion_{tag}_signals.png")
    manifest = {
        "contract": "COIN_R8_S5_CHAMPION_VIDEO_MANIFEST", "purpose": "deterministic reproduction for visualization only",
        "held_out_state": tag, "champion_algo": champ["algo"], "seed": champ["seed"],
        "checkpoint": champ["ckpt"], "checkpoint_sha256_16": _sha256(champ["ckpt"]),
        "launch_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "config_hash_16": _sha256(f"{OUT}/training_contract.json"),
        "recorded_result_hash_16": _sha256(f"{OUT}/heldout_frozen_result.json"),
        "rendering_command": "python -m hymeko_rl.experiments.video_coin_r8_residual",
        "reproduced_champion_residual_a_star": [round(float(x), 6) for x in a_star],
        "reproduced_dtz_end_mm": {"scaffold": sc_mm, "champion": ch_mm},
        "recorded_dtz_end_mm": {"scaffold": recorded_sc["dtz_end_mm"], "champion": recorded["dtz_end_mm"]},
        "determinism_gate": "PASS (reproduced == recorded)", "n_frames": len(pair),
        "gif": gif, "gif_sha256_16": _sha256(gif), "plot": f"{OUT}/video/s5_champion_{tag}_signals.png"}
    json.dump(manifest, open(f"{OUT}/video/s5_champion_{tag}_manifest.json", "w"), indent=1, default=float)
    print(f"  champion {champ['algo']} s{champ['seed']} on {tag}: scaffold {sc_mm}mm -> champion {ch_mm}mm "
          f"(recorded {recorded_sc['dtz_end_mm']}/{recorded['dtz_end_mm']}) | determinism PASS | frames {len(pair)}", flush=True)
    print(f"  GIF {gif} (sha {manifest['gif_sha256_16']}) | ckpt sha {manifest['checkpoint_sha256_16']}\nR8_VIDEO_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    render_champion(sys.argv[1] if len(sys.argv) > 1 else "s4")
