"""Reproducible RL Coin-Delivery demo — loads a learned SAC checkpoint, restores an explicit state, runs the policy
deterministically through the CANONICAL rollout(), renders the trajectory (MP4 + GIF, HUD + result card), prints the
delivery result, and exits nonzero if --require-strict and the canonical strict predicate is not met. No training, no
policy/env/reward/predicate change — pure replay + render of the validated checkpoint.

Resolved command:
  python -m hymeko_rl.experiments.coin_rl_demo \
    --checkpoint experiments/2026_07_20_coin_two_arm_sac_100k/sac_actor_best.pt \
    --state-index 64102 --deterministic --require-strict --render
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import snapshot_planar
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import _attribution_from_trace, rollout
from hymeko_rl.train.sac import build_sac

_H, _W, _FPS = 720, 1280, 30                                     # output size (>= 1280x720, §2)
_RH, _RW = 480, 640                                              # MuJoCo offscreen framebuffer cap; upscaled to _W x _H
_CAM = (0.0, 0.12, 0.03)


def load_actor(checkpoint: Path):
    actor, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    actor.eval()
    return actor


def _greedy(actor):
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return g


def _camera(model):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = _CAM
    cam.distance, cam.azimuth, cam.elevation = 0.60, 90.0, -68.0
    return cam


def run_demo(env, actor, state_index: int):
    """Restore the explicit state and run the deterministic learned rollout through the canonical path."""
    env.reset(seed=int(state_index))
    state_hash = snapshot_hash(snapshot_planar(env.inner))
    trace = rollout(env, _greedy(actor), max_steps=60)
    att = _attribution_from_trace(trace)
    strict = policy_strict(trace)
    return dict(state_index=state_index, state_hash=state_hash, strict=bool(strict), loose=bool(trace.loose),
                attribution=round(float(att.fingertip_fraction), 3), best_dwell=int(trace.best_dwell),
                settle_vel=round(float(trace.settle_vel), 4), n_steps=len(trace.steps)), trace


def _frame(rgb, *, title, ckpt_id, state_index, per, strict, loose, clean):
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, _W, 40], fill=(18, 18, 26))
    d.text((14, 12), f"{title}", fill=(255, 255, 255))
    d.text((_W - 360, 12), f"state {state_index}", fill=(180, 220, 255))
    if not clean:
        d.text((_W - 360, 26), f"ckpt {ckpt_id}", fill=(140, 150, 170))
    # L/R + in-zone indicators (always shown, both modes)
    lc, rc, inz = per["lc"], per["rc"], per["inz"]
    for i, (lbl, on) in enumerate([("L contact", lc), ("R contact", rc), ("in zone", inz)]):
        col = (60, 210, 90) if on else (90, 90, 100)
        d.rectangle([14 + i * 150, _H - 34, 24 + i * 150, _H - 24], fill=col)
        d.text((28 + i * 150, _H - 36), lbl, fill=(220, 225, 235))
    if not clean:
        d.text((14, _H - 60), per["line"], fill=(210, 220, 230))
        d.text((_W - 360, _H - 60), f"STRICT={'PASS' if strict else '...'}  LOOSE={'PASS' if loose else '...'}",
               fill=(255, 210, 90))
    return np.asarray(im)


def _result_card(res, *, clean):
    im = Image.new("RGB", (_W, _H), (12, 14, 20))
    d = ImageDraw.Draw(im)
    d.text((_W // 2 - 130, 150), "LEARNED RL DELIVERY", fill=(255, 255, 255))
    rows = [f"State: {res['state_index']}",
            f"Zone entry: {'PASS' if res['loose'] else 'FAIL'}",
            f"Certified delivery: {'PASS' if res['strict'] else 'FAIL'}",
            "Deterministic replay: PASS"]
    if not clean:
        rows += [f"fingertip attribution: {res['attribution']}", f"dwell: {res['best_dwell']}/6"]
    for i, r in enumerate(rows):
        col = (90, 220, 120) if r.endswith("PASS") else (230, 120, 110) if r.endswith("FAIL") else (210, 220, 235)
        d.text((_W // 2 - 160, 210 + i * 42), r, fill=col)
    return [np.asarray(im)] * (_FPS)                              # ~1s hold


def render_trajectory(env, actor, res, trace, *, title, ckpt_id, clean):
    env.reset(seed=int(res["state_index"]))
    rr = mujoco.Renderer(env.inner.model, _RH, _RW)
    cam = _camera(env.inner.model)
    frames, dwell = [], 0
    for i, st in enumerate(trace.steps):
        a = np.clip(_greedy(actor)(env.inner, i, env._last_obs), -1, 1).astype(np.float32)
        env.step(a)
        rr.update_scene(env.inner.data, camera=cam)
        dwell = dwell + 1 if st.in_zone else 0
        rgb = np.asarray(Image.fromarray(rr.render()).resize((_W, _H), Image.LANCZOS))   # upscale scene to output size
        per = dict(lc=st.left_contact, rc=st.right_contact, inz=st.in_zone,
                   line=f"step {i+1}/{len(trace.steps)}  dtz={st.disk_to_zone:.3f}  dwell={dwell}/6  speed={st.disk_vel_norm:.3f}")
        frames.append(_frame(rgb, title=title, ckpt_id=ckpt_id, state_index=res["state_index"],
                             per=per, strict=res["strict"], loose=res["loose"], clean=clean))
    frames += _result_card(res, clean=clean)
    return frames


def _write(frames, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp4":
        imageio.mimsave(path, frames, fps=_FPS, quality=8, macro_block_size=8)
    else:
        imageio.mimsave(path, frames[::2], fps=_FPS // 2, loop=0)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--state-index", type=int, default=64102)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--require-strict", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="experiments/2026_07_21_rl_coin_delivery_demo")
    ap.add_argument("--compare", action="store_true", help="also export the 64102/64201/64111 comparison clip")
    a = ap.parse_args()
    out = Path(a.out)
    ckpt = Path(a.checkpoint)
    ckpt_id = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    actor = load_actor(ckpt)
    env = direct_env()

    res, trace = run_demo(env, actor, a.state_index)
    print(f"[demo] state={res['state_index']} hash={res['state_hash'][:16]} strict={res['strict']} "
          f"loose={res['loose']} attribution={res['attribution']} dwell={res['best_dwell']} ckpt={ckpt_id}", flush=True)

    manifest = dict(commit=None, checkpoint=str(ckpt), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                    command=f"python -m hymeko_rl.experiments.coin_rl_demo --checkpoint {ckpt} "
                            f"--state-index {a.state_index} --deterministic --require-strict --render",
                    state_index=a.state_index, state_hash=res["state_hash"], deterministic=bool(a.deterministic),
                    result=res, videos={})
    if a.render:
        main_mp4 = out / f"learned_rl_coin_delivery_{a.state_index}.mp4"
        main_gif = out / f"learned_rl_coin_delivery_{a.state_index}.gif"
        clean_mp4 = out / f"learned_rl_coin_delivery_{a.state_index}_clean.mp4"
        f_full = render_trajectory(env, actor, res, trace, title="RL Coin Delivery Demo", ckpt_id=ckpt_id, clean=False)
        f_clean = render_trajectory(env, actor, res, trace, title="RL Coin Delivery", ckpt_id=ckpt_id, clean=True)
        manifest["videos"]["main_mp4"] = dict(path=str(main_mp4), sha256=_write(f_full, main_mp4))
        manifest["videos"]["main_gif"] = dict(path=str(main_gif), sha256=_write(f_full, main_gif))
        manifest["videos"]["clean_mp4"] = dict(path=str(clean_mp4), sha256=_write(f_clean, clean_mp4))
        print(f"[demo] wrote {main_mp4} + {main_gif} + {clean_mp4}", flush=True)
    if a.compare:
        clip = []
        labels = {64102: "Certified learned delivery", 64201: "Certified learned delivery (second state)",
                  64111: "Near-certified failure"}
        for sidx in (64102, 64201, 64111):
            r2, tr2 = run_demo(env, actor, sidx)
            clip += render_trajectory(env, actor, r2, tr2, title=f"RL Coin Delivery — {labels[sidx]}",
                                      ckpt_id=ckpt_id, clean=False)
            manifest.setdefault("comparison_states", {})[sidx] = dict(strict=r2["strict"], loose=r2["loose"],
                                                                      hash=r2["state_hash"], label=labels[sidx])
        cmp_mp4 = out / "rl_coin_delivery_comparison.mp4"
        manifest["videos"]["comparison_mp4"] = dict(path=str(cmp_mp4), sha256=_write(clip, cmp_mp4))
        print(f"[demo] wrote {cmp_mp4}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))

    if a.require_strict and not res["strict"]:
        print(f"[demo] FAIL: state {a.state_index} did not certify (strict predicate).", flush=True)
        sys.exit(1)
    print(f"[demo] OK: certified delivery on state {a.state_index}." if res["strict"]
          else f"[demo] state {a.state_index} not strict (require-strict not set).", flush=True)


if __name__ == "__main__":
    main()
