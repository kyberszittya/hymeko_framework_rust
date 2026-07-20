"""Render presentation videos of the learned two-arm policy with the honest strict-gap HUD overlay (read-only; the
policy/env/predicate are unchanged). MuJoCo frames + a PIL HUD (dwell counter, coin speed, L/R contact, attribution,
body-shove, clean-mechanism, loose/strict). Honest labels — a loose entry is never labelled a strict delivery."""
from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from hymeko_rl.experiments.coin_two_arm_sac import direct_env, policy_strict
from hymeko_rl.train.coin_delivery_actor import (
    _BODY_SHOVE_MAX,
    _DWELL_STEPS,
    _ONE_FINGER_MAX,
    _SETTLE_VEL,
    _attribution_from_trace,
    rollout,
)
from hymeko_rl.train.sac import build_sac

_RUN = Path("experiments/2026_07_20_coin_two_arm_sac_100k")
_VID = _RUN / "videos"
_ATTR_MIN, _H, _W = 0.60, 480, 640
_CAM_LOOKAT = (0.0, 0.12, 0.03)


def load_actor(path: Path):
    a, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    a.load_state_dict(torch.load(path, map_location="cpu"))
    a.eval()
    return a


def greedy_of(actor):
    def g(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return g


def _camera(model):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = _CAM_LOOKAT
    cam.distance, cam.azimuth, cam.elevation = 0.62, 90.0, -70.0
    return cam


def _hud(img, banner, per_step, label):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, _W, 26], fill=(20, 20, 28))
    d.text((6, 6), label, fill=(255, 255, 255))
    y = _H - 118
    d.rectangle([0, y, _W, _H], fill=(0, 0, 0))
    for i, line in enumerate(per_step + [""] + banner):
        col = (255, 210, 90) if line.startswith(("STRICT", "LOOSE")) else (210, 220, 230)
        d.text((8, y + 6 + i * 13), line, fill=col)
    return np.asarray(im)


def render(state_seed: int, ckpt: Path, label: str, gif_name: str) -> dict:
    actor = load_actor(ckpt)
    env = direct_env()
    env.reset(seed=int(state_seed))
    trace = rollout(env, greedy_of(actor), max_steps=60)                 # authoritative metrics
    att = _attribution_from_trace(trace)
    strict = policy_strict(trace)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= _ONE_FINGER_MAX
    banner = [
        f"attribution(fingertip)={ff:.2f} (>= {_ATTR_MIN}: {'PASS' if ff >= _ATTR_MIN else 'FAIL'})",
        f"body-shove={att.alpha_body:.2f} (<= {_BODY_SHOVE_MAX}: {'PASS' if att.alpha_body <= _BODY_SHOVE_MAX else 'FAIL'})",
        f"clean-mechanism={'YES' if clean else 'NO (one-finger)'}   L/R attr={att.alpha_L:.2f}/{att.alpha_R:.2f}",
        f"LOOSE(zone entry)={'YES' if trace.loose else 'NO'}    STRICT(certified)={'YES' if strict else 'NO'}",
    ]
    # re-step deterministically for frames (visualization pass; same trajectory)
    env.reset(seed=int(state_seed))
    renderer = mujoco.Renderer(env.inner.model, _H, _W)
    cam = _camera(env.inner.model)
    frames, dwell = [], 0
    for i, st in enumerate(trace.steps):
        a = np.clip(greedy_of(actor)(env.inner, i, env._last_obs), -1, 1).astype(np.float32)
        env.step(a)
        renderer.update_scene(env.inner.data, camera=cam)
        dwell = dwell + 1 if st.in_zone else 0
        per = [f"step {i+1}/{len(trace.steps)}",
               f"in-zone dwell={dwell}/{_DWELL_STEPS}   coin speed={st.disk_vel_norm:.3f} (settle<= {_SETTLE_VEL})",
               f"contact L={'Y' if st.left_contact else '.'} R={'Y' if st.right_contact else '.'}   dtz={st.disk_to_zone:.3f}"]
        frames.append(_hud(renderer.render(), banner, per, label))
    for _ in range(12):                                                  # hold the final frame
        frames.append(frames[-1])
    _VID.mkdir(parents=True, exist_ok=True)
    gif = _VID / gif_name
    imageio.mimsave(gif, frames, fps=15, loop=0)
    sha = hashlib.sha256(gif.read_bytes()).hexdigest()
    return dict(state_seed=state_seed, label=label, path=str(gif), sha256=sha, frames=len(frames),
                strict=bool(strict), loose=bool(trace.loose), attribution=float(ff),
                body_shove=float(att.alpha_body), clean_mechanism=bool(clean),
                best_dwell=int(trace.best_dwell), settle_vel=float(trace.settle_vel), size_bytes=gif.stat().st_size)


def main() -> None:
    best = _RUN / "sac_actor_best.pt"
    targets = [
        (64102, best, "Learned certified delivery  (state 64102, reproducible 10/10)", "learned_certified_delivery_64102.gif"),
        (64201, best, "Learned bilateral zone entry  (state 64201, also certified)", "learned_bilateral_zone_entry_64201.gif"),
        (64111, best, "Near-certified delivery  (state 64111: fingertip-attribution short of 0.60)", "near_certified_delivery_64111.gif"),
    ]
    import json
    manifest = [render(*t) for t in targets]
    (_VID / "video_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    for m in manifest:
        print(f"  {m['path']}  strict={m['strict']} loose={m['loose']} attr={m['attribution']:.2f} "
              f"dwell={m['best_dwell']} bytes={m['size_bytes']} sha={m['sha256'][:16]}")
    print(f"saved {_VID/'video_manifest.json'}")


if __name__ == "__main__":
    main()
