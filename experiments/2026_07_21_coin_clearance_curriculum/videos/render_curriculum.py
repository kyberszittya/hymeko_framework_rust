"""Render the curriculum SHORT-TRANSPORT result: on a clear-start held state (coin outside target, clearance +0.025),
the curriculum-trained checkpoint delivers (strict) where the base GENERATOR checkpoint fails. Honest labels: this is
short transport (below the +0.030 presentation grade), NOT a clear presentation-distance delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from hymeko_rl.experiments import coin_rl_demo as D
from hymeko_rl.experiments.coin_clearance_curriculum import _STAGES, _clearance
from hymeko_rl.experiments.coin_generator_exp import _greedy, _restore_generated, direct_env
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import _attribution_from_trace, rollout
from hymeko_rl.train.sac import build_sac

_OUT = Path("experiments/2026_07_21_coin_clearance_curriculum/videos")
_CURR = "experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt"
_BASE = "experiments/2026_07_20_coin_generator_generator_s2r0/actor_best.pt"
_TARGET_HASH = "04870b0e0357ecb5"


def _find_state(env):
    for s in _STAGES:
        for c in load_configs(Path(f"experiments/2026_07_21_coin_clearance_curriculum/{s}_held.pkl")):
            _restore_generated(env, c.snapshot)
            from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
            from hymeko_rl.env.planar_snapshot import snapshot_planar
            if snapshot_hash(snapshot_planar(env.inner)).startswith(_TARGET_HASH[:12]):
                return c
    raise RuntimeError("target state not found")


def render(env, actor, cfg, label, name):
    _restore_generated(env, cfg.snapshot)
    D._align_overlay(env.inner)
    clr = _clearance(env.inner)
    tr = rollout(env, _greedy(actor), max_steps=60)
    att = _attribution_from_trace(tr)
    strict = policy_strict(tr)
    _restore_generated(env, cfg.snapshot)
    D._align_overlay(env.inner)
    rr = mujoco.Renderer(env.inner.model, D._RH, D._RW)
    cam = D._camera(env.inner.model)
    frames, dwell, first_contact = [], 0, None
    for i, st in enumerate(tr.steps):
        a = np.clip(_greedy(actor)(env.inner, i, env._last_obs), -1, 1).astype(np.float32)
        env.step(a)
        rr.update_scene(env.inner.data, camera=cam)
        dwell = dwell + 1 if st.in_zone else 0
        if first_contact is None and (st.left_contact or st.right_contact):
            first_contact = i
        tag = "START - coin outside target" if i == 0 else ("FIRST CONTACT" if i == first_contact else
              ("ZONE ENTRY" if st.in_zone and dwell == 1 else ""))
        rgb = np.asarray(D.Image.fromarray(rr.render()).resize((D._W, D._H), D.Image.LANCZOS))
        per = dict(lc=st.left_contact, rc=st.right_contact, inz=st.in_zone,
                   line=f"{tag}  step {i+1}/{len(tr.steps)}  dtz={st.disk_to_zone:.3f}  clearance-start={clr:+.3f}")
        frames.append(D._frame(rgb, title=label, ckpt_id="", state_index=cfg.config_id[:20], per=per,
                               strict=strict, loose=tr.loose, clean=False))
        if i == 0:
            frames += [frames[-1]] * D._FPS                      # ~1s START hold
    res = dict(state_index=cfg.config_id[:24], strict=strict, loose=tr.loose, attribution=round(float(att.fingertip_fraction), 3),
               best_dwell=tr.best_dwell)
    frames += D._result_card(res, clean=False)
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / name
    if p.suffix == ".mp4":
        imageio.mimsave(p, frames, fps=D._FPS, quality=8, macro_block_size=8)
    else:
        imageio.mimsave(p, frames[::2], fps=D._FPS // 2, loop=0)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), strict=bool(strict), clearance=round(float(clr), 4)), frames


def main():
    env = direct_env()
    env._base_override = lambda inner, t: np.zeros(6, np.float32)
    env._delta_override = 1.0
    def load(p):
        a, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
        a.load_state_dict(D.torch.load(p, map_location="cpu"))
        return a
    cfg = _find_state(env)
    curr, base = load(_CURR), load(_BASE)
    man = {}
    man["short_transport"], f_curr = render(env, curr, cfg, "Curriculum SAC - SHORT-TRANSPORT certified delivery", "rl_short_transport_certified_delivery.mp4")
    man["short_transport_gif"], _ = render(env, curr, cfg, "Curriculum SAC - SHORT-TRANSPORT certified delivery", "rl_short_transport_certified_delivery.gif")
    _, f_base = render(env, base, cfg, "STARTING GENERATOR checkpoint - FAILS this clear-start", "_base_fail_tmp.mp4")
    # before-vs-after (same initial hash): base fail then curriculum delivery
    imageio.mimsave(_OUT / "rl_short_transport_before_vs_after.mp4", f_base + f_curr, fps=D._FPS, quality=8, macro_block_size=8)
    bva = _OUT / "rl_short_transport_before_vs_after.mp4"
    man["before_vs_after"] = dict(path=str(bva), sha256=hashlib.sha256(bva.read_bytes()).hexdigest())
    (_OUT / "_base_fail_tmp.mp4").unlink(missing_ok=True)
    (_OUT / "curriculum_video_manifest.json").write_text(json.dumps(man, indent=1, default=float))
    for k, v in man.items():
        print(f"  {k}: {v['path']} strict={v.get('strict')} clr={v.get('clearance')} sha={v['sha256'][:16]}")


if __name__ == "__main__":
    main()
