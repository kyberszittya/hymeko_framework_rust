"""BALLTIP_INTERARM_FILTERED_V1 §7 — deterministic 4-way rollout video (original clamp / POINT sphere / ball nofilter /
ball filtered) from the REAL matched-panel eval rollout.

For a chosen held-out state, renders the deployed controller (`θ_center from the proposal → fixed b=8 search → committed
option → frozen settling pi_0`) on all four robot variants side-by-side, from the IDENTICAL start (canonical E0 handoff
transplanted by the shared qpos layout). The HUD carries the min inter-arm clearance LIVE — so the §6 pass-through
(clearance going negative on the filtered ball where collision would block it) is watchable, not just tabulated. Physics
/ timestep unchanged; frames captured by the same non-behavioral frame_hook as the regression. Reuses the framework
renderers (mujoco.Renderer + topdown_camera + _draw_overlay + compare_gif). Does NOT modify the frozen baseline.
"""
import copy
import json
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import (  # noqa: E402
    LateStart, build_boundary_panel, reconstruct_handoff)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import (  # noqa: E402
    PANEL_VARIANTS, build_variant_rl, min_interarm_clearance, transplant_handoff)
from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_snapshot import snapshot_planar  # noqa: E402
from hymeko_rl.eval.evaluate import _draw_overlay, compare_gif  # noqa: E402
from hymeko_rl.viz.render_planar_gifs import topdown_camera  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
OUT = "reports/2026-07-24-balltip-regression/video"
FAMS = ("contact_retention", "transport", "braking")
PROP_CKPT = f"{D}/carry_proposal_refined.pt"
SEARCH_SEED = 9000
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}
VARIANTS = list(PANEL_VARIANTS)
LABEL = {"canonical_clamp": "E0 clamp (r0.012)", "point_sphere": "POINT sphere (r0.014)",
         "balltip_nofilter": "ball r0.020 (collide)", "balltip_filtered": "ball r0.020 (FILTERED)"}


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _fresh(variant, rl_c, ls):
    if variant == "canonical_clamp":
        return copy.deepcopy(rl_c)
    return transplant_handoff(build_variant_rl(variant, seed=int(ls.seed)), rl_c)


def render_variant(variant, rl_c, gate_c, ls, c_center, i, pi0, base, *, width, height, fps, horizon=160):
    """Render the committed option on ONE variant → (frames, outcome, min_clr). The clearance HUD is drawn from the
    live min inter-arm clearance captured each physics step (negative ⇒ arm-through-arm)."""
    theta, _ = search_select(_fresh(variant, rl_c, ls), gate_c, c_center, pi0, base,
                             np.random.default_rng(SEARCH_SEED + i), b=8, horizon=horizon)
    gate_c = copy.deepcopy(gate_c)               # the render rollout mutates the gate; isolate it (matched across variants)
    rl = _fresh(variant, rl_c, ls)
    renderer = mujoco.Renderer(rl.inner.model, height=height, width=width)
    cam = topdown_camera()
    raw = []

    def hook(phase, strict):
        renderer.update_scene(rl.inner.data, camera=cam)
        clr = min_interarm_clearance(rl.inner.model, rl.inner.data)
        raw.append((np.asarray(renderer.render(), np.uint8), phase, int(strict), float(clr)))

    out = structured_carry_rollout(rl, gate_c, pi0, base, np.asarray(theta, np.float32), horizon=horizon, frame_hook=hook)
    renderer.close()
    min_clr = min((c for *_x, c in raw), default=float("nan"))
    verdict = "DELIVERED (K6)" if out["k6"] else ("handoff, no K6" if out["reached_handoff"] else "no delivery")
    frames = []
    for f, ph, st, clr in raw:
        tag = "  OVERLAP!" if clr < 0 else ""
        frames.append(_draw_overlay(f, [f"robot: {LABEL[variant]}", "coin   search: b=8", f"phase: {ph}",
                                        f"K6: {st}/6   handoff: {'Y' if st >= 1 else '-'}",
                                        f"inter-arm clr: {clr * 100:+.1f} cm{tag}"]))
    if frames:
        card = _draw_overlay(raw[-1][0], [f"robot: {LABEL[variant]}", f"VERDICT: {verdict}", f"min clr: {min_clr * 100:+.1f} cm"])
        frames += [card] * fps
    prov = {"variant": variant, "k6": int(out["k6"]), "reached_handoff": int(out["reached_handoff"]),
            "contain_exit_ct": int(out["contain_exit_ct"]), "min_interarm_clearance": round(min_clr, 5),
            "theta_selected": [round(float(x), 4) for x in theta], "start_state_hash": snapshot_hash(snapshot_planar(rl_c.inner))}
    return frames, out, prov


def render_state(i, panel, pi0, prop, base, W, H, FPS, log):
    rl_c, gate_c, ls = panel[i]
    c_center = prop.theta(rl_c.obs())
    panels, provs = [], []
    for v in VARIANTS:
        fr, out, prov = render_variant(v, rl_c, gate_c, ls, c_center, i, pi0, base, width=W, height=H, fps=FPS)
        panels.append(fr)
        provs.append(prov)
        log(f"    {v:18} K6 {out['k6']} handoff {out['reached_handoff']} min_clr {prov['min_interarm_clearance']:+.4f}", flush=True)
    name = f"balltip_4way_state{i}_seed{int(ls.seed)}"
    compare_gif(panels, f"{OUT}/{name}.gif", fps=FPS)
    T = max(len(p) for p in panels)
    frames = [np.hstack([p[min(t, len(p) - 1)] for p in panels]) for t in range(T)]
    imageio.mimwrite(f"{OUT}/{name}.mp4", frames, fps=FPS, quality=8, macro_block_size=1)
    meta = {"name": name, "state": i, "seed": int(ls.seed), "family": ls.family, "order": VARIANTS,
            "mp4": f"{OUT}/{name}.mp4", "gif": f"{OUT}/{name}.gif", "provenance": provs, "baseline": BASELINE}
    json.dump(meta, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
    log(f"  wrote {name}: mp4+gif ({len(frames)} frames)", flush=True)
    return meta


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    W, H, FPS = (240, 220, 15) if smoke else (300, 280, 18)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    prop = load_proposal(PROP_CKPT)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    raw, _c, _s = build_boundary_panel(pi0, range(14000, 15200), forbidden, want=4 if smoke else 24, families=FAMS,
                                       strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    panel = [(*reconstruct_handoff(pi0, ls, horizon=360)[:2], ls) for ls in raw]
    # two scenes (post gate-fix): state 14 = the filtered VISIBLE pass-through exploit (min_clr −0.010 ⇒ HUD "OVERLAP!",
    # blocked by a collision-on replay) while clamp+collision-on ball fail; state 3 = the honest collision-on ball
    # delivering legitimately (positive clearance) where the clamp fails.
    scenes = [0] if smoke else [14, 3]
    scenes = [s for s in scenes if s < len(panel)]
    print(f"[balltip §7] {len(panel)} states | rendering 4-way at {scenes} | search seed {SEARCH_SEED}", flush=True)
    metas = []
    for i in scenes:
        print(f"  state {i} (seed {int(panel[i][2].seed)}):", flush=True)
        metas.append(render_state(i, panel, pi0, prop, base, W, H, FPS, print))
    json.dump({"contract": "BALLTIP_INTERARM_FILTERED_V1", "section": "§7", "videos": metas, "baseline": BASELINE},
              open(f"{OUT}/manifest.json", "w"), indent=1, default=float)
    print(f"\n  artifacts in {OUT}/\nBALLTIP_VIDEO_DONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
