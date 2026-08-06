"""Videos 04 + 05 — coin/balltip object-variant demonstrations on the FROZEN baselines (BALLTIP_COIN_BASELINE_V1).

Per shape, two rollouts SIDE BY SIDE, each the EXACT scored rollout (structured_carry_rollout's non-behavioral
frame_hook — the codebase's own video observer):
  DEPLOY  = the frozen ball proposal update-0 + fixed b=8 search  (NOT SAC)
  EXPERT  = the 192-shot structured search (the physical/task ceiling)

  04_balltip_disk_sizes.mp4  — small / canonical / large disk. Message: the option language works across sizes, but the
                               canonical proposal is not scale-invariant (deploy degrades off the canonical radius).
  05_balltip_box_variants.mp4 — square / 2:1 / 3:1. The MECHANISM video: on the long rectangle the expert's contact
                               retention rises while the deploy's stays flat — the visual of the O2 diagnosis.

Coin overlay: status bar (shape | ctrl | K6), dtz(t) with the entry tolerance, held-dwell(t) toward K6, a phase/contact
info panel, and a summary card (method / K6 / max-dwell / mean-contact / handoff / budget). Reward is available in the
manifest but is NOT the headline — dtz + dwell + contact say why a rollout delivered or not.
"""
import copy
import json
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
import torch  # noqa: E402

from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import (  # noqa: E402
    CENTER_TOL, HELD_DWELL, structured_carry_rollout, structured_random_best_with_support)
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.manipulation_rig import RobotContract, build_manipulation_rig  # noqa: E402
from hymeko_rl.env.object_spec import ObjectSpec, Shape  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.viz.rollout_overlay import (  # noqa: E402
    InfoPanel, StatusBar, TimeSeriesPanel, assert_trace_render_consistency, encode_clip, hstack, overlay_frames,
    rollout_trace_hash, summary_card)

from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_object_o2 import _ball_tf  # noqa: E402

OUT = "reports/2026-07-24-se3-obstacle-6d1/videos"
CYL_PROP = f"{D}/carry_proposal_balltip_fresh_v1.pt"
BOX_PROP = f"{D}/carry_proposal_o2_box_v1.pt"
FAMS = ("contact_retention", "transport", "braking")
_R, EVAL_H, EXPERT_SHOTS = 0.020, 160, 192


def _setup():
    torch.set_num_threads(1)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    return pi0, base, forbidden


def _cam():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 0.9, -55, 90
    cam.lookat = np.array([0.0, 0.08, 0.0])
    return cam


def render_coin_rollout(rl, gate, pi0, base, theta, *, horizon=EVAL_H, h=360, w=470):
    """Render the EXACT structured_carry_rollout for ``theta`` via its frame_hook, capturing dtz / held-dwell / contact /
    phase per step. Returns (frames, diag, outcome). Non-behavioral (frame_hook only observes).

    IMPORTANT: the hook must render the SAME rl that the rollout STEPS. structured_carry_rollout steps its ``rl``
    argument in place, so we deepcopy ONCE here (isolating the caller's rl between deploy/expert calls) and pass THAT
    copy to both the rollout and the hook. (Previously the rollout stepped a separate deepcopy while the hook rendered
    the un-stepped original → every frame was the static initial state.)"""
    rl = copy.deepcopy(rl)
    gate = copy.deepcopy(gate)
    renderer = mujoco.Renderer(rl.inner.model, height=h, width=w)
    cam = _cam()
    frames, dtz, dwell, contact, phase = [], [], [], [], []

    def hook(ph, s):
        renderer.update_scene(rl.inner.data, camera=cam)
        frames.append(np.asarray(renderer.render(), np.uint8))
        dtz.append(float(rl._dtz()))
        dwell.append(int(s))
        m = rl.inner._planar_metrics
        contact.append(int(bool(m.left_contact or m.right_contact)))
        phase.append(ph)
    out = structured_carry_rollout(rl, gate, pi0, base,
                                   np.asarray(theta, np.float32), horizon=horizon, frame_hook=hook)
    renderer.close()
    cfrac = round(float(np.mean(contact)), 2) if contact else 0.0
    diag = {"dtz": np.asarray(dtz, np.float32), "dwell": np.asarray(dwell, np.float32), "phase": phase,
            "contact": np.asarray(contact), "contact_frac": cfrac, "k6": int(out["k6"]), "max_dwell": int(out["max_dwell"]),
            "handoff": int(out["reached_handoff"]), "contain_exit": int(out["contain_exit_ct"])}
    # VIDEO_TRACE_CONSISTENCY_V1 gate: the rendered frames MUST match the stepped rollout (the 2026-07-24 static-video bug).
    assert_trace_render_consistency(frames, diag["dtz"], label="coin_carry")
    diag["rollout_hash"] = rollout_trace_hash(diag["dtz"], {"k6": diag["k6"], "max_dwell": diag["max_dwell"]})
    return frames, diag, out


def _coin_clip(frames, diag, *, shape, ctrl, budget):
    k6 = "SUCCESS" if diag["k6"] else ("HANDOFF" if diag["handoff"] else "FAIL")
    n = len(diag["dtz"])

    def status(t):
        return "SUCCESS" if (diag["k6"] and t >= n - 2) else ("RUNNING" if t < n - 2 else k6)

    def info(t):
        tt = min(t, n - 1)
        return [f"phase: {diag['phase'][tt]}",
                f"dtz: {diag['dtz'][tt]:.3f}   contact: {int(diag['contact'][tt])}",
                f"K6 progress: {int(diag['dwell'][tt])}/{HELD_DWELL}",
                f"max-dwell {diag['max_dwell']}  contact-frac {diag['contact_frac']}"]
    panels = [
        StatusBar(f"coin | {shape} | CTRL: {ctrl} | B={budget}", status),
        TimeSeriesPanel({"dtz(t)": diag["dtz"]}, title="distance to zone (entry tol dashed)", threshold=CENTER_TOL,
                        size=(260, 135)),
        InfoPanel(info),
    ]
    return overlay_frames(frames, panels)


def _pair(pi0, base, rl, gate, *, shape):
    """Deploy (update-0 + b=8) vs expert (192-shot) side by side, both the exact scored rollout."""
    obs = rl.obs()
    prop = load_proposal(BOX_PROP if "rect" in shape or "square" in shape else CYL_PROP)
    th_d, _od = search_select(copy.deepcopy(rl), copy.deepcopy(gate), prop.theta(obs), pi0, base,
                              np.random.default_rng(7), b=8, horizon=EVAL_H)
    th_e, _oe, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                        np.random.default_rng(7), shots=EXPERT_SHOTS, horizon=EVAL_H)
    fd, dd, outd = render_coin_rollout(rl, gate, pi0, base, th_d)
    fe, de, oute = render_coin_rollout(rl, gate, pi0, base, th_e)
    left = _coin_clip(fd, dd, shape=shape, ctrl="deploy (update-0 + b8)", budget=8)
    right = _coin_clip(fe, de, shape=shape, ctrl="structured expert", budget=EXPERT_SHOTS)
    clip = hstack(left, right)
    clip += summary_card(clip[0].size, f"coin '{shape}': deploy vs structured expert",
                         [("deploy K6 / dwell / contact", f"{dd['k6']} / {dd['max_dwell']} / {dd['contact_frac']}"),
                          ("expert K6 / dwell / contact", f"{de['k6']} / {de['max_dwell']} / {de['contact_frac']}"),
                          ("message", "expert holds contact; deploy is not scale/shape-invariant")], hold=55)
    manifest = {"shape": shape, "deploy": {k: (dd[k] if not isinstance(dd[k], np.ndarray) else None)
                                           for k in ("k6", "max_dwell", "contact_frac", "handoff", "contain_exit")},
                "expert": {k: (de[k] if not isinstance(de[k], np.ndarray) else None)
                           for k in ("k6", "max_dwell", "contact_frac", "handoff", "contain_exit")}}
    return clip, manifest


# The R11.6C exact-zero robot contract: the ball-tip embodiment (POINT geom + ball-tip arm transform),
# orthogonal to the manipuland. Both the video variants and bv/`_make_env` reconstruct under it.
BALLTIP_ROBOT = RobotContract(geom="POINT", arm_mjcf_transform=_ball_tf, label="balltip")


def _reconstruct(pi0, base, forbidden, *, coin_shape="cylinder", hx=_R, hy=None, seed_lo, tries=16,
                 object_spec=None):
    """Pick a DELIVERING demo state with VISIBLE travel: reconstruct several fresh handoffs, run a quick expert, and
    rank by (delivers, longest approach). The fresh-reconstruct distribution rarely delivers, so a demonstration must
    select a deliverable state — and among those, the one with the longest approach (highest ``completion`` = K6-step)
    so the clip shows real motion, not a 0.5 s snap. Scans all candidates (no early break).

    The manipuland is a single :class:`ObjectSpec` (R11.7A unification): ``object_spec`` overrides when given
    (the HyMeKo-sourced object), else it is built from the back-compat ``coin_shape``/``hx``/``hy`` kwargs so
    the ~10 existing coin callers are unchanged. Reconstruction routes through :func:`build_manipulation_rig`
    so both this video path and bv's ``_make_env`` share one object→model generator.

    .. deprecated:: R11.7A
       The loose ``coin_shape``/``hx``/``hy`` kwargs are legacy back-compat for the pre-unification coin
       callers. New callers pass a typed ``object_spec``. Physical removal is deferred to post-U6 (nothing
       legacy is deleted while the object variants are still validated against it)."""
    obj = object_spec if object_spec is not None else ObjectSpec(
        shape=Shape.from_str(coin_shape), radius=hx, radius_y=hy)
    ls, _c, _s = build_boundary_panel(pi0, range(seed_lo, seed_lo + 1600), forbidden, want=tries, families=FAMS,
                                      strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    best = None
    for one in ls:
        try:
            rig = build_manipulation_rig(pi0, one, object_spec=obj, robot=BALLTIP_ROBOT)
        except ValueError:
            continue
        rl, gate = rig.rl, rig.gate
        if int(rl._strict) != 0:
            continue
        _th, out, _sup = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                             np.random.default_rng(3), shots=96, horizon=EVAL_H)
        # rank: delivering first, then the LONGEST approach (completion = when K6 first held) for visible travel
        score = (int(out["k6"]), int(out["completion"]) if out["k6"] else int(out["max_dwell"]))
        if best is None or score > best[2]:
            best = (rl, gate, score)
    if best:
        return best[0], best[1]
    fallback = build_manipulation_rig(pi0, ls[0], object_spec=obj, robot=BALLTIP_ROBOT)
    return fallback.rl, fallback.gate


def video_04_disk_sizes(pi0, base, forbidden):
    sizes = {"small_disk": 0.014, "canonical_disk": 0.020, "large_disk": 0.028}
    clip, manifests, size = [], [], None
    for i, (name, r) in enumerate(sizes.items()):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=r, hy=None, seed_lo=14000 + 900 * i)
        pair, man = _pair(pi0, base, rl, gate, shape=name)
        size = pair[0].size
        clip += summary_card(size, f"DISK: {name} (r={r})", [], hold=18) + pair
        manifests.append({name: man})
    clip += summary_card(size, "04 — disk-size variants: proposal is not scale-invariant",
                         [("shown", "small / canonical / large"), ("deploy", "frozen update-0 + b8"),
                          ("expert", "192-shot structured search")], hold=45)
    path = encode_clip(clip, f"{OUT}/04_balltip_disk_sizes.mp4", fps=30)
    json.dump({"clip": "04_balltip_disk_sizes", "variants": manifests, "commit": "1ef48623"},
              open(f"{OUT}/04_balltip_disk_sizes.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames)")


def video_05_box_variants(pi0, base, forbidden):
    a = math.pi * _R * _R / 4.0
    shapes = {"square_1_1": 1.0, "rect_2_1": 2.0, "rect_3_1": 3.0}
    clip, manifests, size = [], [], None
    for i, (name, asp) in enumerate(shapes.items()):
        hx = math.sqrt(a / asp)
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="box", hx=hx, hy=asp * hx, seed_lo=15000 + 900 * i)
        pair, man = _pair(pi0, base, rl, gate, shape=name)
        size = pair[0].size
        clip += summary_card(size, f"BOX: {name} (aspect {asp:g})", [], hold=18) + pair
        manifests.append({name: man})
    clip += summary_card(size, "05 — box variants (MECHANISM): expert holds contact, deploy stays flat",
                         [("shown", "square / 2:1 / 3:1"), ("evidence", "contact retention rises for the expert"),
                          ("of", "the O2 multimodal-structure diagnosis")], hold=45)
    path = encode_clip(clip, f"{OUT}/05_balltip_box_variants.mp4", fps=30)
    json.dump({"clip": "05_balltip_box_variants", "variants": manifests, "commit": "1ef48623"},
              open(f"{OUT}/05_balltip_box_variants.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames)")


def video_06_clamp_vs_balltip(pi0, base, forbidden):
    """Matched canonical-coin state, two embodiments side by side (NOT winner-loser — different contact strategy):
      CLAMP  = the task-specific E0 CONCAVE_CLAMP baseline (structured expert)  — force-closure hold
      BALLTIP = the generalization embodiment, deploy = FROZEN proposal update-0 + b8 search (NOT SAC) — push/coast
    Both from the SAME landing state; pick one where BOTH deliver so the clip contrasts strategy, not success."""
    ls, _c, _s = build_boundary_panel(pi0, range(16000, 17600), forbidden, want=24, families=FAMS,
                                      strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    prop = load_proposal(CYL_PROP)
    chosen = None
    for one in ls:
        try:
            rlc, gatec, _h, _r = reconstruct_handoff(pi0, one, coin_shape="cylinder", disk_radius_override=_R)
            rlb, gateb, _h2, _r2 = reconstruct_handoff(pi0, one, geom="POINT", arm_mjcf_transform=_ball_tf,
                                                       coin_shape="cylinder", disk_radius_override=_R)
        except ValueError:
            continue
        if int(rlc._strict) != 0 or int(rlb._strict) != 0:
            continue
        thc, oc, _s1 = structured_random_best_with_support(copy.deepcopy(rlc), copy.deepcopy(gatec), pi0, base,
                                                           np.random.default_rng(5), shots=EXPERT_SHOTS, horizon=EVAL_H)
        thb, ob = search_select(copy.deepcopy(rlb), copy.deepcopy(gateb), prop.theta(rlb.obs()), pi0, base,
                                np.random.default_rng(5), b=8, horizon=EVAL_H)
        if oc["k6"] and ob["k6"]:
            chosen = (rlc, gatec, thc, rlb, gateb, thb)
            break
    if chosen is None:
        print("06: no matched state where BOTH deliver found")
        return
    rlc, gatec, thc, rlb, gateb, thb = chosen
    fc, dc, _oc = render_coin_rollout(rlc, gatec, pi0, base, thc)
    fb, db, _ob = render_coin_rollout(rlb, gateb, pi0, base, thb)
    left = _coin_clip(fc, dc, shape="canonical coin", ctrl="CLAMP (E0, task-specific)", budget=EXPERT_SHOTS)
    right = _coin_clip(fb, db, shape="canonical coin", ctrl="BALLTIP (deploy update-0+b8)", budget=8)
    clip = hstack(left, right)
    clip += summary_card(clip[0].size, "06 — clamp vs balltip (matched canonical coin, different contact strategy)",
                         [("CLAMP", "task-specific strong baseline (force-closure hold)"),
                          ("BALLTIP", "generalization embodiment; collision ON; filtering OFF"),
                          ("balltip deploy", "frozen proposal update-0 + b8 search (NOT SAC)"),
                          ("both", f"K6 delivered — clamp cf {dc['contact_frac']} vs balltip cf {db['contact_frac']}")], hold=60)
    path = encode_clip(clip, f"{OUT}/06_clamp_vs_balltip_coin.mp4", fps=30)
    json.dump({"clip": "06_clamp_vs_balltip_coin",
               "clamp": {"k6": dc["k6"], "max_dwell": dc["max_dwell"], "contact_frac": dc["contact_frac"]},
               "balltip": {"k6": db["k6"], "max_dwell": db["max_dwell"], "contact_frac": db["contact_frac"],
                           "deploy": "update-0 + b8 (NOT SAC)"}, "commit": "4dfdb2d6"},
              open(f"{OUT}/06_clamp_vs_balltip_coin.json", "w"), indent=1, default=str)
    print(f"wrote {path} ({len(clip)} frames)")


def main():
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "04"):
        video_04_disk_sizes(pi0, base, forbidden)
    if only in ("all", "05"):
        video_05_box_variants(pi0, base, forbidden)
    if only in ("all", "06"):
        video_06_clamp_vs_balltip(pi0, base, forbidden)
    print("COIN_VARIANT_VIDEOS_DONE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
