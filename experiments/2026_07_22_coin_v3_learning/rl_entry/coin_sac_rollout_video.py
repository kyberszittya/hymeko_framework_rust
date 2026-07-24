"""SAC_AGENT_ROLLOUT_VIDEO_V1 — reproducible video EVIDENCE of the Stage-5b SAC carry-option agent (not a cherry-picked
highlight).

Renders the REAL evaluation rollout — the deployed controller `θ_center → fixed b=8 search → committed push/brake/release
option → frozen settling pi_0` — via a NON-behavioral `frame_hook` on `execute_one_option` (physics/timestep unchanged;
hook-on == hook-off, gated by test_coin_sac_rollout_video). Reuses the framework renderers (`mujoco.Renderer` +
`topdown_camera` + `_draw_overlay` HUD + `compare_gif`). Emits three artifacts — a matched-seed side-by-side (update-0 vs
SAC), a clean SAC K6 success, and an honest failure/near-miss — each MP4 + GIF preview + a provenance JSON pinning the
checkpoint SHA, env seed, start-state hash, search seeds, θ_center and θ_selected. Linked to the frozen baseline
(`executable-hymeko-option-rl-v1` @ 772a11a4).
"""
import copy
import hashlib
import json
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_option_rl import GaussActor, action_to_theta  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_snapshot import snapshot_planar  # noqa: E402
from hymeko_rl.eval.evaluate import _draw_overlay, _write_gif, compare_gif  # noqa: E402
from hymeko_rl.viz.render_planar_gifs import topdown_camera  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
OUT = "reports/2026-07-24-sac-rollout-video"
FAMS = ("contact_retention", "transport", "braking")
SAC_CKPT = f"{D}/carry_rlb_sac_seed3_selected.pt"       # best Stage-5b SAC (seed3, ΔK6 +0.065)
PROP_CKPT = f"{D}/carry_proposal_refined.pt"            # the update-0 proposal (RL init)
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _panel(pi0, seeds, forbidden, want):
    panel, _c, _s = build_boundary_panel(pi0, seeds, forbidden, want=want, families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    out = []
    for ls in panel:
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        out.append((rl, gate, ls))
    return out


def _center(kind, actor, prop, obs):
    if kind == "sac":
        with torch.no_grad():
            a = actor.mean_action(torch.as_tensor(obs[None]).float())[0].numpy()
        return action_to_theta(a)
    return prop.theta(obs)


def render_rollout(rl0, gate0, theta_selected, pi0, base, ctrl_name, *, width, height, fps, horizon=160):
    """Render ONE deployed option execution (the real rollout) → (frames, outcome). The frame_hook captures the live
    mujoco scene each physics step; the HUD + a held verdict card are drawn in a post-pass."""
    rl, gate = copy.deepcopy(rl0), copy.deepcopy(gate0)
    renderer = mujoco.Renderer(rl.inner.model, height=height, width=width); cam = topdown_camera()
    raw = []

    def hook(phase, strict):
        renderer.update_scene(rl.inner.data, camera=cam)
        raw.append((np.asarray(renderer.render(), np.uint8), phase, int(strict)))

    out = structured_carry_rollout(rl, gate, pi0, base, np.asarray(theta_selected, np.float32), horizon=horizon, frame_hook=hook)
    renderer.close()
    verdict = "DELIVERED  (K6 held)" if out["k6"] else ("HANDOFF, no K6" if out["reached_handoff"] else "no delivery")
    frames = [_draw_overlay(f, [f"controller: {ctrl_name}", "object: coin   search: b=8",
                                f"phase: {ph}", f"K6 dwell: {st}/6", f"handoff: {'YES' if st >= 1 else '--'}"]) for f, ph, st in raw]
    if frames:
        card = _draw_overlay(raw[-1][0], [f"controller: {ctrl_name}", "object: coin   search: b=8", f"VERDICT: {verdict}"])
        frames += [card] * fps                                          # ~1 s result card
    return frames, out


def _provenance(kind, ckpt, ls, rl, search_seed, theta_center, theta_selected, out):
    return {"controller": kind, "checkpoint": ckpt.split("/")[-1], "checkpoint_sha256_16": _sha(ckpt),
            "env_seed": int(ls.seed), "prefix_steps": int(ls.prefix_steps), "family": ls.family,
            "start_state_hash": snapshot_hash(snapshot_planar(rl.inner)), "search_seed": int(search_seed),
            "search_budget": 8, "theta_center": [round(float(x), 4) for x in theta_center],
            "theta_selected_provenance": [round(float(x), 4) for x in theta_selected],
            "k6": int(out["k6"]), "reached_handoff": int(out["reached_handoff"]), "contain_exit_ct": int(out["contain_exit_ct"]),
            "baseline": BASELINE}


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)
    import os
    os.makedirs(OUT, exist_ok=True)
    W, H, FPS = (320, 288, 15) if smoke else (420, 380, 20)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    prop = load_proposal(PROP_CKPT)
    actor = GaussActor(); actor.load_state_dict(torch.load(SAC_CKPT, weights_only=False))
    forbidden = set(l.seed for l in _bank(cfg["banks"]["late_train"])) | set(l.seed for l in _bank(cfg["banks"]["late_dev"]))
    panel = _panel(pi0, range(12200, 13600), forbidden, 8 if smoke else 30)
    SEARCH_SEED = 8000
    log(f"[panel] {len(panel)} held-out final states | SAC {SAC_CKPT.split('/')[-1]} sha {_sha(SAC_CKPT)} | proposal sha {_sha(PROP_CKPT)}")

    # classify each state at the fixed b=8 search seed: SAC vs update-0 K6 (the honest scan — no cherry-picking)
    rows = []
    for i, (rl, gate, ls) in enumerate(panel):
        c_sac = _center("sac", actor, prop, rl.obs()); c_up = _center("up", actor, prop, rl.obs())
        th_sac, o_sac = search_select(rl, gate, c_sac, pi0, base, np.random.default_rng(SEARCH_SEED + i), b=8, horizon=160)
        th_up, o_up = search_select(rl, gate, c_up, pi0, base, np.random.default_rng(SEARCH_SEED + i), b=8, horizon=160)
        rows.append({"i": i, "sac_k6": int(o_sac["k6"]), "up_k6": int(o_up["k6"]),
                     "c_sac": c_sac, "c_up": c_up, "th_sac": th_sac, "th_up": th_up})
    n_sac = sum(r["sac_k6"] for r in rows); n_up = sum(r["up_k6"] for r in rows)
    log(f"[scan] SAC K6 {n_sac}/{len(rows)} | update-0 K6 {n_up}/{len(rows)} (fixed search seed {SEARCH_SEED})")

    def pick(pred, default=0):
        return next((r["i"] for r in rows if pred(r)), default)
    i_cmp = pick(lambda r: r["sac_k6"] > r["up_k6"], pick(lambda r: r["sac_k6"] != r["up_k6"], 0))  # RL adds value, else differs
    i_win = pick(lambda r: r["sac_k6"] == 1, i_cmp)
    i_fail = pick(lambda r: r["sac_k6"] == 0, 0)
    log(f"[scenarios] matched-comparison state {i_cmp} | SAC-success state {i_win} | honest-failure state {i_fail}")

    manifest = {"contract": "SAC_AGENT_ROLLOUT_VIDEO_V1", "date": "2026-07-24", "smoke": smoke, "baseline": BASELINE,
                "scan": {"sac_k6": n_sac, "update0_k6": n_up, "n": len(rows), "search_seed": SEARCH_SEED},
                "videos": []}

    def emit(name, panels, provs, side_by_side):
        gif = f"{OUT}/{name}.gif"; mp4 = f"{OUT}/{name}.mp4"
        if side_by_side:
            compare_gif(panels, gif, fps=FPS)
            frames = [np.hstack([p[min(t, len(p) - 1)] for p in panels]) for t in range(max(len(p) for p in panels))]
        else:
            _write_gif(panels[0], gif, fps=FPS); frames = panels[0]
        imageio.mimwrite(mp4, frames, fps=FPS, quality=8, macro_block_size=1)
        json.dump({"name": name, "mp4": mp4, "gif": gif, "provenance": provs}, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
        manifest["videos"].append({"name": name, "mp4": mp4, "gif": gif, "provenance": provs})
        log(f"  wrote {name}: {mp4} + {gif} ({len(frames)} frames)")

    # 1) matched-seed side-by-side: update-0 (left) vs Stage-5b SAC (right), same state + same search seed
    r = rows[i_cmp]; rl, gate, ls = panel[i_cmp]
    f_up, o_up = render_rollout(rl, gate, r["th_up"], pi0, base, "update-0 proposal", width=W, height=H, fps=FPS)
    f_sac, o_sac = render_rollout(rl, gate, r["th_sac"], pi0, base, "Stage-5b SAC", width=W, height=H, fps=FPS)
    emit("01_matched_seed_update0_vs_sac", [f_up, f_sac],
         [_provenance("update-0", PROP_CKPT, ls, rl, SEARCH_SEED + i_cmp, r["c_up"], r["th_up"], o_up),
          _provenance("sac", SAC_CKPT, ls, rl, SEARCH_SEED + i_cmp, r["c_sac"], r["th_sac"], o_sac)], side_by_side=True)

    # 2) clean SAC success — full PUSH→BRAKE→RELEASE→HANDOFF→SETTLE with K6
    r = rows[i_win]; rl, gate, ls = panel[i_win]
    f_win, o_win = render_rollout(rl, gate, r["th_sac"], pi0, base, "Stage-5b SAC", width=W, height=H, fps=FPS)
    emit("02_sac_success_k6", [f_win], [_provenance("sac", SAC_CKPT, ls, rl, SEARCH_SEED + i_win, r["c_sac"], r["th_sac"], o_win)], side_by_side=False)

    # 3) honest failure / near-miss — the remaining failure mode, no cherry-picking
    r = rows[i_fail]; rl, gate, ls = panel[i_fail]
    f_fail, o_fail = render_rollout(rl, gate, r["th_sac"], pi0, base, "Stage-5b SAC", width=W, height=H, fps=FPS)
    emit("03_sac_honest_failure", [f_fail], [_provenance("sac", SAC_CKPT, ls, rl, SEARCH_SEED + i_fail, r["c_sac"], r["th_sac"], o_fail)], side_by_side=False)

    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1, default=float)
    log(f"\n== SAC_AGENT_ROLLOUT_VIDEO_V1 == scan SAC {n_sac}/{len(rows)} vs update-0 {n_up}/{len(rows)}")
    log(f"  1) matched-seed side-by-side (state {i_cmp}): update-0 K6 {o_up['k6']} vs SAC K6 {o_sac['k6']}")
    log(f"  2) SAC success (state {i_win}): K6 {o_win['k6']} handoff {o_win['reached_handoff']}")
    log(f"  3) SAC honest failure (state {i_fail}): K6 {o_fail['k6']} handoff {o_fail['reached_handoff']} exit {o_fail['contain_exit_ct']}")
    log(f"  artifacts + provenance in {OUT}/\nSAC_ROLLOUT_VIDEO_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
