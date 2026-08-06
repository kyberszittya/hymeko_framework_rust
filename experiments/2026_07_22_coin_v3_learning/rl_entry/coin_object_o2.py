"""OBJECT_TO_TARGET_VARIANTS_V1 — O2: square & rectangle on the FRESH-reconstruct distribution (diagnostic).

Ball-tip embodiment FIXED (BALLTIP_COIN_BASELINE_V1). Varies the manipuland to edged objects (equal-area boxes to the
canonical r0.020 cylinder): square 1:1, rectangle 2:1, rectangle 3:1. All states are FRESH per-object reconstructions
(pi_0 replayed on the ball+box), orientation-stratified by the object's disk_rz, on ONE shared bank across arms.

Five canonical arms (frozen reward + K6 certificate throughout; the HOLD reward is a SEPARATE pre-registered ablation in
coin_object_o2_hold.py — NOT here, to keep O2 comparable to O1):
  A pi_0-only            — the clamp-trained base with no carry option (sequence_then_pi0, empty prefix)
  B zero-shot deploy     — the frozen TRANSPLANT-fit proposal + b=8 (cross-object transfer, no O2 fitting)
  C fresh-refit deploy   — an O2 proposal fit on fresh O2 box labels + b=8 (best deployable under the current arch)
  D explicit geometry    — a deterministic geometry-aware push θ (contact-Jacobian → zone; same bounds+cert); open-loop
  E strong expert        — 192-shot structured search (the physical/task ceiling on the SAME fresh states)

The headline is NOT strict K6 alone: per rollout we decompose approach/contact → handoff → transport-retention (contact
frac) → target-entry (dtz_min) → rotation-induced (disk_rz range) → strict delivery, and report the gaps (expert vs
explicit vs fresh-refit vs zero-shot) per shape and per orientation bin. Verdict taxonomy per §10.
"""
import copy
import json
import math
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import (  # noqa: E402
    A_BOUND, T_MAX, T_MIN, structured_carry_rollout, structured_random_best_with_support)
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import BALLTIP_SPEC, build_arm_mjcf  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_grasp_env import with_fingertip_sites  # noqa: E402

OUT = "reports/2026-07-24-o2-square-rectangle-fresh-reconstruct"
OLD_PROP = f"{D}/carry_proposal_balltip_fresh_v1.pt"      # the fresh-refit CYLINDER proposal (O1 true-deploy) = zero-shot arm
O2_PROP = f"{D}/carry_proposal_o2_box_v1.pt"              # the O2 box fresh-refit proposal (arm C)
FAMS, EVAL_H, EXPERT_SHOTS, K = ("contact_retention", "transport", "braking"), 160, 192, 6
_R = 0.020
_A = math.pi * _R * _R / 4.0                              # equal-area target: 4·hx·hy = π r²
SHAPES = {"square_1_1": 1.0, "rect_2_1": 2.0, "rect_3_1": 3.0}   # aspect a = hy/hx
ORIENT = [("axis_0", 0.0), ("inter_22", math.pi / 8), ("diag_45", math.pi / 4), ("random", None)]
CENTER_TOL, SETTLE_VEL, HELD_DWELL = 0.02, 0.06, 6


def _hxy(a):
    hx = math.sqrt(_A / a)
    return hx, a * hx


def _ball_tf(_c):
    return with_fingertip_sites(build_arm_mjcf(BALLTIP_SPEC, "enabled"))


def _rz_adr(rl):
    return rl.inner._disk_x_adr + 2                       # qpos layout: [..arm.., disk_x, disk_y, disk_rz]


def _set_orientation(rl, theta):
    import mujoco
    d = rl.inner.data
    d.qpos[_rz_adr(rl)] = float(theta)
    mujoco.mj_forward(rl.inner.model, d)
    rl.inner._planar_metrics = rl.inner._metrics()


def fresh_o2_bank(pi0, shape, cand_ls, want, log):
    """Fresh per-object reconstruction on the ball+box; set a stratified orientation; keep strict==0 carry handoffs."""
    hx, hy = _hxy(SHAPES[shape])
    panel, oi = [], 0
    for ls in cand_ls:
        try:
            rl, gate, _h, _r = reconstruct_handoff(pi0, ls, geom="POINT", arm_mjcf_transform=_ball_tf,
                                                   coin_shape="box", disk_radius_override=hx, disk_radius_y_override=hy)
        except ValueError:
            continue
        if int(rl._strict) != 0:
            continue
        name, ang = ORIENT[oi % len(ORIENT)]
        oi += 1
        theta = float(np.random.default_rng(1234 + len(panel)).uniform(-math.pi, math.pi)) if ang is None else ang
        _set_orientation(rl, theta)
        if int(rl._strict) != 0:                          # orientation set must not already satisfy/break the carry start
            continue
        panel.append({"rl": rl, "gate": gate, "orient_bin": name, "orient_rad": round(theta, 4), "seed": int(ls.seed)})
        if len(panel) >= want:
            break
    log(f"    [{shape}] fresh box handoffs {len(panel)} (hx {hx:.4f} hy {hy:.4f})")
    return panel


def _stage_metrics(rl, gate, pi0, base, theta):
    """Re-run a committed θ with a non-behavioral hook → stage decomposition (contact frac, dtz_min, disk_rz range)."""
    acc = {"contact": 0, "n": 0, "dtz_min": float(rl._dtz()), "rz0": float(rl.inner.data.qpos[_rz_adr(rl)]),
           "rz_lo": float(rl.inner.data.qpos[_rz_adr(rl)]), "rz_hi": float(rl.inner.data.qpos[_rz_adr(rl)])}

    def hook(_p, _s):
        m = rl.inner._planar_metrics
        acc["contact"] += int(bool(m.left_contact or m.right_contact))
        acc["n"] += 1
        acc["dtz_min"] = min(acc["dtz_min"], float(rl._dtz()))
        rz = float(rl.inner.data.qpos[_rz_adr(rl)])
        acc["rz_lo"] = min(acc["rz_lo"], rz)
        acc["rz_hi"] = max(acc["rz_hi"], rz)

    o = structured_carry_rollout(rl, copy.deepcopy(gate), pi0, base, np.asarray(theta, np.float32), horizon=EVAL_H, frame_hook=hook)
    o["contact_frac"] = round(acc["contact"] / max(1, acc["n"]), 3)
    o["dtz_min"] = round(acc["dtz_min"], 4)
    o["target_entry"] = int(acc["dtz_min"] <= CENTER_TOL)
    o["rot_range"] = round(acc["rz_hi"] - acc["rz_lo"], 4)
    return o


def _geometry_theta(rl, mag=2.8, k_probe=6, t_push=12, t_brake=6):
    """Explicit geometry-aware push θ: finite-difference d(disk_xy)/d(joint torque) via a short contact probe, least-squares
    the push that drives the object along coin→zone, brake = −0.4·push. Observes the true object+zone geometry (arm D)."""
    dir_vec, _dist = rl.inner.direction_to_zone()
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    J = np.zeros((2, 4), np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = mag
        for _ in range(k_probe):
            r2.step(a)
        J[:, j] = (np.asarray(r2.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0) / (mag * k_probe)
    a_push = np.linalg.lstsq(J, np.asarray(dir_vec, np.float32), rcond=None)[0]
    a_push = np.clip(a_push / (np.linalg.norm(a_push) + 1e-9) * mag, -A_BOUND, A_BOUND).astype(np.float32)
    dur = np.clip([t_push, t_brake, 3], T_MIN, T_MAX).astype(np.float32)
    return np.concatenate([a_push, dur[:1], -0.4 * a_push, dur[1:2], np.zeros(4, np.float32), dur[2:3]]).astype(np.float32)


def eval_state(item, old_prop, o2_prop, pi0, base, adim, i):
    """Run all five arms on ONE fresh state; return per-arm outcome + stage metrics."""
    rl, gate = item["rl"], item["gate"]
    obs = rl.obs()
    out = {}
    # A pi_0-only
    o = sequence_then_pi0(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H)
    out["pi0"] = {"k6": int(o["k6"]), "reached_handoff": int(o["reached_handoff"]), "contain_exit_ct": int(o["contain_exit_ct"])}
    # B zero-shot deploy (fresh-cylinder proposal, no O2 fitting) / C fresh-refit O2 proposal — both b=8
    for name, prop in (("zeroshot", old_prop), ("freshrefit", o2_prop)):
        th, _o = search_select(copy.deepcopy(rl), copy.deepcopy(gate), prop.theta(obs), pi0, base,
                               np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
        out[name] = _summ(_stage_metrics(copy.deepcopy(rl), gate, pi0, base, th))
    # D explicit geometry-aware push
    out["explicit"] = _summ(_stage_metrics(copy.deepcopy(rl), gate, pi0, base, _geometry_theta(copy.deepcopy(rl))))
    # E expert (ceiling)
    th_e, _oe, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                        np.random.default_rng(9000 + i), shots=EXPERT_SHOTS, horizon=EVAL_H)
    out["expert"] = _summ(_stage_metrics(copy.deepcopy(rl), gate, pi0, base, th_e))
    return out


def _summ(o):
    return {k: (int(o[k]) if k in ("k6", "reached_handoff", "contain_exit_ct", "touched", "target_entry") else o[k])
            for k in ("k6", "reached_handoff", "contain_exit_ct", "touched", "max_dwell", "completion", "contact_frac", "dtz_min", "rot_range")}


def main(smoke=False):
    import os

    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    old_prop = load_proposal(OLD_PROP)
    want = 6 if smoke else 24

    # ---- C: fit ONE O2 box proposal on fresh box teacher labels (frozen reward; shape-blind under the current arch) ----
    log("[O2] fitting the fresh-refit O2 box proposal (frozen-reward teacher on fresh box handoffs)...")
    tshapes = ["square_1_1"] if smoke else list(SHAPES)
    obs_b, th_b = [], []
    for sh in tshapes:
        hx, hy = _hxy(SHAPES[sh])
        tr_ls, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(8 if smoke else 70),
                                             families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
        rk = {"geom": "POINT", "arm_mjcf_transform": _ball_tf, "coin_shape": "box", "disk_radius_override": hx, "disk_radius_y_override": hy}
        o, t, _p = generate_bank(pi0, base, tr_ls, shots=(24 if smoke else 128), reconstruct_kwargs=rk, log=log)
        obs_b += o
        th_b += t
    log(f"[O2] O2 box teacher labels: {len(obs_b)}")
    kk = min(K, max(2, len(obs_b) // 4))
    o2_prop, fit = fit_proposal(np.asarray(obs_b, np.float32), np.asarray(th_b, np.float32), kk,
                                clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
    save_proposal(o2_prop, O2_PROP)
    log(f"[O2] O2 proposal K={kk} res_mse {fit['res_mse']:.3f} → {O2_PROP}")

    # ---- shared fresh eval bank per shape (orientation-stratified, disjoint from the teacher seeds) ----
    ev_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 60),
                                         families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    results = {}
    for sh in ([tshapes[0]] if smoke else list(SHAPES)):
        bank = fresh_o2_bank(pi0, sh, ev_ls, want, log)
        recs = [{"orient_bin": it["orient_bin"], "orient_rad": it["orient_rad"], **eval_state(it, old_prop, o2_prop, pi0, base, adim, i)}
                for i, it in enumerate(bank)]
        results[sh] = {"n": len(recs), "records": recs, "summary": _aggregate(recs)}
        s = results[sh]["summary"]
        log(f"  [{sh}] n {len(recs)} K6: pi0 {s['pi0']['k6']} zeroshot {s['zeroshot']['k6']} freshrefit {s['freshrefit']['k6']} "
            f"explicit {s['explicit']['k6']} expert {s['expert']['k6']}")

    verdict = _verdict(results)
    manifest = {"contract": "OBJECT_TO_TARGET_VARIANTS_V1", "stage": "O2_square_rectangle", "date": "2026-07-24", "smoke": smoke,
                "embodiment": "collision-on ball-tip (frozen)", "distribution": "fresh_reconstruct", "reward": "FROZEN (canonical; HOLD is a separate ablation)",
                "shapes": {k: {"aspect": v, "hx_hy": list(_hxy(v))} for k, v in SHAPES.items()}, "o2_proposal_fit": fit,
                "results": results, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/o2.json", "w"), indent=1, default=float)
    log(f"\n== O2 square/rectangle (fresh-reconstruct, frozen reward) ==  → {verdict}\n  artifact: {OUT}/o2.json\nOBJECT_O2_DONE")
    return manifest


def _aggregate(recs):
    arms = ("pi0", "zeroshot", "freshrefit", "explicit", "expert")
    n = max(1, len(recs))
    agg = {}
    for a in arms:
        rs = [r[a] for r in recs]
        agg[a] = {"k6": sum(r["k6"] for r in rs), "k6_rate": round(sum(r["k6"] for r in rs) / n, 3),
                  "handoff": sum(r["reached_handoff"] for r in rs), "target_entry": sum(r.get("target_entry", 0) for r in rs) if a != "pi0" else None,
                  "mean_contact_frac": round(float(np.mean([r.get("contact_frac", 0.0) for r in rs])), 3) if a != "pi0" else None,
                  "mean_rot_range": round(float(np.mean([r.get("rot_range", 0.0) for r in rs])), 4) if a != "pi0" else None}
    return agg


def _verdict(results):
    # aggregate across shapes: expert ceiling vs deploy (fresh-refit) and explicit
    exp = np.mean([results[s]["summary"]["expert"]["k6_rate"] for s in results])
    dep = np.mean([results[s]["summary"]["freshrefit"]["k6_rate"] for s in results])
    exl = np.mean([results[s]["summary"]["explicit"]["k6_rate"] for s in results])
    if exp < 0.15:
        return "O2_UNINFORMATIVE_OR_PHYSICAL_LIMIT"
    if dep >= 0.5 * exp:
        return "O2_DEPLOY_GENERALIZES"
    if exp >= 0.3 or exl >= 0.3:
        return "O2_STRUCTURALLY_SOLVABLE_DEPLOY_GAP"
    return "O2_INCONCLUSIVE_FIRST_PASS"


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
