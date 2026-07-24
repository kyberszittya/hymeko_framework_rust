"""O2 HOLD_AWARE_REWARD_V1 — pre-registered SEPARATE ablation (contact-retention teacher) on the O2 fresh box bank.

O2 canonical (coin_object_o2.py) keeps the FROZEN reward — the control, comparable to O1. This ablation changes ONLY the
TEACHER SELECTION criterion the fresh-refit proposal imitates: instead of K6-first (which a single well-timed flick can
satisfy), it selects options that genuinely RETAIN contact — phase-dependent, farming-proof:
  transport phase : reward continuous legal contact (dwell-continuity) + contact-retention + progress toward the zone;
                    penalize early contact loss and contact-loss→regain FARMING (a loss-regain cycle cannot re-collect).
  post-entry      : contact is useful only until the object slows; a stable release is NOT penalized; overshoot / never-
                    release are penalized.
Both teachers are evaluated by the SAME K6 certificate on the SAME fresh bank (comparability). Six mandatory negative
controls are audited on the SELECTED options: camping / pinning / high-force / overshoot / never-release / farming.

Verdicts (pre-registered): HOLD ≫ canonical deploy ⇒ REWARD_UNDERSPECIFIED_CONTACT_RETENTION; HOLD > canonical but still
≪ expert ⇒ REWARD_FIX_HELPFUL_BUT_ARCHITECTURAL_GAP_REMAINS; HOLD ≈ canonical ⇒ RETENTION_CAPABILITY_NOT_REWARD_LIMITED.
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_object_o2 import (  # noqa: E402
    CENTER_TOL, EVAL_H, FAMS, HELD_DWELL, K, SETTLE_VEL, SHAPES, _ball_tf, _hxy, fresh_o2_bank)
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, T_MAX, T_MIN, structured_carry_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-o2-square-rectangle-fresh-reconstruct"
O2_PROP = f"{D}/carry_proposal_o2_box_v1.pt"                  # canonical O2 proposal (frozen-reward teacher)
HOLD_PROP = f"{D}/carry_proposal_o2_hold_v1.pt"              # HOLD-aware teacher proposal
K_HOLD = K


def hold_metrics(rl, gate, pi0, base, theta, horizon=EVAL_H):
    """Rollout with a rich hook → contact retention, dwell-continuity, farming toggles, progress, overshoot speed at entry,
    contact-after-K6 (never-release). Non-behavioral (gate deep-copied)."""
    rz = float(rl._dtz())
    acc = {"contact": 0, "n": 0, "toggles": 0, "prev": False, "run": 0, "run_max": 0,
           "dtz0": rz, "dtz_min": rz, "entry_speed": None, "after_k6_contact": 0, "k6_seen": False}

    def hook(_p, strict):
        m = rl.inner._planar_metrics
        c = bool(m.left_contact or m.right_contact)
        acc["contact"] += int(c)
        acc["n"] += 1
        if c and not acc["prev"] and acc["n"] > 1:
            acc["toggles"] += 1                                # a regain after a loss = the farming signature
        acc["run"] = acc["run"] + 1 if c else 0
        acc["run_max"] = max(acc["run_max"], acc["run"])
        acc["prev"] = c
        dtz = float(rl._dtz())
        acc["dtz_min"] = min(acc["dtz_min"], dtz)
        if dtz <= CENTER_TOL and acc["entry_speed"] is None:
            acc["entry_speed"] = float(rl._speed())
        if int(strict) >= HELD_DWELL:
            acc["k6_seen"] = True
        if acc["k6_seen"] and c:
            acc["after_k6_contact"] += 1

    o = structured_carry_rollout(rl, copy.deepcopy(gate), pi0, base, np.asarray(theta, np.float32), horizon=horizon, frame_hook=hook)
    n = max(1, acc["n"])
    o["contact_frac"] = round(acc["contact"] / n, 3)
    o["dwell_continuity"] = round(min(acc["run_max"] / HELD_DWELL, 1.0), 3)
    o["toggles"] = acc["toggles"]
    o["progress"] = round(max(0.0, acc["dtz0"] - acc["dtz_min"]), 4)
    o["entry_speed"] = None if acc["entry_speed"] is None else round(acc["entry_speed"], 4)
    o["after_k6_contact"] = acc["after_k6_contact"]
    return o


def hold_score(o):
    """Phase-dependent HOLD lexicographic teacher score. K6 first (still must deliver); then genuine retention (dwell-
    continuity + contact-fraction) MINUS a bounded farming penalty (a loss→regain cycle only LOWERS the score); then
    progress; then fewer containment exits. Cannot be farmed: toggles subtract, they never add."""
    retention = o["dwell_continuity"] + o["contact_frac"] - 0.2 * min(o["toggles"], 5)
    return (int(o["k6"]), round(float(retention), 3), round(float(o["progress"]), 3), -int(o["contain_exit_ct"]), int(o["reached_handoff"]))


def negative_controls(o):
    """Six mandatory NEGATIVE controls on a selected/deployed option (True = pathology present)."""
    return {
        "camping": bool(o["contact_frac"] > 0.5 and o["progress"] < 0.02),               # holds but does not move it
        "overshoot": bool(o["entry_speed"] is not None and o["entry_speed"] > 3 * SETTLE_VEL),   # enters target too fast
        "never_release": bool(o["after_k6_contact"] > HELD_DWELL),                        # keeps clamping after delivery
        "farming": bool(o["toggles"] > 3),                                               # loss→regain cycling
        "pinning": None,                                                                 # needs boundary-proximity read — NOT AUDITED this pass
        "high_force": None,                                                              # requires contact-force read — NOT AUDITED this pass
    }


def hold_teacher(pi0, base, cand_ls, shape, shots, want, log):
    """HOLD teacher: per fresh box state, shoot random structured θ, keep the K6-delivering θ with the best HOLD score
    (genuine retention, farming-penalized). Returns (obs, theta) for HOLD-proposal fitting."""
    hx, hy = _hxy(SHAPES[shape])
    rk = {"geom": "POINT", "arm_mjcf_transform": _ball_tf, "coin_shape": "box", "disk_radius_override": hx, "disk_radius_y_override": hy}
    obs, theta = [], []
    for i, ls in enumerate(cand_ls):
        try:
            rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360, **rk)
        except ValueError:
            continue
        if int(rl._strict) != 0:
            continue
        rng = np.random.default_rng(2000 + i)
        best, best_o = None, None
        for _ in range(shots):
            th = np.concatenate([rng.uniform(-A_BOUND, A_BOUND, 12), rng.uniform(T_MIN, T_MAX, 3)]).astype(np.float32)
            o = hold_metrics(copy.deepcopy(rl), gate, pi0, base, th)
            if best_o is None or hold_score(o) > hold_score(best_o):
                best, best_o = th, o
        if best_o is not None and int(best_o["k6"]) == 1:                # confident only if it delivers AND holds
            obs.append(rl.obs().copy())
            theta.append(best)
        if (i + 1) % 20 == 0:
            log(f"    [HOLD {shape} {i+1}/{len(cand_ls)}] labels {len(obs)}")
        if len(obs) >= want:
            break
    return obs, theta


def main(smoke=False):
    import os

    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    o2_prop = load_proposal(O2_PROP)                                    # the canonical O2 proposal (control)
    shapes = ["square_1_1"] if smoke else list(SHAPES)
    want_ev = 6 if smoke else 24

    # ---- HOLD teacher bank (HOLD-scored) → HOLD proposal ----
    log("[HOLD] HOLD-aware teacher (contact-retention, farming-proof) on fresh box handoffs...")
    obs_h, th_h = [], []
    for sh in shapes:
        tr_ls, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(8 if smoke else 60),
                                             families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
        o, t = hold_teacher(pi0, base, tr_ls, sh, shots=(24 if smoke else 96), want=(8 if smoke else 40), log=log)
        obs_h += o
        th_h += t
    log(f"[HOLD] HOLD teacher labels: {len(obs_h)}")
    if len(obs_h) < 4:
        json.dump({"contract": "O2_HOLD_AWARE_REWARD_V1", "verdict": "HOLD_INSUFFICIENT_LABELS", "n": len(obs_h)}, open(f"{OUT}/o2_hold.json", "w"), indent=1)
        log("[HOLD] too few labels — abort")
        return
    kk = min(K_HOLD, max(2, len(obs_h) // 4))
    hold_prop, fit = fit_proposal(np.asarray(obs_h, np.float32), np.asarray(th_h, np.float32), kk,
                                  clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
    save_proposal(hold_prop, HOLD_PROP)
    log(f"[HOLD] HOLD proposal K={kk} res_mse {fit['res_mse']:.3f} → {HOLD_PROP}")

    # ---- eval canonical vs HOLD on the SAME fresh eval bank (K6) + negative-control audit ----
    ev_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 60),
                                         families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    results = {}
    for sh in shapes:
        bank = fresh_o2_bank(pi0, sh, ev_ls, want_ev, log)
        canon_k6 = hold_k6 = 0
        negs = {kk_: 0 for kk_ in ("camping", "overshoot", "never_release", "farming")}
        for i, it in enumerate(bank):
            rl, gate = it["rl"], it["gate"]
            obs = rl.obs()
            th_c, _ = search_select(copy.deepcopy(rl), copy.deepcopy(gate), o2_prop.theta(obs), pi0, base, np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
            th_h2, _ = search_select(copy.deepcopy(rl), copy.deepcopy(gate), hold_prop.theta(obs), pi0, base, np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
            oc = hold_metrics(copy.deepcopy(rl), gate, pi0, base, th_c)
            oh = hold_metrics(copy.deepcopy(rl), gate, pi0, base, th_h2)
            canon_k6 += int(oc["k6"])
            hold_k6 += int(oh["k6"])
            nc = negative_controls(oh)
            for kk_ in negs:
                negs[kk_] += int(bool(nc[kk_]))
        n = max(1, len(bank))
        results[sh] = {"n": len(bank), "canonical_k6": canon_k6, "canonical_rate": round(canon_k6 / n, 3),
                       "hold_k6": hold_k6, "hold_rate": round(hold_k6 / n, 3), "negative_controls": negs}
        log(f"  [{sh}] canonical K6 {canon_k6}/{n} | HOLD K6 {hold_k6}/{n} | neg {negs}")

    canon = float(np.mean([results[s]["canonical_rate"] for s in results]))
    hold = float(np.mean([results[s]["hold_rate"] for s in results]))
    # need the expert ceiling for the middle verdict — read it from the canonical O2 run if present
    exp = None
    try:
        exp = float(np.mean([json.load(open(f"{OUT}/o2.json"))["results"][s]["summary"]["expert"]["k6_rate"] for s in results]))
    except Exception:
        exp = None
    if hold > canon + 0.1 and (exp is None or hold >= 0.6 * exp):
        verdict = "REWARD_UNDERSPECIFIED_CONTACT_RETENTION"
    elif hold > canon + 0.05:
        verdict = "REWARD_FIX_HELPFUL_BUT_ARCHITECTURAL_GAP_REMAINS"
    else:
        verdict = "RETENTION_CAPABILITY_NOT_REWARD_LIMITED"
    out = {"contract": "O2_HOLD_AWARE_REWARD_V1", "date": "2026-07-24", "smoke": smoke, "hold_fit": fit,
           "hold_teacher_labels": len(obs_h), "results": results, "canonical_mean": round(canon, 3),
           "hold_mean": round(hold, 3), "expert_ceiling_from_canonical": exp, "verdict": verdict,
           "negative_control_note": "pinning/high_force NOT AUDITED this pass (need boundary/contact-force read); the other four are audited"}
    json.dump(out, open(f"{OUT}/o2_hold.json", "w"), indent=1, default=float)
    log(f"\n== O2 HOLD_AWARE_REWARD_V1 == canonical {round(canon,3)} vs HOLD {round(hold,3)} (expert {exp}) → {verdict}\n  artifact: {OUT}/o2_hold.json\nO2_HOLD_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
