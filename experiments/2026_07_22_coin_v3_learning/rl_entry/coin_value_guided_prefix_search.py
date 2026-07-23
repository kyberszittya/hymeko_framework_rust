"""VALUE_GUIDED_PREFIX_SEARCH_V1 — can the (validated) learned value be CONVERTED into real K6 gain by a bounded,
critic-guided, receding-horizon search whose candidates are ranked ONLY by the learned signal?

Builds on DIVERGENT_K6_PREFIX_VALUE_AUDIT (POST_PREFIX_TERMINAL_VALUE_ORDERING_CONFIRMED + ONE_STEP_VALUE_TO_ACTION_CREDIT
_UNRESOLVED): the critic's post-prefix value orders divergent K6 consequences above chance, but the one-step action credit
does not. So we ask whether a short (K=2) prefix search, selecting purely by the learned signal, converts that value into
a physical improvement over pi_0.

CRITICAL: the exact simulator is used ONLY for the bounded K=2 lookahead (the search horizon); the terminal value beyond
the prefix is the LEARNED Q_target — never a full exact-rollout-to-episode-end oracle (that would only re-prove the task
is solvable). MANDATORY three-scorer ablation on the SAME candidates disambiguates whether the critic VALUE helps or the
dense reward drives it:
  * REWARD_ONLY          — argmax Σγ^t r_t
  * BOOTSTRAP_VALUE_ONLY — argmax γ^K Q_target(s_K, π_target(s_K))
  * REWARD_PLUS_VALUE    — argmax Σγ^t r_t + γ^K Q_target(s_K, π_target(s_K))
All receding-horizon (execute only the first action, replan). Candidates: pi_0 and support-bounded pi_0±ε·e_i chunks.

PRIMARY result is PHYSICS (K1/K3/K5/K6, max dwell, full-containment exit, speed, effective drift vs pi_0, replay-support
distance), NOT the predicted score. Success = K6 > pi_0 AND exit not worse, across held-out states and seeds. Statistical
safeguards: per-STATE unit + hierarchical bootstrap over seed×state; matched states/seeds; paired ΔK6 classified vs 0 with
an equivalence band; underpower gate; BOTH Arm-A and Arm-B critics. Trust region / certificate gate stay. NOT full TD3/SAC.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_action_perturbation import classify_vs_chance, hierarchical_bootstrap_ci  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    GAMMA,
    make_late_actor55_from_pi0,
    prefix_candidate_rollout,
    receding_horizon_rollout,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
ASCALE = 4.0
K = 2                                                   # minimal divergent prefix length (from the audit)
MAG = 0.08                                              # discovered divergent per-step offset (support-bounded, reported)
H = 20                                                  # control horizon (K6 reachable from strict-3/4/5 within it)
BIG = 1e6                                               # a delivered (terminated) lookahead is the best possible
SCORERS = ("reward", "value", "reward_value")


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _offsets(adim):
    offs = [np.zeros(adim, np.float32)]                 # index 0 == pi_0 (no perturbation)
    for ax in range(adim):
        for sg in (+1, -1):
            o = np.zeros(adim, np.float32); o[ax] = sg * MAG; offs.append(o)
    return offs


def candidate_full_outcomes(templates, i, pi0, base, offsets):
    rl0, gate0 = templates[i]
    return [prefix_candidate_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, off, K, horizon=H)["outcome"]
            for off in offsets]


def informative(outs):
    """Held-out ROBUST-PAIR boundary state: some support-bounded chunk reaches K6 while another clearly fails (robust
    dwell spread, not knife-edge) — the decision matters. pi_0 (offset 0) may itself hold or fail; the primary measure is
    the NET paired ΔK6 vs pi_0 (a value-usable controller must at least not HURT, and ideally convert the pi_0-fail
    subset). Pure pi_0-fail states are too rare (~0.8%) to power on their own, so we measure net + report that subset."""
    k6s = [o["k6"] for o in outs]; dwells = [o["max_dwell"] for o in outs]
    return max(k6s) == 1 and min(dwells) <= HELD_DWELL - 2


def sel_pi0(rl, gate, o55, a_pi0):
    return a_pi0, {"choice": 0, "nonpi0": 0}


def make_sel_search(pi0, base, critic_t, actor_t, offsets, scorer):
    def f(rl, gate, o55, a_pi0):
        R, B = [], []
        for off in offsets:
            r = prefix_candidate_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, off, K, horizon=K)
            R.append(float(sum(GAMMA ** t * x for t, x in enumerate(r["rewardsA"]))))
            if r["terminated_in_prefix"] or r["obs55_K"] is None:
                B.append(BIG)                            # delivered in the lookahead = best terminal value
            else:
                o = torch.as_tensor(r["obs55_K"])[None]
                with torch.no_grad():
                    B.append(float(GAMMA ** r["k_applied"] * critic_t.min_q(o, torch.clamp(actor_t.action_mean(o), -ASCALE, ASCALE))[0]))
        sc = R if scorer == "reward" else (B if scorer == "value" else [a + b for a, b in zip(R, B)])
        b = int(np.argmax(sc))
        return np.clip(a_pi0 + offsets[b], -ASCALE, ASCALE), {"choice": b, "nonpi0": int(b != 0)}
    return f


def buffer_obs_sample(buf, rng, n=300):
    obs = [t["obs"] for tr in buf.trajectories if tr for t in tr]
    if not obs:
        return None
    idx = rng.integers(0, len(obs), min(n, len(obs)))
    return np.asarray([obs[j] for j in idx], np.float32)


def run_controller(templates, i, pi0, base, select, buf_sample, arm="A"):
    rl0, gate0 = templates[i]
    out, infos = receding_horizon_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, select, horizon=H, arm=arm)
    gon = [x for x in infos if x and x.get("gate_on")]
    out["cum_drift"] = round(float(sum(x["drift"] for x in gon)), 4)
    out["nonpi0_rate"] = round(float(np.mean([x.get("nonpi0", 0) for x in gon])) if gon else 0.0, 3)
    if buf_sample is not None and gon:
        d = [float(np.linalg.norm(buf_sample - x["obs55"], axis=1).min()) for x in gon]
        out["support_dist"] = round(float(np.mean(d)), 4)
    else:
        out["support_dist"] = None
    return out


def _cls(per_seed_state, equiv=0.02):
    ci = hierarchical_bootstrap_ci(per_seed_state, stat=np.mean)
    return ci, classify_vs_chance(ci, chance=0.0, equiv=equiv)


def main(seeds=(0, 1, 2), smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]
    adim = pi0.action_dim; offsets = _offsets(adim)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    tcfg = TransactionalConfig(); seeds = list(seeds[:1] if smoke else seeds); arms = ("A",) if smoke else ("A", "B")
    base = make_late_actor55_from_pi0(pi0, trainable=False)

    n_scan, keep_cap = (80, 6) if smoke else (320, 30)
    log("[panel] scanning held-out boundary states, filtering to ROBUST-PAIR (a chunk reaches K6, another clearly fails)...")
    panel, comp, strict_hist = build_boundary_panel(pi0, range(6200, 7600), forbidden, want=n_scan)
    templates = [reconstruct_handoff(pi0, ls, horizon=360)[:2] for ls in panel]
    kept = []
    for i in range(len(templates)):
        if len(kept) >= keep_cap:
            break
        outs = candidate_full_outcomes(templates, i, pi0, base, offsets)
        if informative(outs):
            kept.append(i)
    log(f"[panel] scanned {len(templates)} boundary (strict {strict_hist}); kept {len(kept)} robust-pair states "
        f"(a support-bounded chunk reaches K6, another clearly fails)")

    # pi_0 baseline (seed-independent) — K6 is 0 by the improvable filter; measured for the exit/dwell reference
    pi0_out = [run_controller(templates, i, pi0, base, sel_pi0, None) for i in kept]

    # per seed × arm: three learned-signal receding scorers
    data = {arm: {sc: {"k6": [], "exit": [], "dwell": [], "nonpi0": [], "drift": [], "support": []} for sc in SCORERS} for arm in arms}
    for seed in seeds:
        log(f"[seed {seed}] train Arm {'/'.join(arms)} Markov critics...")
        arts = {}
        for arm in arms:
            _r, arts[arm] = train_arm(pi0, arm, stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        for arm in arms:
            art = arts[arm]; ct, at = art["critic_target"], art["actor_target"]
            bsample = buffer_obs_sample(art["buf"], np.random.default_rng(seed))
            for sc in SCORERS:
                sel = make_sel_search(pi0, base, ct, at, offsets, sc)
                outs = [run_controller(templates, i, pi0, base, sel, bsample, arm=arm) for i in kept]
                data[arm][sc]["k6"].append([o["k6"] for o in outs])
                data[arm][sc]["exit"].append([o["contain_exit_ct"] for o in outs])
                data[arm][sc]["dwell"].append([o["max_dwell"] for o in outs])
                data[arm][sc]["nonpi0"].append([o["nonpi0_rate"] for o in outs])
                data[arm][sc]["drift"].append([o["cum_drift"] for o in outs])
                data[arm][sc]["support"].append([o["support_dist"] for o in outs if o["support_dist"] is not None])
                k6r = float(np.mean(data[arm][sc]["k6"][-1]))
                log(f"  [seed {seed} {arm} {sc:13}] K6 {round(k6r, 3)} nonpi0 {round(float(np.mean(data[arm][sc]['nonpi0'][-1])), 3)} "
                    f"drift {round(float(np.mean(data[arm][sc]['drift'][-1])), 3)}")

    pi0_k6 = [o["k6"] for o in pi0_out]; pi0_exit = [o["contain_exit_ct"] for o in pi0_out]
    log(f"[pi_0 reference] K6 {round(float(np.mean(pi0_k6)), 3)} | mean containment-exit {round(float(np.mean(pi0_exit)), 3)}")

    hidx = [j for j, p in enumerate(pi0_k6) if p == 0]      # pi_0-fail subset (the improvement lens; typically small)
    res = {}
    for arm in arms:
        res[arm] = {}
        for sc in SCORERS:
            dk6 = [[k - p for k, p in zip(seedrow, pi0_k6)] for seedrow in data[arm][sc]["k6"]]     # NET paired ΔK6 vs pi_0
            dexit = [[e - p for e, p in zip(seedrow, pi0_exit)] for seedrow in data[arm][sc]["exit"]]
            head = [[seedrow[j] for j in hidx] for seedrow in data[arm][sc]["k6"]]                  # K6 on pi_0-fail states (=ΔK6)
            ci_k6, cls_k6 = _cls(dk6); ci_ex, _ = _cls(dexit); ci_hr, cls_hr = _cls(head)
            res[arm][sc] = {"K6_rate": hierarchical_bootstrap_ci(data[arm][sc]["k6"], stat=np.mean),
                            "net_dK6_vs_pi0": ci_k6, "class_net_dK6": cls_k6, "dexit_vs_pi0": ci_ex,
                            "pi0fail_n": len(hidx), "dK6_pi0fail": ci_hr, "class_dK6_pi0fail": cls_hr,
                            "nonpi0_rate": hierarchical_bootstrap_ci(data[arm][sc]["nonpi0"], stat=np.mean),
                            "cum_drift": hierarchical_bootstrap_ci(data[arm][sc]["drift"], stat=np.mean),
                            "support_dist": hierarchical_bootstrap_ci([r for r in data[arm][sc]["support"] if r], stat=np.mean)}

    # ── decision tree (per user); improvement is judged on the HEADROOM (pi_0-fail) subset — where K6 can actually rise —
    #     disambiguated by the scorer ablation (does value add over reward) + non-pi0 rate + drift ──
    def net_improves(arm, sc):                              # net K6 up vs pi_0, exit not worse
        r = res[arm][sc]
        return (r["net_dK6_vs_pi0"]["lo"] or -1) > 0 and (r["dexit_vs_pi0"]["hi"] or 0) <= 0.5
    def net_hurts(arm, sc):
        return res[arm][sc]["class_net_dK6"] == "ANTI"
    val_improves = all(net_improves(a, "reward_value") or net_improves(a, "value") for a in arms)
    rew_improves = all(net_improves(a, "reward") for a in arms)
    val_hurts = any(net_hurts(a, "value") or net_hurts(a, "reward_value") for a in arms)
    val_adds = all(res[a]["reward_value"]["net_dK6_vs_pi0"]["stat"] is not None and res[a]["reward"]["net_dK6_vs_pi0"]["stat"] is not None
                   and res[a]["reward_value"]["net_dK6_vs_pi0"]["stat"] > res[a]["reward"]["net_dK6_vs_pi0"]["stat"] + 0.03 for a in arms)
    big_drift = any((res[a][sc]["cum_drift"]["stat"] or 0) > 0.4 for a in arms for sc in ("value", "reward_value"))

    if len(kept) < 8:
        verdict = "VALUE_GUIDED_SEARCH_UNDERPOWERED_TOO_FEW_STATES"; nxt = f"only {len(kept)} robust-pair held-out states (<8) — enlarge scan"
    elif val_improves and (val_adds or not rew_improves):
        verdict = "VALUE_GUIDED_SEARCH_BEATS_PI0_VALUE_USABLE"
        nxt = "value is usable for prospective control — the wall is one-step actor improvement; amortise this search into a sequence/chunk actor or prefix critic"
    elif val_hurts and not rew_improves:
        verdict = "VALUE_GUIDED_SEARCH_HURTS_Q_NOT_USABLE_FOR_PROSPECTIVE_CONTROL"
        nxt = ("ranking by Q(s_K) steers OFF pi_0 into failure (net K6 down, high drift) — the value's consequence-recognition "
               "does NOT transfer to prospective control → a chunk critic learning Q_K(s, a_0:K-1) directly, or a better "
               "temporal-credit target; reward alone " + ("does" if rew_improves else "does not") + " find the save")
    elif rew_improves and not val_adds:
        verdict = "REWARD_DRIVES_IMPROVEMENT_VALUE_ADDS_NOTHING"
        nxt = "critic terminal recognition is real but not usable for prospective control → a sequence critic / better temporal-credit target"
    elif not (val_improves or rew_improves):
        verdict = "NO_SEARCH_CONVERTS_PI0_NEAR_OPTIMAL_ON_BOUNDARY"
        nxt = ("no scorer nets a K6 gain and none clearly hurts — pi_0 is near-optimal on these boundary states (little to "
               "convert)" + ("; large drift suggests support/candidate-generation is the limit" if big_drift else "")
               + ". Q(s_K) recognises consequences post-hoc but a bounded search over these candidates does not plan a gain")
    else:
        verdict = "VALUE_GUIDED_SEARCH_INCONCLUSIVE"; nxt = "CIs straddle 0 or arms disagree — add states/seeds"

    out = {"contract": "VALUE_GUIDED_PREFIX_SEARCH_V1", "date": "2026-07-23", "no_new_campaign": True, "smoke": smoke,
           "method": {"K": K, "mag": MAG, "horizon": H, "candidates": len(offsets), "scorers": list(SCORERS),
                      "lookahead": "K-step exact ONLY; terminal value = learned Q_target (no full exact-rollout oracle)",
                      "bootstrap": "critic_target + actor_target", "per_state_unit": True, "hierarchical_seed_x_state": True,
                      "state_filter": "robust-pair: some chunk K6=1 ∧ another clearly fails (dwell ≤ K6-2); primary = NET paired ΔK6 vs pi_0", "arms": list(arms)},
           "panel": {"scanned": len(templates), "kept_improvable": len(kept), "strict_hist": strict_hist, "all_held_out": True},
           "pi0_reference": {"K6": round(float(np.mean(pi0_k6)), 3), "mean_containment_exit": round(float(np.mean(pi0_exit)), 3)},
           "seeds": seeds, "results": res, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/value_guided_prefix_search_v1.json", "w"), indent=1, default=float)

    log("\n== VALUE_GUIDED_PREFIX_SEARCH_V1 (physics-primary, 3-scorer ablation, per-state hierarchical) ==")
    log(f"  {len(kept)} robust-pair held-out states ({len(hidx)} pi_0-fail) | K={K} mag={MAG} | pi_0 K6 {round(float(np.mean(pi0_k6)), 3)}")
    for arm in arms:
        for sc in SCORERS:
            r = res[arm][sc]
            log(f"  {arm} {sc:13}: K6 {r['K6_rate']['stat']} | net ΔK6 {r['net_dK6_vs_pi0']['stat']} "
                f"CI[{r['net_dK6_vs_pi0']['lo']},{r['net_dK6_vs_pi0']['hi']}] ({r['class_net_dK6']}) | "
                f"pi0fail ΔK6 {r['dK6_pi0fail']['stat']}(n{r['pi0fail_n']}) | Δexit {r['dexit_vs_pi0']['stat']} | "
                f"nonpi0 {r['nonpi0_rate']['stat']} drift {r['cum_drift']['stat']}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/value_guided_prefix_search_v1.json\nVALUE_GUIDED_PREFIX_SEARCH_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
