"""DIVERGENT_K6_PREFIX_VALUE_AUDIT_V1 — when a short, physically-produced action prefix leads to genuinely different
K6/dwell, does the critic's VALUE (its Bellman-target bootstrap) rank the better consequence higher, and how much of any
apparent success is just the observed reward?

Renamed from "one-step action-ranking": this does NOT test the one-step critic's action ranking. Q(s0,a0) assumes the
critic's own continuation policy follows a0, but here K−1 artificial offset actions follow — so a first-action-Q failure
is continuation mismatch, not necessarily a critic defect. What this audits is whether the critic's bootstrap + reward can
VALUE a physically-produced short prefix. Design (all review safeguards):

  * CRITIC-INDEPENDENT generator — fixed K-step matched-norm actuator-offset prefixes, then frozen pi_0.
  * PHYSICS FIRST — roll every candidate; form matched, ROBUST (margin ≥2 on dwell/containment) primary-certifier
    divergent pairs (K6 | Δdwell | containment) BEFORE any critic is consulted; pairs matched on EFFECTIVE post-clamp norm.
  * DECOMPOSE the score — report separately: R_prefix (reward-only, critic-independent), bootstrap-Q (critic value at s_K
    under the TARGET critic + TARGET actor, i.e. the trainer's Bellman contract), G = R_prefix + γ^k·bootstrap, empirical
    full-return (critic-independent), first-action-Q (ONLINE critic, continuation-mismatch DIAGNOSTIC only). If R_prefix
    alone already ranks well, a high G is not the critic's merit.
  * PER-STATE statistical unit — pairwise accuracy per physical state, then a hierarchical bootstrap over
    critic-seed × state (pairs within a state are correlated; never flattened).
  * DISCOVERY vs EVAL states; smallest (K, mag) meeting a PRE-REGISTERED robust-pair yield (not the biggest/most-OOD).
  * SEPARATE Arm-A and Arm-B verdicts + cross-arm consistency; CI-vs-chance done with an equivalence band (a CI that
    merely contains 0.5 is INCONCLUSIVE, never "defective").
"""
import copy
import json
import sys
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_action_perturbation import (  # noqa: E402
    bootstrap_ci,
    classify_vs_chance,
    hierarchical_bootstrap_ci,
    lex_better,
    lex_key,
    primary_divergence,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    GAMMA,
    _aug,
    make_late_actor55_from_pi0,
    prefix_candidate_rollout,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
ASCALE = 4.0
H = 40
K_SET = (2, 4, 8)                                        # prefix lengths, ascending (prefer smallest)
MAG_SET = (0.01, 0.03, 0.08)                            # per-step offset magnitudes, ascending (prefer smallest)
MIN_ROBUST_FRAC = 0.35                                  # pre-registered: choose smallest (K,mag) with ≥35% states robust-paired
MIN_ROBUST_PAIRS = 10                                   # ...and at least this many robust pairs on DISCOVERY
NORM_TOL = 0.25                                         # pairs matched to within 25% effective cumulative norm
EQUIV = 0.05                                            # equivalence band around chance (0.5±0.05)
MIN_EVAL_STATES = 8                                     # gate: fewer robust-paired states ⇒ verdict is UNDERPOWERED, period


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _directions(adim):
    return [(f"e{ax}{'+' if sg > 0 else '-'}", ax, sg) for ax in range(adim) for sg in (+1, -1)]


def _offset(adim, ax, sg, mag):
    off = np.zeros(adim, np.float32); off[ax] = sg * mag; return off


def robust_divergence(a, b):
    """Margin ≥2 divergence — jitter-robust vs the dwell/speed threshold knife-edge (excludes dwell±1 that could flip on
    a 0.05999-vs-0.06001 speed reading)."""
    return abs(int(a["max_dwell"]) - int(b["max_dwell"])) >= 2 or abs(int(a["contain_exit_ct"]) - int(b["contain_exit_ct"])) >= 2


def norm_matched(ca, cb, tol=NORM_TOL):
    m = (ca["cum_eff_dev"] + cb["cum_eff_dev"]) / 2.0
    return abs(ca["cum_eff_dev"] - cb["cum_eff_dev"]) <= tol * m + 1e-6


def candidate_physics(pi0, base, templates, i, adim, K, mag):
    rl0, gate0 = templates[i]; out = []
    for name, ax, sg in _directions(adim):
        off = _offset(adim, ax, sg, mag)
        r = prefix_candidate_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, off, K, horizon=H)
        out.append({"name": name, "ax": ax, "sg": sg, "outcome": r["outcome"], "rewardsA": r["rewardsA"],
                    "rewardsB": r["rewardsB"], "obs55_K": r["obs55_K"], "terminated": r["terminated_in_prefix"],
                    "k_applied": r["k_applied"], "cum_eff_dev": r["cum_eff_dev"],
                    "full_returnA": r["full_returnA"], "full_returnB": r["full_returnB"]})
    return out


def state_pairs(cands):
    """Matched, robust, primary-certifier-divergent candidate pairs. Returns (all_pairs, primary, robust)."""
    all_n, primary, robust = 0, [], []
    for a, b in combinations(range(len(cands)), 2):
        all_n += 1
        oa, ob = cands[a]["outcome"], cands[b]["outcome"]
        if lex_key(oa) == lex_key(ob) or not norm_matched(cands[a], cands[b]):
            continue
        if primary_divergence(oa, ob):
            better = lex_better(oa, ob); primary.append((a, b, better))
            if robust_divergence(oa, ob):
                robust.append((a, b, better))
    return all_n, primary, robust


def _pair_acc(pairs, vals):
    """Fraction of pairs where the physically-better candidate has the higher value (0.5 on ties / missing). None if the
    pair set is empty."""
    scored = []
    for a, b, better_is_a in pairs:
        hi, lo = (a, b) if better_is_a else (b, a)
        va, vb = vals[hi], vals[lo]
        if va is None or vb is None:
            continue
        scored.append(1.0 if va > vb else (0.0 if va < vb else 0.5))
    return float(np.mean(scored)) if scored else None


def reward_values(cands, arm):
    r = "rewardsA" if arm == "A" else "rewardsB"; fr = "full_returnA" if arm == "A" else "full_returnB"
    R = [float(np.sum([GAMMA ** t * x for t, x in enumerate(c[r])])) for c in cands]
    full = [c[fr] for c in cands]
    return R, full


def critic_values(cands, R, critic_online, critic_t, actor_t, base, first_obs55):
    """G (n-step, target-contract bootstrap), bootstrap-Q (target critic value at s_K, None if terminated), first-action-Q
    (online critic — continuation-mismatch diagnostic)."""
    G, bootQ, firstQ = [], [], []
    for idx, c in enumerate(cands):
        if c["terminated"] or c["obs55_K"] is None:
            bootQ.append(None); G.append(R[idx])                       # terminated: n-step return is the reward (mask=0)
        else:
            o = torch.as_tensor(c["obs55_K"])[None]
            with torch.no_grad():
                q = float(critic_t.min_q(o, torch.clamp(actor_t.action_mean(o), -ASCALE, ASCALE))[0])
            bootQ.append(q); G.append(R[idx] + GAMMA ** c["k_applied"] * q)
        with torch.no_grad():
            firstQ.append(float(critic_online.min_q(torch.as_tensor(first_obs55)[None],
                                                    torch.as_tensor(c["_a0"])[None])[0]))
    return G, bootQ, firstQ


def choose_config(pi0, base, templates, disc_idx, adim, log):
    scores = {}
    for K in K_SET:
        for mag in MAG_SET:
            sw, tot, devs = 0, 0, []
            for i in disc_idx:
                cands = candidate_physics(pi0, base, templates, i, adim, K, mag)
                _a, _p, robust = state_pairs(cands)
                if robust:
                    sw += 1; tot += len(robust)
                devs += [c["cum_eff_dev"] for c in cands]
            scores[(K, mag)] = {"states_robust": sw, "robust_pairs": tot, "mean_cum_eff_dev": round(float(np.mean(devs)), 4)}
            log(f"    K={K} mag={mag}: states_robust {sw}/{len(disc_idx)} robust_pairs {tot} eff_dev {scores[(K, mag)]['mean_cum_eff_dev']}")
    frac = MIN_ROBUST_FRAC * len(disc_idx)
    ok = [(K, mag) for K in K_SET for mag in MAG_SET
          if scores[(K, mag)]["states_robust"] >= frac and scores[(K, mag)]["robust_pairs"] >= MIN_ROBUST_PAIRS]
    if ok:
        return ok[0], scores, "preregistered-min-yield"                # K_SET/MAG_SET ascending → smallest that qualifies
    best = max(scores, key=lambda km: (scores[km]["states_robust"], scores[km]["robust_pairs"]))
    return best, scores, "fallback-max-yield-NO-config-met-threshold"


def _agg(per_seed_state, stat=np.mean):
    return hierarchical_bootstrap_ci(per_seed_state, stat=stat)


def main(seeds=(0, 1, 2), smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]
    adim = pi0.action_dim
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    tcfg = TransactionalConfig(); seeds = list(seeds[:1] if smoke else seeds)
    base = make_late_actor55_from_pi0(pi0, trainable=False)

    n_disc, n_eval = (8, 12) if smoke else (24, 120)
    log(f"[panel] held-out boundary panel ({n_disc} DISCOVERY + {n_eval} EVAL)...")
    panel, comp, strict_hist = build_boundary_panel(pi0, range(6200, 7400), forbidden, want=n_disc + n_eval)
    templates = [reconstruct_handoff(pi0, ls, horizon=360)[:2] for ls in panel]
    disc_idx, eval_idx = list(range(n_disc)), list(range(n_disc, len(templates)))
    log(f"[panel] n={len(panel)} families={comp} strict={strict_hist} | discovery={len(disc_idx)} eval={len(eval_idx)}")

    log("[discovery] choosing SMALLEST (K,mag) meeting pre-registered robust-pair yield...")
    (Kbest, magbest), disc_scores, choose_reason = choose_config(pi0, base, templates, disc_idx, adim, log)
    log(f"[discovery] chosen K={Kbest} mag={magbest} ({choose_reason})")

    # ── EVAL: freeze candidates + robust matched pairs + critic-INDEPENDENT reward/return values, before any critic ──
    log("[eval] rolling EVAL candidates + freezing robust matched divergent pairs...")
    ev = []  # per state: {cands, robust_pairs, robust_nonterm_pairs, first_obs55, RA, RB, fullA, fullB, meta}
    yields = {"all": 0, "primary": 0, "robust": 0}
    for i in eval_idx:
        cands = candidate_physics(pi0, base, templates, i, adim, Kbest, magbest)
        rl0, _g = templates[i]; s0 = int(rl0._strict); fobs = _aug(rl0.obs(), s0)
        a0 = torch.clamp(base.action_mean(torch.as_tensor(fobs)[None])[0], -ASCALE, ASCALE).numpy()
        for c in cands:
            c["_a0"] = np.clip(a0 + _offset(adim, c["ax"], c["sg"], magbest), -ASCALE, ASCALE).astype(np.float32)
        all_n, primary, robust = state_pairs(cands)
        nonterm = [(a, b, t) for (a, b, t) in robust if not cands[a]["terminated"] and not cands[b]["terminated"]]
        yields["all"] += all_n; yields["primary"] += len(primary); yields["robust"] += len(robust)
        RA, fullA = reward_values(cands, "A"); RB, fullB = reward_values(cands, "B")
        ev.append({"cands": cands, "robust": robust, "nonterm": nonterm, "fobs": fobs,
                   "RA": RA, "RB": RB, "fullA": fullA, "fullB": fullB,
                   "meta": {"state": i, "strict0": s0, "dtz0": round(rl0._dtz(), 4), "speed0": round(rl0._speed(), 4),
                            "n_robust": len(robust), "mean_cum_eff_dev": round(float(np.mean([c["cum_eff_dev"] for c in cands])), 4)}})
    states_with_pairs = [e for e in ev if e["robust"]]
    log(f"[eval] pairs all→primary→robust: {yields['all']}→{yields['primary']}→{yields['robust']} "
        f"over {len(states_with_pairs)}/{len(eval_idx)} states with a robust pair")

    # critic-INDEPENDENT metrics (reward-only, full-return) — per state, seed-invariant
    ind = {"A": {"R": [], "full": []}, "B": {"R": [], "full": []}}
    for e in states_with_pairs:
        for arm in ("A", "B"):
            Rv, fullv = (e["RA"], e["fullA"]) if arm == "A" else (e["RB"], e["fullB"])
            ind[arm]["R"].append(_pair_acc(e["robust"], Rv)); ind[arm]["full"].append(_pair_acc(e["robust"], fullv))

    # critic-DEPENDENT metrics per seed
    dep = {"A": {"G": [], "boot": [], "firstQ": []}, "B": {"G": [], "boot": [], "firstQ": []}}
    for seed in seeds:
        log(f"[seed {seed}] train Arm A + Arm B Markov critics...")
        _rA, artA = train_arm(pi0, "A", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        _rB, artB = train_arm(pi0, "B", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        for arm, art in (("A", artA), ("B", artB)):
            co, ct, at = art["critic"], art["critic_target"], art["actor_target"]
            gs, bs, fs = [], [], []
            for e in states_with_pairs:
                R = e["RA"] if arm == "A" else e["RB"]
                G, bootQ, firstQ = critic_values(e["cands"], R, co, ct, at, base, e["fobs"])
                gs.append(_pair_acc(e["robust"], G)); bs.append(_pair_acc(e["nonterm"], bootQ)); fs.append(_pair_acc(e["robust"], firstQ))
            dep[arm]["G"].append(gs); dep[arm]["boot"].append(bs); dep[arm]["firstQ"].append(fs)
            log(f"  [seed {seed} {arm}] per-state mean acc: G {round(np.nanmean([x for x in gs if x is not None]), 3)} "
                f"boot {round(np.nanmean([x for x in bs if x is not None]) if any(x is not None for x in bs) else float('nan'), 3)} "
                f"first-Q {round(np.nanmean([x for x in fs if x is not None]), 3)}")

    # ── aggregate: critic-independent = state bootstrap; critic-dependent = hierarchical seed×state ──
    res = {}
    for arm in ("A", "B"):
        res[arm] = {"R_prefix": bootstrap_ci(ind[arm]["R"], stat=np.mean), "full_return": bootstrap_ci(ind[arm]["full"], stat=np.mean),
                    "G_nstep": _agg(dep[arm]["G"]), "bootstrap_Q": _agg(dep[arm]["boot"]), "first_action_Q_diag": _agg(dep[arm]["firstQ"])}
        res[arm]["class_G"] = classify_vs_chance(res[arm]["G_nstep"], equiv=EQUIV)
        res[arm]["class_bootstrap_Q"] = classify_vs_chance(res[arm]["bootstrap_Q"], equiv=EQUIV)
        res[arm]["class_R_prefix"] = classify_vs_chance(res[arm]["R_prefix"], equiv=EQUIV)

    # ── verdict: per-arm + cross-arm; centre on the CRITIC's marginal value (bootstrap-Q), reward as baseline ──
    cg = {arm: res[arm]["class_bootstrap_Q"] for arm in ("A", "B")}
    reward_ranks = res["A"]["class_R_prefix"] == "ABOVE" or res["B"]["class_R_prefix"] == "ABOVE"
    if len(states_with_pairs) < MIN_EVAL_STATES:
        verdict = "DIVERGENT_K6_UNDERPOWERED_TOO_FEW_ROBUST_STATES"
        nxt = (f"only {len(states_with_pairs)} EVAL states had a robust divergent pair (< {MIN_EVAL_STATES}); the per-state "
               "unit is too small to conclude. Small in-support prefixes rarely produce robust K6 divergence — enlarge the "
               "held-out panel or accept a larger (more OOD) offset with support diagnostics.")
    elif cg["A"] == cg["B"] == "ABOVE":
        verdict = "CRITIC_VALUE_RANKS_DIVERGENT_K6_CONSEQUENCE"
        nxt = ("the critic's value (Bellman-target bootstrap) ranks the physically-better consequence above chance on BOTH "
               "arms — value understanding of terminally-relevant alternatives EXISTS; the remaining gap is converting it "
               "(actor search / temporal credit / a sequence critic), not value fidelity. Note reward-only also ranks — "
               "report the critic marginal.")
    elif cg["A"] in ("ANTI",) or cg["B"] in ("ANTI",):
        verdict = "CRITIC_VALUE_ANTIRANKS_DIVERGENT_K6"
        nxt = "critic value ranks the WORSE consequence higher on ≥1 arm → PAIRED_LOCAL_ACTION_RANKING_CRITIC / pessimism directly justified"
    elif "INCONCLUSIVE" in cg.values() or cg["A"] != cg["B"]:
        verdict = "CRITIC_VALUE_RANKING_INCONCLUSIVE_OR_ARM_DISAGREEMENT"
        nxt = f"bootstrap-Q A={cg['A']} B={cg['B']} — add EVAL states/seeds or reconcile arms before concluding"
    else:
        verdict = "CRITIC_VALUE_EQUIVALENT_TO_CHANCE_ON_DIVERGENT_K6"
        nxt = ("critic value is statistically equivalent to chance on the consequence (reward "
               + ("does" if reward_ranks else "does not") + " rank) → paired-difference ranking loss justified")

    out = {"contract": "DIVERGENT_K6_PREFIX_VALUE_AUDIT_V1", "date": "2026-07-23", "no_new_campaign": True, "smoke": smoke,
           "reframe": "audits critic bootstrap+reward VALUE of a physical prefix; first-action-Q is a continuation-mismatch diagnostic, NOT a one-step critic verdict",
           "method": {"critic_independent_generator": True, "physics_first": True, "per_state_unit": True,
                      "hierarchical_bootstrap_seed_x_state": True, "target_contract_bootstrap": "critic_target + actor_target",
                      "robust_margin": ">=2 dwell/containment", "norm_matched_tol": NORM_TOL, "equiv_band": EQUIV,
                      "chosen_K": Kbest, "chosen_mag": magbest, "choose_reason": choose_reason, "horizon": H},
           "panel": {"n": len(panel), "families": comp, "strict_hist": strict_hist, "all_held_out": True},
           "discovery_scores": {f"K{k}_mag{m}": v for (k, m), v in disc_scores.items()},
           "eval_pair_yield": yields, "eval_states_with_robust_pair": len(states_with_pairs), "seeds": seeds,
           "A": res["A"], "B": res["B"], "cross_arm_bootstrapQ": {"A": cg["A"], "B": cg["B"]},
           "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/divergent_k6_prefix_value_audit_v1.json", "w"), indent=1, default=float)

    log("\n== DIVERGENT_K6_PREFIX_VALUE_AUDIT_V1 (per-state, decomposed, target-contract, per-arm) ==")
    log(f"  panel n={len(panel)} strict={strict_hist} | K={Kbest} mag={magbest} | robust pairs={yields['robust']} over {len(states_with_pairs)} states")
    for arm in ("A", "B"):
        r = res[arm]
        log(f"  critic {arm}: G {r['G_nstep']['stat']} CI[{r['G_nstep']['lo']},{r['G_nstep']['hi']}] ({r['class_G']}) | "
            f"bootstrap-Q {r['bootstrap_Q']['stat']} CI[{r['bootstrap_Q']['lo']},{r['bootstrap_Q']['hi']}] ({r['class_bootstrap_Q']}) | "
            f"R_prefix {r['R_prefix']['stat']} ({r['class_R_prefix']}) | first-Q(diag) {r['first_action_Q_diag']['stat']}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/divergent_k6_prefix_value_audit_v1.json\nDIVERGENT_K6_PAIRS_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
