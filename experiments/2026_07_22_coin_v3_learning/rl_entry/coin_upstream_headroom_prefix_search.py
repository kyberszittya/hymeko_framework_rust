"""UPSTREAM_HEADROOM_PREFIX_SEARCH_V1 — re-run the exact critic-guided prefix-search ablation on UPSTREAM states (strict
1–2) where pi_0 has measurable headroom, with no oracle filtering of the eval panel.

VALUE_GUIDED_PREFIX_SEARCH found pi_0 near-optimal on the settling boundary (strict 3–5), so there was nothing to convert.
Here the panel is earlier states (strict 1–2), kept UNFILTERED (the candidate bank may be tuned on a discovery panel, but
the EVAL panel is not filtered to contain a good chunk). Three metrics are reported SEPARATELY:
  * candidate coverage    — in how many pi_0-fail states does the bank contain ≥1 K6-improving chunk (oracle, not a filter),
  * selection quality     — of those, how many the scorer actually FINDS,
  * unconditional ΔK6      — net K6 change over the WHOLE held-out panel.

Same candidates, five controllers: PI_0, RANDOM_VALID (isolates scorer intelligence vs a merely-good generator),
REWARD_ONLY, BOOTSTRAP_VALUE_ONLY, REWARD_PLUS_VALUE. Strata kept SEPARATE by (strict × family). Both arms; 3 seeds;
per-state hierarchical bootstrap; equivalence-band CI vs 0.
"""
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_action_perturbation import classify_vs_chance, hierarchical_bootstrap_ci  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0, train_arm  # noqa: E402
from hymeko_rl.coin_delivery.coin_prefix_search import (  # noqa: E402
    buffer_obs_sample,
    candidate_outcomes,
    make_sel_random,
    make_sel_search,
    offsets,
    run_controller,
    sel_pi0,
)
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
K, MAG, H = 2, 0.08, 24
STRICT_UP = (1, 2)
FAMS_UP = ("target_entry", "braking", "settling_dwell")
LEARNED = ("reward", "value", "reward_value")
MIN_STATES, MIN_COVERED = 8, 4


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _metrics(out):
    return {"k1": out["k1"], "k3": out["k3"], "k5": out["k5"], "k6": out["k6"], "dwell": out["max_dwell"],
            "exit": out["contain_exit_ct"], "speed": out["mean_speed"], "drift": out.get("cum_drift", 0.0),
            "nonpi0": out.get("nonpi0_rate", 0.0)}


def _agg(per_seed_state, stat=np.mean):
    return hierarchical_bootstrap_ci(per_seed_state, stat=stat)


def main(seeds=(0, 1, 2), smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]
    adim = pi0.action_dim; offs = offsets(adim, MAG)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    tcfg = TransactionalConfig(); seeds = list(seeds[:1] if smoke else seeds); arms = ("A",) if smoke else ("A", "B")
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    want = 40 if smoke else 220

    log("[panel] scanning held-out UPSTREAM boundary states (strict 1–2), NO oracle filter on eval...")
    panel, comp, strict_hist = build_boundary_panel(pi0, range(6200, 8000), forbidden, want=want,
                                                    families=FAMS_UP, strict_primary=STRICT_UP, strict_fill=(), per_seed_cap=4)
    templates, strata = [], []
    for ls in panel:
        rl, gate, _h, _rec = reconstruct_handoff(pi0, ls, horizon=360)
        templates.append((rl, gate)); strata.append(f"strict{int(rl._strict)}/{ls.family}")
    n = len(templates)
    log(f"[panel] kept {n} unfiltered upstream states | strata { {k: v for k, v in sorted(Counter(strata).items())} }")

    # pi_0 baseline + oracle candidate coverage (seed-independent; coverage is NOT used to filter)
    pi0_out = [run_controller(templates, i, pi0, base, sel_pi0, None, H) for i in range(n)]
    pi0_k6 = [o["k6"] for o in pi0_out]
    cand_k6 = [[o["k6"] for o in candidate_outcomes(templates, i, pi0, base, offs, K, H)] for i in range(n)]
    pi0fail = [i for i in range(n) if pi0_k6[i] == 0]
    covered = [i for i in pi0fail if max(cand_k6[i]) == 1]                 # a K6-improving chunk EXISTS
    log(f"[baseline] pi_0 K6 {round(float(np.mean(pi0_k6)), 3)} (headroom target 0.2–0.8) | pi_0-fail {len(pi0fail)}/{n} | "
        f"candidate-coverage {len(covered)}/{len(pi0fail)} (bank has an improving chunk)")

    # controllers per seed (random is critic-independent, re-seeded per seed) × arm (learned scorers)
    rand = {m: [] for m in ("k6", "exit", "cov_sel")}
    dep = {arm: {sc: {m: [] for m in ("k6", "exit", "drift", "nonpi0", "cov_sel", "f2s", "s2f")} for sc in LEARNED} for arm in arms}
    for seed in seeds:
        log(f"[seed {seed}] train Arm {'/'.join(arms)} Markov critics...")
        arts = {arm: train_arm(pi0, arm, stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)[1] for arm in arms}
        rout = [run_controller(templates, i, pi0, base, make_sel_random(offs, 1000 + seed), None, H) for i in range(n)]
        rand["k6"].append([o["k6"] for o in rout]); rand["exit"].append([o["contain_exit_ct"] for o in rout])
        rand["cov_sel"].append([rout[i]["k6"] for i in covered])
        log(f"  [seed {seed} RANDOM_VALID] K6 {round(float(np.mean(rand['k6'][-1])), 3)} sel {round(float(np.mean(rand['cov_sel'][-1])) if covered else 0.0, 3)}")
        for arm in arms:
            art = arts[arm]; ct, at = art["critic_target"], art["actor_target"]
            bs = buffer_obs_sample(art["buf"], np.random.default_rng(seed))
            for sc in LEARNED:
                sel = make_sel_search(pi0, base, ct, at, offs, K, sc)
                outs = [run_controller(templates, i, pi0, base, sel, bs, H) for i in range(n)]
                m = [_metrics(o) for o in outs]
                dep[arm][sc]["k6"].append([x["k6"] for x in m]); dep[arm][sc]["exit"].append([x["exit"] for x in m])
                dep[arm][sc]["drift"].append([x["drift"] for x in m]); dep[arm][sc]["nonpi0"].append([x["nonpi0"] for x in m])
                dep[arm][sc]["cov_sel"].append([m[i]["k6"] for i in covered])
                dep[arm][sc]["f2s"].append([int(pi0_k6[i] == 0 and m[i]["k6"] == 1) for i in range(n)])
                dep[arm][sc]["s2f"].append([int(pi0_k6[i] == 1 and m[i]["k6"] == 0) for i in range(n)])
                log(f"  [seed {seed} {arm} {sc:13}] K6 {round(float(np.mean(dep[arm][sc]['k6'][-1])), 3)} "
                    f"sel {round(float(np.mean(dep[arm][sc]['cov_sel'][-1])) if covered else 0.0, 3)} "
                    f"f→s {int(np.sum(dep[arm][sc]['f2s'][-1]))} s→f {int(np.sum(dep[arm][sc]['s2f'][-1]))}")

    # ── aggregate (unconditional net ΔK6 vs pi_0, selection quality on covered states, per stratum) ──
    def net_dk6(k6rows):
        return [[k - pi0_k6[i] for i, k in enumerate(row)] for row in k6rows]
    res = {"RANDOM_VALID": {"K6": _agg(rand["k6"]), "net_dK6": _agg(net_dk6(rand["k6"])),
                            "selection_quality": _agg(rand["cov_sel"]) if covered else None}}
    for arm in arms:
        for sc in LEARNED:
            ci_net, cls = classify_and(net_dk6(dep[arm][sc]["k6"]))
            res[f"{arm}/{sc}"] = {"K6": _agg(dep[arm][sc]["k6"]), "net_dK6": ci_net, "class_net_dK6": cls,
                                  "selection_quality": _agg(dep[arm][sc]["cov_sel"]) if covered else None,
                                  "fail_to_success": _agg(dep[arm][sc]["f2s"], stat=np.sum),
                                  "success_to_fail": _agg(dep[arm][sc]["s2f"], stat=np.sum),
                                  "drift": _agg(dep[arm][sc]["drift"]), "nonpi0_rate": _agg(dep[arm][sc]["nonpi0"])}
    strata_k6 = {}
    for st in sorted(set(strata)):
        idx = [i for i in range(n) if strata[i] == st]
        row = {"n": len(idx), "pi0_K6": round(float(np.mean([pi0_k6[i] for i in idx])), 3)}
        for arm in arms:
            for sc in LEARNED:
                vals = [np.mean([seedrow[i] for i in idx]) for seedrow in dep[arm][sc]["k6"]]
                row[f"{arm}/{sc}_K6"] = round(float(np.mean(vals)), 3)
        strata_k6[st] = row

    verdict, nxt = _decision(res, arms, len(templates), len(covered), covered)
    out = {"contract": "UPSTREAM_HEADROOM_PREFIX_SEARCH_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "method": {"K": K, "mag": MAG, "horizon": H, "strict": list(STRICT_UP), "families": list(FAMS_UP),
                      "eval_unfiltered": True, "controls": ["pi_0", "RANDOM_VALID"], "scorers": list(LEARNED),
                      "bootstrap": "target critic + target actor", "arms": list(arms), "per_state_hierarchical": True},
           "panel": {"n": n, "strata": {k: v for k, v in sorted(Counter(strata).items())}, "all_held_out": True},
           "baseline": {"pi0_K6": round(float(np.mean(pi0_k6)), 3), "pi0_fail": len(pi0fail),
                        "candidate_coverage": len(covered), "coverage_of_fail": round(len(covered) / max(len(pi0fail), 1), 3)},
           "seeds": seeds, "results": res, "strata_K6": strata_k6, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/upstream_headroom_prefix_search_v1.json", "w"), indent=1, default=float)

    log("\n== UPSTREAM_HEADROOM_PREFIX_SEARCH_V1 (strict 1–2, unfiltered eval, coverage/selection/unconditional) ==")
    log(f"  n={n} | pi_0 K6 {round(float(np.mean(pi0_k6)), 3)} | pi_0-fail {len(pi0fail)} | candidate-coverage {len(covered)}")
    for name in ["RANDOM_VALID"] + [f"{a}/{s}" for a in arms for s in LEARNED]:
        r = res[name]; sel = r["selection_quality"]["stat"] if r["selection_quality"] else None
        cls = r.get("class_net_dK6", "")
        log(f"  {name:18}: K6 {r['K6']['stat']} | net ΔK6 {r['net_dK6']['stat']} CI[{r['net_dK6']['lo']},{r['net_dK6']['hi']}] {cls} | sel-quality {sel}")
    log("  strata pi_0 K6: " + " ".join(f"{st}={strata_k6[st]['pi0_K6']}(n{strata_k6[st]['n']})" for st in strata_k6))
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/upstream_headroom_prefix_search_v1.json\nUPSTREAM_HEADROOM_DONE")
    return out


def classify_and(net_rows, equiv=0.03):
    ci = hierarchical_bootstrap_ci(net_rows, stat=np.mean)
    return ci, classify_vs_chance(ci, chance=0.0, equiv=equiv)


def _decision(res, arms, n, n_covered, covered):
    if n < MIN_STATES:
        return "UPSTREAM_UNDERPOWERED_TOO_FEW_STATES", f"only {n} upstream states (<{MIN_STATES}) — widen the scan"
    if n_covered < MIN_COVERED:
        return "NO_CANDIDATE_COVERAGE_GENERATOR_OR_K_LIMIT", \
               f"the bank contains an improving chunk in only {n_covered} pi_0-fail states — not a scorer problem; longer K, a better candidate-generator, or an earlier residual/MPC"

    def val(a):
        return res[f"{a}/value"]
    def rv(a):
        return res[f"{a}/reward_value"]
    def rew(a):
        return res[f"{a}/reward"]
    def net_up(r):
        return (r["net_dK6"]["lo"] or -1) > 0
    def beats_random(r):
        rr = res["RANDOM_VALID"]["net_dK6"]["stat"] or 0
        return (r["net_dK6"]["stat"] or 0) > rr + 0.03
    val_up = all(net_up(val(a)) or net_up(rv(a)) for a in arms)
    rew_up = all(net_up(rew(a)) for a in arms)
    val_intelligent = all(beats_random(val(a)) or beats_random(rv(a)) for a in arms)
    rv_best = all((rv(a)["net_dK6"]["stat"] or -1) > max(val(a)["net_dK6"]["stat"] or -1, rew(a)["net_dK6"]["stat"] or -1) + 0.03 for a in arms)
    sel = all((val(a)["selection_quality"] and (val(a)["selection_quality"]["stat"] or 0) > 0.5) for a in arms) if covered else False

    if val_up and val_intelligent:
        return "VALUE_TO_POLICY_CONVERTER_FOUND", "value converts AND beats random — amortise this search into a chunk actor / option (the search is now proven to work)"
    if rv_best:
        return "REWARD_PLUS_VALUE_SYNERGY", "critic adds prospectively and the signals combine — LEARN the scaling/gating, don't just sum"
    if rew_up and not val_up:
        return "REWARD_GUIDED_SEARCH_IS_THE_OPERATOR", "reward-guided MPC/search is the working improvement operator → distil it into a chunk actor; the value alone is not prospective"
    if not (val_up or rew_up) and sel is False and n_covered >= MIN_COVERED:
        return "COVERAGE_EXISTS_BUT_NO_SCORER_SELECTS_IT_CHUNK_CRITIC_JUSTIFIED", \
               "an improving chunk exists but no scorer finds it → a direct chunk critic Q_K(s, a_0:K-1) or a pairwise prefix-ranker is now directly justified"
    return "UPSTREAM_SEARCH_INCONCLUSIVE", "CIs straddle 0 / arms disagree — add states/seeds or inspect strata"


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
