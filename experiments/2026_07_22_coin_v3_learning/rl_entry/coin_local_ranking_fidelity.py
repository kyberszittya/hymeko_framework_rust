"""LOCAL_ACTION_RANKING_FIDELITY_V2 — POWERED, held-out, boundary-conditioned local action-ranking test.

V1 measured the local one-step ranking correctly but was underpowered (DEV n=12, arbitrary ε, coarse certifier barely
moved). V2 adds discrimination WITHOUT introducing a new measurement error:
  * PANEL — a large, exclusively HELD-OUT in-distribution panel (seeds disjoint from both critic-train 6000–6088 and the
    old dev 6100–6148), biased to strict∈{3,4,5} boundary states (mid-dwell, at the CENTER_TOL/SETTLE_VEL edge) where a
    single action change CAN flip the certifier — so K6/dwell can actually move.
  * ε FROM HISTORY — perturbation norms are the empirical accepted transactional actor-update action-drift {p50, p90, p99}
    plus the trust cap step_max (largest safe single-action deviation), not arbitrary values.
  * PRIMARY CERTIFICATE reordered per review: K6 ≻ max_dwell ≻ true full-containment exit (dtz≤CENTER_TOL) ≻ speed ≻ dtz
    (dtz only the last tiebreaker). Sensitivity under pure K6 / pure max_dwell.
  * STILL a single candidate action at t=0 then frozen pi_0 (deepcopy-verified); STILL state-wise, ε-stratified.
  * ≥3 seeds; HIERARCHICAL bootstrap over seeds AND states (no single seed drives the verdict).
  * both Arm-A and Arm-B critics.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_action_perturbation import (  # noqa: E402
    actuator_basis_delta,
    critic_grad_delta,
    eps_from_drifts,
    hierarchical_bootstrap_ci,
    spearman,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart, _sha, reconstruct_handoff, replay_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    _aug,
    make_late_actor55_from_pi0,
    one_step_candidate_outcome,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
H = 30
ASCALE = 4.0
FAMS = ("target_entry", "braking", "settling_dwell")


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


# ── pre-registered physical certificate (higher = better): K6 ≻ dwell ≻ fewer full-containment exits ≻ slower ≻ closer ──
def lex_key(o):
    return (int(o["k6"]), int(o["max_dwell"]), -int(o["contain_exit_ct"]), -float(o["mean_speed"]), -float(o["final_dtz"]))


def lex_better(a, b):
    return lex_key(a) > lex_key(b)


def lex_ranks(outcomes):
    keys = [lex_key(o) for o in outcomes]; order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = [0.0] * len(keys); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def build_boundary_panel(pi0, seeds, held_out_forbidden, *, want, per_seed_cap=3, strict_primary=(3, 4, 5), strict_fill=(2,)):
    """Scan HELD-OUT seeds; collect gate-active, in-family handoffs at strict∈strict_primary (boundary, mid-dwell);
    top up with strict_fill if short. Returns (list[LateStart], composition, strict_hist). Asserts held-out disjointness."""
    picked, comp, strict_hist = [], Counter(), Counter()
    for pool in (strict_primary, strict_fill):
        for s in seeds:
            if len(picked) >= want:
                break
            assert s not in held_out_forbidden, f"seed {s} is a critic-train/dev seed — not held out"
            taken = 0
            for rec in replay_pi0(pi0, int(s), horizon=360):
                if taken >= per_seed_cap or len(picked) >= want:
                    break
                if rec.gate_mult != 1.0 or rec.family not in FAMS or rec.strict not in pool:
                    continue
                picked.append(LateStart(seed=int(s), prefix_steps=rec.step, family=rec.family, obs_sha=_sha(rec.obs),
                                        base_sha=_sha(rec.base), causal_sha=_sha(rec.causal_state), gate_state=rec.gate_state))
                comp[rec.family] += 1; strict_hist[rec.strict] += 1; taken += 1
        if len(picked) >= want:
            break
    return picked, dict(comp), dict(sorted(strict_hist.items()))


def accepted_drift_epsilons(pi0, stage, tb, db, tcfg, *, seed=0):
    """ε from the EMPIRICAL accepted transactional actor-update action-drift (one instrumented training)."""
    sink = []
    train_arm(pi0, "A", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, accepted_sink=sink, log=lambda *a: None)
    return eps_from_drifts([r["step_p95"] for r in sink], tcfg.step_max)


def actuator_specs(adim, epsilons):
    return [(f"act{ax}{'+' if sg > 0 else '-'}@{eps}", ax, sg, eps) for ax in range(adim) for sg in (+1, -1) for eps in epsilons]


def roll(templates, pi0, base, i, first_action):
    rl, gate = templates[i]
    return one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, first_action, arm="A", horizon=H)


def _dq(critic, o55, a_cand, q0):
    with torch.no_grad():
        return round(float(critic.min_q(o55, a_cand)[0]) - q0, 4)


def state_candidates_by_eps(critic, base, o55, a0, q0, adim, i, epsilons, act_phys, cgrad_phys):
    by = {eps: [] for eps in epsilons}; a0v = a0[0]
    for name, ax, sg, eps in actuator_specs(adim, epsilons):
        a_cand = torch.clamp(a0v + actuator_basis_delta(adim, ax, sg, eps), -ASCALE, ASCALE)[None]
        by[eps].append({"dQ": _dq(critic, o55, a_cand, q0), "outcome": act_phys[(i, name)]})
    for eps in epsilons:
        delta = critic_grad_delta(base, critic, eps)(o55)[0]
        a_cand = torch.clamp(a0v + delta, -ASCALE, ASCALE)[None]
        by[eps].append({"dQ": _dq(critic, o55, a_cand, q0), "outcome": cgrad_phys[(i, eps)]})
    return by


def state_fidelity_eps(cands, base_out):
    outs = [c["outcome"] for c in cands]; dqs = np.array([c["dQ"] for c in cands])
    disc = any(lex_key(o) != lex_key(base_out) for o in outs)
    dwell_disc = len(set(o["max_dwell"] for o in outs) | {base_out["max_dwell"]}) > 1
    k6_disc = len(set(o["k6"] for o in outs) | {base_out["k6"]}) > 1
    top = cands[int(dqs.argmax())]; pos = [c for c in cands if c["dQ"] > 1e-9]
    return {"rho_lex": spearman(dqs, lex_ranks(outs)),
            "rho_dwell": spearman(dqs, [o["max_dwell"] for o in outs]),
            "rho_k6": spearman(dqs, [o["k6"] for o in outs]),
            "top1_beats": (lex_better(top["outcome"], base_out) if disc else None),
            "sign_agree": (float(np.mean([lex_better(c["outcome"], base_out) for c in pos])) if pos and disc else None),
            "discriminating": disc, "dwell_moves": dwell_disc, "k6_moves": k6_disc}


def critic_fidelity(critic, base, templates, obs55, a0, fams_list, adim, epsilons, act_phys, cgrad_phys, pi0):
    per = []
    for i in range(len(templates)):
        o55 = obs55[i:i + 1]
        with torch.no_grad():
            q0 = float(critic.min_q(o55, a0[i:i + 1])[0])
        base_out = roll(templates, pi0, base, i, a0[i].numpy())
        by = state_candidates_by_eps(critic, base, o55, a0[i:i + 1], q0, adim, i, epsilons, act_phys, cgrad_phys)
        per.append({"family": fams_list[i], "eps": {eps: state_fidelity_eps(by[eps], base_out) for eps in epsilons}})
    return per


def aggregate(seed_perstate, epsilons, key):
    """Hierarchical bootstrap over seeds AND states, per ε. seed_perstate = list (per seed) of per-state dicts."""
    out = {}
    for eps in epsilons:
        per_seed = [[s["eps"][eps][key] for s in per if s["eps"][eps]["discriminating"]] for per in seed_perstate]
        out[str(eps)] = hierarchical_bootstrap_ci(per_seed, stat=(np.mean if key in ("top1_beats", "sign_agree") else np.median))
    return out


def discrimination_report(seed_perstate, epsilons):
    out = {}
    for eps in epsilons:
        flat = [s["eps"][eps] for per in seed_perstate for s in per]
        n = len(flat)
        out[str(eps)] = {"disc_frac": round(np.mean([f["discriminating"] for f in flat]), 3),
                         "dwell_moves_frac": round(np.mean([f["dwell_moves"] for f in flat]), 3),
                         "k6_moves_frac": round(np.mean([f["k6_moves"] for f in flat]), 3), "n": n}
    return out


def _verdict(agg_rho_A, sign_A, disc_A, eps_local):
    """Gate on PRIMARY-certifier discrimination first: if a single local action never moves K6/dwell, the probe cannot
    test the critic's K6-ranking, and near-zero ρ on the fine (terminal-irrelevant) tiebreakers is NOT a critic defect —
    a K6-faithful critic should be indifferent to action changes that do not change the terminal outcome."""
    d = disc_A[str(eps_local)]
    if d["k6_moves_frac"] < 0.05 and d["dwell_moves_frac"] < 0.05:
        return "PRIMARY_CERTIFIER_UNMOVED_LOCAL_TEST_INCONCLUSIVE_ON_K6", \
               ("a single accepted-drift-scale action does not move K6/dwell (the certificate discriminates only via "
                "terminal-IRRELEVANT speed/dtz/exit tiebreakers); ρ≈0 there is correct indifference, NOT infidelity. "
                "Before concluding on fidelity or building a ranking critic, need a probe where candidate actions "
                "actually change K6/dwell — e.g. a bounded multi-step candidate window, which is a different test.")
    ci = agg_rho_A[str(eps_local)]
    if ci["stat"] is None:
        return "NO_LOCAL_PHYSICAL_SIGNAL", "even the boundary panel gives no local discrimination — widen the window"
    lo, hi, s = ci["lo"], ci["hi"], (sign_A[str(eps_local)]["stat"] or 0.0)
    if lo is not None and lo >= 0.5 and s >= 0.5:
        return "LOCAL_ACTION_RANKING_PHYSICALLY_FAITHFUL", "matched TD3 & SAC warranted (keep the trust region as a safety wall)"
    if hi is not None and hi < 0.5:
        return "STRONG_LOCAL_RANKING_FIDELITY_NOT_DEMONSTRATED", \
               "critic ranking below the strong bar even powered → PAIRED_LOCAL_ACTION_RANKING_CRITIC_V1 (learn a_i≻a_j from short rollouts); pessimism as a 2nd arm; keep trust region"
    return "LOCAL_RANKING_SIGNAL_WEAK_AND_UNDERPOWERED", "CI still spans the 0.5 bar — add seeds/states before concluding"


def _plot(seed_perstate_A, seed_perstate_B, epsilons, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                          # noqa: BLE001
        print(f"[plot] skipped ({e})", flush=True); return False
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    e0 = epsilons[0]
    for lab, sp, col in (("A", seed_perstate_A, "#457b9d"), ("B", seed_perstate_B, "#e76f51")):
        rho = [s["eps"][e0]["rho_lex"] for per in sp for s in per if s["eps"][e0]["discriminating"] and s["eps"][e0]["rho_lex"] is not None]
        ax[0].hist(rho, bins=np.linspace(-1, 1, 15), alpha=0.6, color=col, label=f"critic {lab} (n={len(rho)})")
    ax[0].axvline(0, color="#888", lw=1); ax[0].axvline(0.5, color="#2a9d8f", ls="--", lw=1, label="faithful≥0.5")
    ax[0].set_xlabel(f"per-state Spearman(ΔQ, K6-lex certificate), ε={e0} (smallest accepted drift)")
    ax[0].set_ylabel("states"); ax[0].set_title("powered held-out boundary panel — local ranking fidelity"); ax[0].legend(fontsize=8)
    x = np.arange(len(epsilons)); w = 0.36
    for j, (lab, sp, col) in enumerate((("A", seed_perstate_A, "#457b9d"), ("B", seed_perstate_B, "#e76f51"))):
        ag = aggregate(sp, epsilons, "rho_lex")
        vals = [ag[str(e)]["stat"] or 0.0 for e in epsilons]
        lo = [(ag[str(e)]["stat"] or 0.0) - (ag[str(e)]["lo"] or 0.0) for e in epsilons]
        hi = [(ag[str(e)]["hi"] or 0.0) - (ag[str(e)]["stat"] or 0.0) for e in epsilons]
        ax[1].bar(x + (j - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3, color=col, label=f"critic {lab}")
    ax[1].axhline(0.5, color="#2a9d8f", ls="--", lw=1); ax[1].axhline(0.0, color="#888", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(e) for e in epsilons], fontsize=8); ax[1].set_xlabel("ε (empirical accepted actor-drift)")
    ax[1].set_ylabel("hierarchical median ρ(ΔQ, lex)"); ax[1].set_ylim(-1, 1)
    ax[1].set_title("ε-stratified, hierarchical bootstrap (seeds×states)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".svg", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig); return True


def main(seeds=(0, 1, 2), smoke=False, want=72):
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
    want = 24 if smoke else want
    base = make_late_actor55_from_pi0(pi0, trainable=False)

    log(f"[panel] scanning held-out seeds (≥6200, disjoint from {len(forbidden)} train/dev seeds) for strict-boundary states...")
    panel, comp, strict_hist = build_boundary_panel(pi0, range(6200, 7200), forbidden, want=want)
    fams_list = [ls.family for ls in panel]
    log(f"[panel] n={len(panel)} families={comp} strict_hist={strict_hist}")

    log("[ε] deriving perturbation norms from empirical accepted actor-drift...")
    epsilons, eps_info = accepted_drift_epsilons(pi0, stage, tb, db, tcfg)
    log(f"[ε] {epsilons}  ({eps_info})")

    templates, obs55 = [], []
    for ls in panel:
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        templates.append((rl, gate)); obs55.append(_aug(rec.obs, rl._strict))
    obs55 = torch.as_tensor(np.asarray(obs55, np.float32))
    with torch.no_grad():
        a0 = base.action_mean(obs55)

    log(f"[physics] {len(epsilons) * adim * 2} actuator one-step candidates × {len(panel)} states (shared)...")
    act_phys = {}
    for i in range(len(templates)):
        for name, ax, sg, eps in actuator_specs(adim, epsilons):
            fa = torch.clamp(a0[i] + actuator_basis_delta(adim, ax, sg, eps), -ASCALE, ASCALE).numpy()
            act_phys[(i, name)] = roll(templates, pi0, base, i, fa)

    per_A, per_B = [], []
    for seed in seeds:
        log(f"[seed {seed}] train Arm A + Arm B Markov critics...")
        _rA, artA = train_arm(pi0, "A", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        _rB, artB = train_arm(pi0, "B", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        for art, sink in ((artA, per_A), (artB, per_B)):
            critic = art["critic"]; cgrad_phys = {}
            for i in range(len(templates)):
                for eps in epsilons:
                    delta = critic_grad_delta(base, critic, eps)(obs55[i:i + 1])[0]
                    cgrad_phys[(i, eps)] = roll(templates, pi0, base, i, torch.clamp(a0[i] + delta, -ASCALE, ASCALE).numpy())
            sink.append(critic_fidelity(critic, base, templates, obs55, a0, fams_list, adim, epsilons, act_phys, cgrad_phys, pi0))
        ag = aggregate([per_A[-1]], epsilons, "rho_lex")
        log(f"  [seed {seed} A] per-ε median ρ_lex: " + " ".join(f"{e}={ag[str(e)]['stat']}" for e in epsilons))

    rho_A = aggregate(per_A, epsilons, "rho_lex"); rho_B = aggregate(per_B, epsilons, "rho_lex")
    sign_A = aggregate(per_A, epsilons, "sign_agree"); sign_B = aggregate(per_B, epsilons, "sign_agree")
    top_A = aggregate(per_A, epsilons, "top1_beats")
    disc_A = discrimination_report(per_A, epsilons)
    verdict, nxt = _verdict(rho_A, sign_A, disc_A, epsilons[0])
    fig_path = f"{D}/local_ranking_fidelity_v2.svg"; plotted = _plot(per_A, per_B, epsilons, fig_path)
    out = {"contract": "LOCAL_ACTION_RANKING_FIDELITY_V2", "date": "2026-07-23", "no_new_campaign": True, "smoke": smoke,
           "method": {"one_step_candidate": True, "held_out_boundary_panel": True, "eps_from_accepted_drift": eps_info,
                      "certificate": "lexicographic K6 ≻ max_dwell ≻ -contain_exit(CENTER_TOL) ≻ -speed ≻ -dtz",
                      "hierarchical_bootstrap_seeds_x_states": True, "critics": ["A_markov", "B_terminal_aligned"], "horizon": H},
           "panel": {"n": len(panel), "families": comp, "strict_hist": strict_hist, "all_held_out": True},
           "epsilons": epsilons, "seeds": seeds,
           "discrimination_A": disc_A,
           "rho_lex_A": rho_A, "rho_lex_B": rho_B, "sign_agree_A": sign_A, "sign_agree_B": sign_B, "top1_beats_A": top_A,
           "sensitivity_A_eps0": {"rho_dwell": aggregate(per_A, epsilons, "rho_dwell")[str(epsilons[0])],
                                  "rho_k6": aggregate(per_A, epsilons, "rho_k6")[str(epsilons[0])]},
           "verdict": verdict, "next_lever": nxt, "figure": fig_path if plotted else None}
    json.dump(out, open(f"{D}/local_ranking_fidelity_v2.json", "w"), indent=1, default=float)

    log("\n== LOCAL_ACTION_RANKING_FIDELITY_V2 (powered held-out boundary; hierarchical seeds×states) ==")
    log(f"  panel n={len(panel)} strict={strict_hist} families={comp} | ε={epsilons}")
    log(f"  discrimination @ε{epsilons[0]}: any {disc_A[str(epsilons[0])]['disc_frac']} | dwell-moves {disc_A[str(epsilons[0])]['dwell_moves_frac']} | K6-moves {disc_A[str(epsilons[0])]['k6_moves_frac']}")
    for lab, rho, sign in (("A", rho_A, sign_A), ("B", rho_B, sign_B)):
        log(f"  critic {lab}: " + " | ".join(f"ε{e}: ρ_lex {rho[str(e)]['stat']} CI[{rho[str(e)]['lo']},{rho[str(e)]['hi']}]" for e in epsilons))
    log(f"  Arm-A sensitivity @ε{epsilons[0]}: ρ_dwell {out['sensitivity_A_eps0']['rho_dwell']['stat']} ρ_k6 {out['sensitivity_A_eps0']['rho_k6']['stat']} | top1>base {top_A[str(epsilons[0])]['stat']} sign {sign_A[str(epsilons[0])]['stat']}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/local_ranking_fidelity_v2.json{' + ' + fig_path if plotted else ''}\nLOCAL_RANKING_FIDELITY_V2_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
