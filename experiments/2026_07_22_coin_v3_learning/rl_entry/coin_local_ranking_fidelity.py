"""LOCAL_ACTION_RANKING_FIDELITY_V1 (corrected + safeguarded) — is the trained Markov critic's LOCAL action-ranking
physically faithful, IN-DISTRIBUTION, measured the right way?

Method (all review safeguards in):
  * ONE-STEP candidate — apply the perturbed action only at t=0, then the *frozen* pi_0 baseline.
  * critic-gradient direction computed ONCE at s0; Arm-A and Arm-B critic-gradient candidates get their OWN physical
    rollouts (they are critic-specific); only the identical actuator-basis candidates share physics across critics.
  * STATE-WISE ranking fidelity — per-state Spearman(ΔQ, physical) / top-1 physical win / +ΔQ sign-agreement, bootstrap
    CI *over states*, STRATIFIED BY EPSILON (0.005 / 0.01 / 0.02 / 0.04 reported separately — local vs larger-local are
    different questions).
  * PRE-REGISTERED physical target — a fixed lexicographic terminal certificate (max_dwell ≻ fewer target-exits ≻ closer
    containment ≻ slower), NOT a post-hoc weighted scalar. Raw K6/dwell/exit/dtz/speed reported; ρ also reported under two
    predeclared sensitivity scalarizations (pure max_dwell, pure K6).
  * TRAIN-ID vs DEV-ID split — the terminal verdict uses DEV-ID only; TRAIN-ID reported separately. DEV target_entry has
    n=1, so no family-level conclusion is drawn there.
  * evaluated against BOTH Arm-A and Arm-B (terminal-aligned) Markov critics.

No new campaign; one 4000-update critic per arm per seed; deepcopy of once-reconstructed handoff templates.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_action_perturbation import (  # noqa: E402
    actuator_basis_delta,
    bootstrap_ci,
    critic_grad_delta,
    spearman,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    _aug,
    make_late_actor55_from_pi0,
    one_step_candidate_outcome,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
NORMS = [0.005, 0.01, 0.02, 0.04]
H = 30                                                            # rollout horizon (8–30; 30 matches the certifier eval)
ASCALE = 4.0


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


# ── pre-registered physical target: fixed lexicographic terminal certificate (higher = better) ──
def lex_key(o):
    """more dwell ≻ fewer target-exits ≻ closer containment ≻ slower settle. No weights, no tuning."""
    return (int(o["max_dwell"]), -int(o["exit_ct"]), -float(o["final_dtz"]), -float(o["mean_speed"]))


def lex_better(a, b):
    return lex_key(a) > lex_key(b)


def lex_ranks(outcomes):
    """Average ranks by lex_key (higher key → higher rank; identical keys share the average rank)."""
    keys = [lex_key(o) for o in outcomes]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = [0.0] * len(keys); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def actuator_specs(adim):
    return [(f"act{ax}{'+' if sg > 0 else '-'}@{eps}", ax, sg, eps) for ax in range(adim) for sg in (+1, -1) for eps in NORMS]


def roll(templates, pi0, base, i, first_action):
    rl, gate = templates[i]
    return one_step_candidate_outcome(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, first_action, arm="A", horizon=H)


def _dq(critic, o55, a_cand, q0):
    with torch.no_grad():
        return round(float(critic.min_q(o55, a_cand)[0]) - q0, 4)


def state_candidates_by_eps(critic, base, o55, a0, q0, adim, i, act_phys, cgrad_phys):
    """For ONE state and ONE critic, candidates grouped by ε: 8 actuator (shared physics) + 1 critic-gradient (own)."""
    by = {eps: [] for eps in NORMS}; a0v = a0[0]
    for name, ax, sg, eps in actuator_specs(adim):
        a_cand = torch.clamp(a0v + actuator_basis_delta(adim, ax, sg, eps), -ASCALE, ASCALE)[None]
        by[eps].append({"name": name, "dQ": _dq(critic, o55, a_cand, q0), "outcome": act_phys[(i, name)]})
    for eps in NORMS:
        delta = critic_grad_delta(base, critic, eps)(o55)[0]
        a_cand = torch.clamp(a0v + delta, -ASCALE, ASCALE)[None]
        by[eps].append({"name": f"cgrad@{eps}", "dQ": _dq(critic, o55, a_cand, q0), "outcome": cgrad_phys[(i, eps)]})
    return by


def state_fidelity_eps(cands, base_out):
    """Per-state, per-ε fidelity. rho_* are None for degenerate (no-signal) states so the caller can exclude them."""
    outs = [c["outcome"] for c in cands]; dqs = np.array([c["dQ"] for c in cands])
    disc = any(lex_key(o) != lex_key(base_out) for o in outs)          # did any candidate change the physics vs baseline?
    top = cands[int(dqs.argmax())]; pos = [c for c in cands if c["dQ"] > 1e-9]
    return {"rho_lex": spearman(dqs, lex_ranks(outs)),
            "rho_dwell": spearman(dqs, [o["max_dwell"] for o in outs]),
            "rho_k6": spearman(dqs, [o["k6"] for o in outs]),
            "top1_beats": (lex_better(top["outcome"], base_out) if disc else None),
            "sign_agree": (float(np.mean([lex_better(c["outcome"], base_out) for c in pos])) if pos and disc else None),
            "discriminating": disc}


def critic_fidelity(critic, base, templates, obs55, a0, tags, fams_list, adim, act_phys, cgrad_phys, pi0):
    per = []
    for i in range(len(templates)):
        o55 = obs55[i:i + 1]
        with torch.no_grad():
            q0 = float(critic.min_q(o55, a0[i:i + 1])[0])
        base_out = roll(templates, pi0, base, i, a0[i].numpy())
        by = state_candidates_by_eps(critic, base, o55, a0[i:i + 1], q0, adim, i, act_phys, cgrad_phys)
        per.append({"family": fams_list[i], "panel": tags[i], "base_k6": base_out["k6"],
                    "eps": {eps: state_fidelity_eps(by[eps], base_out) for eps in NORMS}})
    return per


def summarize(per_state, panel):
    sub = [s for s in per_state if s["panel"] == panel]
    out = {"n": len(sub)}
    for eps in NORMS:
        disc = [s for s in sub if s["eps"][eps]["discriminating"]]
        out[str(eps)] = {"n_disc": len(disc),
                         "median_rho_lex": bootstrap_ci([s["eps"][eps]["rho_lex"] for s in disc]),
                         "median_rho_dwell": bootstrap_ci([s["eps"][eps]["rho_dwell"] for s in disc]),
                         "median_rho_k6": bootstrap_ci([s["eps"][eps]["rho_k6"] for s in disc]),
                         "top1_beats_rate": bootstrap_ci([s["eps"][eps]["top1_beats"] for s in disc], stat=np.mean),
                         "sign_agree_rate": bootstrap_ci([s["eps"][eps]["sign_agree"] for s in disc], stat=np.mean)}
    return out


def family_dev(per_state, eps):
    fam = {}
    for name in sorted(set(s["family"] for s in per_state if s["panel"] == "dev")):
        d = [s for s in per_state if s["panel"] == "dev" and s["family"] == name and s["eps"][eps]["discriminating"]]
        fam[name] = {"n_disc": len(d), "median_rho_lex": bootstrap_ci([s["eps"][eps]["rho_lex"] for s in d])}
    return fam


def _verdict(dev_A):
    """CI-aware classification on DEV-ID, Arm-A critic, at the LOCAL scale ε=0.005. The faithfulness bar is ρ_lex≥0.5.
    We only call 'faithful' if the CI lower bound clears the bar, and 'not-faithful' only if the CI upper bound is below
    it — otherwise the measurement is underpowered (CI spans the bar) and we say so rather than manufacture a verdict."""
    ci = dev_A["0.005"]["median_rho_lex"]
    top1 = dev_A["0.005"]["top1_beats_rate"]["stat"]; sign = dev_A["0.005"]["sign_agree_rate"]["stat"]
    nxt_unfaithful = "more on-distribution critic data / paired-difference (ranking-loss) critic / a pessimistic critic; " \
                     "and give the local test more physical discrimination (dwell/K6 rarely move under one nudge)"
    if ci["stat"] is None:
        return "NO_LOCAL_PHYSICAL_SIGNAL", "the one-step local test lacks discrimination — widen the candidate step or horizon"
    lo, hi = ci["lo"], ci["hi"]
    if lo is not None and lo >= 0.5:
        return "LOCAL_ACTION_RANKING_PHYSICALLY_FAITHFUL", "matched TD3 & SAC warranted (keep the trust region as a safety wall)"
    if hi is not None and hi < 0.5:
        near0 = hi >= 0.0
        label = "LOCAL_RANKING_NOT_FAITHFUL_NEAR_ZERO" if near0 else "LOCAL_RANKING_ANTICORRELATED"
        return label, nxt_unfaithful
    return "LOCAL_FIDELITY_INCONCLUSIVE_UNDERPOWERED", \
           f"CI spans the 0.5 bar (n small); add seeds/states before concluding. top1>base {top1}, +ΔQ-sign {sign} (~chance=0.5)"


def _plot(seed_results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                          # noqa: BLE001 — optional
        print(f"[plot] skipped ({e})", flush=True); return False
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for arm, col in (("A", "#457b9d"), ("B", "#e76f51")):
        rho = [s["eps"][0.005]["rho_lex"] for s in seed_results[0][arm] if s["panel"] == "dev"
               and s["eps"][0.005]["discriminating"] and s["eps"][0.005]["rho_lex"] is not None]
        ax[0].hist(rho, bins=np.linspace(-1, 1, 13), alpha=0.6, color=col, label=f"critic {arm} (n={len(rho)})")
    ax[0].axvline(0, color="#888", lw=1); ax[0].axvline(0.5, color="#2a9d8f", ls="--", lw=1, label="faithful≥0.5")
    ax[0].set_xlabel("per-state Spearman(ΔQ, lex certificate), DEV ε=0.005"); ax[0].set_ylabel("states")
    ax[0].set_title("state-wise LOCAL action-ranking fidelity (DEV)"); ax[0].legend(fontsize=8)
    x = np.arange(len(NORMS)); w = 0.36
    for j, (arm, col) in enumerate((("A", "#457b9d"), ("B", "#e76f51"))):
        dev = summarize(seed_results[0][arm], "dev")
        vals = [dev[str(e)]["median_rho_lex"]["stat"] or 0.0 for e in NORMS]
        lo = [(dev[str(e)]["median_rho_lex"]["stat"] or 0.0) - (dev[str(e)]["median_rho_lex"]["lo"] or 0.0) for e in NORMS]
        hi = [(dev[str(e)]["median_rho_lex"]["hi"] or 0.0) - (dev[str(e)]["median_rho_lex"]["stat"] or 0.0) for e in NORMS]
        ax[1].bar(x + (j - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3, color=col, label=f"critic {arm}")
    ax[1].axhline(0.5, color="#2a9d8f", ls="--", lw=1); ax[1].axhline(0.0, color="#888", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(e) for e in NORMS]); ax[1].set_xlabel("perturbation norm ε")
    ax[1].set_ylabel("median per-state ρ(ΔQ, lex)"); ax[1].set_ylim(-1, 1)
    ax[1].set_title("ε-stratified fidelity (DEV, bootstrap CI over states)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".svg", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig); return True


def main(seeds=(0, 1), smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]
    fams = tuple(stage["families"]); adim = pi0.action_dim
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    train_p = [ls for ls in tb if ls.family in fams]; dev_p = [ls for ls in db if ls.family in fams]
    panel = train_p + dev_p; tags = ["train"] * len(train_p) + ["dev"] * len(dev_p)
    tcfg = TransactionalConfig(); seeds = list(seeds[:1] if smoke else seeds)
    base = make_late_actor55_from_pi0(pi0, trainable=False)

    templates, obs55, fams_list = [], [], []
    for ls in panel:
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        templates.append((rl, gate)); obs55.append(_aug(rec.obs, rl._strict)); fams_list.append(ls.family)
    obs55 = torch.as_tensor(np.asarray(obs55, np.float32))
    with torch.no_grad():
        a0 = base.action_mean(obs55)
    dev_fam = {f: [t for t, fm in zip(tags, fams_list) if t == "dev" and fm == f].__len__() for f in fams}
    log(f"[panel] train-ID {len(train_p)} dev-ID {len(dev_p)} | dev families {dev_fam} (target_entry n=1: no family claim) H={H} seeds={seeds}")

    log("[physics] rolling 32 actuator one-step candidates (arm-independent, shared)...")
    act_phys = {}
    for i in range(len(templates)):
        for name, ax, sg, eps in actuator_specs(adim):
            fa = torch.clamp(a0[i] + actuator_basis_delta(adim, ax, sg, eps), -ASCALE, ASCALE).numpy()
            act_phys[(i, name)] = roll(templates, pi0, base, i, fa)

    seed_results = []
    for seed in seeds:
        log(f"[seed {seed}] train Arm A + Arm B Markov critics (4000 upd each)...")
        _rA, artA = train_arm(pi0, "A", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        _rB, artB = train_arm(pi0, "B", stage, tb, db, seed=seed, tcfg=tcfg, return_artifacts=True, log=lambda *a: None)
        res = {}
        for arm, art in (("A", artA), ("B", artB)):
            critic = art["critic"]
            cgrad_phys = {}                                            # critic-specific one-step critic-gradient physics
            for i in range(len(templates)):
                for eps in NORMS:
                    delta = critic_grad_delta(base, critic, eps)(obs55[i:i + 1])[0]
                    fa = torch.clamp(a0[i] + delta, -ASCALE, ASCALE).numpy()
                    cgrad_phys[(i, eps)] = roll(templates, pi0, base, i, fa)
            res[arm] = critic_fidelity(critic, base, templates, obs55, a0, tags, fams_list, adim, act_phys, cgrad_phys, pi0)
            dev = summarize(res[arm], "dev")
            log(f"  [seed {seed} {arm}] DEV per-ε median ρ_lex: " +
                " ".join(f"{e}={dev[str(e)]['median_rho_lex']['stat']}(n{dev[str(e)]['n_disc']})" for e in NORMS) +
                f" | ε0.005 top1>base {dev['0.005']['top1_beats_rate']['stat']} sign {dev['0.005']['sign_agree_rate']['stat']}")
        seed_results.append(res)

    dev_A = summarize(seed_results[0]["A"], "dev"); dev_B = summarize(seed_results[0]["B"], "dev")
    train_A = summarize(seed_results[0]["A"], "train")
    verdict, nxt = _verdict(dev_A)
    fig_path = f"{D}/local_ranking_fidelity_v1.svg"; plotted = _plot(seed_results, fig_path)
    out = {"contract": "LOCAL_ACTION_RANKING_FIDELITY_V1", "date": "2026-07-23", "no_new_campaign": True, "smoke": smoke,
           "method": {"one_step_candidate": True, "critic_grad_once_at_s0": True, "cgrad_rollout_per_critic": True,
                      "state_wise": True, "eps_stratified": True, "physical_target": "pre-registered lexicographic: max_dwell ≻ -exit ≻ -dtz ≻ -speed",
                      "sensitivity_scalarizations": ["max_dwell", "k6"], "train_dev_split": True, "verdict_panel": "DEV-ID",
                      "critics": ["A_markov", "B_terminal_aligned"], "rollout_horizon": H},
           "panel": {"train_id": len(train_p), "dev_id": len(dev_p), "dev_families": dev_fam,
                     "dev_target_entry_n": dev_fam.get("target_entry", 0)},
           "seeds": seeds, "dev_A": dev_A, "dev_B": dev_B, "train_A_reference": train_A,
           "dev_A_family_eps0005": family_dev(seed_results[0]["A"], 0.005),
           "all_seeds_dev": [{"A": summarize(s["A"], "dev"), "B": summarize(s["B"], "dev")} for s in seed_results],
           "verdict": verdict, "next_lever": nxt, "figure": fig_path if plotted else None}
    json.dump(out, open(f"{D}/local_ranking_fidelity_v1.json", "w"), indent=1, default=float)

    log("\n== LOCAL_ACTION_RANKING_FIDELITY_V1 (one-step, state-wise, ε-stratified, lexicographic, DEV verdict) ==")
    for arm, dv in (("A", dev_A), ("B", dev_B)):
        log(f"  critic {arm} DEV: " + " | ".join(
            f"ε{e}: ρ_lex {dv[str(e)]['median_rho_lex']['stat']} CI[{dv[str(e)]['median_rho_lex']['lo']},{dv[str(e)]['median_rho_lex']['hi']}]"
            f" top1>base {dv[str(e)]['top1_beats_rate']['stat']}" for e in NORMS))
    log(f"  (sensitivity ε0.005 DEV Arm-A: ρ_dwell {dev_A['0.005']['median_rho_dwell']['stat']} ρ_k6 {dev_A['0.005']['median_rho_k6']['stat']})")
    log(f"  TRAIN-ID ref Arm-A ε0.005 ρ_lex {train_A['0.005']['median_rho_lex']['stat']} (reported separately, not in verdict)")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/local_ranking_fidelity_v1.json{' + ' + fig_path if plotted else ''}\nLOCAL_RANKING_FIDELITY_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
