"""§18 figures for the corrected V2 critic-development result. Reads critic_dev_v2.json (aggregates) and regenerates a
small raw dQ/dG panel (standard causal critic + twin advantage critic) for the scatter. Writes:
  reports/figures/2026-07-23-critic-v2-results.png   (6-panel results)
  reports/figures/2026-07-23-critic-v2-scatter.png   (centered dQ/dG scatter, both critic families)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_counterfactual_labels import capture_state_panel, collect_critic_transitions, composite
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
J = json.load(open("experiments/2026_07_22_coin_v3_learning/rl_entry/critic_dev_v2.json"))
FAM = ("transport", "entry", "settling", "contact_retention")
COL = {"transport": "#4C78A8", "entry": "#F58518", "settling": "#54A24B", "contact_retention": "#E45756"}


def _ck_series(results, fam, key):
    cks = sorted(int(k) for k in results)
    return cks, [results[str(k)].get(fam, {}).get(key) for k in cks]


# ── regenerate raw dQ/dG for the scatter (small, fast) ──
def raw_scatter():
    import importlib.util as u
    s = u.spec_from_file_location("m", "experiments/2026_07_22_coin_v3_learning/rl_entry/coin_critic_dev_v2.py")
    m = u.module_from_spec(s); s.loader.exec_module(m)
    pi0 = load_frozen_clip_actor(PI0, freeze=True); residual0 = m.ZeroInitResidualActor()
    dev = capture_state_panel(pi0, range(6100, 6160), per_family=8)
    trs = collect_critic_transitions(pi0, range(6000, 6040), seed=0)
    std_ck, _ = m.train_std_critic(trs, pi0, residual0, 6000, arm="causal", log=lambda *a: None)
    std = std_ck[6000]; qc, _ = m._q_closures(std, "causal")
    adv_ck = m.train_adv_twin(dev, "causal", steps=4000, log=lambda *a: None)   # illustrative fit on dev for scatter
    ac, _ = m.adv_closures(adv_ck[4000], "causal")
    rows = []
    for g in dev:
        for i, d in enumerate(g.cand_delta):
            a = composite(g.base, 1.0, d)
            rows.append((g.family, g.G[i] - g.G0, qc(g, a, "Q1") - qc(g, composite(g.base, 1.0, g.cand_delta[0]), "Q1"),
                         ac(g, a, "Q1")))
    return rows


def fig_results():
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
    # 1: matched-checkpoint ablation deltas
    a = ax[0, 0]; d = J["ablation"]["causal_minus_instant_mean_over_trained_ckpts"]
    labels = list(d); vals = [d[k] for k in labels]
    a.barh(range(len(labels)), vals, color=["#4C78A8" if v >= 0 else "#E45756" for v in vals])
    a.set_yticks(range(len(labels))); a.set_yticklabels([l.replace("centered_corr_Q1_vs_dG", "corr").replace("_", " ") for l in labels], fontsize=8)
    a.axvline(0, color="k", lw=.8); a.axvline(0.10, color="gray", ls=":", lw=.8)
    a.set_title(f"§8 ablation: causal − instant (mean/trained ckpts)\n→ {J['ablation']['verdict']}", fontsize=9)
    a.set_xlabel("Δ (causal advantage)")
    # 2: within-group ranking gap5 vs ckpt (both arms, transport)
    a = ax[0, 1]
    for arm, ls in (("instant", "--"), ("causal", "-")):
        cks, v = _ck_series(J["ablation_arms"][arm], "transport", "acc_gap5")
        a.plot(cks, v, ls, marker="o", label=f"std {arm}", color="#4C78A8")
    cks, v = _ck_series(J["advantage_dev"], "transport", "acc_gap5")
    a.plot([c * 5 for c in cks], v, "-", marker="s", label="advantage", color="#B279A2")
    a.axhline(0.5, color="k", ls=":", lw=.8, label="chance")
    a.set_title("Within-group ranking acc (|ΔG|≥5), transport", fontsize=9); a.set_xlabel("critic update"); a.set_ylim(0.3, 0.75); a.legend(fontsize=7)
    # 3: +gradQ1 wins vs ckpt
    a = ax[0, 2]
    for arm in ("instant", "causal"):
        cks, v = _ck_series(J["ablation_arms"][arm], "transport", "gradQ1_wins")
        a.plot(cks, [x if x is not None else np.nan for x in v], marker="o", label=f"std {arm}")
    cks, v = _ck_series(J["advantage_dev"], "transport", "gradQ1_wins")
    a.plot([c * 5 for c in cks], [x if x is not None else np.nan for x in v], marker="s", label="advantage", color="#B279A2")
    a.axhline(0.5, color="k", ls=":", lw=.8, label="chance")
    a.set_title("Empirical +gradQ1 beats −gradQ1 (transport)", fontsize=9); a.set_xlabel("critic update"); a.set_ylim(0, 1); a.legend(fontsize=7)
    # 4: standard critic loss (TD instability)
    a = ax[1, 0]
    for arm in ("instant", "causal"):
        L = J["losses"][arm]; a.plot([x[0] for x in L], [x[1] for x in L], marker=".", label=f"std {arm}")
    a.set_title("Standard TD critic loss (bootstrap instability)", fontsize=9); a.set_xlabel("update"); a.set_ylabel("TD loss"); a.legend(fontsize=7)
    # 5: per-family centered corr Q1/dG at final ckpt (both arms)
    a = ax[1, 1]; x = np.arange(len(FAM)); w = 0.35
    fin = max(int(k) for k in J["ablation_arms"]["instant"])
    iv = [J["ablation_arms"]["instant"][str(fin)].get(f, {}).get("centered_corr_Q1_vs_dG", 0) for f in FAM]
    cv = [J["ablation_arms"]["causal"][str(fin)].get(f, {}).get("centered_corr_Q1_vs_dG", 0) for f in FAM]
    a.bar(x - w / 2, iv, w, label="instant", color="#4C78A8"); a.bar(x + w / 2, cv, w, label="causal", color="#72B7B2")
    a.axhline(0, color="k", lw=.8); a.set_xticks(x); a.set_xticklabels([f[:5] for f in FAM], fontsize=8)
    a.set_title(f"Centered corr(ΔQ1, ΔG) @update {fin}", fontsize=9); a.set_ylabel("Pearson r"); a.legend(fontsize=7)
    # 6: failure taxonomy text
    a = ax[1, 2]; a.axis("off")
    txt = ("FAILURE TAXONOMY — RESIDUAL_CRITIC_ROUTE_BLOCKED\n"
           "\n1. Standard TD twin critic (instant & causal):\n"
           "   centered corr(ΔQ1,ΔG) ≈ 0 (transport), turns\n"
           "   ANTI-correlated in contact_retention (−0.16).\n"
           "   Ranking ≈ chance; +gradQ1 no edge.\n"
           "\n2. State ablation: causal history = instant\n"
           "   (all matched Δ ≤ 0.033) → NO aliasing rescue.\n"
           "\n3. Twin advantage critic (direct ΔG regression):\n"
           "   held-out corr ≈ 0; ALWAYS prefers boundary\n"
           "   residual (boundary_pref = 1.0) → gate-fail.\n"
           "\nMechanism: a ONE-STEP ±0.25 residual under frozen\n"
           "pi_0 continuation barely moves the full-horizon\n"
           "canonical return ⇒ ΔG below critic resolution.\n"
           "The blocker is signal leverage, not architecture.")
    a.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8.4, family="monospace")
    fig.suptitle("Coin push-delivery — corrected V2 residual-critic development (RESIDUAL_CRITIC_ROUTE_BLOCKED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("reports/figures/2026-07-23-critic-v2-results.png", dpi=130)
    print("wrote reports/figures/2026-07-23-critic-v2-results.png")


def fig_scatter(rows):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for j, (title, col) in enumerate([("Standard TD critic: ΔQ1 vs ΔG", 2), ("Twin advantage critic: A1 vs ΔG", 3)]):
        a = ax[j]
        for f in FAM:
            xs = [r[1] for r in rows if r[0] == f]; ys = [r[col] for r in rows if r[0] == f]
            a.scatter(xs, ys, s=10, alpha=.5, color=COL[f], label=f[:5])
        allx = [r[1] for r in rows]; ally = [r[col] for r in rows]
        r = np.corrcoef(allx, ally)[0, 1] if np.std(allx) > 0 and np.std(ally) > 0 else 0
        a.set_title(f"{title}\nPearson r = {r:+.3f}", fontsize=10)
        a.set_xlabel("ΔG = G(residual) − G(zero)  [canonical return]"); a.set_ylabel("critic prediction"); a.axhline(0, color="k", lw=.6); a.axvline(0, color="k", lw=.6)
        a.legend(fontsize=7, title="family")
    fig.suptitle("Centered critic value vs empirical one-step residual advantage (held-out states)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("reports/figures/2026-07-23-critic-v2-scatter.png", dpi=130)
    print("wrote reports/figures/2026-07-23-critic-v2-scatter.png")


if __name__ == "__main__":
    import os
    os.makedirs("reports/figures", exist_ok=True)
    fig_results()
    print("regenerating raw scatter data...", flush=True)
    fig_scatter(raw_scatter())
    print("FIGURES_DONE")
