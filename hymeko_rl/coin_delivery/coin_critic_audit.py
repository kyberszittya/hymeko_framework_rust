"""ACTOR-RELEVANT critic development audit (§10) — shared by the instantaneous and causal arms and by the standard and
advantage critics. All metrics are CENTERED per state group (dQ vs dG relative to the zero residual) and RANKED only
WITHIN a group (never across unrelated physical states). The load-bearing test is empirical: does a tiny step along
``+grad Q1`` beat a step along ``-grad Q1`` in canonical continuation return, more often than chance.

``q_of(group, action, head)`` returns Q1/Q2/min for a candidate; ``grad_of(group)`` returns dQ1/d(action) at the base.
Both are closures the caller builds for the specific critic + representation, so this module is critic-agnostic.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_counterfactual_labels import (
    RESIDUAL_BOUND,
    composite,
    counterfactual_return,
)

FAMILIES = ("transport", "entry", "settling", "contact_retention")


def _bootstrap_ci(vals, stat=np.mean, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _group_arrays(g, q_of):
    G = np.asarray(g.G, float); dg = G - g.G0
    q1 = np.array([q_of(g, a, "Q1") for a in _cand_actions(g)])
    q2 = np.array([q_of(g, a, "Q2") for a in _cand_actions(g)])
    return G, dg, q1, q1 - q1[0], q2, q2 - q2[0]


def _cand_actions(g):
    return [composite(g.base, 1.0, d) for d in g.cand_delta]


def _within_pair_acc(q, G, thr):
    ok = tot = 0
    m = len(G)
    for i in range(m):
        for j in range(i + 1, m):
            if abs(G[i] - G[j]) < thr or G[i] == G[j]:
                continue
            tot += 1; ok += int((q[i] > q[j]) == (G[i] > G[j]))
    return ok, tot


def _grad_win(g, grad_of, rl, pi0):
    grad = grad_of(g)
    if grad is None or np.linalg.norm(grad) < 1e-9:
        return None
    u = grad / np.linalg.norm(grad); eps = 0.1 * RESIDUAL_BOUND
    ap = composite(g.base, 1.0, eps * u); am = composite(g.base, 1.0, -eps * u)
    gp, _ = counterfactual_return(rl, pi0, g.snap, ap)
    gm, _ = counterfactual_return(rl, pi0, g.snap, am)
    if gp == gm:
        return None
    return int(gp > gm)


def audit_family(groups, q_of, grad_of, rl, pi0, *, run_grad=True):
    """Per-family actor-relevant metrics with bootstrap CIs by state group."""
    res = {}
    for fam in FAMILIES:
        gs = [g for g in groups if g.family == fam]
        if not gs:
            res[fam] = {"n": 0}; continue
        dq1_pool, dg_pool, dq2_pool = [], [], []
        allp = [0, 0]; gap = {1: [0, 0], 5: [0, 0], 10: [0, 0]}; oq = [0, 0]
        harmful_rej, top1_regret, sel_worse0, sel_break, boundary_pref = [], [], [], [], []
        grad_wins = []
        for g in gs:
            G, dg, q1, dq1, q2, dq2 = _group_arrays(g, q_of)
            dq1_pool += dq1.tolist(); dg_pool += dg.tolist(); dq2_pool += dq2.tolist()
            a, t = _within_pair_acc(q1, G, 0.0); allp[0] += a; allp[1] += t
            for thr in (1, 5, 10):
                a, t = _within_pair_acc(q1, G, thr); gap[thr][0] += a; gap[thr][1] += t
            ao, to = _within_pair_acc(q2, q1, 0.0)  # Q1/Q2 ordering agreement uses q1 as "truth"
            oq[0] += ao; oq[1] += to
            sel = int(np.argmax(q1))
            harmful_rej.append(int(G[sel] > np.percentile(G, 25)))
            top1_regret.append(float(np.max(G) - G[sel]))
            sel_worse0.append(int(G[sel] < g.G0))
            sel_break.append(int(not g.outcomes[sel]["contact_persist"]))
            boundary_pref.append(int(g.cand_meta[sel]["magnitude"] >= 0.25))
            if run_grad:
                w = _grad_win(g, grad_of, rl, pi0)
                if w is not None:
                    grad_wins.append(w)
        corr1 = float(np.corrcoef(dq1_pool, dg_pool)[0, 1]) if np.std(dq1_pool) > 1e-9 and np.std(dg_pool) > 1e-9 else 0.0
        corr2 = float(np.corrcoef(dq2_pool, dg_pool)[0, 1]) if np.std(dq2_pool) > 1e-9 and np.std(dg_pool) > 1e-9 else 0.0
        res[fam] = {
            "n": len(gs),
            "centered_corr_Q1_vs_dG": round(corr1, 3), "centered_corr_Q2_vs_dG": round(corr2, 3),
            "allpair_acc": round(allp[0] / max(allp[1], 1), 3),
            "acc_gap1": round(gap[1][0] / max(gap[1][1], 1), 3),
            "acc_gap5": round(gap[5][0] / max(gap[5][1], 1), 3),
            "acc_gap10": round(gap[10][0] / max(gap[10][1], 1), 3),
            "q1q2_order_agree": round(oq[0] / max(oq[1], 1), 3),
            "harmful_rej": round(float(np.mean(harmful_rej)), 3),
            "harmful_rej_ci": [round(x, 3) for x in _bootstrap_ci(harmful_rej)],
            "top1_regret": round(float(np.mean(top1_regret)), 3),
            "prob_sel_worse_than_zero": round(float(np.mean(sel_worse0)), 3),
            "prob_sel_breaks_contact": round(float(np.mean(sel_break)), 3),
            "boundary_pref": round(float(np.mean(boundary_pref)), 3),
            "gradQ1_wins": round(float(np.mean(grad_wins)), 3) if grad_wins else None,
            "gradQ1_wins_ci": [round(x, 3) for x in _bootstrap_ci(grad_wins)] if grad_wins else None,
            "grad_n": len(grad_wins),
        }
    return res
