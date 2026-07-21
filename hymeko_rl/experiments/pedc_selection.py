"""PED-C — Clean-Observation Proposal Selection (post-R47 obs fix).

The corrected §12 decomposition froze O_planner 0.812 / O_target 0.688 / O_decoded 0.688 / O_selected 0.500: proposal
representation, memorization and held generalization all pass, and the primary remaining gap is SELECTION
(Δ_selection = O_decoded − O_selected = 0.188). This module rebuilds the selector datasets from clean, history-
independent observations with the identity contract (`arcrl_corpus`), and first answers the reproducibility gate
(SEL0 vs SEL1) before any objective engineering — per the standing rule "do not call selection the binding bottleneck
unless the gap remains reproducible."

Design:
  * The COUNTERFACTUAL OUTCOME MATRIX (§1) is the reusable artifact: for each state, execute EVERY decoded proposal head
    (guarded by the state identity) and store a multi-dimensional outcome (stable two-finger recovery is the primary
    label, NOT re-touch). Every selector then evaluates by LOOKUP against this matrix — no re-execution, so ranking
    diagnostics (§8) and variance decomposition (§2) are cheap and exact.
  * Selectors score each proposal INDEPENDENTLY (`_PlanScorer`), so they are permutation-invariant in head order — the
    property the head-index classifier (`_train_selector`) lacked. A regression test pins this.
  * Split by immutable physical-state seed (§3): decoder-train, selector-train and held-eval seed ranges are disjoint.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from hymeko_rl.env.contact_formation_env import ContactFormationEnv
from hymeko_rl.experiments.arcrl_corpus import assert_oracle_ordering, oracle_decomposition
from hymeko_rl.experiments.exp_coin_toss_arcrl import (
    _NLRP_H,
    _NLRP_K,
    _OUT,
    _PlanScorer,
    _build_plan_corpus_cached,
    _C1_HORIZON,
    _ctx,
    _expand,
    _load_pkl_bank,
    _pm,
    _privileged,
    _reset_checked,
    _roll_env,
    _shoot_ref_held,
    _train_plan_actor,
    c1_config,
)


# ── §1 multi-dimensional counterfactual outcome ─────────────────────────────────────────────────────────────────
def proposal_outcome(env, seed: int, plan: np.ndarray, seg_len: int, lo, hi) -> dict:
    """Execute one decoded proposal (open-loop, exact reset) and record the multi-dimensional outcome. Primary success
    is STABLE two-finger recovery (handoff-ready), NOT re-touch. `reached` (momentary both-contact) is recorded
    separately and is explicitly not the success label."""
    env.reset(seed=int(seed)); actions = _expand(plan, seg_len)
    stable = harm = reached = False; rt = len(actions); persist = 0; m = None
    for t, a in enumerate(actions):
        _o, _r, term, trunc, info = env.step(np.clip(a, lo, hi).astype(np.float32))
        m = _pm(env); both = bool(m.left_contact) and bool(m.right_contact)
        reached = reached or both; persist += int(both)
        harm = harm or bool(info.get("safety_violation"))
        if info.get("handoff_ready"):
            stable = True; rt = t; break
        if term or trunc:
            break
    slip = float(np.hypot(float(m.disk_vel[0]), float(m.disk_vel[1]))) if m is not None else 0.0
    imbal = abs(float(m.left_tip_dist) - float(m.right_tip_dist)) if m is not None else 0.0
    arm = float(m.arm_speed) if m is not None else 0.0
    return {"stable": bool(stable), "reached": bool(reached), "harm": bool(harm), "persistence": int(persist),
            "rt": int(rt), "slip": round(slip, 4), "imbalance": round(imbal, 4), "arm": round(arm, 4),
            "controllable": bool(arm < 1.5 and not harm)}


# ── §1 counterfactual outcome matrix + deployable per-proposal features (§5) ─────────────────────────────────────
def _proposal_features(obs: np.ndarray, plan: np.ndarray) -> np.ndarray:
    """Deployable per-proposal summary (§5): the flattened proposal + gentle-ness / spread statistics. Uses NO
    privileged, outcome, planner, or seed information."""
    flat = plan.flatten()
    return np.concatenate([flat, [float(np.linalg.norm(flat)), float(np.abs(flat).max()), float(flat.std())]]).astype(np.float32)


def build_state_records(decoder, seeds, bank_path: str, env, seg_len, lo, hi, role: str) -> list:
    """For each seed: reset (guarded), read the decoder's M proposals, execute every one, and store the state's obs,
    privileged vector, proposals, per-proposal deployable features, and the full counterfactual outcome row Y[state]."""
    from hymeko_rl.experiments.arcrl_corpus import StateID, sim_hash
    recs = []
    for sd in seeds:
        obs, _i = env.reset(seed=int(sd)); priv = _privileged(env).astype(np.float32)
        sid = StateID(seed=int(sd), sim_hash=sim_hash(obs), run_id=f"{role}:pedc", role=role,
                      contact_regime=int(round(float(priv[0]))) + int(round(float(priv[1]))),
                      drift_depth=round(float((priv[3] + priv[4]) / 2.0), 4))
        with torch.no_grad():
            plans, logit = decoder(torch.as_tensor(obs[None].astype(np.float32)))
        pl = plans[0].numpy(); m = pl.shape[0]
        _reset_checked(env, sid)                                         # §0 boundary guard before executing this state
        outs = [proposal_outcome(env, sd, pl[h], seg_len, lo, hi) for h in range(m)]
        feats = np.stack([_proposal_features(obs, pl[h]) for h in range(m)])
        recs.append({"sid": sid, "obs": obs.astype(np.float32), "priv": priv, "plans": pl, "feats": feats,
                     "logit": logit[0].numpy(), "outcomes": outs, "Y_stable": np.array([int(o["stable"]) for o in outs])})
    return recs


def record_stats(recs: list) -> dict:
    """Per-state selection structure (§1): success fraction, all-fail rate, multi-success rate, mean margin."""
    ns = [int(r["Y_stable"].sum()) for r in recs]
    return {"n_states": len(recs), "mean_proposals": round(float(np.mean([len(r["Y_stable"]) for r in recs])), 2),
            "mean_successes": round(float(np.mean(ns)), 2), "all_fail_frac": round(float(np.mean([n == 0 for n in ns])), 3),
            "multi_success_frac": round(float(np.mean([n >= 2 for n in ns])), 3),
            "oracle_best_of_m": round(float(np.mean([n >= 1 for n in ns])), 3)}


# ── selectors (§4 baselines + §6 C0) evaluated by LOOKUP against the matrix ──────────────────────────────────────
def _select_success(recs: list, choose) -> dict:
    """Given a per-state head-choice function `choose(rec) -> head_index`, the selected-success rate + ranking
    diagnostics (§8): top-1 hit, MRR of the first success under the induced order, regret vs oracle."""
    sel = 0; regret = []
    for r in recs:
        Y = r["Y_stable"]; h = int(choose(r)); sel += int(Y[h])
        regret.append(int(Y.max()) - int(Y[h]))
    return {"selected": round(sel / len(recs), 3), "mean_regret": round(float(np.mean(regret)), 3)}


def _rank_diagnostics(recs: list, score_of) -> dict:
    """§8 listwise diagnostics from a per-proposal score function `score_of(rec) -> np.ndarray(m,)`: top-1/2/3 hit,
    MRR of the first successful proposal under the ranking, regret."""
    t1 = t2 = t3 = 0; rr = []; regret = []
    for r in recs:
        Y = r["Y_stable"]; s = np.asarray(score_of(r)); order = np.argsort(-s)
        ranked = Y[order]
        t1 += int(ranked[:1].any()); t2 += int(ranked[:2].any()); t3 += int(ranked[:3].any())
        hit = np.where(ranked == 1)[0]
        rr.append(1.0 / (hit[0] + 1) if hit.size else 0.0)
        regret.append(int(Y.max()) - int(Y[order[0]]))
    n = len(recs)
    return {"top1": round(t1 / n, 3), "top2": round(t2 / n, 3), "top3": round(t3 / n, 3),
            "mrr": round(float(np.mean(rr)), 3), "mean_regret": round(float(np.mean(regret)), 3)}


def heuristic_choose(rec) -> int:
    """§4 S2: a declared deterministic deployable rule — pick the GENTLEST proposal (smallest action L2 norm). Uses only
    the proposal tensor (deployable), no outcome/privileged/planner info."""
    return int(np.argmin([float(np.linalg.norm(rec["plans"][h])) for h in range(len(rec["Y_stable"]))]))


def random_expected(recs: list) -> float:
    """§4 S0: expected selected-success of a uniform random choice = mean success fraction per state."""
    return round(float(np.mean([r["Y_stable"].mean() for r in recs])), 3)


# ── §6 C0 pointwise scorer (clean per-proposal BCE + within-state pairwise ranking), permutation-invariant ───────
def train_scorer_c0(train_recs: list, ctx_key: str, seed: int = 0, epochs: int = 500) -> "_PlanScorer":
    """C0: independent per-proposal stable-recovery probability, obs- (S4) or privileged- (S5/P-SEL) conditioned. Adds a
    within-state pairwise ranking term (§8 needs correct WITHIN-state order, not just global AUROC)."""
    torch.manual_seed(seed)
    ctx = torch.as_tensor(np.stack([r[ctx_key] for r in train_recs for _ in range(len(r["Y_stable"]))]))
    plan = torch.as_tensor(np.stack([r["feats"][h] for r in train_recs for h in range(len(r["Y_stable"]))]))
    y = torch.as_tensor(np.array([int(r["outcomes"][h]["stable"]) for r in train_recs for h in range(len(r["Y_stable"]))], np.float32))
    grp = np.array([i for i, r in enumerate(train_recs) for _ in range(len(r["Y_stable"]))])
    scorer = _PlanScorer(ctx.shape[1], plan.shape[1]); opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for _ in range(epochs):
        s = scorer(ctx, plan); loss = torch.nn.functional.binary_cross_entropy_with_logits(s, y)
        for g in np.unique(grp):
            idx = np.where(grp == g)[0]
            if len(idx) >= 2:
                si = s[idx]; yi = y[idx]; d = si[:, None] - si[None, :]
                mask = (yi[:, None] > yi[None, :] + 0.2).float()
                if mask.sum() > 0:
                    loss = loss + 0.5 * (torch.nn.functional.softplus(-d) * mask).sum() / mask.sum()
        loss.backward(); opt.step(); opt.zero_grad()
    return scorer


def scorer_scores(scorer, rec, ctx_key: str) -> np.ndarray:
    ctx = torch.as_tensor(np.repeat(rec[ctx_key][None].astype(np.float32), len(rec["Y_stable"]), 0))
    with torch.no_grad():
        return scorer(ctx, torch.as_tensor(rec["feats"])).numpy()


def _env(fingertip_geometry: str = "POINT"):
    return ContactFormationEnv(_roll_env(fingertip_geometry), _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False),
                               _ctx()["contract"], horizon=_C1_HORIZON, cfg=c1_config(), seed=0)


# ── Phase 1 — reproducibility gate (§0/§1/§2/§4): is Δ_selection reproducible? SEL0 vs SEL1 ──────────────────────
def pedc_gate(decoder_seeds=(0, 1, 2), selector_seeds=(0, 1, 2, 3, 4),
              held=range(62_500, 62_516), sel_train=range(63_000, 63_060)) -> dict:
    """Clean-obs reproducibility gate. For each of ≥3 decoders (V1), build the held counterfactual matrix (oracle) and
    train ≥5 C0 selectors (V0) → the joint decoder×selector variance (V2) of the SELECTION gap. Baselines S0 random /
    S2 heuristic / S3 oracle. Verdict SEL0 (no reproducible gap) vs SEL1 (gap confirmed)."""
    seg = max(1, _NLRP_H // _NLRP_K); env = _env(); lo, hi = env.action_space.low, env.action_space.high
    # PED-B decoder corpus (first 20 states of the cached nested pool — identical seeds/order, avoids a rebuild)
    corpus = _build_plan_corpus_cached(list(range(62_000, 62_312)), "c1_heldseed_bank.pkl", "pedf_pool_160.pkl", role="train")[:20]
    obs_dim = len(corpus[0].obs); act_dim = corpus[0].plans.shape[-1]
    planner = _shoot_ref_held(list(held))
    per_decoder = []
    for ds in decoder_seeds:
        dec = _train_plan_actor(corpus, obs_dim, act_dim, 6, seed=ds)
        held_recs = build_state_records(dec, held, "c1_heldseed_bank.pkl", env, seg, lo, hi, "held")
        train_recs = build_state_records(dec, sel_train, "c1_heldseed_bank.pkl", env, seg, lo, hi, "train")
        stats = record_stats(held_recs); oracle = stats["oracle_best_of_m"]
        s0 = random_expected(held_recs); s2 = _select_success(held_recs, heuristic_choose)["selected"]
        sel_scores = []
        for ss in selector_seeds:
            sc = train_scorer_c0(train_recs, "obs", seed=ss)
            r = _select_success(held_recs, lambda rec, sc=sc: int(np.argmax(scorer_scores(sc, rec, "obs"))))
            sel_scores.append(r["selected"])
        per_decoder.append({"decoder_seed": ds, "oracle": oracle, "record_stats": stats, "S0_random": s0,
                            "S2_heuristic": s2, "S1_selected_per_seed": sel_scores,
                            "S1_selected_median": round(float(np.median(sel_scores)), 3),
                            "S1_selected_worst": round(min(sel_scores), 3)})
        print(f"  [pedc-gate] decoder{ds}: oracle {oracle} | S0 {s0} S2 {s2} | S1 {sel_scores} "
              f"(med {per_decoder[-1]['S1_selected_median']})", flush=True)
    oracles = [d["oracle"] for d in per_decoder]
    all_sel = [v for d in per_decoder for v in d["S1_selected_per_seed"]]
    sel_med = float(np.median(all_sel)); sel_worst = float(min(all_sel)); oracle_med = float(np.median(oracles))
    gap_med = round(oracle_med - sel_med, 3)
    assert_oracle_ordering(oracle_med, sel_med)
    reproducible = gap_med >= 0.08 and (oracle_med - sel_worst) >= 0.08
    verdict = "SEL1_selection_gap_reproducible" if reproducible else "SEL0_no_reproducible_gap"
    decomp = oracle_decomposition(planner["shoot"], oracle_med, oracle_med, sel_med)
    out = {"planner_shoot": planner["shoot"], "v12": planner["v12"], "decoder_seeds": list(decoder_seeds),
           "selector_seeds": list(selector_seeds), "per_decoder": per_decoder,
           "oracle_median": round(oracle_med, 3), "oracle_range": [round(min(oracles), 3), round(max(oracles), 3)],
           "selected_median": round(sel_med, 3), "selected_worst": round(sel_worst, 3),
           "selected_ci95": [round(float(np.percentile(all_sel, 2.5)), 3), round(float(np.percentile(all_sel, 97.5)), 3)],
           "gap_median": gap_med, "oracle_decomposition": decomp, "verdict": verdict}
    _OUT.mkdir(parents=True, exist_ok=True); (_OUT / "pedc_gate.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[pedc-GATE] oracle {round(oracle_med,3)} [{out['oracle_range']}] | selected {round(sel_med,3)} "
          f"(worst {round(sel_worst,3)}, ci {out['selected_ci95']}) | gap {gap_med} → {verdict}", flush=True)
    return out


# ── Phase 2 — §6 objectives (C0/C1/C2/C3), §7 hard negatives, §9 F0/F1, §10 P-SEL ───────────────────────────────
def _flatten(recs, ctx_key):
    """Per-proposal training tensors: (ctx, feats, y_stable, group-index)."""
    ctx = torch.as_tensor(np.stack([r[ctx_key] for r in recs for _ in range(len(r["Y_stable"]))]))
    feats = torch.as_tensor(np.stack([r["feats"][h] for r in recs for h in range(len(r["Y_stable"]))]))
    y = torch.as_tensor(np.array([int(r["outcomes"][h]["stable"]) for r in recs for h in range(len(r["Y_stable"]))], np.float32))
    grp = np.array([i for i, r in enumerate(recs) for _ in range(len(r["Y_stable"]))])
    return ctx, feats, y, grp


def _pair_loss(s, y, idx, feats, hard_neg: bool):
    """§6 C1 / §7: within-state success>failure ranking hinge; hard_neg upweights pairs close in action space."""
    si, yi = s[idx], y[idx]; d = si[:, None] - si[None, :]
    mask = (yi[:, None] > yi[None, :] + 0.2).float()
    if mask.sum() == 0:
        return torch.zeros((), dtype=s.dtype)
    w = torch.ones_like(mask)
    if hard_neg:                                                    # emphasize near-in-feature success/failure pairs
        fi = feats[idx]; fd = torch.cdist(fi, fi); w = w * torch.exp(-fd)
    wm = w * mask
    return (torch.nn.functional.softplus(-d) * wm).sum() / wm.sum().clamp_min(1.0)


def _list_loss(s, y, idx):
    """§6 C2 listwise: distribute target mass over ALL successful proposals in the state (not one arbitrary index)."""
    si, yi = s[idx], y[idx]
    if yi.sum() == 0:
        return torch.zeros((), dtype=s.dtype)
    tgt = yi / yi.sum()
    return -(tgt * torch.log_softmax(si, 0)).sum()


def train_ranker(recs, ctx_key: str, objective: str, seed: int = 0, epochs: int = 500, hard_neg: bool = False):
    """C0 pointwise-BCE · C1 pairwise · C2 listwise — all produce a scalar-per-proposal `_PlanScorer` (permutation
    invariant). One entry point, objective-dispatched (no per-objective function proliferation)."""
    torch.manual_seed(seed); ctx, feats, y, grp = _flatten(recs, ctx_key)
    sc = _PlanScorer(ctx.shape[1], feats.shape[1]); opt = torch.optim.Adam(sc.parameters(), lr=1e-3)
    groups = [np.where(grp == g)[0] for g in np.unique(grp)]
    for _ in range(epochs):
        s = sc(ctx, feats)
        if objective == "c0":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(s, y)
        elif objective == "c1":
            loss = sum(_pair_loss(s, y, idx, feats, hard_neg) for idx in groups if len(idx) >= 2) / max(1, len(groups))
        elif objective == "c2":
            loss = sum(_list_loss(s, y, idx) for idx in groups) / max(1, len(groups))
        else:
            raise ValueError(f"unknown objective {objective}")
        loss.backward(); opt.step(); opt.zero_grad()
    return sc


class VectorCritic(torch.nn.Module):
    """§6 C3: predict [p_recovery(reached), p_delivery(stable), q_contact(controllable), p_harm] per proposal."""
    def __init__(self, ctx_dim: int, plan_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(ctx_dim + plan_dim, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 4))

    def forward(self, ctx, plan):
        return self.net(torch.cat([ctx, plan], -1))


def train_vector_critic(recs, ctx_key: str, seed: int = 0, epochs: int = 500):
    torch.manual_seed(seed); ctx, feats, _y, _g = _flatten(recs, ctx_key)
    keys = ("reached", "stable", "controllable", "harm")
    tgt = torch.as_tensor(np.array([[int(r["outcomes"][h][k]) for k in keys]
                                    for r in recs for h in range(len(r["Y_stable"]))], np.float32))
    crit = VectorCritic(ctx.shape[1], feats.shape[1]); opt = torch.optim.Adam(crit.parameters(), lr=1e-3)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(crit(ctx, feats), tgt)
        loss.backward(); opt.step(); opt.zero_grad()
    return crit


def _vector_scores(crit, rec, ctx_key):
    ctx = torch.as_tensor(np.repeat(rec[ctx_key][None].astype(np.float32), len(rec["Y_stable"]), 0))
    with torch.no_grad():
        return torch.sigmoid(crit(ctx, torch.as_tensor(rec["feats"]))).numpy()   # (m,4): recovery,delivery,contact,harm


def select_vector(crit, rec, ctx_key) -> int:
    """§6 C3 predeclared LEXICOGRAPHIC rule: maximize predicted delivery, break ties by lower harm then higher contact."""
    v = _vector_scores(crit, rec, ctx_key)
    key = -v[:, 1] * 1.0 + v[:, 3] * 0.5 - v[:, 2] * 0.25          # delivery ↑, harm ↓, contact ↑ (lexicographic-ish)
    return int(np.argmin(key))


def _ndcg(Y, s) -> float:
    order = np.argsort(-np.asarray(s)); g = Y[order]
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float((g * disc).sum()); ideal = float((np.sort(Y)[::-1] * disc).sum())
    return dcg / ideal if ideal > 0 else 1.0


def eval_objective(held_recs, score_of, by_group: bool = True) -> dict:
    """Selected success + §8 ranking diagnostics (top-1/2/3, MRR, NDCG, regret) + per-regime/drift, from lookup."""
    diag = _rank_diagnostics(held_recs, score_of)
    ndcg = float(np.mean([_ndcg(r["Y_stable"], score_of(r)) for r in held_recs]))
    sel = _select_success(held_recs, lambda r: int(np.argmax(score_of(r))))
    out = {"selected": sel["selected"], "mean_regret": sel["mean_regret"], "ndcg": round(ndcg, 3), **diag}
    if by_group:
        reg = {0: [], 1: [], 2: []}
        for r in held_recs:
            Y = r["Y_stable"]; h = int(np.argmax(score_of(r))); reg[r["sid"].contact_regime].append(int(Y[h]))
        out["selected_by_regime"] = {str(k): (round(float(np.mean(v)), 3) if v else None) for k, v in reg.items()}
    return out


def two_stage_f0_f1(feas_scorer, crit, rec, ctx_key_f, ctx_key_q) -> int:
    """§9 F0 feasibility (predict any stable recovery) → F1 quality (best delivery/low-harm among feasible)."""
    fs = scorer_scores(feas_scorer, rec, ctx_key_f); feasible = np.where(fs > 0.0)[0]
    if feasible.size == 0:
        return int(np.argmax(fs))
    v = _vector_scores(crit, rec, ctx_key_q)
    q = v[feasible, 1] - 0.5 * v[feasible, 3]                       # delivery − harm among feasible
    return int(feasible[int(np.argmax(q))])


def pedc_objectives(decoder_seeds=(0, 1, 2), obj_seeds=(0, 1, 2),
                    held=range(62_500, 62_516), sel_train=range(63_000, 63_060)) -> dict:
    """§6-§10 objective campaign: compare C0/C1/C2/C3 + hard-neg + F0/F1 vs the current selector and oracle, with a
    privileged P-SEL upper bound (§10). Multi-seed over decoders × objective seeds. → SEL2/SEL3/SEL4."""
    seg = max(1, _NLRP_H // _NLRP_K); env = _env(); lo, hi = env.action_space.low, env.action_space.high
    corpus = _build_plan_corpus_cached(list(range(62_000, 62_312)), "c1_heldseed_bank.pkl", "pedf_pool_160.pkl", role="train")[:20]
    obs_dim = len(corpus[0].obs); act_dim = corpus[0].plans.shape[-1]
    planner = _shoot_ref_held(list(held))
    objectives = ("c0", "c1", "c2", "c1_hard")
    agg: dict = {k: [] for k in (*objectives, "C3_vector", "current", "F0F1", "PSEL_priv", "oracle", "S0_random")}
    for ds in decoder_seeds:
        dec = _train_plan_actor(corpus, obs_dim, act_dim, 6, seed=ds)
        held_recs = build_state_records(dec, held, "c1_heldseed_bank.pkl", env, seg, lo, hi, "held")
        train_recs = build_state_records(dec, sel_train, "c1_heldseed_bank.pkl", env, seg, lo, hi, "train")
        agg["oracle"].append(record_stats(held_recs)["oracle_best_of_m"]); agg["S0_random"].append(random_expected(held_recs))
        for os_ in obj_seeds:
            for obj in objectives:
                sc = train_ranker(train_recs, "obs", obj.replace("_hard", ""), seed=os_, hard_neg=obj.endswith("hard"))
                agg[obj].append(eval_objective(held_recs, lambda r, s=sc: scorer_scores(s, r, "obs"))["selected"])
            cur = train_scorer_c0(train_recs, "obs", seed=os_)
            agg["current"].append(eval_objective(held_recs, lambda r, s=cur: scorer_scores(s, r, "obs"))["selected"])
            crit = train_vector_critic(train_recs, "obs", seed=os_)
            agg["C3_vector"].append(_select_success(held_recs, lambda r, c=crit: select_vector(c, r, "obs"))["selected"])
            feas = train_ranker(train_recs, "obs", "c0", seed=os_)
            agg["F0F1"].append(_select_success(held_recs, lambda r, f=feas, c=crit: two_stage_f0_f1(f, c, r, "obs", "obs"))["selected"])
            psel = train_ranker(train_recs, "priv", "c2", seed=os_)   # §10 privileged upper bound
            agg["PSEL_priv"].append(eval_objective(held_recs, lambda r, s=psel: scorer_scores(s, r, "priv"))["selected"])
        print(f"  [pedc-obj] decoder{ds} done (oracle {agg['oracle'][-1]})", flush=True)
    stat = lambda xs: {"median": round(float(np.median(xs)), 3), "worst": round(float(min(xs)), 3),   # noqa: E731
                       "best": round(float(max(xs)), 3)}
    res = {k: stat(v) for k, v in agg.items()}
    oracle_med = res["oracle"]["median"]; cur_med = res["current"]["median"]
    best_obj = max(objectives + ("C3_vector", "F0F1"), key=lambda k: res[k]["median"])
    best_med = res[best_obj]["median"]
    sel2 = best_med > cur_med + 0.03
    sel3 = best_med >= 0.63 and (oracle_med - best_med) <= 0.06 and res[best_obj]["worst"] >= 0.5
    sel4 = (oracle_med - best_med) <= 0.03
    verdict = "SEL4_reaches_oracle" if sel4 else "SEL3_closes_most_of_gap" if sel3 else "SEL2_objective_improves" if sel2 else "SEL1_gap_persists"
    out = {"planner_shoot": planner["shoot"], "oracle_median": oracle_med, "current_median": cur_med,
           "best_objective": best_obj, "best_median": best_med, "psel_priv_median": res["PSEL_priv"]["median"],
           "results": res, "SEL2_objective_improves": sel2, "SEL3_closes_gap": sel3, "SEL4_reaches_oracle": sel4,
           "verdict": verdict}
    _OUT.mkdir(parents=True, exist_ok=True); (_OUT / "pedc_objectives.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[pedc-OBJ] oracle {oracle_med} | current {cur_med} | best {best_obj} {best_med} | P-SEL(priv) "
          f"{res['PSEL_priv']['median']} → {verdict}", flush=True)
    return out


# ── §14 Route-C audit — is the gap within-state INDISTINGUISHABILITY, or selector generalization? ───────────────
def _within_state_auc(recs, score_of) -> float:
    """Mean within-state ranking AUROC: fraction of (success, failure) pairs in a state where the score ranks the
    success higher. 0.5 = indistinguishable; 1.0 = perfectly separable. Averaged over states that have both."""
    aucs = []
    for r in recs:
        Y = r["Y_stable"]; s = np.asarray(score_of(r))
        pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        if pos.size and neg.size:
            wins = sum(float(s[p] > s[n]) + 0.5 * float(s[p] == s[n]) for p in pos for n in neg)
            aucs.append(wins / (pos.size * neg.size))
    return round(float(np.mean(aucs)), 3) if aucs else float("nan")


def _feat_separation(recs) -> float:
    """Mean within-state action-feature distance between successful and failing proposals — are they even different?"""
    ds = []
    for r in recs:
        Y = r["Y_stable"]; f = r["feats"]; pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        if pos.size and neg.size:
            ds.append(float(np.mean([np.linalg.norm(f[p] - f[n]) for p in pos for n in neg])))
    return round(float(np.mean(ds)), 3) if ds else float("nan")


def pedc_route_c(decoder_seeds=(0, 1, 2), held=range(62_500, 62_516), sel_train=range(63_000, 63_060)) -> dict:
    """The discriminating test for the plateau: fit the privileged C2 critic, then measure within-state success/fail
    separation on TRAIN vs HELD. train≈0.5 → indistinguishable (RC-IND); train high, held≈0.5 → selector generalization
    (RC-GEN); both high yet selected<oracle → an eval/optimization bug. Plus single- vs multi-success gap breakdown."""
    seg = max(1, _NLRP_H // _NLRP_K); env = _env(); lo, hi = env.action_space.low, env.action_space.high
    corpus = _build_plan_corpus_cached(list(range(62_000, 62_312)), "c1_heldseed_bank.pkl", "pedf_pool_160.pkl", role="train")[:20]
    obs_dim = len(corpus[0].obs); act_dim = corpus[0].plans.shape[-1]
    rows = []
    for ds in decoder_seeds:
        dec = _train_plan_actor(corpus, obs_dim, act_dim, 6, seed=ds)
        tr = build_state_records(dec, sel_train, "c1_heldseed_bank.pkl", env, seg, lo, hi, "train")
        hd = build_state_records(dec, held, "c1_heldseed_bank.pkl", env, seg, lo, hi, "held")
        priv = train_ranker(tr, "priv", "c2", seed=0); obsc = train_ranker(tr, "obs", "c2", seed=0)
        # single vs multi-success held gap
        single = [r for r in hd if r["Y_stable"].sum() == 1]; multi = [r for r in hd if r["Y_stable"].sum() >= 2]
        sel_single = _select_success(single, lambda r, s=priv: int(np.argmax(scorer_scores(s, r, "priv"))))["selected"] if single else None
        sel_multi = _select_success(multi, lambda r, s=priv: int(np.argmax(scorer_scores(s, r, "priv"))))["selected"] if multi else None
        rows.append({"decoder_seed": ds,
                     "priv_auc_train": _within_state_auc(tr, lambda r: scorer_scores(priv, r, "priv")),
                     "priv_auc_held": _within_state_auc(hd, lambda r: scorer_scores(priv, r, "priv")),
                     "obs_auc_train": _within_state_auc(tr, lambda r: scorer_scores(obsc, r, "obs")),
                     "obs_auc_held": _within_state_auc(hd, lambda r: scorer_scores(obsc, r, "obs")),
                     "feat_sep_held": _feat_separation(hd),
                     "n_single": len(single), "n_multi": len(multi),
                     "priv_sel_single": sel_single, "priv_sel_multi": sel_multi})
        print(f"  [route-c] dec{ds}: priv-AUC train {rows[-1]['priv_auc_train']} held {rows[-1]['priv_auc_held']} | "
              f"obs-AUC train {rows[-1]['obs_auc_train']} held {rows[-1]['obs_auc_held']} | "
              f"single {sel_single}({len(single)}) multi {sel_multi}({len(multi)})", flush=True)
    med = lambda k: round(float(np.median([r[k] for r in rows if r[k] is not None])), 3) if any(r[k] is not None for r in rows) else None  # noqa: E731
    pat = med("priv_auc_train"); pah = med("priv_auc_held")
    if pat is not None and pat < 0.6:
        verdict = "RC_INDISTINGUISHABLE_even_privileged_cannot_separate_on_train"
    elif pah is not None and pah < pat - 0.1:
        verdict = "RC_SELECTOR_GENERALIZATION_train_separates_held_does_not"
    else:
        verdict = "RC_SEPARABLE_but_selection_underperforms_check_eval"
    out = {"per_decoder": rows, "priv_auc_train_median": pat, "priv_auc_held_median": pah,
           "obs_auc_train_median": med("obs_auc_train"), "obs_auc_held_median": med("obs_auc_held"),
           "priv_sel_single_median": med("priv_sel_single"), "priv_sel_multi_median": med("priv_sel_multi"),
           "verdict": verdict}
    _OUT.mkdir(parents=True, exist_ok=True); (_OUT / "pedc_route_c.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[pedc-ROUTE-C] priv-AUC train {pat} held {pah} | obs-AUC train {med('obs_auc_train')} held {med('obs_auc_held')} "
          f"| single-succ sel {med('priv_sel_single')} multi-succ sel {med('priv_sel_multi')} → {verdict}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    fn = pedc_route_c if "--route-c" in sys.argv else pedc_objectives if "--objectives" in sys.argv else pedc_gate
    fn()
