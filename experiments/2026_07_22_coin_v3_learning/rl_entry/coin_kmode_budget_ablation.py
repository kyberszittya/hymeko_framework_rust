"""EQUAL_BUDGET_KMODE_ABLATION_V1 — the decisive Track B physical test of the multimodal runtime.

Question (user, verbatim): single-head proposal + total budget B  vs  K-mode proposal (K=2/4/6) + the SAME total
budget B. The K-mode arms MUST NOT get more total candidates just because they have more modes — `allocate_budget`
splits the one budget B across the modes (≥1 each), so every K spends exactly B rollouts.

Binds the FROZEN OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1 (`MultimodalBudgetSearch`) to the coin contact task with NO
new runtime code: the O2 box proposal's K=6 templates become K strategy modes; ONE shared candidate generator
(structured jitter, identical to `structured_random_around`) and ONE shared lexicographic scorer (`structured_score`)
serve every K, so K=1 is a faithful single-head search and the only thing that varies across arms is the number of
proposal modes. Grade = the PHYSICAL K6 delivery certificate (not a fit loss) on the fresh-reconstruct O2 box states.

Success criterion: K-mode + search yields a physically higher fixed-budget K6 deploy than the single-head proposal at
the SAME compute. Run on the O2 boxes now (a distribution already physically validated, no new geometry); the
triangular-prism (stronger vertex/edge multimodality) is the follow-up once the mesh manipuland lands.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
import torch  # noqa: E402

from hymeko_rl.coin_delivery.coin_carry_proposal import denorm_theta, load_proposal  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import (  # noqa: E402
    A_BOUND, T_MAX, T_MIN, structured_carry_rollout, structured_score)
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.option_rl import MultimodalBudgetSearch, ProposalMode  # noqa: E402 — the FROZEN runtime

from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_object_o2 import SHAPES, fresh_o2_bank  # noqa: E402 — reuse the O2 fresh-reconstruct panel (uses _ball_tf internally)

OUT = "reports/2026-07-24-kmode-budget-ablation"
O2_PROP = f"{D}/carry_proposal_o2_box_v1.pt"
K_VALUES, BUDGET, EVAL_H = (1, 2, 4, 6), 12, 160


class LexScore:
    """Adapts the coin's lexicographic ``structured_score`` tuple to the runtime's float-scorer contract: it compares
    lexicographically against other LexScores and always beats the ``-inf`` sentinel the search seeds ``best`` with.
    ``__float__`` is a monotone summary for logging only — the ORDERING is the tuple, so the search picks exactly the
    candidate ``structured_score`` would. # Invariants: total order consistent with tuple comparison."""

    __slots__ = ("t",)

    def __init__(self, t):
        self.t = tuple(t)

    def __gt__(self, other):
        return self.t > other.t if isinstance(other, LexScore) else True   # any real outcome > the -inf seed

    def __eq__(self, other):
        return isinstance(other, LexScore) and self.t == other.t

    def __float__(self):
        return float(self.t[0] * 1e6 + self.t[1] * 1e3 + self.t[2])        # k6 ≻ handoff ≻ dwell summary


class CoinJitterGenerator:
    """`CandidateGenerator`: Gaussian jitter around a θ centre, IDENTICAL to ``structured_random_around`` (amp std 0.6,
    dur std 2.0, clip to legal bounds) — so a single mode's search reproduces the coin's canonical local search."""

    def __init__(self, std_amp=0.6, std_dur=2.0):
        self.std_amp, self.std_dur = std_amp, std_dur

    def sample(self, center, n, rng):
        center = np.asarray(center, np.float32)
        out = np.empty((int(n), 15), np.float32)
        for i in range(int(n)):
            theta = center + np.concatenate([rng.normal(0, self.std_amp, 12), rng.normal(0, self.std_dur, 3)]).astype(np.float32)
            theta[0:12] = np.clip(theta[0:12], -A_BOUND, A_BOUND)
            theta[12:15] = np.clip(theta[12:15], T_MIN, T_MAX)
            out[i] = theta
        return out


class CoinCarryScorer:
    """`CandidateScorer` bound to ONE fresh eval state: roll the committed θ (fresh deepcopy of rl AND gate every time —
    the gate-contamination fix) and score by ``structured_score``. Returns (LexScore, outcome-dict-with-k6)."""

    def __init__(self, rl, gate, pi0, base, horizon=EVAL_H):
        self.rl, self.gate, self.pi0, self.base, self.horizon = rl, gate, pi0, base, horizon

    def score(self, cand, rng):
        o = structured_carry_rollout(copy.deepcopy(self.rl), copy.deepcopy(self.gate), self.pi0, self.base,
                                     np.asarray(cand, np.float32), horizon=self.horizon)
        return LexScore(structured_score(o)), o


class TemplateKModeProposal:
    """`MultimodalProposalPolicy`: expose the O2 proposal's residual-adjusted templates as K strategy modes. K=1 returns
    ONLY the classifier-argmax template (≡ ``proposal.theta(obs)`` — the single-head baseline); K>1 returns the top-K
    templates by classifier probability, each mode keyed to its template index (mode_id) for order-invariant search."""

    def __init__(self, proposal, obs, k):
        self.proposal, self.obs, self.k = proposal, np.asarray(obs, np.float32), int(k)

    def _all_centers_probs(self):
        o = self.obs[None]
        p = self.proposal
        with torch.no_grad():
            probs = torch.softmax(p.clf(torch.as_tensor(o)), -1).numpy()[0]     # (K,)
            eye = np.eye(p.K, dtype=np.float32)
            centers = [denorm_theta(p.templates_norm[i] + p.residual(torch.as_tensor(o), torch.as_tensor(eye[i][None])).numpy())[0]
                       for i in range(p.K)]
        return centers, probs

    def modes(self, _obs):
        centers, probs = self._all_centers_probs()
        order = np.argsort(-probs)[: self.k]                                    # top-k modes by classifier prob
        return [ProposalMode(float(probs[i]), np.asarray(centers[i], np.float32), None, int(i)) for i in order]


def run_arm(rl, gate, proposal, pi0, base, k, budget, rng):
    """One (state, K) physical arm at EQUAL total budget: MultimodalBudgetSearch over the top-K template modes, budget
    split across them (≥1 each, ∝prob), keep the global-best committed θ. Returns (k6, reached_handoff, selected_mode,
    per_mode_budget)."""
    scorer = CoinCarryScorer(rl, gate, pi0, base)
    prop = TemplateKModeProposal(proposal, rl.obs(), k)
    prov = MultimodalBudgetSearch(CoinJitterGenerator(), scorer, budget=budget).select(prop, rl.obs(), rng)
    return {"k6": int(prov.outcome.get("k6", 0)), "reached_handoff": int(prov.outcome.get("reached_handoff", 0)),
            "selected_mode": int(prov.selected_mode), "per_mode_budget": list(prov.per_mode_budget)}


def _paired_bootstrap(diffs, iters=5000, seed=0):
    """Bootstrap 95% CI of the mean paired (K6_hi − K6_lo) difference over states×seeds."""
    d = np.asarray(diffs, np.float64)
    if len(d) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)]
    return {"mean": round(float(d.mean()), 4), "lo": round(float(np.percentile(boot, 2.5)), 4),
            "hi": round(float(np.percentile(boot, 97.5)), 4), "n": int(len(d))}


def main(smoke=False):
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    proposal = load_proposal(O2_PROP)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    want = 6 if smoke else 24
    seeds = (0,) if smoke else (0, 1)                         # 2 search-seeds ⇒ PILOT cap (seed-aware, no single-draw)
    shapes = ["square_1_1"] if smoke else list(SHAPES)
    ks = (1, 6) if smoke else K_VALUES

    ev_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 60),
                                         families=("contact_retention", "transport", "braking"),
                                         strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    results = {}
    for sh in shapes:
        bank = fresh_o2_bank(pi0, sh, ev_ls, want, log)
        recs = []
        for i, it in enumerate(bank):
            rec = {"orient_bin": it["orient_bin"]}
            for k in ks:
                arms = [run_arm(it["rl"], it["gate"], proposal, pi0, base, k, BUDGET,
                                np.random.default_rng(70000 + 1000 * s + i)) for s in seeds]
                rec[f"K{k}"] = {"k6": [a["k6"] for a in arms], "reached": [a["reached_handoff"] for a in arms],
                               "modes": [a["selected_mode"] for a in arms], "budget": arms[0]["per_mode_budget"]}
            recs.append(rec)
        n_draws = len(recs) * len(seeds)
        k6_rate = {f"K{k}": round(sum(sum(r[f"K{k}"]["k6"]) for r in recs) / max(1, n_draws), 3) for k in ks}
        lo, hi = min(ks), max(ks)
        diffs = [r[f"K{hi}"]["k6"][j] - r[f"K{lo}"]["k6"][j] for r in recs for j in range(len(seeds))]
        results[sh] = {"n_states": len(recs), "n_draws": n_draws, "k6_rate": k6_rate,
                       f"boot_K{hi}_minus_K{lo}": _paired_bootstrap(diffs), "records": recs}
        log(f"  [{sh}] n {len(recs)}×{len(seeds)}seed  K6-rate " +
            " ".join(f"K{k}={k6_rate[f'K{k}']}" for k in ks) +
            f"  Δ(K{hi}-K{lo}) {results[sh][f'boot_K{hi}_minus_K{lo}']}")

    manifest = {"contract": "EQUAL_BUDGET_KMODE_ABLATION_V1", "date": "2026-07-24", "smoke": smoke,
                "runtime": "OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1 (frozen)", "distribution": "O2_fresh_reconstruct_box",
                "embodiment": "collision-on ball-tip (frozen)", "budget_total": BUDGET, "K_values": list(ks),
                "search_seeds": list(seeds), "grade": "physical K6 delivery certificate", "results": results}
    json.dump(manifest, open(f"{OUT}/kmode_budget.json", "w"), indent=1, default=float)
    log(f"\n== EQUAL_BUDGET_KMODE_ABLATION (O2 boxes, budget {BUDGET}) ==\n  artifact: {OUT}/kmode_budget.json\nKMODE_BUDGET_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
