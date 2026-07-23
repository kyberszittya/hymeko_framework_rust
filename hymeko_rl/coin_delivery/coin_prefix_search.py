"""Reusable prefix-search controllers for the coin value→policy-conversion audits (VALUE_GUIDED_PREFIX_SEARCH,
UPSTREAM_HEADROOM_PREFIX_SEARCH).

A critic-INDEPENDENT candidate generator ({pi_0} ∪ {pi_0 ± ε·e_i}) + receding-horizon selectors that rank the candidates
by the LEARNED signal only (the exact simulator is used solely for the bounded K-step lookahead; the terminal value is the
learned Q_target — never a full exact-rollout-to-episode-end oracle). Controls: pi_0 (no selection) and RANDOM_VALID (picks
a uniformly random candidate — isolates whether the scorer selects intelligently vs the candidate-generator merely being
good). Learned scorers: REWARD_ONLY, BOOTSTRAP_VALUE_ONLY, REWARD_PLUS_VALUE.
"""
import copy

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_markov_ablation_train import (
    GAMMA,
    prefix_candidate_rollout,
    receding_horizon_rollout,
)

ASCALE = 4.0
BIG = 1e6                                                 # a delivered (terminated) lookahead is the best possible terminal value


def offsets(adim, mag):
    """Candidate action offsets: index 0 == pi_0 (zero), then ±ε·e_i on each actuator."""
    offs = [np.zeros(adim, np.float32)]
    for ax in range(adim):
        for sg in (+1, -1):
            o = np.zeros(adim, np.float32); o[ax] = sg * mag; offs.append(o)
    return offs


def sel_pi0(rl, gate, o55, a_pi0):
    return a_pi0, {"choice": 0, "nonpi0": 0}


def make_sel_random(offs, seed):
    """RANDOM_VALID: pick a uniformly random candidate each step (deterministic given ``seed``)."""
    rng = np.random.default_rng(seed)
    def f(rl, gate, o55, a_pi0):
        b = int(rng.integers(0, len(offs)))
        return np.clip(a_pi0 + offs[b], -ASCALE, ASCALE), {"choice": b, "nonpi0": int(b != 0)}
    return f


def make_sel_search(pi0, base, critic_t, actor_t, offs, k_prefix, scorer):
    """Learned-signal receding scorer. scorer ∈ {reward, value, reward_value}. Terminal value = Q_target(s_K, π_target)."""
    def f(rl, gate, o55, a_pi0):
        r_pref, boot = [], []
        for off in offs:
            r = prefix_candidate_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, off, k_prefix, horizon=k_prefix)
            r_pref.append(float(sum(GAMMA ** t * x for t, x in enumerate(r["rewardsA"]))))
            if r["terminated_in_prefix"] or r["obs55_K"] is None:
                boot.append(BIG)
            else:
                o = torch.as_tensor(r["obs55_K"])[None]
                with torch.no_grad():
                    boot.append(float(GAMMA ** r["k_applied"] * critic_t.min_q(o, torch.clamp(actor_t.action_mean(o), -ASCALE, ASCALE))[0]))
        sc = r_pref if scorer == "reward" else (boot if scorer == "value" else [a + b for a, b in zip(r_pref, boot)])
        b = int(np.argmax(sc))
        return np.clip(a_pi0 + offs[b], -ASCALE, ASCALE), {"choice": b, "nonpi0": int(b != 0)}
    return f


def candidate_outcomes(templates, i, pi0, base, offs, k_prefix, horizon):
    """Full-episode certifier outcome of each candidate (offset prefix then pi_0). Used for oracle candidate-coverage —
    NOT for filtering the eval panel."""
    rl0, gate0 = templates[i]
    return [prefix_candidate_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, off, k_prefix, horizon=horizon)["outcome"]
            for off in offs]


def buffer_obs_sample(buf, rng, n=300):
    obs = [t["obs"] for tr in buf.trajectories if tr for t in tr]
    if not obs:
        return None
    idx = rng.integers(0, len(obs), min(n, len(obs)))
    return np.asarray([obs[j] for j in idx], np.float32)


def run_controller(templates, i, pi0, base, select, buf_sample, horizon):
    """One receding-horizon episode from state ``i``; augments the certifier outcome with drift / non-pi_0 rate / replay
    support distance derived from the per-step infos."""
    rl0, gate0 = templates[i]
    out, infos = receding_horizon_rollout(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, select, horizon=horizon)
    gon = [x for x in infos if x and x.get("gate_on")]
    out["cum_drift"] = round(float(sum(x["drift"] for x in gon)), 4)
    out["nonpi0_rate"] = round(float(np.mean([x.get("nonpi0", 0) for x in gon])) if gon else 0.0, 3)
    if buf_sample is not None and gon:
        out["support_dist"] = round(float(np.mean([float(np.linalg.norm(buf_sample - x["obs55"], axis=1).min()) for x in gon])), 4)
    else:
        out["support_dist"] = None
    return out
