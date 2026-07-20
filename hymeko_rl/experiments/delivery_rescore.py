"""Delivery re-score — re-evaluate the existing SEARCH-TENSOR / KANC selector artifacts under the CORRECTED task metric
(delivery to the target zone: `in_zone` = disk_to_zone <= zone_half), NOT the handoff/grasp submetric.

Task-definition correction (2026-07-20): the coin task is DELIVERY to the zone; handoff/contact is an acquisition
submetric and cannot be task success. This module re-rolls the existing candidate pool (the frozen decoders' proposals,
exact depth-2 replay) capturing per-branch delivery, and re-scores every selection mode under `delivery_success`. NO
training, NO RL, NO env/reward change — measurement + offline selection only.

Decisive question first: does the FULL candidate pool contain ANY delivery (full depth-2 delivery rate), or is the pool
acquisition-only? Then: does a delivery-dominant selection order convert acquisition into delivery, and does the tensor
prior (acquisition-trained) help delivery at all?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hymeko_rl.experiments import jfa0, kanc0
from hymeko_rl.experiments.exp_coin_toss_arcrl import _NLRP_H, _NLRP_K, _expand
from hymeko_rl.experiments.jfa_rl1 import _q4, build_jfa0_checkpoint, deploy_score
from hymeko_rl.experiments.loop3 import UnionSMDP
from hymeko_rl.experiments.pedc_selection import _env, train_ranker
from hymeko_rl.experiments.pedg_relational import _single_success

D = Path("experiments/2026_07_20_delivery_rescore")
_IDXS = kanc0._IDXS
_K = 8                                                                              # search budget (pairs), matches SEARCH-TENSOR-1


def _exec(smdp, j, inner) -> dict:
    """Execute one proposal from the current SMDP state; capture handoff + DELIVERY (in_zone, disk_to_zone)."""
    pl, _f = smdp._propose(smdp._obs); plan = pl[int(j)]
    handoff = deliv = done = False; min_dtz = 1e9; final_dtz = float(inner._planar_metrics.disk_to_zone); obs = smdp._obs
    for a in _expand(plan, smdp.seg):
        obs, _r, term, trunc, info = smdp.env.step(np.clip(a, smdp.lo, smdp.hi).astype(np.float32))
        m = inner._planar_metrics; dtz = float(m.disk_to_zone)
        min_dtz = min(min_dtz, dtz); final_dtz = dtz; deliv = deliv or bool(m.in_zone); smdp._steps += 1
        if info.get("handoff_ready"):
            handoff = done = True; break
        if term or trunc or smdp._steps >= smdp.step_budget:
            done = True; break
    smdp._obs = obs; smdp._nprop += 1
    return {"obs": obs.astype(np.float32), "handoff": handoff, "deliv": deliv, "min_dtz": min_dtz, "final_dtz": final_dtz,
            "done": done}


def _state_tree(smdp, inner, r, idxs) -> dict:
    """Exact-replay depth-2 candidate pool for one state, capturing per-branch delivery + successor features."""
    sd = int(r["sid"].seed); smdp.reset(sd); start_dtz = float(inner._planar_metrics.disk_to_zone); firsts = []
    for pos_j, j in enumerate(idxs):
        smdp.reset(sd); e1 = _exec(smdp, j, inner); entry = {"pos": pos_j, "e1": e1, "seconds": [], "succ_feats": None}
        if not e1["done"]:
            entry["succ_feats"] = smdp._propose(e1["obs"])[1][list(idxs)].astype(np.float32)
            for jp in idxs:
                smdp.reset(sd); _exec(smdp, j, inner); e2 = _exec(smdp, jp, inner)
                entry["seconds"].append({"handoff": e1["handoff"] or e2["handoff"], "deliv": e1["deliv"] or e2["deliv"],
                                         "min_dtz": min(e1["min_dtz"], e2["min_dtz"]), "final_dtz": e2["final_dtz"]})
        firsts.append(entry)
    return {"sid": r["sid"], "obs": r["obs"], "feats": r["feats"], "actor_oh": r["actor_oh"],
            "start_dtz": start_dtz, "firsts": firsts}


def _any_deliv(tree: dict) -> bool:
    return any(fr["e1"]["deliv"] or any(s["deliv"] for s in fr["seconds"]) for fr in tree["firsts"])


def delivery_corpus(smdp: UnionSMDP, recs: list, idxs=_IDXS, cache: str | None = None) -> list:
    """Per state: exact-replay depth-2 candidate pool with per-branch DELIVERY (handoff/deliv/min_dtz/final_dtz), the
    successor features for tensor ranking, and start_dtz. Physics executed once, cached."""
    import pickle
    path = D / "corpus" / cache if cache else None
    if path and path.exists():
        return pickle.loads(path.read_bytes())
    inner = smdp.env._env; out = []
    for i, r in enumerate(recs):
        out.append(_state_tree(smdp, inner, r, idxs))
        if (i + 1) % 20 == 0:
            print(f"  [delivery-corpus:{cache or 'held'}] {i + 1}/{len(recs)} states | any-deliv Σ "
                  f"{sum(_any_deliv(o) for o in out)}", flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(pickle.dumps(out))
    return out


def _pairs(tree: dict) -> list:
    """Flatten a state's depth-2 tree into candidate pairs: (handoff, deliv, min_dtz, final_dtz, first_pos, jp_pos)."""
    out = []
    for fr in tree["firsts"]:
        e1 = fr["e1"]
        if e1["done"]:
            out.append({"handoff": e1["handoff"], "deliv": e1["deliv"], "min_dtz": e1["min_dtz"],
                        "final_dtz": e1["final_dtz"], "first": fr["pos"], "jp": None})
        for jp, s in enumerate(fr["seconds"]):
            out.append({**s, "first": fr["pos"], "jp": jp})
    return out


def _tensor_prior(tensor, tree: dict) -> np.ndarray:
    """Pair prior = α·Q_H1_C0(first) + β·Q_H0_C0(second) — the SEARCH-TENSOR-1 acquisition-trained prior, over the
    same candidate order as `_pairs`."""
    qf = tensor.q_tensor(tree["obs"], tree["feats"], tree["actor_oh"]); aoh = _actor_oh(); scores = []
    for fr in tree["firsts"]:
        h1 = float(qf[fr["pos"], 0, 1])
        if fr["e1"]["done"]:
            scores.append(h1)                                                      # terminal pair (matches _pairs order)
        if fr["succ_feats"] is not None:
            qs = tensor.q_tensor(fr["e1"]["obs"], fr["succ_feats"], aoh)            # successor = obs after the first move
            for jp in range(len(fr["seconds"])):
                scores.append(h1 + float(qs[jp, 0, 0]))
    return np.array(scores)


def _actor_oh(idxs=_IDXS) -> np.ndarray:
    return kanc0._actor_onehot(np.array([0 if j < 6 else 1 for j in idxs], np.int64))


def _score_mode(trees: list, order_fn, k: int = _K) -> dict:
    """Evaluate a selection mode: order the pairs per state, take top-k, DELIVERY = any top-k pair delivers. Reports the
    delivery + handoff + distance metrics."""
    deliv = hf = 0; fdtz = []; mdtz = []; prog = []; grasp_no_deliv = 0; n = len(trees)
    for t in trees:
        pr = _pairs(t)
        if not pr:
            continue
        idx = order_fn(t, pr)[:k]; sel = [pr[i] for i in idx]
        d = any(p["deliv"] for p in sel); h = any(p["handoff"] for p in sel)
        deliv += int(d); hf += int(h); grasp_no_deliv += int(h and not d)
        best = min(sel, key=lambda p: p["final_dtz"])
        fdtz.append(best["final_dtz"]); mdtz.append(min(p["min_dtz"] for p in sel))
        prog.append(t["start_dtz"] - best["final_dtz"])
    return {"delivery_success": round(deliv / n, 4), "handoff_success": round(hf / n, 4),
            "grasp_no_delivery_rate": round(grasp_no_deliv / n, 4),
            "final_disk_to_zone_med": round(float(np.median(fdtz)), 4), "min_disk_to_zone_med": round(float(np.median(mdtz)), 4),
            "delivery_progress_med": round(float(np.median(prog)), 4)}


def _order_full(t, pr):
    return list(range(len(pr)))                                                    # full search = evaluate all (k>=|pr|)


def _order_random(rng):
    def f(t, pr):
        return list(rng.permutation(len(pr)))
    return f


def _order_tensor(tensor):
    def f(t, pr):
        s = _tensor_prior(tensor, t)
        return list(np.argsort(-s))
    return f


def _order_delivery_dominant(tensor):
    """delivery_success > delivery_progress > handoff > tensor prior. The prior is SUBORDINATE — it only breaks ties
    among equal delivery/progress; it cannot override a zero-delivery plateau."""
    def f(t, pr):
        s = _tensor_prior(tensor, t) if tensor is not None else np.zeros(len(pr))
        key = [(-int(p["deliv"]), p["final_dtz"], -int(p["handoff"]), -s[i]) for i, p in enumerate(pr)]
        return list(np.lexsort(([k[3] for k in key], [k[2] for k in key], [k[1] for k in key], [k[0] for k in key])))
    return f


def _old_selector(m0):
    """The deployed handoff-oriented selector: order pairs by the FIRST proposal's M0 deploy score (acquisition)."""
    def f(t, pr):
        q4 = _q4(m0, t["obs"], t["feats"]); ds = deploy_score(q4)
        return list(np.argsort([-ds[p["first"]] for p in pr]))
    return f


def run(n_held: int = 90) -> dict:
    D.mkdir(parents=True, exist_ok=True); (D / "manifests").mkdir(exist_ok=True)
    env = _env(); seg = max(1, _NLRP_H // _NLRP_K); lo, hi = env.action_space.low, env.action_space.high
    old_dec, new_dec, _o, act_dim = jfa0._decoders(env, seg, lo, hi)
    smdp = UnionSMDP(env, old_dec, new_dec)
    tu = jfa0.build_union_corpus(old_dec, new_dec, range(63_000, 63_120), env, seg, lo, hi, "train", "train.pkl")
    h0u = jfa0.build_union_corpus(old_dec, new_dec, range(64_000, 64_000 + n_held), env, seg, lo, hi, "held", "h0.pkl")
    kh0 = kanc0.build_kanc_corpus(smdp, h0u, cache="kh0.pkl")
    ktr = kanc0.build_kanc_corpus(smdp, tu, cache="ktrain.pkl")
    trees = delivery_corpus(smdp, kh0, cache="h0_delivery.pkl")
    osc = train_ranker(_single_success(jfa0.old_only_records(tu))[:90] or jfa0.old_only_records(tu), "obs", "c0", seed=0)
    m0 = build_jfa0_checkpoint(tu, osc, seed=0)
    obs_dim = kh0[0]["obs"].shape[0]; plan_dim = kh0[0]["feats"].shape[1]
    tn = kanc0.train_tensor(kanc0.KANCTensor(obs_dim, plan_dim, actor_dim=2, hidden=48), ktr, seed=0)
    sc = kanc0.train_tensor(kanc0.KANCTensor(obs_dim, plan_dim, actor_dim=2, hidden=48), ktr, seed=0, scramble="S4")
    rng = np.random.default_rng(0)

    full = max(len(_pairs(t)) for t in trees)
    modes = {
        "A_old_handoff_selector": _score_mode(trees, _old_selector(m0), k=_K),
        "full_depth2_delivery_ceiling": _score_mode(trees, _order_full, k=full),
        "C_tensor_guided_topk": _score_mode(trees, _order_tensor(tn), k=_K),
        "D_scrambled_guided_topk": _score_mode(trees, _order_tensor(sc), k=_K),
        "E_random_guided_topk": _score_mode(trees, _order_random(rng), k=_K),
        "B_delivery_dominant_exact": _score_mode(trees, _order_delivery_dominant(None), k=full),
        "delivery_dominant_tensor_topk": _score_mode(trees, _order_delivery_dominant(tn), k=_K),
    }
    ceiling = modes["full_depth2_delivery_ceiling"]["delivery_success"]
    verdict = ("B_no_candidate_delivers_generator_insufficient" if ceiling <= 0.0
               else "A_candidates_deliver_delivery_dominant_selection_matters")
    out = {"n_held": len(trees), "search_k": _K, "full_pairs": full, "delivery_ceiling_full_depth2": ceiling,
           "modes": modes, "verdict": verdict}
    (D / "manifests" / "delivery_rescore.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n=== DELIVERY RE-SCORE (metric = in_zone, zone delivery) | n={len(trees)} ===", flush=True)
    for name, m in modes.items():
        print(f"  {name:32s} delivery {m['delivery_success']:.3f} | handoff {m['handoff_success']:.3f} | "
              f"grasp-no-deliv {m['grasp_no_delivery_rate']:.3f} | final_dtz {m['final_disk_to_zone_med']:.3f} "
              f"progress {m['delivery_progress_med']:.3f}", flush=True)
    print(f"[DELIVERY] full-depth2 delivery ceiling {ceiling} → {verdict}", flush=True)
    return out


if __name__ == "__main__":
    run()
