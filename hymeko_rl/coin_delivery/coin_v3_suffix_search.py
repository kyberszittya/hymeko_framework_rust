"""KATO14 CPU-parallel success-certified suffix search (v3 expert-strengthening §5-§11).

For each BC-reached grasp state, run CEM over a compact residual-knot suffix (applied via env.step, open-loop),
ranked lexicographically by the strict K=6 certificate, then replay-certify the discovered suffix from the ORIGINAL
neutral reset (prefix + suffix, all through env.step, no state injection during the certified rollout). Parallel over
states, checkpointed (skips completed states), continuous JSONL logging, restartable.

Usage:  python coin_v3_suffix_search.py --stage headline --workers 14 --out RESULTS_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent  # the deployed source root
sys.path.insert(0, str(REPO))

BUNDLE_HASH = "6664ac459cca8f62"
HEADLINE = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)
HELDOUT = tuple(range(4000, 4030))
N_KNOTS = 10          # residual control knots over the transport suffix
SUFFIX_STEPS = 200
ACT_DIM = 6           # transport residual dim (env.step raw)


def _bc():
    """The frozen E_valselect APPROACH (the reliable grasp policy) — the strengthened teacher's approach (§10)."""
    from hymeko_rl.experiments.coin_neutral_start import _e_approach_actor
    return _e_approach_actor()


def _env():
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0)
    return env, cf, cf._env


def _roll_bc_to_grasp(env, cf, inner, e, seed, grasp_hold=3):
    """The frozen E approach drives from neutral to the TRANSITION state (bilateral grasp_hold OR the 160-step cap —
    exactly the composed-chain handoff point). Always returns the snapshot (qpos,qvel) + whether a firm bilateral
    grasp formed. The strengthened-teacher search targets this transition state (where the handoff would take over)."""
    env.set_stage(0)
    env.reset(seed=int(seed))
    bi = 0
    grasped = False
    touched = False
    for _k in range(160):
        m = inner._planar_metrics
        touched = touched or bool(m.left_contact or m.right_contact)
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        if bi >= grasp_hold:
            grasped = True
            break
        a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
    return inner.data.qpos.copy(), inner.data.qvel.copy(), grasped, touched


def _knots_to_actions(knots):
    """(N_KNOTS, ACT_DIM) → (SUFFIX_STEPS, ACT_DIM) piecewise-linear interpolation."""
    xp = np.linspace(0, SUFFIX_STEPS - 1, len(knots))
    x = np.arange(SUFFIX_STEPS)
    return np.stack([np.interp(x, xp, knots[:, d]) for d in range(knots.shape[1])], axis=1).astype(np.float32)


def _rollout_suffix(env, cf, inner, qpos, qvel, actions, prefix_touched=False):
    """Restore the grasp state, run the open-loop residual suffix via env.step; return the strict-lexo objective.
    ``robot_touched`` is initialised from the real prefix contact and UPDATED from actual suffix contacts (via
    _cert_step) — NOT forced — so the search certificate matches the from-neutral replay certificate."""
    import mujoco

    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step
    inner.data.qpos[:] = qpos
    inner.data.qvel[:] = qvel
    mujoco.mj_forward(inner.model, inner.data)
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    cert = DeliveryCertifier(initial_clearance=0.05)
    cert.robot_touched = bool(prefix_touched)                # honest: carry the real prefix attribution
    cf._obs(np.zeros(4, np.float32))
    best_dwell = 0
    min_dtz = 9.0
    min_speed = 9.0
    for t in range(SUFFIX_STEPS):
        cert.update(_cert_step(inner, cf))
        best_dwell = max(best_dwell, cert.delivery_dwell)
        m = inner._planar_metrics
        min_dtz = min(min_dtz, float(m.disk_to_zone))
        sp = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        min_speed = min(min_speed, sp)
        if cert.delivery_certified:
            return {"strict": True, "dwell": 6, "min_dtz": min_dtz, "min_speed": min_speed, "steps": t + 1,
                    "effort": float(np.mean(np.abs(actions[:t + 1])))}
        env.step(np.asarray(actions[t], np.float32))
        if not np.all(np.isfinite(inner.data.qvel)):
            return {"strict": False, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": 9.0, "nan": True,
                    "effort": 9.0}
    return {"strict": False, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": min_speed,
            "effort": float(np.mean(np.abs(actions)))}


def _lexo_key(r):
    # rank: strict → dwell → -min_dtz → -min_speed → -effort  (higher is better)
    return (int(r["strict"]), r["dwell"], -r["min_dtz"], -r.get("min_speed", 9.0), -r.get("effort", 9.0))


def _handoff_nominal(env, cf, inner, qpos, qvel):
    """The frozen handoff's residual sequence from the grasp state (a CEM seed / retrieval baseline)."""
    import mujoco

    from hymeko_rl.coin_delivery.full_action_dataset import _handoff_transport
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    tfn = _greedy_action_fn(_handoff_transport())
    inner.data.qpos[:] = qpos
    inner.data.qvel[:] = qvel
    mujoco.mj_forward(inner.model, inner.data)
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    o = cf._obs(np.zeros(4, np.float32))
    seq = []
    for _t in range(SUFFIX_STEPS):
        a = np.asarray(tfn(env, o, None), np.float32)
        seq.append(a[:ACT_DIM].copy())
        o = env.step(a)[0]
    seq = np.asarray(seq)
    idx = np.linspace(0, len(seq) - 1, N_KNOTS).astype(int)
    return seq[idx]                                          # (N_KNOTS, ACT_DIM) handoff knots


def cem_search_state(seed, *, pop=48, iters=12, elite=8, sigma0=0.7, budget_mult=1, log=None):
    """CEM over the suffix knots for one BC-grasp state. Returns the best result + the accepted (strict) suffix."""
    env, cf, inner = _env()
    bc = _bc()
    qpos, qvel, grasped, touched = _roll_bc_to_grasp(env, cf, inner, bc, seed)   # transition state (always valid)
    nominal = _handoff_nominal(env, cf, inner, qpos, qvel)   # (N_KNOTS, ACT_DIM)
    mean = nominal.flatten().copy()
    sigma = np.full_like(mean, sigma0)
    best = {"strict": False, "dwell": 0, "min_dtz": 9.0}
    best_knots = None
    rng = np.random.default_rng(seed)
    rollouts = 0
    for it in range(iters * budget_mult):
        cand = rng.normal(mean, sigma, size=(pop, mean.size)).astype(np.float32)
        cand[0] = mean                                       # keep the current mean (handoff-seeded on it 0)
        scored = []
        for c in cand:
            r = _rollout_suffix(env, cf, inner, qpos, qvel, _knots_to_actions(c.reshape(N_KNOTS, ACT_DIM)),
                                prefix_touched=touched)
            rollouts += 1
            scored.append((_lexo_key(r), c, r))
        scored.sort(key=lambda x: x[0])
        top = scored[-elite:]
        el = np.stack([c for _k, c, _r in top])
        mean, sigma = el.mean(0), el.std(0) + 0.05
        bk, bc_knots, br = scored[-1]
        if _lexo_key(br) > _lexo_key(best):
            best, best_knots = br, bc_knots.copy()
        if log:
            log(f"seed {seed} it {it + 1}: strict={best['strict']} dwell={best['dwell']} "
                f"min_dtz={best['min_dtz']:.4f} rollouts={rollouts}")
        if best["strict"]:
            break
    cls = ("SEARCH_SUCCESS" if best["strict"]
           else "HORIZON_LIMIT" if best.get("min_dtz", 9) < 0.02
           else "CONTACT_STATE_UNRECOVERABLE_WITH_TESTED_SEARCH" if best["dwell"] == 0
           else "SEARCH_BUDGET_EXHAUSTED")
    return {"seed": int(seed), "grasped": bool(grasped), "strict": bool(best["strict"]), "best_dwell": int(best["dwell"]),
            "min_dtz": float(best["min_dtz"]), "min_speed": float(best.get("min_speed", 9.0)),
            "rollouts": rollouts, "classification": cls,
            "suffix_knots": best_knots.reshape(N_KNOTS, ACT_DIM).tolist() if (best["strict"] and best_knots is not None) else None}


def _run_one(args_tuple):
    seed, pop, iters, elite, budget_mult, logdir = args_tuple
    t0 = time.time()
    logf = open(Path(logdir) / f"search_{seed}.log", "a")
    try:
        res = cem_search_state(seed, pop=pop, iters=iters, elite=elite, budget_mult=budget_mult,
                               log=lambda m: (logf.write(m + "\n"), logf.flush()))
        res["wall_s"] = round(time.time() - t0, 1)
        return res
    finally:
        logf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["headline", "heldout"], default="headline")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--budget_mult", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    seeds = HEADLINE if args.stage == "headline" else HELDOUT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "logs"
    logdir.mkdir(exist_ok=True)
    done_path = out / f"{args.stage}_results.jsonl"
    done = {json.loads(ln)["seed"] for ln in done_path.read_text().splitlines()} if done_path.exists() else set()
    todo = [s for s in seeds if s not in done]
    print(f"[{args.stage}] bundle {BUNDLE_HASH} | {len(todo)}/{len(seeds)} states to search | workers {args.workers}",
          flush=True)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import multiprocessing as mp
    tasks = [(s, args.pop, args.iters, args.elite, args.budget_mult, str(logdir)) for s in todo]
    with mp.Pool(args.workers) as pool, open(done_path, "a") as jf:
        for res in pool.imap_unordered(_run_one, tasks):
            jf.write(json.dumps(res) + "\n")
            jf.flush()
            print(f"  seed {res['seed']}: {res['classification']} strict={res.get('strict')} "
                  f"dwell={res.get('best_dwell')} wall={res.get('wall_s')}s", flush=True)
    allres = [json.loads(ln) for ln in done_path.read_text().splitlines()]
    n_success = sum(r["strict"] for r in allres)
    print(f"[{args.stage}] SEARCH DONE: {n_success}/{len(seeds)} strict-K6 suffixes found", flush=True)
    (out / f"{args.stage}_summary.json").write_text(json.dumps(
        {"stage": args.stage, "bundle_hash": BUNDLE_HASH, "n_states": len(seeds), "n_success": n_success,
         "results": allres}, indent=1))


if __name__ == "__main__":
    main()
