"""§8-10 success-certified DAgger on the deployed BC's OWN reached states.

The initial phase-balanced BC clones accurately but delivers ~1/9: it drives the coin partway then it rolls out of the
strict zone (covariate shift in transport→settle, because 57/96 base demos are open-loop CEM suffixes — feedforward,
not feedback). DAgger closes the gap by querying corrective labels ON THE STATES THE BC ACTUALLY REACHES:

  1. Roll the CURRENT BC from neutral; if it already delivers (strict K=6), skip — NOT a divergence (§8).
  2. Otherwise snapshot the BC's transition state and CEM-search a strict-K=6 suffix FROM THAT reached state
     (``cem_from_snapshot`` — the same certified search, re-seeded on the BC's distribution, not the E-approach's).
  3. Replay-certify: re-roll BC-approach + suffix FROM NEUTRAL (no injection) and require natural strict K=6.
  4. Record the full certified trajectory (obs=node_features flat 48, executed ctrl 4, runtime phase) as new labels.

The recorded labels live on the BC's own state distribution — the standard DAgger fix for compounding error. Iterate
(retrain phase-balanced on base ∪ DAgger labels → re-query) until headline ≥6/9 or the stop criterion.

Usage: python coin_v3_dagger.py --bc BC.pt --bank train_query --workers 14 --out DAGGER_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coin_v3_dataset_gen import _phase  # noqa: E402
from coin_v3_suffix_search import (  # noqa: E402
    ACT_DIM, N_KNOTS, SUFFIX_STEPS, _bc_action_fn, _env, _knots_to_actions, _roll_to_transition, cem_from_snapshot,
)

_BC_PATH = None   # set by main, read by workers (process-local torch load)


def _load_bc(path):
    from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
    bc = FullActionBC()
    bc.load_state_dict(torch.load(path, map_location="cpu"))
    bc.eval()
    return bc


def _bc_roll_full(bc, seed, horizon=360, grasp_hold=3):
    """Roll the BC from neutral to ``horizon``. Snapshot the transition state (bilateral grasp_hold OR the 160-cap) and
    also report whether the BC's OWN continuation delivers (so we skip states that need no correction). One rollout."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance
    env, cf, inner = _env()
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    bi = 0
    touched = False
    snap = None
    grasped = False
    for k in range(horizon):
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        touched = touched or bool(m.left_contact or m.right_contact)
        if snap is None:
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            if bi >= grasp_hold or k == 159:
                snap = (inner.data.qpos.copy(), inner.data.qvel.copy(), touched)
                grasped = bi >= grasp_hold
        if cert.delivery_certified:
            return {"delivered": True, "snap": snap, "grasped": grasped, "touched": touched}
        inner.step(np.asarray(bc.act(np.asarray(inner.node_features(), np.float32).flatten()), np.float32))
    if snap is None:
        snap = (inner.data.qpos.copy(), inner.data.qvel.copy(), touched)
    return {"delivered": False, "snap": snap, "grasped": grasped, "touched": touched}


def _record_certified_bc_traj(seed, bc, suffix_knots):
    """From-neutral replay: BC drives the approach to the transition, then the open-loop certified suffix drives the
    transport; record (obs, executed ctrl, runtime phase) at every step. Admissible iff natural strict K=6 (no
    injection). Returns (obs, act, phase, delivered)."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance
    env, cf, inner = _env()
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    obs, act, ph = [], [], []
    touched = False
    start_dtz = float(inner._planar_metrics.disk_to_zone)
    bi = 0
    for _k in range(160):                                     # BC approach (its own reached states)
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        touched = touched or bool(m.left_contact or m.right_contact)
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        if bi >= 3:
            break
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        inner.step(np.asarray(bc.act(nf), np.float32))
        obs.append(nf)
        act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
        ph.append(_phase(inner, cert.delivery_dwell, touched, start_dtz))
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)   # transition to transport
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    cf._obs(np.zeros(4, np.float32))
    suffix = _knots_to_actions(np.asarray(suffix_knots, np.float32))
    delivered = False
    for t in range(SUFFIX_STEPS):
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        touched = touched or bool(m.left_contact or m.right_contact)
        if cert.delivery_certified:
            delivered = True
            break
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        env.step(np.asarray(suffix[t], np.float32))
        obs.append(nf)
        act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
        ph.append(_phase(inner, cert.delivery_dwell, touched, start_dtz))
    return (np.asarray(obs, np.float32), np.asarray(act, np.float32), np.asarray(ph, np.int8), delivered)


def dagger_query_state(seed, outdir, *, pop=48, iters=12, elite=8):
    """One state: skip if the BC already delivers, else search a certified corrective suffix from the BC's reached
    state and replay-certify + record. Returns a status dict; writes ``dagger_{seed}.npz`` on a certified label."""
    bc = _load_bc(_BC_PATH)
    roll = _bc_roll_full(bc, seed)
    if roll["delivered"]:
        return {"seed": int(seed), "outcome": "BC_ALREADY_DELIVERS"}
    env, cf, inner = _env()
    qpos, qvel, grasped, touched = _roll_to_transition(env, cf, inner, _bc_action_fn(bc), seed)
    res = cem_from_snapshot(seed, env, cf, inner, qpos, qvel, grasped, touched, pop=pop, iters=iters, elite=elite)
    if not res["strict"]:
        return {"seed": int(seed), "outcome": "SEARCH_FAILED", "classification": res["classification"],
                "min_dtz": res["min_dtz"]}
    obs, act, ph, deliv = _record_certified_bc_traj(seed, bc, res["suffix_knots"])
    if not deliv:
        return {"seed": int(seed), "outcome": "REPLAY_NOT_CERTIFIED", "min_dtz": res["min_dtz"]}
    p = Path(outdir) / f"dagger_{seed}.npz"
    np.savez_compressed(p, obs=obs, act=act, phase=ph)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {"seed": int(seed), "outcome": "DAGGER_CERTIFIED", "steps": int(len(act)), "sha16": sha}


def _run_one(args_tuple):
    seed, outdir, pop, iters, elite = args_tuple
    t0 = time.time()
    r = dagger_query_state(seed, outdir, pop=pop, iters=iters, elite=elite)
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main():
    global _BC_PATH
    from hymeko_rl.coin_delivery import coin_v3_seed_banks as sb
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", required=True)
    ap.add_argument("--bank", default="train_query")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _BC_PATH = str(Path(args.bc).resolve())
    seeds = {"train_query": sb.TRAIN_QUERY, "validation": sb.VALIDATION, "headline": sb.HEADLINE}[args.bank]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "dagger_index.jsonl"
    done = {json.loads(ln)["seed"] for ln in done_path.read_text().splitlines()} if done_path.exists() else set()
    todo = [s for s in seeds if s not in done]
    bc_sha = hashlib.sha256(Path(_BC_PATH).read_bytes()).hexdigest()[:16]
    print(f"[dagger {args.bank}] BC {bc_sha} | {len(todo)}/{len(seeds)} states | workers {args.workers}", flush=True)
    torch.set_num_threads(1)
    import multiprocessing as mp
    tasks = [(s, str(out), args.pop, args.iters, args.elite) for s in todo]
    with mp.Pool(args.workers) as pool, open(done_path, "a") as jf:
        for r in pool.imap_unordered(_run_one, tasks):
            jf.write(json.dumps(r) + "\n")
            jf.flush()
            print(f"  seed {r['seed']}: {r['outcome']} steps={r.get('steps')} wall={r.get('wall_s')}s", flush=True)
    idx = [json.loads(ln) for ln in done_path.read_text().splitlines()]
    tally = {}
    for r in idx:
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    (out / "dagger_manifest.json").write_text(json.dumps(
        {"bank": args.bank, "bc_sha16": bc_sha, "bundle_hash": sb.BUNDLE_HASH, "n_states": len(seeds),
         "tally": tally, "search": {"pop": args.pop, "iters": args.iters, "elite": args.elite,
                                    "n_knots": N_KNOTS, "act_dim": ACT_DIM, "suffix_steps": SUFFIX_STEPS},
         "index": idx}, indent=1))
    print(f"[dagger {args.bank}] DAGGER DONE: {tally}", flush=True)


if __name__ == "__main__":
    main()
