"""Replay-certified FULL-trajectory dataset generator for the strengthened v3 teacher (§2-§3).

For each seed, roll the COMPLETE neutral→K=6 trajectory (E-approach → transport) with the best-of teacher — the frozen
handoff if it delivers, else a CEM-searched residual suffix — recording every step (obs, executed action, runtime
phase, contact, dwell, robot_touched, strict). The recorded trajectory IS the from-neutral replay (no injection); it
is admissible only if it reaches natural strict K=6. Phase labels are DERIVED FROM RUNTIME PREDICATES (§3), not
timestep ranges. Parallel over seeds, checkpointed.

Usage: python coin_v3_dataset_gen.py --bank train_query --workers 14 --out DATASET_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

CENTER_TOL = 0.02
ZONE_HALF = 0.04
SETTLE_VEL = 0.06
SUFFIX_STEPS = 200
ACT_DIM = 6

# phase codes (§3) — runtime-predicate derived
PHASES = ["APPROACH", "CONTACT_ACQUISITION", "BILATERAL_OR_STABLE_CONTACT", "TRANSPORT",
          "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]


def _phase(inner, dwell, touched_ever, start_dtz):
    m = inner._planar_metrics
    dtz = float(m.disk_to_zone)
    bilateral = bool(m.left_contact and m.right_contact)
    any_c = bool(m.left_contact or m.right_contact)
    if dwell > 0:
        return 6                                             # STRICT_DWELL
    if dtz <= CENTER_TOL:
        return 5                                             # SETTLING (centered, not yet dwelling)
    if dtz <= ZONE_HALF:
        return 4                                             # TARGET_ENTRY (in zone, not centered)
    if touched_ever and dtz < start_dtz - 0.01:
        return 3                                             # TRANSPORT (carrying toward zone)
    if bilateral:
        return 2                                             # BILATERAL_OR_STABLE_CONTACT
    if any_c or touched_ever:
        return 1                                             # CONTACT_ACQUISITION
    return 0                                                 # APPROACH


def _e_and_handoff():
    from hymeko_rl.coin_delivery.full_action_dataset import _handoff_transport
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _e_approach_actor
    return _e_approach_actor(), _greedy_action_fn(_handoff_transport())


def _record_full_trajectory(seed, transport):
    """Roll neutral→(E-approach)→transport, recording per-step. ``transport`` is either the handoff action_fn or an
    open-loop suffix (SUFFIX_STEPS, ACT_DIM). Returns the trajectory dict; ``delivered`` iff natural strict K=6."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    e, _tfn = _e_and_handoff()
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    obs, act, ph, contact, dwell_h, touched_h, strict_h = [], [], [], [], [], [], []
    touched = False
    start_dtz = float(inner._planar_metrics.disk_to_zone)
    bi = 0
    # --- APPROACH (E-approach to transition) ---
    for _k in range(160):
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        touched = touched or bool(m.left_contact or m.right_contact)
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        if bi >= 3:
            break
        nf = np.asarray(inner.node_features(), np.float32)
        a = e.action_mean(torch.as_tensor(nf[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
        obs.append(nf.flatten())
        act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
        ph.append(_phase(inner, cert.delivery_dwell, touched, start_dtz))
        contact.append(int(m.left_contact) + int(m.right_contact))
        dwell_h.append(int(cert.delivery_dwell))
        touched_h.append(int(touched))
        strict_h.append(0)
    # --- TRANSPORT ---
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    o = cf._obs(np.zeros(4, np.float32))
    delivered = False
    suffix_actions = None if callable(transport) else _knots_to_actions(transport)
    for t in range(SUFFIX_STEPS):
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        touched = touched or bool(m.left_contact or m.right_contact)
        if cert.delivery_certified:
            delivered = True
            break
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        raw = np.asarray(transport(env, o, None) if callable(transport) else suffix_actions[t], np.float32)
        o = env.step(raw)[0]
        obs.append(nf)
        act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
        ph.append(_phase(inner, cert.delivery_dwell, touched, start_dtz))
        contact.append(int(m.left_contact) + int(m.right_contact))
        dwell_h.append(int(cert.delivery_dwell))
        touched_h.append(int(touched))
        strict_h.append(int(cert.delivery_dwell > 0))
    return {"seed": int(seed), "obs": np.asarray(obs, np.float32), "act": np.asarray(act, np.float32),
            "phase": np.asarray(ph, np.int8), "contact": np.asarray(contact, np.int8),
            "dwell": np.asarray(dwell_h, np.int16), "touched": np.asarray(touched_h, np.int8),
            "strict": np.asarray(strict_h, np.int8), "delivered": bool(delivered), "steps": len(act)}


def _knots_to_actions(knots):
    knots = np.asarray(knots, np.float32)
    xp = np.linspace(0, SUFFIX_STEPS - 1, len(knots))
    x = np.arange(SUFFIX_STEPS)
    return np.stack([np.interp(x, xp, knots[:, d]) for d in range(knots.shape[1])], axis=1).astype(np.float32)


def certified_trajectory_for_seed(seed):
    """Best-of teacher: handoff first (fast); else CEM-search a suffix. Return a REPLAY-CERTIFIED full trajectory or
    a NO-DELIVERY record. Everything through env.step, no injection in the recorded trajectory."""
    _e, tfn = _e_and_handoff()
    tr = _record_full_trajectory(seed, tfn)                  # handoff-first
    if tr["delivered"]:
        tr["teacher"] = "handoff"
        return tr
    from coin_v3_suffix_search import cem_search_state       # search fallback
    sr = cem_search_state(seed, pop=64, iters=25, elite=10)
    if sr.get("strict") and sr.get("suffix_knots"):
        tr2 = _record_full_trajectory(seed, sr["suffix_knots"])   # re-roll from neutral (= replay cert)
        if tr2["delivered"]:
            tr2["teacher"] = "search"
            return tr2
    tr["teacher"] = "none"
    tr["classification"] = sr.get("classification", "NO_HANDOFF")
    return tr


def _run_one(args_tuple):
    seed, outdir = args_tuple
    tr = certified_trajectory_for_seed(seed)
    if tr["delivered"]:
        p = Path(outdir) / f"traj_{seed}.npz"
        np.savez_compressed(p, obs=tr["obs"], act=tr["act"], phase=tr["phase"], contact=tr["contact"],
                            dwell=tr["dwell"], touched=tr["touched"], strict=tr["strict"])
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    else:
        sha = None
    return {"seed": int(seed), "delivered": tr["delivered"], "teacher": tr.get("teacher"), "steps": tr["steps"],
            "sha16": sha, "classification": tr.get("classification")}


def main():
    from hymeko_rl.coin_delivery import coin_v3_seed_banks as sb
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = {"train_query": sb.TRAIN_QUERY, "validation": sb.VALIDATION, "final_test": sb.FINAL_TEST,
             "headline": sb.HEADLINE, "coverage_v1": sb.COVERAGE_PANEL_V1}[args.bank]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "index.jsonl"
    done = {json.loads(ln)["seed"] for ln in done_path.read_text().splitlines()} if done_path.exists() else set()
    todo = [s for s in seeds if s not in done]
    print(f"[{args.bank}] bundle {sb.BUNDLE_HASH} | {len(todo)}/{len(seeds)} trajectories to generate", flush=True)
    torch.set_num_threads(1)
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool, open(done_path, "a") as jf:
        for r in pool.imap_unordered(_run_one, [(s, str(out)) for s in todo]):
            jf.write(json.dumps(r) + "\n")
            jf.flush()
            print(f"  seed {r['seed']}: delivered={r['delivered']} teacher={r['teacher']} steps={r['steps']}",
                  flush=True)
    idx = [json.loads(ln) for ln in done_path.read_text().splitlines()]
    n_cert = sum(r["delivered"] for r in idx)
    by_teacher = {t: sum(r["teacher"] == t for r in idx if r["delivered"]) for t in ("handoff", "search")}
    (out / "dataset_manifest.json").write_text(json.dumps(
        {"bank": args.bank, "bundle_hash": sb.BUNDLE_HASH, "semantic_graph_fp": sb.SEMANTIC_GRAPH_FP,
         "obs_contract": sb.OBS_CONTRACT, "action_contract": sb.ACTION_CONTRACT, "n_states": len(seeds),
         "n_certified": n_cert, "by_teacher": by_teacher, "index": idx}, indent=1))
    print(f"[{args.bank}] DATASET DONE: {n_cert}/{len(seeds)} certified full trajectories ({by_teacher})", flush=True)


if __name__ == "__main__":
    main()
