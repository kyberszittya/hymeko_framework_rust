"""Replay-certification of searched suffixes from the ORIGINAL neutral reset (v3 expert-strengthening §9 CERT-B).

For each accepted (strict) suffix from the search, reset with the original seed, re-run the DETERMINISTIC BC to its
grasp state (no injection), then apply the searched open-loop residual suffix — all through env.step. Require natural
strict K=6 termination. This is the independent certification (run on KATO15) that a candidate is a genuine
neutral-to-K6 trajectory, not a snapshot-only success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
N_KNOTS = 10
SUFFIX_STEPS = 200
ACT_DIM = 6


def _knots_to_actions(knots):
    knots = np.asarray(knots, np.float32)
    xp = np.linspace(0, SUFFIX_STEPS - 1, len(knots))
    x = np.arange(SUFFIX_STEPS)
    return np.stack([np.interp(x, xp, knots[:, d]) for d in range(knots.shape[1])], axis=1).astype(np.float32)


def replay_certify(seed, suffix_knots) -> dict:
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, _e_approach_actor, neutral_env
    e = _e_approach_actor()
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    bi = 0
    grasped = False
    for _k in range(160):                                    # frozen E approach drives from neutral (NO injection)
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        if bi >= 3:
            grasped = True
            break
        a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    actions = _knots_to_actions(suffix_knots)
    cf._obs(np.zeros(4, np.float32))
    delivered = False
    for t in range(SUFFIX_STEPS):
        cert.update(_cert_step(inner, cf))
        if cert.delivery_certified:
            delivered = True
            break
        env.step(np.asarray(actions[t], np.float32))
        if not np.all(np.isfinite(inner.data.qvel)):
            break
    return {"seed": int(seed), "grasped_from_neutral": bool(grasped),
            "replay_strict_delivery": bool(delivered and grasped), "robot_touched": bool(cert.robot_touched)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    summ = json.loads(Path(args.search_summary).read_text())
    accepted = [r for r in summ["results"] if r.get("strict") and r.get("suffix_knots")]
    print(f"replay-certifying {len(accepted)} search-accepted suffixes (bundle {summ.get('bundle_hash')})", flush=True)
    certs = []
    for r in accepted:
        c = replay_certify(r["seed"], r["suffix_knots"])
        certs.append(c)
        print(f"  seed {c['seed']}: replay_strict={c['replay_strict_delivery']} grasp={c['grasped_from_neutral']}",
              flush=True)
    n_cert = sum(c["replay_strict_delivery"] for c in certs)
    Path(args.out).write_text(json.dumps({"bundle_hash": summ.get("bundle_hash"), "n_accepted": len(accepted),
                                          "n_replay_certified": n_cert, "certs": certs}, indent=1))
    print(f"REPLAY-CERTIFIED {n_cert}/{len(accepted)} from neutral", flush=True)


if __name__ == "__main__":
    main()
