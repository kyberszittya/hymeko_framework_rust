"""COIN transport/settle failure diagnosis (2026-07-22, v3 expert-strengthening §4) — distinguish the mechanisms
behind FULL_ACTION_BC_NOT_ESTABLISHED_AT_TRANSPORT_SETTLE before rebuilding: weak teacher vs suffix-coverage gap vs
action averaging vs observation insufficiency vs a contact-mechanics ceiling.

Probes (see the report for the verdict): P1 = frozen handoff delivery from EXPERT vs BC grasp states (coverage gap);
P4 = transport obs→action multimodality (averaging / observation insufficiency); P3 = bounded action-sequence search
from the expert grasp state (weak-teacher vs ceiling — a NEGATIVE here is search-budget-limited, NOT proven physical).
"""
from __future__ import annotations

import numpy as np
import torch


def _chain():
    from hymeko_rl.coin_delivery.full_action_dataset import _handoff_transport
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _e_approach_actor, neutral_env
    env, cf = neutral_env(prefix_steps=0)
    return env, cf, cf._env, _e_approach_actor(), _greedy_action_fn(_handoff_transport())


def _to_grasp(env, cf, inner, e, seed, *, bc=None, grasp_hold: int = 3):
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    bi = 0
    for _k in range(160):
        m = inner._planar_metrics
        cert.update(_cert_step(inner, cf))
        bi = bi + 1 if (m.left_contact and m.right_contact) else 0
        if bi >= grasp_hold:
            break
        nf = np.asarray(inner.node_features(), np.float32)
        a = bc.act(nf.flatten()) if bc is not None else e.action_mean(torch.as_tensor(nf[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
    return cert


def p1_handoff_from_grasp(seeds, *, bc=None) -> dict:
    """P1: does the frozen handoff transport deliver strict K=6 from the (expert- or BC-) reached grasp states?"""
    from hymeko_rl.experiments.coin_neutral_start import _cert_step
    env, cf, inner, e, tfn = _chain()
    deliv = 0
    per = []
    for s in seeds:
        cert = _to_grasp(env, cf, inner, e, s, bc=bc)
        cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
        cf._t = 0
        cf._both_hist = []
        env._suffix_t = 0
        env._prev_dtz = env._dtz()
        env._prev_both = env._both()
        o = cf._obs(np.zeros(4, np.float32))
        for _t in range(200):
            cert.update(_cert_step(inner, cf))
            if cert.delivery_certified:
                break
            o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
        deliv += int(cert.delivery_certified)
        per.append((int(s), bool(cert.delivery_certified)))
    return {"deliver": deliv, "n": len(list(seeds)), "per_seed": per}


def p4_transport_action_ambiguity(train_npz: str, *, k: int = 10, samples: int = 200, seed: int = 0) -> dict:
    """P4: neighbourhood action spread for transport observations (low ⇒ unimodal, not an averaging/aliasing problem)."""
    d = np.load(train_npz)
    to, ta = d["obs"][d["phase"] == 1], d["act"][d["phase"] == 1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(to), min(samples, len(to)), replace=False)
    spreads = []
    for i in idx:
        near = np.argsort(np.linalg.norm(to - to[i], axis=1))[:k]
        spreads.append(float(np.mean(np.std(ta[near], axis=0))))
    return {"nbhd_action_spread": float(np.mean(spreads)), "global_action_std": float(ta.std()),
            "ratio": float(np.mean(spreads) / (ta.std() + 1e-9))}
