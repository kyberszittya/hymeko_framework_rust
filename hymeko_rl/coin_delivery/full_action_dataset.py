"""CANONICAL FULL-ACTION expert dataset (2026-07-22, v3 learning §2–§3) — roll out the frozen composed learned chain
(E_valselect approach → handoff transport) on the canonical v3 robot and record the ACTUAL executed actuator command
at every step, from true neutral, keeping only complete physically-executed successful trajectories.

Contract: obs = flattened ``node_features`` (48), target = ``u_expert_executed`` = ``inner.data.ctrl`` (the 4 arm
actuators) — NOT a residual, not a scripted delta, not reconstructed post-hoc. All motion is through ``env.step`` /
``inner.step``; the strict K=6 certificate decides success. The deployed BC will run ``u = policy(node_features)`` with
no expert online. Train / validation / headline seed pools are disjoint.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

_HANDOFF = "experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt"
# disjoint seed pools (headline test = coin_neutral_start._HEADLINE)
TRAIN_SEEDS = tuple(range(2000, 2200))
VAL_SEEDS = tuple(range(3000, 3040))


def _handoff_transport():
    from hymeko_rl.train.sac import build_sac
    tr, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    tr.load_state_dict(torch.load(_HANDOFF, weights_only=True))
    tr.eval()
    return tr


def collect_full_action_demos(seeds, *, grasp_hold: int = 3) -> list[dict]:
    """Roll the composed chain per seed; return one record per seed with obs/action/phase/contact traces + delivered.

    Each ``(obs, act)`` pair is aligned: obs is ``node_features`` BEFORE the step, act is the executed ``ctrl`` AFTER.
    """
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, _e_approach_actor, neutral_env
    e, tfn = _e_approach_actor(), _greedy_action_fn(_handoff_transport())
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    out: list[dict] = []
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        obs: list[np.ndarray] = []
        act: list[np.ndarray] = []
        phase: list[int] = []
        contact: list[int] = []
        bi = 0
        for _k in range(160):                                    # phase 0: learned E approach
            cert.update(_cert_step(inner, cf))
            m = inner._planar_metrics
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            if bi >= grasp_hold:
                break
            nf_graph = np.asarray(inner.node_features(), np.float32)      # (6,8) for the E-approach actor
            a = e.action_mean(torch.as_tensor(nf_graph[None]))[0].detach().numpy()
            inner.step(np.asarray(a, np.float32))
            obs.append(nf_graph.flatten())                               # store flat 48 as the BC input
            act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
            phase.append(0)
            contact.append(int(m.left_contact) + int(m.right_contact))
        cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
        cf._t = 0
        cf._both_hist = []
        env._suffix_t = 0
        env._prev_dtz = env._dtz()
        env._prev_both = env._both()
        o = cf._obs(np.zeros(4, np.float32))
        for _t in range(200):                                    # phase 1: learned transport → strict K=6
            cert.update(_cert_step(inner, cf))
            if cert.delivery_certified:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
            obs.append(nf)
            act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
            m = inner._planar_metrics
            phase.append(1)
            contact.append(int(m.left_contact) + int(m.right_contact))
        out.append({"seed": int(s), "obs": np.asarray(obs, np.float32), "act": np.asarray(act, np.float32),
                    "phase": np.asarray(phase, np.int8), "contact": np.asarray(contact, np.int8),
                    "delivered": bool(cert.delivery_certified), "steps": len(act)})
    return out


def build_dataset(seeds, *, successful_only: bool = True) -> dict:
    """Concatenate demos into a flat (N, 48) obs / (N, 4) act dataset with per-sample phase + trajectory id."""
    demos = collect_full_action_demos(seeds)
    kept = [d for d in demos if d["delivered"]] if successful_only else demos
    if not kept:
        return {"obs": np.zeros((0, 48), np.float32), "act": np.zeros((0, 4), np.float32),
                "phase": np.zeros((0,), np.int8), "traj": np.zeros((0,), np.int32),
                "n_traj": 0, "n_seeds": len(list(seeds)), "delivered_seeds": []}
    obs = np.concatenate([d["obs"] for d in kept])
    act = np.concatenate([d["act"] for d in kept])
    phase = np.concatenate([d["phase"] for d in kept])
    traj = np.concatenate([np.full(len(d["act"]), i, np.int32) for i, d in enumerate(kept)])
    return {"obs": obs, "act": act, "phase": phase, "traj": traj, "n_traj": len(kept),
            "n_seeds": len(list(seeds)), "delivered_seeds": [d["seed"] for d in kept],
            "obs_sha16": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()[:16]}


def save_dataset(ds: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=ds["obs"], act=ds["act"], phase=ds["phase"], traj=ds["traj"])
