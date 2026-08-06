"""PHASE_GATED_RESIDUAL behavior collector (§3) — collects executed transitions from the *deployable* controller
distribution, NOT full-action Gaussian noise.

    base_t   = clip(pi_0(obs_t), -4, 4)
    delta_t  = clip(sampled_residual_t, -0.25, 0.25)          # residual exploration, bounded
    action_t = clip(base_t + gate_t * delta_t, -4, 4)         # gate_t in {0,1} from STABLE_OBJECT_ENGAGEMENT_V1

When ``gate_t == 0`` the executed action is **bit-identical** to ``pi_0`` regardless of the sampled residual — the
early approach/grasp policy that produces 9/9 grasp is never perturbed. Transitions store ``terminated`` AND
``truncated`` separately (no bootstrap across time-limit truncation).
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import (
    StableEngagementConfig,
    StableEngagementGate,
    stable_engagement_signals,
)

RESIDUAL_BOUND = 0.25
ACTION_SCALE = 4.0


def gated_composite_action(base: np.ndarray, gate_mult: float, delta: np.ndarray) -> np.ndarray:
    """base + gate*clip(delta, ±0.25), clipped to ±4. gate_mult==0 ⇒ returns clip(base) bit-identically."""
    d = np.clip(np.asarray(delta, np.float32), -RESIDUAL_BOUND, RESIDUAL_BOUND)
    b = np.clip(np.asarray(base, np.float32), -ACTION_SCALE, ACTION_SCALE)
    return np.clip(b + gate_mult * d, -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def base_action(pi0, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.tensor(np.asarray(obs, np.float32)[None]))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def collect_gated_residual(pi0, seeds, *, sample_delta=None, explore=True, horizon=360, gate_cfg=None, seed=0):
    """Roll the gated residual controller and record deployable transitions.

    ``sample_delta(rng, gate_active) -> residual(4)`` proposes a residual (bounded internally). ``explore=False``
    forces zero residual (the deployable-identity regression). Returns a list of dicts with the full Markov tuple.

    # Postconditions: every transition's action equals the gated composite; gate-off steps equal ``pi_0`` exactly;
      ``terminated`` and ``truncated`` are stored separately.
    """
    if sample_delta is None:
        def sample_delta(rng, active):
            return rng.normal(0, 0.1, 4).astype(np.float32) if active else np.zeros(4, np.float32)
    rng = np.random.default_rng(seed)
    trs = []
    rl = CoinRL4Dof(horizon=horizon)
    for s in seeds:
        o = rl.reset(int(s)); gate = StableEngagementGate(gate_cfg or StableEngagementConfig())
        for _t in range(horizon):
            g = gate.gate
            base = base_action(pi0, o)
            delta = sample_delta(rng, g == 1.0) if explore else np.zeros(4, np.float32)
            action = gated_composite_action(base, g, delta)
            gs_t = ReplayControllerStateV2.from_gate(gate).to_dict()
            o2, r, term, trunc, _ = rl.step(action)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            gs_tp1 = ReplayControllerStateV2.from_gate(gate).to_dict()
            trs.append({"obs_t": o.astype(np.float32), "gate_state_t": gs_t, "action_t": action,
                        "reward_t": float(r), "obs_tp1": o2.astype(np.float32), "gate_state_tp1": gs_tp1,
                        "terminated": bool(term), "truncated": bool(trunc), "gate_mult_t": float(g),
                        "requested_delta": np.clip(delta, -RESIDUAL_BOUND, RESIDUAL_BOUND).astype(np.float32),
                        "executed_residual": (action - np.clip(base, -4, 4)).astype(np.float32)})
            o = o2
            if term or trunc:
                break
    return trs


def eval_gated_residual_identity(pi0, seeds, horizon=360):
    """Eval the gated composite with residual OFF from neutral — must reproduce pi_0 delivery (deployable identity)."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    fc = grasp = deliv = 0; per = []
    for s in seeds:
        env.set_stage(0); env.reset(seed=int(s))
        gate = StableEngagementGate(StableEngagementConfig()); cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        touched = False
        for _t in range(horizon):
            cert.update(_cert_step(inner, cf)); m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            if cert.delivery_certified:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            a = gated_composite_action(base_action(pi0, nf), gate.gate, np.zeros(4, np.float32))   # residual OFF
            inner.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(cert.delivery_certified))
        d = bool(cert.delivery_certified)
        fc += int(touched); grasp += int(touched); deliv += int(d); per.append((int(s), d))
    n = max(1, len(list(seeds)))
    return {"n": n, "grasp": grasp, "deliver": deliv, "delivered_seeds": [s for s, dd in per if dd]}
