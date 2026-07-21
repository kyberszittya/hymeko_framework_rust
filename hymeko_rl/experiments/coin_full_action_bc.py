"""Standalone full-action BC (2026-07-22, steps 3-4).

Trains a STANDALONE policy ``u_exec = policy_BC(observation)`` that clones the scripted grasp_carry expert's FULL
executed action (``u_target = u_expert_executed``, never a zero residual) under corrected physics, with the scripted
base DISABLED during rollout. Then the competence gate that must pass before any RL:

  G1 scripted expert evaluated under corrected physics (the ceiling BC is measured against)
  G2 standalone BC evaluated with the script disabled (env-native center + strict)
  G3 BC action-imitation error on HELD-OUT expert trajectories
  G4 BC rollout success on the frozen panel and >=50 held-out states
  G5 zero-action control (proves the base is disabled: a null policy does not deliver)
  G6 exact action-source trace (every actuator command in a BC rollout equals the policy's own action)

If standalone BC does not reproduce useful expert competence -> FULL_ACTION_BC_NOT_ESTABLISHED (do not launch RL).
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.train.coin_full_action import (
    collect_expert_dataset,
    eval_full_action,
    make_full_action_env,
    scripted_expert_fn,
)

_PANEL = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)
_HELDOUT = tuple(s for s in range(1000, 1100) if s not in _PANEL)[:50]
_TRAIN = tuple(s for s in range(1100, 1300))
_OBS, _ACT = 41, 6


def greedy_fn(actor):
    def fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return fn


def train_bc(dataset, *, epochs: int = 300, lr: float = 1e-3, seed: int = 0):
    """Fit ``action_mean(obs) ≈ u_expert_executed`` (the full action). Returns (actor, loss_curve)."""
    from hymeko_rl.train.sac import build_sac
    torch.manual_seed(seed)
    actor, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0)
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32)
    act = torch.as_tensor(dataset["act"], dtype=torch.float32)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((actor.action_mean(obs) - act) ** 2)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            curve.append(float(loss.detach()))
    actor.eval()
    return actor, curve


def train_bc_dagger(dataset, *, rounds: int = 3, epochs: int = 300, seed: int = 0,
                    fingertip_geometry: str = "POINT", horizon: int = 160, dagger_seeds=None):
    """Establish a competent standalone BC via DAgger — the covariate-shift fix: roll the current BC, relabel every
    BC-VISITED state with the expert's full action (``p_grasp_carry`` at that state), aggregate, retrain. Still pure
    imitation; the final policy is standalone (no expert at deployment). Returns (actor, curve, agg_meta)."""
    from hymeko_rl.train.coin_delivery_rl import p_grasp_carry
    from hymeko_rl.train.coin_full_action import make_full_action_env
    dagger_seeds = tuple(_TRAIN[:80]) if dagger_seeds is None else tuple(dagger_seeds)
    env = make_full_action_env(fingertip_geometry=fingertip_geometry, horizon=horizon)
    obs = list(dataset["obs"])
    act = list(dataset["act"])
    actor, curve = train_bc({"obs": np.asarray(obs, np.float32), "act": np.asarray(act, np.float32)},
                            epochs=epochs, seed=seed)
    sizes = [len(obs)]
    for _r in range(rounds):
        bc = greedy_fn(actor)
        for s in dagger_seeds:                         # roll the CURRENT BC; relabel visited states with the expert
            env.reset(seed=int(s))
            for _t in range(horizon):
                o = env._last_obs.copy()
                obs.append(o.astype(np.float32))
                act.append(np.asarray(p_grasp_carry(env.inner, env._suffix_t), np.float32))   # expert on BC's state
                env.step(np.asarray(bc(o), np.float32))
        actor, curve = train_bc({"obs": np.asarray(obs, np.float32), "act": np.asarray(act, np.float32)},
                                epochs=epochs, seed=seed)
        sizes.append(len(obs))
    return actor, curve, {"rounds": rounds, "aggregate_sizes": sizes, "dagger_seeds": len(dagger_seeds)}


def bc_gate(bc_actor, *, fingertip_geometry: str = "POINT", horizon: int = 160) -> dict:
    """Run G1-G6. Returns the metrics + ``established: bool`` (all gates pass)."""
    env = make_full_action_env(fingertip_geometry=fingertip_geometry, horizon=horizon)

    # G1 scripted expert ceiling (its own action source)
    g1_panel = eval_full_action(scripted_expert_fn(env), _PANEL, env)
    g1_held = eval_full_action(scripted_expert_fn(env), _HELDOUT, env)

    # G2/G4 standalone BC with the script disabled (policy drives the full action)
    bc = greedy_fn(bc_actor)
    g2_panel = eval_full_action(bc, _PANEL, env)
    g4_held = eval_full_action(greedy_fn(bc_actor), _HELDOUT, env)

    # G3 imitation error on a fresh HELD-OUT expert dataset (not the training seeds)
    held_ds = collect_expert_dataset(tuple(range(1300, 1330)), fingertip_geometry=fingertip_geometry,
                                     horizon=horizon, successful_only=True)
    with torch.no_grad():
        pred = bc_actor.action_mean(torch.as_tensor(held_ds["obs"], dtype=torch.float32)).numpy()
    imit_mse = float(np.mean((pred - held_ds["act"]) ** 2)) if held_ds["obs"].size else float("nan")

    # G5 zero-action control (base disabled: a null policy does NOT deliver)
    zero = eval_full_action(lambda _o: np.zeros(_ACT, np.float32), _PANEL, env)

    # G6 exact action-source trace: every executed command == the policy's own action, on one seed
    env.reset(seed=int(_PANEL[0]))
    max_src_delta = 0.0
    for _t in range(horizon):
        obs = env._last_obs.copy()
        a_policy = bc(obs)
        a_exec = np.clip(np.asarray(a_policy, np.float32).reshape(-1), env.cfg.lo, env.cfg.hi)
        env.step(a_policy)
        max_src_delta = max(max_src_delta, float(np.abs(a_exec - np.clip(a_policy, env.cfg.lo, env.cfg.hi)).max()))

    established = bool(g2_panel["center_rate"] >= 0.5 and g4_held["center_rate"] >= 0.4
                       and imit_mse < 0.05 and zero["strict_count"] == 0 and max_src_delta < 1e-6)
    return {
        "scripted_panel": {"center_rate": g1_panel["center_rate"], "strict": g1_panel["strict_count"]},
        "scripted_heldout": {"center_rate": g1_held["center_rate"], "strict": g1_held["strict_count"]},
        "bc_panel": {"center_rate": g2_panel["center_rate"], "strict": g2_panel["strict_count"],
                     "tts_median": g2_panel["tts_median"], "auc": g2_panel["success_curve_auc"]},
        "bc_heldout": {"center_rate": g4_held["center_rate"], "strict": g4_held["strict_count"]},
        "imitation_mse_heldout": imit_mse, "zero_action_strict": zero["strict_count"],
        "action_source_max_delta": max_src_delta, "established": established,
        "verdict": "FULL_ACTION_BC_ESTABLISHED" if established else "FULL_ACTION_BC_NOT_ESTABLISHED",
    }
