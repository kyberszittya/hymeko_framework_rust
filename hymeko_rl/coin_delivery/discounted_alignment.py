"""COIN discounted reward alignment (Option B §9) — the OUTER canonical K=6 Coin env's v3 reward must rank the
DISCOUNTED return of strict K=6 delivery above every non-success behavior class, with no repeatable non-success loop
able to farm a higher finite-horizon return.

Faithfulness contract (directive §1): every reward is the value returned by :class:`CoinDeliveryTrainEnv.step`
(``inner.reward_spec.evaluate`` on the v3 spec, under the canonical K=6 strict predicate + terminal). No reward is
re-derived. The strict-delivery REFERENCE is a controlled demonstration that reaches the canonical strict state
(centered ``dtz<=center_tol`` ∧ settled ``speed<settle_vel`` ∧ robot-attributed) and holds it for the K=6 dwell — it
measures the reward a delivery EARNS, independent of whether a learned policy can reach it (the transport env is
contact-mechanics-limited; §3 forbids classifying a frozen policy's non-delivery as reward misalignment).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig

Action = np.ndarray
Controller = Callable[[object, int], Action]

_HORIZON_CAP = 120                       # matched to DeliveryRLConfig.horizon (recorded in the manifest)


# ── bundle provenance ───────────────────────────────────────────────────────────────────────────────────────────
def _sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def bundle_hashes(cfg: DeliveryRLConfig) -> dict:
    """The exact bundle identity every discounted result is bound to (directive §1)."""
    from hymeko_rl.env.planar_grasp_env import _CANONICAL_ROBOT_V2, _PLANAR_ENV, read_control_contract
    reward = cfg.canonical_reward_file
    parts = {
        "reward_spec": _sha16(reward), "robot_spec": _sha16(_CANONICAL_ROBOT_V2),
        "scene_spec": _sha16(_PLANAR_ENV), "control_contract": read_control_contract(),
        "held_dwell_steps": cfg.held_dwell_steps, "center_tol": cfg.center_tol, "settle_vel": cfg.settle_vel,
        "horizon": cfg.horizon, "reward_source": cfg.reward_source, "success_contract": cfg.success_contract,
    }
    combined = hashlib.sha256(repr(sorted(parts.items(), key=lambda kv: kv[0])).encode()).hexdigest()[:16]
    return {**parts, "combined_bundle_hash": combined}


def resolve_gammas() -> dict[str, float]:
    """Read γ from the actual SAC and TD3 configs (directive §2 — not assumed from memory)."""
    from hymeko_rl.train.ddpg import OffPolicyConfig
    from hymeko_rl.train.sac import SACConfig
    return {"sac": float(SACConfig().gamma), "td3": float(OffPolicyConfig().gamma)}


# ── rollout (consumes the OUTER env reward) ─────────────────────────────────────────────────────────────────────
def _rollout(env, controller: Controller, *, seed: int, horizon: int) -> dict:
    """Roll ``controller`` (a 6-DoF suffix action law) through the OUTER env; collect per-step reward + status."""
    env._base_override = lambda inner, t: np.asarray(controller(inner, t), np.float32)
    try:
        env.reset(seed=seed)
        rewards: list[float] = []
        terminated = truncated = False
        for _ in range(horizon):
            _o, r, terminated, truncated, _info = env.step(np.zeros(6, np.float32))  # raw=0 → a_exec = controller
            rewards.append(float(r))
            if terminated or truncated:
                break
        return {"rewards": rewards, "steps": len(rewards), "terminated": bool(terminated),
                "truncated": bool(truncated), "final_dwell": int(env._strict_dwell)}
    finally:
        env._base_override = None


def strict_delivery_reference(env, *, seed: int) -> dict:
    """Deterministic strict-delivery reference: latch robot-attribution via the acquisition grasp, then place the
    coin at the zone at rest and hold — the canonical strict state (centered ∧ settled ∧ touched) for the K=6 dwell,
    firing the v3 terminal. Every reward is the OUTER env's ``step`` reward. Physically valid over the held window
    (coin at rest, dynamics respected); the placement is the controlled demonstration that reaches the state."""
    inner = env.inner
    dxa = inner._disk_x_adr
    env._base_override = lambda i, t: np.array([0, 0, 0, 0.3, 0, 0], np.float32)   # gentle grasp hold
    try:
        env.reset(seed=seed)
        rewards: list[float] = []
        _o, r0, term, trunc, _ = env.step(np.zeros(6, np.float32))                 # 1 step: latch _robot_touched
        rewards.append(float(r0))
        inner.data.qpos[:4] = [1.5, -1.5, -1.5, 1.5]                               # retract arms clear of the zone
        inner.data.qpos[dxa:dxa + 2] = [inner._zone_x, inner._zone_y]              # coin at the zone centre, at rest
        inner.data.qvel[:] = 0.0
        mujoco.mj_forward(inner.model, inner.data)
        env._base_override = lambda i, t: np.zeros(6, np.float32)                  # no disturbance → coin stays settled
        for _ in range(env.cfg.held_dwell_steps + 2):
            _o, r, term, trunc, _ = env.step(np.zeros(6, np.float32))
            rewards.append(float(r))
            if term or trunc:
                break
        return {"rewards": rewards, "steps": len(rewards), "terminated": bool(term),
                "truncated": bool(trunc), "final_dwell": int(env._strict_dwell)}
    finally:
        env._base_override = None


# ── behavior corpus (non-success suffix controllers) ────────────────────────────────────────────────────────────
def _toward(inner, gain: float = 1.0, sq: float = 0.8) -> Action:
    d, _n = inner.direction_to_zone()
    return np.array([d[0] * gain, d[1] * gain, 0.0, sq, 0.0, 0.0], np.float32)


def _away(inner, gain: float = 1.0, sq: float = 0.8) -> Action:
    d, _n = inner.direction_to_zone()
    return np.array([-d[0] * gain, -d[1] * gain, 0.0, sq, 0.0, 0.0], np.float32)


FAILURE_CONTROLLERS: dict[str, Controller] = {
    "zero_action":        lambda inner, t: np.zeros(6, np.float32),
    "movement_away":      lambda inner, t: _away(inner),
    "reach_without_hold": lambda inner, t: _toward(inner, sq=0.0),                     # reach, never grasp/hold
    "grasp_and_stall":    lambda inner, t: np.array([0, 0, 0, 0.9, 0, 0], np.float32),  # squeeze, no transport
    "oscillation":        lambda inner, t: (_toward(inner) if (t // 3) % 2 == 0 else _away(inner)),
    "center_then_exit":   lambda inner, t: (_toward(inner) if t < 30 else _away(inner)),
    "timeout":            lambda inner, t: np.array([0, 0, 0, 0.3, 0, 0], np.float32),  # minimal motion to the cap
    "contact_drop":       lambda inner, t: (_toward(inner) if t < 20 else _toward(inner, sq=0.0)),
    "approach_retreat":   lambda inner, t: (_toward(inner) if (t // 20) % 2 == 0 else _away(inner)),
    "zone_entry_exit":    lambda inner, t: (_toward(inner, gain=1.5) if (t // 8) % 2 == 0 else _away(inner, gain=1.5)),
}
# the loops whose one full cycle we probe for farming (directive §7): (name, cycle_len)
FARMING_LOOPS = {"oscillation": 6, "approach_retreat": 40, "zone_entry_exit": 16}


def discounted_return(rewards: list[float], gamma: float) -> float:
    return float(sum((gamma ** t) * r for t, r in enumerate(rewards)))


def cycle_upper_bound(cycle_rewards: list[float], gamma: float) -> float:
    """Infinite repeated-cycle discounted upper bound Σ_k γ^{kL}·(discounted cycle) = cycle / (1 - γ^L)."""
    L = len(cycle_rewards)
    cyc = discounted_return(cycle_rewards, gamma)
    return float(cyc / (1.0 - gamma ** L)) if L > 0 and gamma ** L < 1.0 else float("inf")
