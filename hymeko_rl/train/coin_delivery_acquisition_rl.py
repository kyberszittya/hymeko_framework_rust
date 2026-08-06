"""COIN-DELIVERY-OVERNIGHT-2 PART V — acquisition-subtask RL env + reward + BC (the gated RL substrate).

Authorized by the STRICT acquisition gate (12/19 stable acquisitions, geometry load-bearing). This is an ACQUISITION
SUBTASK env: success = stable two-finger acquisition (dwell), reward reflects APPROACH+CONTACT+STABLE (never delivery).
Handoff/contact is a phase event; the chained delivery is measured SEPARATELY (and was 0 — acquisition here has no
downstream task value, so RL success on this env is an acquisition-subtask claim, never a delivery claim).

Reuses train.ppo / train.ddpg (TD3+BC) / train.sac (guarded) unchanged. NO env-dynamics/reward-of-record/CORE change —
the acquisition reward is a NEW subtask reward for this NEW subtask env, not a change to the delivery reward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.experiments.pedc_selection import _env
from hymeko_rl.train.coin_delivery_acquisition import ARM_BODY, MID_TO_COIN, scramble_geometry


@dataclass(frozen=True)
class AcqRewardConfig:
    w_approach: float = 8.0        # potential-based fingertip→coin approach progress (Φ = −mid_to_coin)
    w_contact: float = 0.5         # one-shot first valid fingertip contact
    w_stable: float = 5.0          # one-shot stable two-finger acquisition (success)
    w_loss: float = 0.5            # one-time contact-loss event
    w_stall: float = 0.02          # windowed approach stagnation
    w_collision: float = 1.0       # invalid arm-body collision / safety
    stall_eps: float = 1e-3
    stall_window: int = 8
    stable_dwell: int = 6


def acquisition_reward(prev_mid: float, mid: float, *, first_contact_now: bool, stable_now: bool, dropped: bool,
                       stalled: bool, collision: bool, cfg: AcqRewardConfig) -> float:
    """Potential-based acquisition reward (Φ = −mid_to_coin). Approach progress + one-shot contact/stable events −
    loss − stall − collision. No accumulating reward for merely maintaining contact (anti-farming)."""
    r = cfg.w_approach * (prev_mid - mid)
    if first_contact_now:
        r += cfg.w_contact
    if stable_now:
        r += cfg.w_stable
    if dropped:
        r -= cfg.w_loss
    if stalled:
        r -= cfg.w_stall
    if collision:
        r -= cfg.w_collision
    return float(r)


class AcquisitionRLEnv:
    """Acquisition-subtask RolloutEnv over a pool of hard pre-contact states. success = stable two-finger dwell."""

    def __init__(self, seeds, *, horizon: int = 120, cfg: AcqRewardConfig | None = None, seed: int = 0,
                 scramble_perm: "np.ndarray | None" = None) -> None:
        self.env = _env()
        self.inner = self.env._env
        self.pool = list(seeds)
        self.horizon = horizon
        self.max_steps = horizon                       # for off-policy trainers' generic eval hook
        self.cfg = cfg or AcqRewardConfig()
        self._rng = np.random.default_rng(seed)
        self._scramble = scramble_perm
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _read(self, obs: np.ndarray) -> np.ndarray:
        return obs if self._scramble is None else scramble_geometry(obs, self._scramble)

    def reset(self, *, seed: "int | None" = None) -> tuple[np.ndarray, dict]:
        pick = int(self._rng.choice(self.pool)) if seed is None else int(self.pool[seed % len(self.pool)])
        obs, info = self.env.reset(seed=pick)
        self.env._horizon = self.horizon + 4
        self._t = 0
        self._had_contact = False
        self._had_both = False
        self._prev_both = False
        self._dropped_once = False
        self._stag = 0
        self._dwell = 0
        self._prev_mid = float(np.hypot(*np.asarray(obs)[list(MID_TO_COIN)]))
        self._last_obs = np.asarray(obs, np.float32)
        return self._read(self._last_obs), info

    def _reward_flags(self, info: dict, mid: float) -> tuple:
        """The acquisition-reward event flags this step: (first_contact, stable, dropped, stalled, collision), updating
        the run-state (dwell / contact / drop-edge / stagnation counters)."""
        m = self.inner._planar_metrics
        left, right, both = bool(m.left_contact), bool(m.right_contact), bool(m.left_contact and m.right_contact)
        first_contact_now = (not self._had_contact) and (left or right)
        self._had_contact = self._had_contact or left or right
        self._stag = self._stag + 1 if abs(self._prev_mid - mid) < self.cfg.stall_eps else 0
        stalled = self._stag >= self.cfg.stall_window and not self._had_both
        dropped = (not self._dropped_once) and self._prev_both and (not both)
        self._dropped_once = self._dropped_once or dropped
        self._dwell = self._dwell + 1 if both else 0
        stable_now = self._dwell >= self.cfg.stable_dwell
        collision = bool(self._last_obs[ARM_BODY] > 0.5) or bool(info.get("safety_violation"))
        self._prev_both = both
        self._had_both = self._had_both or both
        return first_contact_now, stable_now, dropped, stalled, collision

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _r, _term, _trunc, info = self.env.step(np.clip(action, -1, 1).astype(np.float32))
        self._last_obs = np.asarray(obs, np.float32)
        self._t += 1
        mid = float(np.hypot(*self._last_obs[list(MID_TO_COIN)]))
        first_contact_now, stable_now, dropped, stalled, collision = self._reward_flags(info, mid)
        reward = acquisition_reward(self._prev_mid, mid, first_contact_now=first_contact_now, stable_now=stable_now,
                                    dropped=dropped, stalled=stalled, collision=collision, cfg=self.cfg)
        self._prev_mid = mid
        terminated = bool(stable_now or info.get("safety_violation"))
        truncated = bool(self._t >= self.horizon)
        return self._read(self._last_obs), reward, terminated, truncated, {"stable_acquisition": stable_now,
                                                                           "first_contact": self._had_contact}


def eval_acq_rate(policy_action, seeds, *, horizon: int = 120, scramble_perm=None) -> dict:
    """Roll a greedy policy (obs->action) on the acquisition env for each seed; report stable-acquisition rate."""
    env = AcquisitionRLEnv(seeds, horizon=horizon, scramble_perm=scramble_perm)
    n_stable = 0
    for i in range(len(seeds)):
        obs, _info = env.reset(seed=i)
        ok = False
        for _ in range(horizon):
            obs, _r, term, trunc, info = env.step(policy_action(obs))
            ok = ok or bool(info["stable_acquisition"])
            if term or trunc:
                break
        n_stable += int(ok)
    return {"n": len(seeds), "n_stable": n_stable, "rate": round(n_stable / max(1, len(seeds)), 4)}
