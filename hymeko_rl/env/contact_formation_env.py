"""Contact-formation micro-MDP (v12) — a short-horizon gym env that starts from sampled PRE-CONTACT states and asks a
cooperative controller to FORM balanced two-fingertip contact, terminating at a handoff-ready state or a safety
violation. Observations are the canonical z_team ACTOR subset; actions are the 6-d cooperative contact action mapped
to motors by CooperativeContactController; the reward is the contact-formation cooperative objective (separate from
the frozen TaskMonitor). This is the training substrate for contact-SAC / TD3 / ES — env-agnostic, so `train_sac`
runs on it unchanged."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from hymeko_rl.agents.cooperative_contact_controller import ACTION_DIM, CooperativeContactController
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar
from hymeko_rl.eval.team_tensor import ACTOR_FIELDS, TeamTensorTracker, project
from hymeko_rl.train.contact_formation_objective import (
    ContactFormationConfig,
    contact_step_reward,
    hold_shaping,
    is_handoff_ready,
    is_safety_violation,
)

_HIGH_COIN_Y = 0.16


def sample_precontact_snapshots(env: Any, mlp: Any, seeds: tuple[int, ...], *, near_band: float = 0.07,
                                max_steps: int = 120) -> list[dict]:
    """Roll the frozen approach MLP on each seed until PRE-CONTACT (a fingertip within ``near_band`` of the coin but
    not yet both-contact); snapshot that state. Returns ``[{snap, coin_y, tag}]`` — the micro-MDP start bank."""
    bank: list[dict] = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        coin_y = float(env._planar_metrics.disk_pos[1])
        snap = None
        for _t in range(max_steps):
            m = env._planar_metrics
            min_tip = min(float(m.left_tip_dist), float(m.right_tip_dist))
            if min_tip < near_band and not (m.left_contact and m.right_contact):
                snap = snapshot_planar(env)
                break
            with torch.no_grad():
                a = mlp.action_mean(torch.as_tensor(np.asarray(env.node_features(), np.float32)[None]))[0].numpy()
            obs, _r, term, trunc, _info = env.step(np.asarray(a, np.float32))
            if term or trunc:
                break
        if snap is not None:
            bank.append({"snap": snap, "coin_y": coin_y, "high_coin_y": coin_y >= _HIGH_COIN_Y})
    return bank


class ContactFormationEnv(gym.Env):
    """Short-horizon contact-formation micro-MDP over the cooperative contact action. Reset restores a random
    pre-contact snapshot; step applies the cooperative action → motors, scores the contact-formation reward, and
    terminates at a handoff-ready grip or a safety (arm/body) violation."""

    def __init__(self, planar_env: Any, bank: list[dict], contract: Any, *, horizon: int = 40,
                 cfg: "ContactFormationConfig | None" = None, seed: int = 0) -> None:
        if not bank:
            raise ValueError("contact-formation bank is empty — no pre-contact snapshots sampled")
        self._env = planar_env
        self._bank = bank
        self._contract = contract
        self._horizon = int(horizon)
        self._cfg = cfg or ContactFormationConfig()
        self._ctrl = CooperativeContactController(planar_env)
        self._tracker = TeamTensorTracker()
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(ACTOR_FIELDS),), dtype=np.float32)
        self._t = 0
        self._both_hist: list[bool] = []
        self._prev_coin = np.zeros(2)
        self._high = False
        self._streak = 0            # v12b: consecutive both-contact run
        self._of_run = 0            # v12b: consecutive one-fingertip run
        self._had_both = False      # v12b: both-contact was established at some point

    def _obs(self, action4: np.ndarray) -> np.ndarray:
        rec = self._tracker.step(self._env, action4, "CONTACT")
        return project(rec, ACTOR_FIELDS)

    def _restore(self, item: dict) -> None:
        """Restore the inner env to a bank snapshot. Canonical (K0): a direct restore. A K1 subclass overrides this to
        PAD the canonical snapshot into the larger (distal-DOF) state so the SAME held corpus is paired across K0/K1."""
        restore_planar(self._env, item["snap"])

    def reset(self, *, seed: "int | None" = None, options: Any = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        item = self._bank[int(self._rng.integers(len(self._bank)))]
        self._restore(item)
        self._tracker.reset(self._env)
        self._t = 0
        self._both_hist = []
        self._prev_coin = np.asarray(self._env._planar_metrics.disk_pos[:2], np.float64)
        self._high = bool(item["high_coin_y"])
        self._streak = 0
        self._of_run = 0
        self._had_both = False
        return self._obs(np.zeros(4, np.float32)), {"high_coin_y": self._high}

    def _physics_motor(self, arm_motor: np.ndarray) -> np.ndarray:
        """The full actuator command stepped into physics. Canonical (K0): the 4-DoF arm motor IS the whole command.
        A pad-aware subclass (K1) overrides this to append the distal pad-orientation actuators (arm servos + `_obs`
        stay driven by ``arm_motor``; only the physics command grows). # Postconditions: length == model.nu."""
        return arm_motor

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        motor = self._ctrl.action(self._env, action)
        self._env.step(self._physics_motor(motor))
        m = self._env._planar_metrics
        coin = np.asarray(m.disk_pos[:2], np.float64)
        disp = float(np.linalg.norm(coin - self._prev_coin))
        self._prev_coin = coin
        left, right = bool(m.left_contact), bool(m.right_contact)
        both = left and right
        prev_both = self._both_hist[-1] if self._both_hist else False
        self._both_hist.append(both)
        self._t += 1
        # v12b hold counters (harmless when the hold weights are 0)
        self._streak = self._streak + 1 if both else 0
        self._of_run = self._of_run + 1 if (left != right) else 0
        flicker = bool(prev_both and not both)
        lost_after_both = bool(self._had_both and not both)
        self._had_both = self._had_both or both
        reward = contact_step_reward(m, disp, self._cfg)
        reward += hold_shaping(self._streak, flicker, self._of_run, lost_after_both, self._cfg)
        handoff = is_handoff_ready(self._both_hist, self._cfg)
        safety = is_safety_violation(m)
        if handoff:
            reward += self._cfg.w_handoff
        if safety:
            reward -= self._cfg.w_body
        terminated = bool(handoff or safety)
        truncated = bool(self._t >= self._horizon)
        info = {"handoff_ready": handoff, "safety_violation": safety, "both": both, "high_coin_y": self._high}
        return self._obs(motor), reward, terminated, truncated, info
